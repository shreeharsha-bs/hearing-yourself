import os
import sys
import tempfile
import time
import torch
import torch.nn as nn
import numpy as np
import librosa
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from librosa.filters import mel as librosa_mel_fn

# Add MeanVC paths to sys.path so we can import the speaker verification init
WORKSPACE_DIR = Path("/Users/shreeharshabs/Desktop/hearing_yourself")
MEANVC_DIR = WORKSPACE_DIR / "MeanVC"
sys.path.insert(0, str(MEANVC_DIR))
sys.path.insert(0, str(MEANVC_DIR / "src"))

# Handle environment-specific 'src' package conflict by dynamically injecting MeanVC/src
try:
    import src
    src_dir = str(MEANVC_DIR / "src")
    if hasattr(src, "__path__") and src_dir not in src.__path__:
        src.__path__.append(src_dir)
except ImportError:
    pass

# Monkeypatch torchaudio.set_audio_backend to bypass s3prl deprecation crash
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None

from src.runtime.speaker_verification.verification import init_model as init_sv_model

# Initialize FastAPI App
app = FastAPI(title="MeanVC Voice Conversion Backend")

# Enable CORS for the client-side HTML to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MeanVC Model Helper Functions (copied from run_rt.py for perfect TorchScript compatibility)
def _amp_to_db(x, min_level_db):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    min_level = torch.ones_like(x) * min_level
    return 20 * torch.log10(torch.maximum(min_level, x))

def _normalize(S, max_abs_value, min_db):
    return torch.clamp((2 * max_abs_value) * ((S - min_db) / (-min_db)) - max_abs_value, -max_abs_value, max_abs_value)

class MelSpectrogramFeatures(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=1024, win_size=640, hop_length=160, n_mels=80, fmin=0, fmax=8000, center=True):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.mel_basis = {}
        self.hann_window = {}

    def forward(self, y):
        dtype_device = str(y.dtype) + '_' + str(y.device)
        fmax_dtype_device = str(self.fmax) + '_' + dtype_device
        wnsize_dtype_device = str(self.win_size) + '_' + dtype_device
        if fmax_dtype_device not in self.mel_basis:
            mel = librosa_mel_fn(sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
            self.mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
        if wnsize_dtype_device not in self.hann_window:
            self.hann_window[wnsize_dtype_device] = torch.hann_window(self.win_size).to(dtype=y.dtype, device=y.device)

        spec = torch.stft(y, self.n_fft, hop_length=self.hop_length, win_length=self.win_size, window=self.hann_window[wnsize_dtype_device],
                        center=self.center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        spec = torch.matmul(self.mel_basis[fmax_dtype_device], spec)
        spec = _amp_to_db(spec, -115) - 20
        spec = _normalize(spec, 1, -115)
        return spec

def extract_fbanks(wav, sample_rate=16000, mel_bins=80, frame_length=25, frame_shift=12.5):
    wav = wav * (1 << 15)
    wav = torch.from_numpy(wav).unsqueeze(0)
    fbanks = kaldi.fbank(
        wav,
        frame_length=frame_length,
        frame_shift=frame_shift,
        snip_edges=True,
        num_mel_bins=mel_bins,
        energy_floor=0.0,
        dither=0.0,
        sample_frequency=sample_rate,
    )
    fbanks = fbanks.unsqueeze(0)
    return fbanks

# Global model state
models_loaded = False
asr_model = None
vc_model = None
vocoder_model = None
sv_model = None
mel_extractor = None

def load_models():
    global asr_model, vc_model, vocoder_model, sv_model, mel_extractor, models_loaded
    if models_loaded:
        return
        
    print("Loading MeanVC models...")
    device = 'cpu'
    if torch.backends.mps.is_available():
        # MPS is fully supported in their TorchScript jit files, but let's default to cpu for absolute stability
        # and fallback to MPS if desired. CPU is fast enough for 14M param model anyway!
        pass
        
    try:
        asr_model = torch.jit.load(str(MEANVC_DIR / "src" / "ckpt" / "fastu2++.pt")).to(device)
        vc_model = torch.jit.load(str(MEANVC_DIR / "src" / "ckpt" / "meanvc_200ms.pt")).to(device)
        vocoder_model = torch.jit.load(str(MEANVC_DIR / "src" / "ckpt" / "vocos.pt")).to(device)
        
        sv_model_path = str(MEANVC_DIR / "src" / "runtime" / "speaker_verification" / "ckpt" / "wavlm_large_finetune.pth")
        sv_model = init_sv_model('wavlm_large', sv_model_path).to(device)
        sv_model.eval()
        
        mel_extractor = MelSpectrogramFeatures(
            sample_rate=16000, n_fft=1024, win_size=640, hop_length=160, 
            n_mels=80, fmin=0, fmax=8000, center=True
        ).to(device)
        
        models_loaded = True
        print("MeanVC models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e

# In-Memory Voice Conversion Pipeline Class
class InMemoryVCPipeline:
    def __init__(self, target_wav_16k, steps=2):
        self.steps = steps
        if self.steps == 1:
            self.timesteps = torch.tensor([1.0, 0.0])
        elif self.steps == 2:
            self.timesteps = torch.tensor([1.0, 0.8, 0.0])
        else:
            self.timesteps = torch.linspace(1.0, 0.0, self.steps + 1)
            
        decoding_chunk_size = 5
        num_decoding_left_chunks = 2
        subsampling = 4
        context = 7
        stride = subsampling * decoding_chunk_size
        self.required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        self.CHUNK = 160 * stride # 3200
        self.vc_chunk = int(decoding_chunk_size * 4) # 20
        self.vocoder_overlap = 3
        upsample_factor = 160
        self.vocoder_wav_overlap = (self.vocoder_overlap - 1) * upsample_factor # 320
        self.down_linspace = torch.linspace(1, 0, steps=self.vocoder_wav_overlap, out=None).numpy()
        self.up_linspace = torch.linspace(0, 1, steps=self.vocoder_wav_overlap, out=None).numpy()
        
        self.samples_cache_len = 720
        
        # Calculate target speaker embedding
        with torch.no_grad():
            ref_wav_tensor = torch.from_numpy(target_wav_16k).unsqueeze(0).float()
            self.vc_spk_emb = sv_model(ref_wav_tensor)
            
            prompt_mel = mel_extractor(ref_wav_tensor)
            prompt_mel = prompt_mel.transpose(1, 2)
            self.vc_prompt_mel = prompt_mel
            
        self.init_cache()

    def init_cache(self):
        self.samples_cache = None
        self.att_cache = torch.zeros((0, 0, 0, 0), device='cpu')
        self.cnn_cache = torch.zeros((0, 0, 0, 0), device='cpu')
        self.asr_offset = 0
        self.encoder_output_cache = None
        self.vc_offset = 0
        self.vc_cache = None
        self.vc_kv_cache = None
        self.vocoder_cache = None
        self.last_wav = None

    def reset_cache(self):
        self.asr_offset = 20
        self.vc_offset = 120

    def inference_one_chunk(self, samples):
        with torch.no_grad():
            if self.samples_cache is None:
                required_len = self.CHUNK + self.samples_cache_len
                if len(samples) < required_len:
                    samples = np.pad(samples, (0, required_len - len(samples)), mode='constant')
            else: 
                samples = np.concatenate((self.samples_cache, samples))
            self.samples_cache = samples[-self.samples_cache_len:]
            fbanks = extract_fbanks(samples, frame_shift=10).float()
            
            (encoder_output, self.att_cache, self.cnn_cache) = asr_model.forward_encoder_chunk(
                fbanks, self.asr_offset, self.required_cache_size, self.att_cache, self.cnn_cache)

            self.asr_offset += encoder_output.size(1)
            if self.encoder_output_cache is None:
                encoder_output = torch.cat([encoder_output[:, 0:1, :], encoder_output], dim=1)
            else:
                encoder_output = torch.cat([self.encoder_output_cache, encoder_output], dim=1)
            self.encoder_output_cache = encoder_output[:, -1:, :]
            encoder_output_upsample = encoder_output.transpose(1, 2)
            encoder_output_upsample = torch.nn.functional.interpolate(encoder_output_upsample, size=self.vc_chunk + 1, mode='linear', align_corners=True)
            encoder_output_upsample = encoder_output_upsample.transpose(1, 2)
            encoder_output_upsample = encoder_output_upsample[:, 1:, :]
            
            x = torch.randn(1, encoder_output_upsample.shape[1], 80, device='cpu', dtype=encoder_output_upsample.dtype)
            
            for i in range(self.steps):
                t = self.timesteps[i]
                r = self.timesteps[i+1]
                t_tensor = torch.full((1,), t, device=x.device)
                r_tensor = torch.full((1,), r, device=x.device)
            
                u, tmp_kv_cache = vc_model(x, t_tensor, r_tensor, cache=self.vc_cache, cond=encoder_output_upsample, spks=self.vc_spk_emb,
                    prompts=self.vc_prompt_mel, offset=self.vc_offset, kv_cache=self.vc_kv_cache)
                
                x = x - (t - r) * u
            self.vc_kv_cache = tmp_kv_cache
            self.vc_offset += x.shape[1]
            self.vc_cache = x

            VC_KV_CACHE_MAX_LEN = 100
            if self.vc_offset > 40 and self.vc_kv_cache[0][0].shape[2] > VC_KV_CACHE_MAX_LEN:
                for i in range(len(self.vc_kv_cache)):
                    new_k = self.vc_kv_cache[i][0][:, :, -VC_KV_CACHE_MAX_LEN:, :]
                    new_v = self.vc_kv_cache[i][1][:, :, -VC_KV_CACHE_MAX_LEN:, :]
                    self.vc_kv_cache[i] = (new_k, new_v)

            mel = x.transpose(1, 2)

            if self.vocoder_cache is not None:
                mel = torch.cat([self.vocoder_cache, mel], dim=-1)
            self.vocoder_cache = mel[:, :, -self.vocoder_overlap:]
            mel = (mel + 1) / 2
            wav = vocoder_model.decode(mel).squeeze()
            wav = wav.detach().cpu().numpy()
            
            if self.last_wav is not None:
                front_wav = wav[:self.vocoder_wav_overlap]
                smooth_front_wav = self.last_wav * self.down_linspace + front_wav * self.up_linspace
                new_wav = np.concatenate([smooth_front_wav, wav[self.vocoder_wav_overlap:-self.vocoder_wav_overlap]], axis=0)
            else:
                new_wav = wav[:-self.vocoder_wav_overlap]
            self.last_wav = wav[-self.vocoder_wav_overlap:]

            return new_wav

    def convert(self, source_wav_16k):
        self.init_cache()
        chunks = []
        offset = 0
        total_samples = len(source_wav_16k)
        
        i = 0
        while offset < total_samples:
            samples = source_wav_16k[offset:offset+self.CHUNK]
            
            if len(samples) < self.CHUNK:
                samples = np.pad(samples, (0, self.CHUNK - len(samples)), mode='constant')
            
            # The first chunk needs an extra 720 samples of following audio to seed cache
            if i == 0:
                next_720 = source_wav_16k[offset+self.CHUNK:offset+self.CHUNK+720]
                if len(next_720) < 720:
                    next_720 = np.pad(next_720, (0, 720 - len(next_720)), mode='constant')
                samples = np.concatenate([samples, next_720])
                offset += self.CHUNK + 720
            else:
                offset += self.CHUNK
                
            if i % 50 == 0 and i != 0:
                self.reset_cache()
                
            vc_wav = self.inference_one_chunk(samples)
            chunks.append(vc_wav)
            i += 1
            
        if self.last_wav is not None:
            chunks.append(self.last_wav)
            
        return np.concatenate(chunks)

# FastAPI Start up loading
@app.on_event("startup")
def startup_event():
    try:
        load_models()
    except Exception as e:
        print(f"Warning: Models could not be loaded at startup: {e}. They will be loaded on demand.")

# Endpoints
@app.get("/health")
def health():
    global models_loaded
    if not models_loaded:
        try:
            load_models()
        except Exception as e:
            return JSONResponse(status_code=500, content={"status": "error", "message": f"Models failing to load: {str(e)}"})
    return {"status": "ok", "models_loaded": models_loaded}

@app.get("/target_voices")
def get_target_voices():
    # Return built-in reference voices
    example_test_wav = MEANVC_DIR / "src" / "runtime" / "example" / "test.wav"
    voices = []
    if example_test_wav.exists():
        voices.append({
            "id": "builtin_test",
            "name": "Built-in Speaker (ASLP Test)",
            "gender": "Male",
            "description": "Standard zero-shot speaker packaged with MeanVC repository."
        })
    return voices

@app.post("/convert")
async def convert_audio(
    whisper: UploadFile = File(...),
    target: UploadFile = File(None),
    steps: int = Form(2)
):
    global models_loaded
    if not models_loaded:
        try:
            load_models()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MeanVC models are not loaded: {str(e)}")

    # Create temporary files to process
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as whisper_tmp, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as target_tmp:
         
        try:
            # Write uploaded whisper file
            whisper_content = await whisper.read()
            whisper_tmp.write(whisper_content)
            whisper_tmp.flush()
            whisper_path = whisper_tmp.name
            
            # Write or select target reference file
            if target is not None:
                target_content = await target.read()
                target_tmp.write(target_content)
                target_tmp.flush()
                target_path = target_tmp.name
            else:
                # Default fallback target voice (test.wav)
                builtin_path = MEANVC_DIR / "src" / "runtime" / "example" / "test.wav"
                if not builtin_path.exists():
                    raise HTTPException(status_code=400, detail="No target reference file provided and built-in fallback is missing.")
                target_path = str(builtin_path)
            
            print(f"Received conversion request: whisper={whisper.filename}, target={target.filename if target else 'BUILT-IN'}, steps={steps}")
            
            # Load and resample source whisper clip to 16kHz mono (which MeanVC requires)
            try:
                whisper_wav, sr = librosa.load(whisper_path, sr=16000)
            except Exception as e:
                # If librosa fails (e.g. browser WebM/Opus issue), try loading via torchaudio or soundfile
                print(f"Librosa load failed, attempting torchaudio fallback: {e}")
                wav_tensor, sr = torchaudio.load(whisper_path)
                # Resample if necessary
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                    wav_tensor = resampler(wav_tensor)
                whisper_wav = wav_tensor.mean(dim=0).numpy()

            # Load and resample target reference clip to 16kHz
            try:
                target_wav, sr = librosa.load(target_path, sr=16000)
            except Exception as e:
                print(f"Librosa target load failed, attempting torchaudio fallback: {e}")
                wav_tensor, sr = torchaudio.load(target_path)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                    wav_tensor = resampler(wav_tensor)
                target_wav = wav_tensor.mean(dim=0).numpy()
                
            # Initialize our in-memory conversion pipeline
            pipeline = InMemoryVCPipeline(target_wav, steps=steps)
            
            # Run conversion
            print("Running in-memory voice conversion...")
            start_time = time.time()
            converted_wav = pipeline.convert(whisper_wav)
            print(f"Conversion complete! Processed in {time.time() - start_time:.2f} seconds.")
            
            # Save the resulting WAV to a temporary output file
            output_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = output_tmp.name
            output_tmp.close()
            
            # Save the numpy array as a 16kHz mono wav file using soundfile
            import soundfile as sf
            sf.write(output_path, converted_wav, 16000)
            
            # Return the generated WAV file directly
            return FileResponse(
                output_path,
                media_type="audio/wav",
                filename="converted_speech.wav"
            )
            
        except Exception as e:
            print(f"Error in conversion: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Conversion error: {str(e)}")
            
        finally:
            # Cleanup temp files
            try:
                os.unlink(whisper_tmp.name)
                os.unlink(target_tmp.name)
            except OSError:
                pass

@app.post("/upload_target")
async def upload_target(target: UploadFile = File(...)):
    # Create a unique filename in temp directory
    suffix = Path(target.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await target.read()
        tmp.write(content)
        tmp.flush()
        tmp_name = tmp.name
    print(f"Uploaded custom target voice saved to: {tmp_name}")
    return {"target_id": tmp_name}

@app.websocket("/ws/stream")
async def websocket_endpoint(
    websocket: WebSocket,
    target_id: str = "builtin_test",
    steps: int = 2
):
    await websocket.accept()
    print(f"WebSocket client connected! target_id={target_id}, steps={steps}")
    
    # Initialize pipeline
    global models_loaded
    if not models_loaded:
        try:
            load_models()
        except Exception as e:
            await websocket.close(code=1011, reason=f"Models failing to load: {str(e)}")
            return
            
    # Resolve target voice path
    target_path = None
    if target_id == "builtin_test" or target_id == "builtin":
        builtin_path = MEANVC_DIR / "src" / "runtime" / "example" / "test.wav"
        if builtin_path.exists():
            target_path = str(builtin_path)
    elif os.path.exists(target_id):
        target_path = target_id
    else:
        # Fallback to built-in test voice
        builtin_path = MEANVC_DIR / "src" / "runtime" / "example" / "test.wav"
        if builtin_path.exists():
            target_path = str(builtin_path)
            
    if target_path is None:
        await websocket.close(code=1008, reason="Built-in target voice reference missing.")
        return
        
    try:
        # Load reference speaker wav resampled to 16kHz
        try:
            target_wav, sr = librosa.load(target_path, sr=16000)
        except Exception as e:
            print(f"Librosa target load failed, attempting torchaudio fallback: {e}")
            wav_tensor, sr = torchaudio.load(target_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                wav_tensor = resampler(wav_tensor)
            target_wav = wav_tensor.mean(dim=0).numpy()
            
        pipeline = InMemoryVCPipeline(target_wav, steps=steps)
        chunk_count = 0
        
        while True:
            # Receive binary float32 PCM chunk (3200 samples * 4 bytes = 12800 bytes)
            data = await websocket.receive_bytes()
            if len(data) == 0:
                continue
                
            pcm_chunk = np.frombuffer(data, dtype=np.float32)
            
            # Run in-memory voice conversion for this single chunk
            vc_wav = pipeline.inference_one_chunk(pcm_chunk)
            
            # Keep attention cache stable by resetting periodically (every 50 chunks / 10s)
            chunk_count += 1
            if chunk_count % 50 == 0:
                pipeline.reset_cache()
                
            # Send back raw Float32 PCM bytes
            await websocket.send_bytes(vc_wav.astype(np.float32).tobytes())
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"Error in WebSocket streaming: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
