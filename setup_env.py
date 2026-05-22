import os
import sys
import subprocess
import urllib.request
import re
from pathlib import Path

def run_cmd(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {result.stderr}")
        raise RuntimeError(result.stderr)
    return result.stdout.strip()

def download_from_google_drive(file_id, destination):
    print(f"Downloading Google Drive file '{file_id}' to '{destination}'...")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Google Drive download link
    url = "https://docs.google.com/uc?export=download"
    
    # Create request session or download using urllib
    session_url = f"{url}&id={file_id}"
    req = urllib.request.Request(session_url, headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        # First request to check for virus scan warning page (confirm token)
        with urllib.request.urlopen(req) as response:
            content = response.read()
            html = content.decode('utf-8', errors='ignore')
            
            # Find the confirmation token
            confirm_token = None
            # Google Drive large file warning usually contains a form or link with confirm=XXXX
            match = re.search(r'confirm=([0-9A-Za-z_-]+)', html)
            if match:
                confirm_token = match.group(1)
                
            if confirm_token:
                download_url = f"{url}&id={file_id}&confirm={confirm_token}"
            else:
                # If no token is found, maybe it's direct or we need to look closer
                download_url = session_url
                
            print(f"Constructed download URL: {download_url}")
            
            # Download file
            download_req = urllib.request.Request(download_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(download_req) as download_response, open(destination, 'wb') as f:
                # Get file size if available
                file_size = download_response.info().get('Content-Length')
                if file_size:
                    file_size = int(file_size)
                    print(f"File size: {file_size / (1024*1024):.2f} MB")
                
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB
                while True:
                    chunk = download_response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if file_size:
                        print(f"Downloaded: {downloaded / file_size * 100:.1f}% ({downloaded / (1024*1024):.1f} MB)", end="\r", flush=True)
                    else:
                        print(f"Downloaded: {downloaded / (1024*1024):.1f} MB", end="\r", flush=True)
                        
            print(f"\nSuccessfully downloaded to {destination}!")
            
    except Exception as e:
        print(f"Error downloading Google Drive file: {e}")
        # Fallback to direct curl download
        print("Attempting fallback using curl...")
        try:
            # We can curl it
            curl_cmd = [
                "curl", "-L",
                "-o", str(destination),
                f"https://docs.google.com/uc?export=download&id={file_id}"
            ]
            run_cmd(curl_cmd)
            print("Successfully downloaded via fallback curl!")
        except Exception as curl_err:
            print(f"Fallback curl also failed: {curl_err}")
            raise e

def main():
    workspace = Path("/Users/shreeharshabs/Desktop/hearing_yourself")
    meanvc_dir = workspace / "MeanVC"
    
    # Use User's existing Conda environment 'qwen2'
    print("--- 1. Selecting Conda Environment 'qwen2' ---")
    conda_python = Path("/Users/shreeharshabs/miniconda3/envs/qwen2/bin/python")
    
    if not conda_python.exists():
        print(f"Conda python at {conda_python} not found. Falling back to local .venv...")
        venv_dir = workspace / ".venv"
        if not venv_dir.exists():
            run_cmd(["uv", "venv", str(venv_dir)], cwd=str(workspace))
        conda_python = venv_dir / "bin" / "python"
        
    print(f"Using Python executable: {conda_python}")

    # 2. Install Missing Dependencies in Conda Environment
    print("\n--- 2. Installing Missing Dependencies ---")
    dependencies = [
        "librosa",
        "einops",
        "x-transformers",
        "tqdm",
        "PyYAML",
        "omegaconf",
        "transformers",
        "accelerate",
        "huggingface-hub",
        "fastapi",
        "uvicorn",
        "python-multipart"
    ]
    
    # We use uv to install dependencies inside the conda environment!
    run_cmd(["uv", "pip", "install", "--python", str(conda_python)] + dependencies, cwd=str(workspace))
    print("Dependencies verified and installed successfully!")

    # 3. Download HuggingFace Checkpoints
    print("\n--- 3. Downloading Hugging Face Models ---")
    # We run the python script via our conda environment python
    run_cmd([str(conda_python), "download_ckpt.py"], cwd=str(meanvc_dir))
    print("Hugging Face models downloaded successfully!")

    # 4. Download Speaker Verification Model from Google Drive
    print("\n--- 4. Downloading Speaker Verification Model (WavLM) ---")
    gdrive_file_id = "1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP"
    destination_path = meanvc_dir / "src" / "runtime" / "speaker_verification" / "ckpt" / "wavlm_large_finetune.pth"
    
    if not destination_path.exists():
        download_from_google_drive(gdrive_file_id, destination_path)
    else:
        print("Speaker verification model (WavLM) already exists. Skipping download.")

    print("\n--- Setup Complete! Environment is fully ready! ---")

if __name__ == "__main__":
    main()
