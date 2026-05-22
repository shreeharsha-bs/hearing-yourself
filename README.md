# ✧ Are you hearing yourself? Zero-Shot Whisper-to-Voiced Speech Converter


---

## 🚀 Quick Start Guide

### 1. Requirements & Setup
Ensure you have `miniconda` or a standard Python 3.10+ environment installed.

```bash
# Clone the repository
git clone https://github.com/shreeharshabs/hearing-yourself.git
cd hearing-yourself

# Set up environment and dependencies
python setup_env.py
```

### 2. Start the Backend Server
The FastAPI backend serves the offline synthesis API and the streaming WebSocket endpoints on port `8000`.

```bash
python server.py
```

### 3. Launch the Dashboard
Open the `aaf_voice_masking_demo.html` file in any modern web browser:

```bash
# macOS shortcut to open in default browser
open aaf_voice_masking_demo.html
```

---

## 🧪 Usage Instructions

### Offline Conversion
1. Connect your headphones.
2. Click **Record Whisper** and whisper a short phrase (up to 15s). Click **Stop Recording**.
3. Under **Reference Target Voice**, choose a built-in speaker profile or upload your own custom `.wav` target clip.
4. Click **✧ Synthesize Voiced Speech ✧**.
5. Once synthesized, click **Play via Headphones (Wet)** to route the converted voice through the live DSP slider chain!

### Live Neural Stream
1. Connect your headphones (essential to avoid feedback howling!).
2. In the **Live Neural Stream** panel, click **✧ Go Live (Start Stream) ✧**.
3. Speak whispers directly into your microphone. Converted voiced speech will play back instantly.
4. Adjust the **Pitch Shift**, **Formant Shift (VTL)**, or **Reverb** sliders on the dashboard in real time!
