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
