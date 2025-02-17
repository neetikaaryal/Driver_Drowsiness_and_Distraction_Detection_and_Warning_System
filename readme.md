# Driver Drowsiness Detection with WebRTC

This project implements a driver drowsiness detection system using aiortc for WebRTC streaming. The system processes webcam video, detects drowsiness based on eye aspect ratio (EAR), and alerts the user when signs of fatigue are detected.

---

## Installation Guide

### 1. Install Python 3.11
Ensure you have Python 3.11 installed on your system. You can check your version with:

```bash
python3 --version
```

If you don't have Python 3.11, download and install it from [python.org](https://www.python.org/downloads/release/python-3110/).

Alternatively, you can install it using a package manager:

- **Ubuntu/Linux**:
  ```bash
  sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev
  ```
- **MacOS** (with Homebrew):
  ```bash
  brew install python@3.11
  ```
- **Windows**:
  Download the installer from [python.org](https://www.python.org/downloads/windows/) and ensure `Add Python to PATH` is checked.

### 2. Create a Virtual Environment

After installing Python 3.11, create a virtual environment to isolate dependencies:

```bash
python3.11 -m venv venv
```

Activate the virtual environment:

- **Linux/macOS**:
  ```bash
  source venv/bin/activate
  ```
- **Windows (Command Prompt)**:
  ```cmd
  venv\Scripts\activate
  ```
- **Windows (PowerShell)**:
  ```powershell
  venv\Scripts\Activate.ps1
  ```

### 3. Install Dependencies

Once the virtual environment is activated, install the required Python packages:

```bash
pip install --upgrade pip
pip install aiohttp aiortc opencv-python mediapipe numpy scipy av
```

---

## Running the Program

```bash 
python main4.py
```