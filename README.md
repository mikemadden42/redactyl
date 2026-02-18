# Redactyl

Redactyl is a hardware-accelerated AI privacy shield for Linux & Windows. It uses your Intel Core Ultra NPU to scan your screen in real-time and automatically redacts sensitive information (API keys, passwords, secrets) before they can be leaked during screen shares or recordings.

## Prerequisites

### 1. Hardware Drivers

**For Ubuntu 24.04+:**

To utilize the Intel AI Boost NPU and Arc GPU, you must install the user-space compute runtimes. OpenVINO also requires `libxcb` to render the PyQt6 overlay.

```bash
sudo apt update && sudo apt install -y \
    intel-opencl-icd libze-intel-gpu1 libze1 libtbb12 \
    libxcb-cursor0

# Grant hardware access to your user
sudo usermod -aG render,video $USER

```

*Note: A system reboot is recommended after changing user groups.*

**For Windows 11:**
Simply download and install the official [Intel® NPU Driver](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html). No additional system packages are required.

### 2. Python Environment

This project uses `uv` for lightning-fast dependency management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

```

---

## Getting Started

### 1. Bootstrap the Models

Run the setup script to fetch the OCR models. This script will download the CPU recognition model and compile the detection model specifically for your NPU.

```bash
uv run src/load_models.py

```

### 2. Verify Hardware

Ensure OpenVINO can see your NPU and GPU:

```bash
uv run src/check_hardware.py

```

### 3. Launch the Shield

Run the main application:

```bash
uv run src/redactyl.py

```

* **Normal Mode:** Solid red redaction on detected secrets.
* **Debug Mode:** `uv run src/redactyl.py --show-all` (Shows green boxes around all detected text).

---

## Shortcuts & Aliases

To launch Redactyl instantly from any terminal, add this alias to your `~/.bashrc` or `~/.zshrc`:

```bash
alias shield-up='cd ~/src/redactyl && uv run src/redactyl.py'

```

---

## Monitoring

**On Linux:**

To see your NPU in action, you can use the [nputop](https://github.com/ZoLArk173/nputop) tool:

```bash
cd nputop
cargo run
```
**On Windows 11:**

Open the Task Manager, navigate to the Performance tab, and scroll down to view the native Intel(R) AI Boost (NPU) usage graphs.

## V2 Roadmap

* **Virtual Camera Architecture:** Pipe redacted video to a virtual webcam to avoid the "Observer Effect".
* **Optical Flow Tracking:** Stick redaction boxes to moving windows smoothly.
* **Application Targeting:** Only scan specific windows (e.g., VS Code or Terminal) to save battery.
