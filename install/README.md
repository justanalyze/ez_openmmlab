# Installation Guide

## Prerequisites

### 1. Install uv (Required)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> **Why uv?** uv is 10-100x faster than pip and provides better dependency resolution. It's the modern standard for Python package management.

### 2. Create Virtual Environment

```bash
# Create virtual environment
uv venv

# Activate (Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

---

## Quick Install

### GPU (CUDA 11.7)

```bash
curl -sSL https://raw.githubusercontent.com/JustAnalyze/ez_openmmlab/main/install/install-gpu.sh | bash
```

### CPU

```bash
curl -sSL https://raw.githubusercontent.com/JustAnalyze/ez_openmmlab/main/install/install-cpu.sh | bash
```

---
## Manual Installation (Without uv)

If you prefer not to use uv, you can install manually with pip:

### Step 0: Create Virtual Environment

**Using Python 3.10 (Recommended):**
```bash
# Create virtual environment with Python 3.10
python3.10 -m venv .venv

# Activate (Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools==80 wheel
```

**Using Python 3.9:**
```bash
# Create virtual environment with Python 3.9
python3.9 -m venv .venv

# Activate (Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools==80 wheel
```

> ⚠️ **Important:** Make sure you're in the activated virtual environment (you should see `(.venv)` in your terminal prompt) before proceeding with the installation steps below.

---

### GPU (CUDA 11.7)
```bash
# Step 1: Install PyTorch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/torch_stable.html

# Step 2: Install MMCV
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html

# Step 3: Install chumpy
pip install git+https://github.com/JustAnalyze/chumpy.git@master

# Step 4: Install ez-openmmlab
pip install ez-openmmlab
```

### CPU
```bash
# Step 1: Install PyTorch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/torch_stable.html

# Step 2: Install MMCV
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html

# Step 3: Install chumpy
pip install git+https://github.com/JustAnalyze/chumpy.git@master

# Step 4: Install ez-openmmlab
pip install ez-openmmlab
```

---

## Requirements

- Python 3.9 or 3.10
- uv (for automated installation)
- Virtual environment
- NVIDIA GPU with CUDA 11.7 (for GPU version)
- Git

---

## Troubleshooting

### Virtual environment not activated

Make sure you see `(.venv)` at the beginning of your terminal prompt. If not:
```bash
source .venv/bin/activate  # Linux
.venv\Scripts\activate      # Windows
```

### uv not found after installation

Restart your terminal or run:
```bash
source $HOME/.cargo/env
```

### Python version issues

Check your Python version:
```bash
python --version
```

If you need to use a specific Python version (3.9 or 3.10), create the virtual environment with that version explicitly:
```bash
python3.10 -m venv .venv  # For Python 3.10
# or
python3.9 -m venv .venv   # For Python 3.9
```
