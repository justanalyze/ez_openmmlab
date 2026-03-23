#!/usr/bin/env bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}"
cat <<"EOF"
╔════════════════════════════════════════════╗
║                                            ║
║       ez-openmmlab Installation            ║
║          GPU/CUDA 11.7 Setup               ║
║                                            ║
╚════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}╔════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ⚠️  WARNING: No virtual environment!  ⚠️  ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════╝${NC}\n"
    echo -e "${YELLOW}Please create and activate a virtual environment first:${NC}\n"
    echo -e "  ${CYAN}uv venv${NC}"
    echo -e "  ${CYAN}source .venv/bin/activate${NC}\n"
    exit 1
fi

echo -e "${GREEN}✓ Virtual environment detected${NC}\n"

# Check for uv
if ! command -v uv &>/dev/null; then
    echo -e "${RED}╔════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║          ⚠️  uv not found!  ⚠️             ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════╝${NC}\n"
    echo -e "${YELLOW}Please install uv first:${NC}\n"
    echo -e "  ${CYAN}curl -LsSf https://astral.sh/uv/install.sh | sh${NC}\n"
    echo -e "${YELLOW}Then restart your terminal and try again.${NC}\n"
    exit 1
fi

echo -e "${GREEN}✓ uv detected${NC}\n"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 9 ] || [ "$PYTHON_MINOR" -gt 10 ]; then
    echo -e "${RED}❌ Python 3.9 or 3.10 required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}\n"

# Installation
echo -e "${YELLOW}[1/5] Installing PyTorch (CUDA 11.7)...${NC}"
uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu117
echo -e "${GREEN}✓ PyTorch installed${NC}\n"

echo -e "${YELLOW}[2/5] Installing MMCV (CUDA 11.7)...${NC}"
uv pip install mmcv==2.1.0 \
    --find-links https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
echo -e "${GREEN}✓ MMCV installed${NC}\n"

echo -e "${YELLOW}[3/5] Installing setuptools...${NC}"
uv pip install setuptools wheel
echo -e "${GREEN}✓ setuptools installed${NC}\n"

echo -e "${YELLOW}[4/5] Installing chumpy...${NC}"
uv pip install git+https://github.com/JustAnalyze/chumpy.git@master
echo -e "${GREEN}✓ chumpy installed${NC}\n"

echo -e "${YELLOW}[5/5] Installing ez-openmmlab...${NC}"
uv pip install ez-openmmlab
echo -e "${GREEN}✓ ez-openmmlab installed${NC}\n"

# Verify
echo -e "${BLUE}Verifying installation...${NC}"
python3 <<'PYEOF'
import sys
try:
    import torch, mmcv, ez_openmmlab
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ MMCV: {mmcv.__version__}")
    print(f"✓ ez-openmmlab: {ez_openmmlab.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ Verification failed: {e}")
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}╔════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║         Installation Successful! 🎉        ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}\n"
else
    echo -e "\n${RED}Installation failed!${NC}\n"
    exit 1
fi
