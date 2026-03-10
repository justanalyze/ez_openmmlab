#!/bin/bash

# ez_openmmlab Installation Script
# This script automates the installation and bootstrapping process to ensure
# all complex build dependencies for OpenMMLab are handled correctly.

set -e

# --- Configuration ---
PROJECT_NAME="ez_openmmlab"
REQUIRED_UV_VERSION="0.1.0" # Minimum recommended

# --- Colors for Output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Welcome to the ez_openmmlab Installer!${NC}"

# 1. Check for uv
if ! command -v uv &>/dev/null; then
    echo -e "${RED}❌ 'uv' is not installed.${NC}"
    echo -e "Please install it first: ${YELLOW}curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
fi

# 2. Detect Hardware / Ask User
echo -e "\n${YELLOW}Select your installation type:${NC}"
echo "1) GPU (NVIDIA CUDA 11.7)"
echo "2) CPU"
read -p "Enter choice [1-2]: " choice

case $choice in
1)
    EXTRA="gpu"
    TORCH_INDEX="https://download.pytorch.org/whl/cu117"
    echo -e "\n✅ ${GREEN}Targeting GPU (CUDA 11.7)${NC}"
    ;;
2)
    EXTRA="cpu"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    echo -e "\n✅ ${GREEN}Targeting CPU${NC}"
    ;;
*)
    echo -e "${RED}Invalid choice. Exiting.${NC}"
    exit 1
    ;;
esac

# 3. Bootstrap Build Dependencies (The "EZ" Insurance)
echo -e "\n${YELLOW}🛠️  Bootstrapping build dependencies...${NC}"
# uv pip install will automatically create a .venv if one doesn't exist
uv pip install setuptools==80 wheel

# 4. Install PyTorch + Core Vision Libraries
echo -e "\n${YELLOW}🔥 Installing PyTorch stack (${EXTRA})...${NC}"
uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url $TORCH_INDEX

# 5. Final Project Sync
echo -e "\n${YELLOW}🔄 Finalizing project synchronization...${NC}"
# This will install mmdet, mmpose (as editables), mmcv, and other dependencies
uv sync --extra $EXTRA

# 6. Verification
echo -e "\n${YELLOW}🔍 Verifying installation...${NC}"
if uv run --extra $EXTRA python -c "import ez_openmmlab; import mmdet; import mmpose; print('Verification Successful!')" &>/dev/null; then
    echo -e "${GREEN}✅ Installation complete and verified!${NC}"
    echo -e "\nTo get started, try running one of the demos:"
    echo -e "${YELLOW}uv run --extra $EXTRA python demos/demo_rtmpose.py${NC}"
    echo -e "\nOr try the CLI:"
    echo -e "${YELLOW}uv run --extra $EXTRA ez-mmlab predict rtmdet_tiny demos/demo.jpg${NC}"
else
    echo -e "${RED}❌ Verification failed.${NC} There might be an issue with the environment."
    echo -e "Please check the error logs or try: ${YELLOW}uv sync --extra $EXTRA --force-reinstall${NC}"
fi

echo -e "\n${GREEN}Enjoy using ez_openmmlab!${NC}"
