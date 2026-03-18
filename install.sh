#!/usr/bin/env bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 9 ] || [ "$PYTHON_MINOR" -gt 10 ]; then
    echo -e "${RED}❌ Python 3.9 or 3.10 required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}\n"

# Step 1: PyTorch
echo -e "${YELLOW}[1/3] Installing PyTorch (CUDA 11.7)...${NC}"
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu117 \
    --no-cache-dir

echo -e "${GREEN}✓ PyTorch installed${NC}\n"

# Step 2: MMCV
echo -e "${YELLOW}[2/3] Installing MMCV (CUDA 11.7)...${NC}"
pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html \
    --no-cache-dir

echo -e "${GREEN}✓ MMCV installed${NC}\n"

# Step 3: ez-openmmlab
echo -e "${YELLOW}[3/3] Installing ez-openmmlab...${NC}"
pip install ez-openmmlab

echo -e "${GREEN}✓ ez-openmmlab installed${NC}\n"

# Verify installation
echo -e "${BLUE}Verifying installation...${NC}"
python3 <<'PYEOF'
import sys
try:
    import torch
    import mmcv
    import ez_openmmlab
    
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ MMCV: {mmcv.__version__}")
    print(f"✓ ez-openmmlab: {ez_openmmlab.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠  Warning: CUDA not available (CPU mode)")
        
except Exception as e:
    print(f"❌ Verification failed: {e}")
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}╔════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║    Installation completed successfully!    ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}\n"
else
    echo -e "\n${RED}Installation verification failed!${NC}\n"
    exit 1
fi
