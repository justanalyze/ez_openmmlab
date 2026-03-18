#!/usr/bin/env bash
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
cat <<"EOF"
╔════════════════════════════════════════════╗
║                                            ║
║       ez-openmmlab Installation            ║
║             CPU Setup                      ║
║                                            ║
╚════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${YELLOW}[1/3] Installing PyTorch (CPU)...${NC}"
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cpu \
    --no-cache-dir

echo -e "${GREEN}✓ PyTorch installed${NC}\n"

echo -e "${YELLOW}[2/3] Installing MMCV (CPU)...${NC}"
pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html \
    --no-cache-dir

echo -e "${GREEN}✓ MMCV installed${NC}\n"

echo -e "${YELLOW}[3/3] Installing ez-openmmlab...${NC}"
pip install ez-openmmlab

echo -e "${GREEN}✓ ez-openmmlab installed${NC}\n"

# Verify
echo -e "${BLUE}Verifying installation...${NC}"
python3 -c "import torch, mmcv, ez_openmmlab; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ MMCV: {mmcv.__version__}'); print(f'✓ ez-openmmlab: {ez_openmmlab.__version__}')"

echo -e "\n${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║    Installation completed successfully!    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}\n"
