#!/usr/bin/env bash
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
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

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 9 ] || [ "$PYTHON_MINOR" -gt 10 ]; then
    echo -e "${RED}❌ Python 3.9 or 3.10 required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}\n"

# Check if uv is available
UV_AVAILABLE=false
if command -v uv &>/dev/null; then
    UV_AVAILABLE=true
fi

# Ask user to choose installation method
echo -e "${CYAN}Choose installation method:${NC}"
if [ "$UV_AVAILABLE" = true ]; then
    echo -e "  ${GREEN}1)${NC} uv (recommended - faster)"
    echo -e "  ${YELLOW}2)${NC} pip (traditional)"
else
    echo -e "  ${YELLOW}1)${NC} pip (uv not detected)"
    echo -e "  ${GREEN}2)${NC} Install uv first, then use uv"
fi
echo ""
read -p "Enter choice [1-2]: " install_choice

# Install uv if requested
if [ "$UV_AVAILABLE" = false ] && [ "$install_choice" = "2" ]; then
    echo -e "\n${CYAN}Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo -e "${GREEN}✓ uv installed${NC}\n"
    UV_AVAILABLE=true
    install_choice="1"
fi

# Determine which tool to use
if [ "$UV_AVAILABLE" = true ] && [ "$install_choice" = "1" ]; then
    INSTALLER="uv pip"
    echo -e "\n${GREEN}Using uv for installation (faster!)${NC}\n"
else
    INSTALLER="pip"
    echo -e "\n${YELLOW}Using pip for installation${NC}\n"
    # Ensure setuptools is available for pip
    echo -e "${YELLOW}Ensuring setuptools is available...${NC}"
    pip install --upgrade setuptools -q
fi

# Step 1: PyTorch
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}[1/4] Installing PyTorch (CPU)...${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
$INSTALLER install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cpu

echo -e "${GREEN}✓ PyTorch installed${NC}\n"

# Step 2: MMCV
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}[2/4] Installing MMCV (CPU)...${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [ "$INSTALLER" = "uv pip" ]; then
    $INSTALLER install mmcv==2.1.0 \
        --find-links https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
else
    $INSTALLER install mmcv==2.1.0 \
        -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html \
        --no-cache-dir
fi

echo -e "${GREEN}✓ MMCV installed${NC}\n"

# Step 3: setuptools (if using uv, needed for mmengine)
if [ "$INSTALLER" = "uv pip" ]; then
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Installing setuptools (required by mmengine)...${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    $INSTALLER install setuptools wheel
    echo -e "${GREEN}✓ setuptools installed${NC}\n"
fi

# Step 4: chumpy (fixed version)
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}[3/4] Installing chumpy (fixed version)...${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
$INSTALLER install git+https://github.com/JustAnalyze/chumpy.git@master

echo -e "${GREEN}✓ chumpy installed${NC}\n"

# Step 5: ez-openmmlab
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}[4/4] Installing ez-openmmlab...${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
$INSTALLER install ez-openmmlab

echo -e "${GREEN}✓ ez-openmmlab installed${NC}\n"

# Verify
echo -e "${BLUE}"
echo "════════════════════════════════════════"
echo "  Verifying Installation"
echo "════════════════════════════════════════"
echo -e "${NC}"

python3 <<'PYEOF'
import sys
try:
    import torch
    import mmcv
    import ez_openmmlab
    
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ MMCV: {mmcv.__version__}")
    print(f"✓ ez-openmmlab: {ez_openmmlab.__version__}")
    
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
