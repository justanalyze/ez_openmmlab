#!/usr/bin/env bash
set -e

echo "════════════════════════════════════════"
echo "  Building ez-openmmlab for PyPI"
echo "════════════════════════════════════════"
echo ""

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info src/*.egg-info

# Build the package
echo "📦 Building wheel..."
uv build --wheel

echo ""
echo "Built packages:"
ls -lh dist/

echo ""
echo "════════════════════════════════════════"
echo "  Testing the build"
echo "════════════════════════════════════════"
echo ""

# Create test environment
echo "Creating test environment..."
python3 -m venv test_env
source test_env/bin/activate

# Install PyTorch first (required)
echo "Installing PyTorch (CPU for testing)..."
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu -q

# Install MMCV
echo "Installing MMCV..."
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html -q

# Install the wheel
echo "Installing ez-openmmlab from wheel..."
pip install dist/*.whl -q

# Test import
echo "Testing import..."
python3 -c "import ez_openmmlab; print(f'✓ Successfully imported ez-openmmlab {ez_openmmlab.__version__}')"

deactivate
rm -rf test_env

echo ""
echo "════════════════════════════════════════"
echo "  Publish to PyPI?"
echo "════════════════════════════════════════"
echo ""
echo "Options:"
echo "  1) Test on TestPyPI first (recommended)"
echo "  2) Publish to PyPI directly"
echo "  3) Cancel"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
1)
    echo ""
    echo "Publishing to TestPyPI..."
    uv publish --publish-url https://test.pypi.org/legacy/
    echo ""
    echo "✓ Published to TestPyPI!"
    echo ""
    echo "Test install with:"
    echo "  pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu"
    echo "  pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html"
    echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple ez-openmmlab"
    ;;
2)
    echo ""
    read -p "Are you sure you want to publish to PyPI? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
        echo "Publishing to PyPI..."
        uv publish
        echo ""
        echo "✓ Published to PyPI!"
        echo ""
        echo "Users can now install with:"
        echo "  curl -sSL https://raw.githubusercontent.com/JustAnalyze/ez_openmmlab/main/install.sh | bash"
    else
        echo "Cancelled."
    fi
    ;;
*)
    echo "Cancelled."
    ;;
esac
