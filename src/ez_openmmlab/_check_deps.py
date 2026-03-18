"""Runtime dependency checker for PyTorch and MMCV."""

import sys
from typing import NoReturn


def _print_install_instructions() -> NoReturn:
    """Print installation instructions and exit."""
    print("\n" + "=" * 70)
    print("  ez-openmmlab requires PyTorch and MMCV to be installed first")
    print("=" * 70)
    print("\n🚀 Quick Install (GPU/CUDA 11.7):")
    print(
        "\n  curl -sSL https://raw.githubusercontent.com/JustAnalyze/ez_openmmlab/main/install.sh | bash"
    )
    print("\n📦 Or install manually:")
    print("\n  GPU (CUDA 11.7):")
    print("  ---------------")
    print("  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \\")
    print("      --index-url https://download.pytorch.org/whl/cu117")
    print("  pip install mmcv==2.1.0 \\")
    print("      -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html")
    print("  pip install ez-openmmlab")
    print("\n  CPU:")
    print("  ----")
    print("  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \\")
    print("      --index-url https://download.pytorch.org/whl/cpu")
    print("  pip install mmcv==2.1.0 \\")
    print("      -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html")
    print("  pip install ez-openmmlab")
    print(
        "\n📚 Documentation: https://github.com/JustAnalyze/ez_openmmlab#installation"
    )
    print("=" * 70 + "\n")
    sys.exit(1)


def check_dependencies() -> None:
    """Check if required dependencies are installed."""
    # Check PyTorch
    try:
        import torch
    except ImportError:
        print("\n❌ PyTorch is not installed!")
        _print_install_instructions()

    # Check MMCV
    try:
        import mmcv
    except ImportError:
        print("\n❌ MMCV is not installed!")
        _print_install_instructions()

    # Warn if CUDA not available
    if not torch.cuda.is_available():
        print("\n⚠️  Warning: CUDA is not available!")
        print("    GPU acceleration is disabled. Inference will be slow.")
        print("    If you have a GPU, ensure CUDA drivers are installed.")
        print()
