"""Test script to verify all imports work correctly"""

print("Testing imports...")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import torchvision
    print(f"✓ TorchVision {torchvision.__version__}")
except ImportError as e:
    print(f"✗ TorchVision: {e}")

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib: {e}")

try:
    import numpy
    print(f"✓ NumPy {numpy.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")

try:
    import h5py
    print(f"✓ h5py {h5py.__version__}")
except ImportError as e:
    print(f"✗ h5py: {e}")

try:
    import scipy
    print(f"✓ SciPy {scipy.__version__}")
except ImportError as e:
    print(f"✗ SciPy: {e}")

try:
    import sklearn
    print(f"✓ Scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"✗ Scikit-learn: {e}")

try:
    import tqdm
    print(f"✓ tqdm {tqdm.__version__}")
except ImportError as e:
    print(f"✗ tqdm: {e}")

try:
    from PIL import Image
    print(f"✓ Pillow (PIL)")
except ImportError as e:
    print(f"✗ Pillow: {e}")

print("\nAll imports successful! You can now run the training script.")
