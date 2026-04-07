import torch
import numpy as np


def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # Create a simple tensor
    x = torch.tensor([1, 2, 3, 4, 5])
    print("\nTensor:", x)
    print("Tensor shape:", x.shape)
    print("Tensor dtype:", x.dtype)

    # Simple operation
    y = x * 2
    print("\nMultiplied by 2:", y)

    # Create a random tensor
    random_tensor = torch.randn(3, 3)
    print("\nRandom 3x3 tensor:")
    print(random_tensor)


if __name__ == "__main__":
    main()
