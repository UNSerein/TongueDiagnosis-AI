import sys
import torch
import cv2
import numpy as np

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("OpenCV:", cv2.__version__)
print("Numpy:", np.__version__)