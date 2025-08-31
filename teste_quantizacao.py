import torch
from torchvision import datasets, transforms
from SVMR_Dataset import ImageQuantizer

# Carregar MNIST (um exemplo)
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
img_tensor, _ = mnist[0]           # [1, H, W], float in [0,1]

# Test CPU
q_cpu = ImageQuantizer.quantize(img_tensor, levels=2, method='uniform', dithering=False)
unique_cpu = torch.unique(q_cpu)
print("CPU unique values:", unique_cpu)

# Test GPU (se disponível)
if torch.cuda.is_available():
    img_cuda = img_tensor.to('cuda')
    q_cuda = ImageQuantizer.quantize(img_cuda, levels=2, method='uniform', dithering=False)
    unique_cuda = torch.unique(q_cuda.cpu())
    print("GPU unique values:", unique_cuda)
else:
    print("CUDA não disponível para teste GPU.")



q_otsu = ImageQuantizer.quantize(img_tensor, method='otsu')  # levels não é necessário para otsu
print("Otsu unique:", torch.unique(q_otsu))