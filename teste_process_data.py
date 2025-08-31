from SVMR_Dataset import ProcessedDataset
from torchvision import datasets, transforms
import torch

transform = transforms.Compose([transforms.ToTensor()])
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
subset = torch.utils.data.Subset(full_dataset, list(range(20)))

low_res_config = {
    'target_size': (28, 28),
    'resize_alg': None,            # ou Image.BICUBIC
    'image_format': None,
    'quality': None,
    'quantization_levels': 2,
    'quantization_method': 'uniform'
}

pd = ProcessedDataset(subset, cache_dir="./cache_test", cache_rebuild=True, **low_res_config)
# pd.data shape [N, C, H, W]
vals = torch.unique(pd.data)
print("ProcessedDataset unique values:", vals)
# Assegurar que são 0.0 e 1.0
assert set([float(x) for x in vals.cpu().numpy()]).issubset({0.0, 1.0}), "Não está binarizado!"
print("Processado: OK — imagens binárias apenas (0/1).")
