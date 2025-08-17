import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from qiskit.primitives import StatevectorSampler as Sampler
from torchvision.transforms import functional as F
from sklearnex import patch_sklearn
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from sklearn.svm import SVR, LinearSVR, SVC
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from joblib import dump, load
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
patch_sklearn()

# Hierarchical Configuration
config_seed=42
data_sample_default = 0.01
data_split_default = 0.1

CONFIG = {
    "global": {
        #"default_model": "linearsvr",
        #"default_model": "svc",
        #"default_model": "svr",
        "default_model": "nn",
        "base_results": "superres",
        "seed": config_seed,
        "cmap": 'viridis',
        "compress": 9,
        "n_jobs": -1,
        "show_samples": {
            "show_comparison_grid": False,
            "show_index_consistency": False,
            "show_regression_results": True
        },
    },
    "data": {
        "data_split": data_split_default,
        "data_sample": data_sample_default,
        "patch": {
            "original_size": 4,
            "original_stride": 2,
            "resized_size": 2,
            "resized_stride": 1
        },
        "image": {
            "original_size": (28, 28),
            "resized_size": 14,
            "resize": "1"
        }, # CONFIG['data']['image']['original_size']
        "classes": [2, 5, 8],
        "samples_per_class": 1
    },
    "models": {
        "linearsvr": {
            "data_split": 0.5,
            "data_sample": 0.1,
            "C": 0.01,
            "max_iter": 100000,
            "loss":  'epsilon_insensitive', #loss{‘epsilon_insensitive’, ‘squared_epsilon_insensitive’}, default=’epsilon_insensitive’
            "tol": 1e-7,
            "intercept_scaling": 2,
            "random_state": config_seed,
            "clip": True
        },
        "svr": {
            "data_split": 0.04,
            "data_sample": 0.1,
            "kernel": "rbf",
            "C": 0.1,
            "epsilon": 0.01,
            "max_iter": -1,
            "gamma": "auto",
            "tol": 1e-6,
            "clip": True
        },"d": MLPRegressor(),
        "nn": {  # Example for potential neural network config
            "data_split": 0.002,
            "data_sample": 0.1,
            "hidden_layers": [8,  16],
            "learning_rate": "adaptive", # 'adaptive', 'constant', 'invscaling'
            "batch_size": "auto",
            "warm_start": True,
            #"max_iter": 1000,
            "tot": 1e-7,
            #"n_iter_no_change": 1000,
            #"early_stopping": False,
            "clip": True
        },
        "svc": {  # Example for potential SVC config
            "data_split": 0.05,
            "data_sample": 0.1,
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale"
        },
        "qsvr": {
            "data_split": 0.01,
            "data_sample": 0.1,
            "kernel": "precomputed"
            #"backend": "statevector_simulator",  # Just the name now
            #"feature_map": "ZZFeatureMap",
            #"shots": 1024,
            #"tol": 1e-6
        }
    },
    "visualization": {
        "default_figsize": (20, 10),
        "comparison_figsize": (14, 6),
        "index_consistency_figsize": (10, 5)
    }
}

# Initialize random seeds
np.random.seed(CONFIG["global"]["seed"])
torch.manual_seed(CONFIG["global"]["seed"])


def extract_patches(image: torch.Tensor, patch_size: int, stride: int) -> torch.Tensor:
    """
    Extract patches from tensor [C, H, W] with improved efficiency.

    Args:
        image: Input tensor of shape (channels, height, width)
        patch_size: Size of square patches to extract
        stride: Step size between patches

    Returns:
        Tensor of patches with shape (num_patches, channels, patch_size, patch_size)
    """
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D tensor (C,H,W), got {image.shape}")

    if patch_size > image.shape[1] or patch_size > image.shape[2]:
        raise ValueError(f"Patch size {patch_size} exceeds image dimensions {image.shape[1:]}")

    # Efficient patch extraction using unfold
    patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.contiguous().view(-1, image.shape[0], patch_size, patch_size)

    return patches


def combine_patches(patches, original_size, patch_size, stride, normalize=True):
    """
    Optimized patch recombination with improved dimension handling
    """
    C = patches.shape[1]
    H, W = original_size

    # More efficient reshape (avoiding movedim)
    patches = patches.permute(1, 2, 3, 0).reshape(C * patch_size ** 2, -1).unsqueeze(0)

    output = torch.nn.functional.fold(
        patches,
        output_size=(H, W),
        kernel_size=patch_size,
        stride=stride
    )

    if normalize and stride < patch_size:
        ones = torch.ones_like(patches, memory_format=torch.contiguous_format)
        norm = torch.nn.functional.fold(
            ones,
            output_size=(H, W),
            kernel_size=patch_size,
            stride=stride
        )
        output = output.div_(norm)

    return output.squeeze(0)


def show_reconstruction(original, reconstructed):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(original.squeeze())
    ax1.set_title('Original')
    ax2.imshow(reconstructed.squeeze())
    ax2.set_title('Reconstructed')
    plt.show()


# 2. Fixed transform pipelines
transf_train = {
    'original_img': transforms.ToTensor(),
    'resized_img': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((14, 14), interpolation=transforms.InterpolationMode.NEAREST_EXACT)
    ]),
    'original_patch': transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: extract_patches(x, 4, 2))
    ]),
    'resized_patch': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((14, 14), interpolation=transforms.InterpolationMode.NEAREST_EXACT),
        transforms.Lambda(lambda x: extract_patches(x, 2, 1))
    ]),
    'flat_pairs': transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda img: (
            extract_patches(F.resize(img, (14, 14), interpolation=transforms.InterpolationMode.NEAREST_EXACT), 2, 1).view(-1, 4),  # X
            extract_patches(img, 4, 2).view(-1, 16)  # y
        ))
    ])
}

transf_test = transf_train

# 3. Dataset loading remains the same
datasets_train = {
    name: datasets.MNIST('./data', train=True, download=True, transform=tr)
    for name, tr in transf_train.items()
}
datasets_test = {
    name: datasets.MNIST('./data', train=False, download=True, transform=tr)
    for name, tr in transf_test.items()
}

#dt_split= CONFIG["models"][CONFIG["global"]["default_model"]]["data_split"]
dt_split= CONFIG.get("models")
__x = CONFIG.get("global",{})
__x = __x.get("default_model", {})
print('Model:', __x)
dt_sample=dt_split.get(__x, {}).get("data_sample", data_sample_default)
dt_split=dt_split.get(__x, {}).get("data_split", data_split_default)

# 3. Compute how many samples correspond to the desired fraction
n_total_train = int(len(datasets_train['original_img'])*dt_sample)
print('Train elements:', n_total_train)
sample_size_train = int(n_total_train * dt_split)  # e.g., int(60000 * 0.01) = 600
print('Train elements samples:', sample_size_train)
train_indices = np.random.choice(len(datasets_train['original_img']), sample_size_train, replace=False)
subset_train = {name: Subset(ds, train_indices) for name, ds in datasets_train.items()}
# ------------------------------
n_total_test = int(len(datasets_test['original_img'])*dt_sample)
print('Test elements:', n_total_test)
sample_size_test = int(n_total_test * (1-dt_split))  # e.g., int(60000 * (1-0.01)) = 60000-600
print('Test elements samples:', sample_size_test)
test_indices = np.random.choice(len(datasets_test['original_img']), sample_size_test, replace=False)
subset_test = {name: Subset(ds, test_indices) for name, ds in datasets_test.items()}


def get_class_samples(original_dataset, classes, samples_per_class=3):
    """Get balanced samples from specified classes in original dataset"""
    indices = {cls: [] for cls in classes}
    for idx, (_, label) in enumerate(original_dataset):
        if label in classes and len(indices[label]) < samples_per_class:
            indices[label].append(idx)
    return [idx for cls in classes for idx in indices[cls]]


def show_comparison_grid(samples, columns_def=None, figsize=(20, 10)):
    """
    Flexible comparison grid with dynamic columns
    columns_def: List of tuples (data_key, title, [plot_kwargs])
    # In show_comparison_grid()
    """

    # Default columns (maintain original behavior)
    if columns_def is None:
        cmap_color = {'cmap': CONFIG["global"]["cmap"]}
        columns_def = [
            ('original_img', lambda s: f"Class {s['label']}\nOriginal", cmap_color),
            ('resized_img', "Resized", cmap_color),
            ('original_patch', lambda s: f"Original Patch\n{s['original_patch'].shape[-2:]}", cmap_color),
            ('resized_patch', lambda s: f"Resized Patch\n{s['resized_patch'].shape[-2:]}", cmap_color),
            ('reconstructed_original', "Recon from\nOriginal Patches", cmap_color),
            ('reconstructed_resized', "Recon from\nResized Patches", cmap_color)
        ]

    n_col = len(columns_def)
    n_row = len(samples)

    plt.figure(figsize=figsize)

    for row_idx, sample in enumerate(samples):
        for col_idx, col_def in enumerate(columns_def):
            ax = plt.subplot(n_row, n_col, row_idx * n_col + col_idx + 1)

            # Unpack column definition
            data_key, title_func, plot_kwargs = (
                col_def if len(col_def) == 3 else
                (*col_def, {})
            )

            # In show_comparison_grid()
            if data_key not in sample:
                raise KeyError(f"Key '{data_key}' not found in sample data")

            # Get data to show
            data = sample[data_key].squeeze()

            # Generate title
            title = (
                title_func(sample) if callable(title_func)
                else f"{title_func}"  # Handle string titles
            )

            # Show image with parameters
            plt.imshow(data.numpy() if torch.is_tensor(data) else data, **plot_kwargs)
            plt.title(title)
            plt.axis('off')

    plt.tight_layout()
    plt.show()


# Configuration
classes = [2, 5, 8]
samples_per_class = 1

# Get indices from ORIGINAL dataset (before any transforms)
class_indices = get_class_samples(subset_train['original_img'], classes=classes, samples_per_class=samples_per_class)

# Verify class labels match
print("Selected indices:", class_indices)
for idx in class_indices:
    print(f"Index {idx} - True class:", subset_train['original_img'][idx][1])


def select_informative_patch(patches, return_index=False):
    """Select patch with highest variance and optionally return index"""
    patches_flat = patches.view(patches.size(0), -1).float()
    patch_variances = torch.var(patches_flat, dim=1, keepdim=False)
    max_idx = torch.argmax(patch_variances)
    return (patches[max_idx].unsqueeze(0), max_idx) if return_index else patches[max_idx].unsqueeze(0)


def get_consistent_samples(classes, samples_per_class=3):
    """Get samples that maintain class consistency across all transformations"""
    # Get indices from original dataset

    original_dataset = subset_train['original_img']
    class_indices = get_class_samples(original_dataset, classes, samples_per_class)

    samples = []
    # tqdm(range(n_samples), desc=f"Processing {dataset_type}")
    for idx in tqdm(class_indices, desc=f"Processing consistent samples {len(class_indices)}"):
        # Verify label consistency
        true_label = original_dataset[idx][1]

        # Get original patch and its index
        orig_patches = subset_train['original_patch'][idx][0]
        orig_patch, orig_idx = select_informative_patch(orig_patches, return_index=True)

        # Get resized patch using SAME index
        resz_patches = subset_train['resized_patch'][idx][0]
        resz_patch = resz_patches[orig_idx].unsqueeze(0)  # Use original's index

        sample_data = {
            'label': true_label,
            'original_img': subset_train['original_img'][idx][0],
            'resized_img': subset_train['resized_img'][idx][0],
            # Take first patch only for visualization
            'original_patch': orig_patch,
            'resized_patch': resz_patch,
            'patch_index': orig_idx.item(),
            'reconstructed_original': combine_patches(
                subset_train['original_patch'][idx][0],
                original_size=(28, 28),
                patch_size=4,
                stride=2
            ),
            'reconstructed_resized': combine_patches(
                subset_train['resized_patch'][idx][0],
                original_size=(14, 14),
                patch_size=2,
                stride=1
            )
        }
        samples.append(sample_data)

    # Verify all samples match requested classes
    assert all(s['label'] in classes for s in samples), "Class mismatch detected!"
    return samples


# Usage

cmap_color = {'cmap': 'viridis'}
columns_def = [
    ('original_img', lambda s: f"Class {s['label']}\nOriginal", cmap_color),
    ('resized_img', "Resized", {'cmap': 'viridis'}),
    ('original_patch', lambda s: f"Original Patch\n{s['original_patch'].shape[-2:]}", cmap_color),
    ('resized_patch', lambda s: f"Resized Patch\n{s['resized_patch'].shape[-2:]}", cmap_color),
    ('reconstructed_original', "Recon from\nOriginal Patches", cmap_color),
    ('reconstructed_resized', "Recon from\nResized Patches", cmap_color)
]
samples = get_consistent_samples(classes=classes, samples_per_class=1)

if CONFIG["global"].get("show_samples", {}).get("show_comparison_grid", True):
    show_comparison_grid(samples, columns_def=columns_def, figsize=(14, 6))


# Example visualization of index consistency
def show_index_consistency(sample):
    """Show patch index consistency between original and resized"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Original patch grid - squeeze to remove channel dimension
    orig_recon = combine_patches(subset_train['original_patch'][0][0], (28, 28), 4, 2).squeeze(0)  # Now (28,28)
    ax1.imshow(orig_recon, cmap=CONFIG["global"]["cmap"])
    ax1.set_title(f"Original Patches\nSelected index: {sample['patch_index']}")

    # Resized patch grid - squeeze to remove channel dimension
    resz_recon = combine_patches(subset_train['resized_patch'][0][0], (14, 14), 2, 1).squeeze(0)  # Now (14,14)
    ax2.imshow(resz_recon, cmap=CONFIG["global"]["cmap"])
    ax2.set_title(f"Resized Patches\nUsing same index: {sample['patch_index']}")

    # Highlight selected index
    for ax, (recon, ps) in zip([ax1, ax2], [(orig_recon, 4), (resz_recon, 2)]):
        y = (sample['patch_index'] // 13) * ps
        x = (sample['patch_index'] % 13) * ps
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), ps, ps,
                                   fill=False, edgecolor='r', linewidth=2))

    plt.show()


samples = get_consistent_samples(classes=[2], samples_per_class=1)

if CONFIG["global"].get("show_samples", {}).get("show_index_consistency", True):
    show_index_consistency(samples[0])

# Added at the end of your code



def make_model_filename(model_key: str, size_sample: int, use_hash: bool = True,
                        extension: str = ".joblib.model") -> str:
    """
    Build a unique filename for the model based on its configuration in CONFIG.

    Parameters:
    - model_key (str): The key identifying the model in CONFIG["models"].
    - size_sample (int): The number of samples used to train the model.
    - use_hash (bool): If True, use a hash of the parameters in the filename;
                        if False, include the full parameter string.

    Returns:
    - str: The generated filename.
    """
    model_config = CONFIG["models"].get(model_key)
    if not model_config:
        raise ValueError(f"Model configuration for '{model_key}' not found in CONFIG.")

    # Create parameter string
    params = model_config.copy()
    params["size_sample"] = size_sample
    sorted_params = sorted(params.items())
    param_str = "_".join(f"{k}{v}" for k, v in sorted_params)
    param_identifier = hashlib.md5(param_str.encode()).hexdigest() if use_hash else param_str

    # Create directory structure
    base_dir = os.path.join(CONFIG["global"]["base_results"], model_key)
    os.makedirs(base_dir, exist_ok=True)

    # Construct filename
    filename = f"{model_key}_v0_{param_identifier}_size_sample{size_sample}{extension}"
    return os.path.join(base_dir, filename)


model_s = CONFIG["global"]["default_model"]
model_key = CONFIG["models"][model_s]
sample_size = len(subset_train['original_patch'])

model_fname = make_model_filename(model_s, sample_size, use_hash=False)
image_fname = make_model_filename(model_s, sample_size, use_hash=False, extension=".png")
metrics_fname = make_model_filename(model_s, sample_size, use_hash=False, extension=".json")


import numba

@numba.jit(nopython=True, fastmath=True, cache=True)
def quantize_y_for_classification(y, n_bins=256):
    """Converts a continuous target y (0-1) to integer classes."""
    return (y * (n_bins - 1)).astype(np.uint8)

N_BINS = 256
DEQUANTIZE_LUT = np.arange(N_BINS, dtype=np.float32) / (N_BINS - 1)
def dequantize_y_lut(y):
    return DEQUANTIZE_LUT[y]

@numba.jit(nopython=True, fastmath=True, cache=True)
def dequantize_y_from_classification(y, n_bins=256):
    """Converts predicted integer classes back to continuous targets."""
    return y.astype(np.float32) / (n_bins - 1)

def train_patch_regressor(original_subsets, resized_subsets):
    """Train model to predict original 4×4 patches from resized 2×2 patches"""

    # 1. Prepare data
    X_list, y_list = [], []
    for idx in range(len(resized_subsets)):
        rp = resized_subsets[idx][0]   # [169,1,2,2]
        op = original_subsets[idx][0]  # [169,1,4,4]
        X_list.append(rp.view(-1, 4))   # flatten 2×2→4
        y_list.append(op.view(-1, 16))  # flatten 4×4→16

    X = torch.cat(X_list).numpy()
    y = torch.cat(y_list).numpy()

    # 2. Dispatch based on default_model
    model_s = CONFIG["global"]["default_model"]
    params  = CONFIG["models"][model_s]

    if model_s == "qsvr":
        feature_range_ = (0, np.pi)
        quantum_scaler = MinMaxScaler(feature_range=(0, np.pi))
        # Quantum Processing
        X_size = len(X)
        X_train_size = int(X_size*0.9)
        #X_text_size = X_size - X_train_size
        X_train_quantum = quantum_scaler.fit_transform(X[:X_train_size])
        X_test_quantum = quantum_scaler.transform(X[X_train_size:])

        # Initialize quantum kernel components
        sampler = Sampler()
        print('sampler: ', sampler)
        fidelity = ComputeUncompute(sampler=sampler)
        feature_map = ZZFeatureMap(feature_dimension=X.shape[1], reps=2)
        quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

        K_train = quantum_kernel.evaluate(x_vec=X_train_quantum)
        K_test = quantum_kernel.evaluate(x_vec=X_test_quantum, y_vec=X_train_quantum)


    else:
        # — Classical path
        # pick base class
        if model_s == "svr":
            Base = SVR
        elif model_s == "linearsvr":
            Base = LinearSVR
        elif model_s == "svc":
            Base = SVC
        elif model_s == "nn":
            Base = MLPRegressor
        else:
            raise KeyError(f"Unknown model '{model_s}'")

        # filter only valid kwargs
        valid = {
            k: params[k]
            for k in params
            if k in Base().get_params()
        }
        base_est = Base(**valid)

        # wrap SVMs in MultiOutputRegressor
        if model_s in ("svr", "linearsvr"):
            estimator = MultiOutputRegressor(
                base_est,
                n_jobs=CONFIG["global"]["n_jobs"]
            )
        elif model_s in ("svc"):
            n_bins = 256
            regressor_ = MultiOutputClassifier(
                base_est,
                n_jobs=CONFIG["global"]["n_jobs"]
            )
            estimator = TransformedTargetRegressor(
                regressor=regressor_,
                func=quantize_y_for_classification,
                inverse_func=dequantize_y_lut
                #inverse_func = dequantize_y_from_classification
            )

        else:
            # MLPRegressor supports multi-output itself
            estimator = base_est
        feature_range_=(0,1)

    # 3. Build the scaling + estimator pipeline
    model = make_pipeline(
        MinMaxScaler(clip=params.get("clip", False), feature_range=feature_range_),
        #StandardScaler(),
        estimator
    )

    # 4. Fit & return
    print('Start training: ')
    start_time = time.time()
    model.fit(X, y)
    training_time = time.time() - start_time
    print('Time to train: ', training_time)
    return model


import time
import json

# Train the model
patch_regressor = None
if not os.path.exists(model_fname):
    start_time = time.time()
    patch_regressor = train_patch_regressor(subset_train['original_patch'], subset_train['resized_patch'])
    training_time = time.time() - start_time
    dump(patch_regressor, model_fname, compress=CONFIG["global"]["compress"])

    # Save metrics
    metrics = {
        "training_time_seconds": training_time,
        "model_parameters": CONFIG["models"][model_s],
        "data_size": sample_size,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(metrics_fname, 'w') as f:
        json.dump(metrics, f, indent=2)

else:
    patch_regressor = load(model_fname)
    if os.path.exists(metrics_fname):
        with open(metrics_fname, 'r') as f:
            metrics = json.load(f)
            print(f"Model loaded - Previous training time: {metrics['training_time_seconds']:.2f} seconds")

# patch_regressor = train_patch_regressor(subsets['original_patch'], subsets['resized_patch'])


def predict_and_reconstruct(sample_idx, dataset_type="train"):
    """Predict patches and reconstruct full image (now dataset-aware)"""
    # Select correct dataset
    dataset = subset_train if dataset_type == "train" else subset_test

    # Get resized patches from correct dataset
    resized_patches = dataset['resized_patch'][sample_idx][0]

    # Predict original patches
    X = resized_patches.view(-1, 4).numpy()
    predicted = patch_regressor.predict(X)

    if CONFIG["models"][model_s].get("clip", False):
        predicted = np.clip(predicted, 0, 1)

    # Reconstruct image from predicted patches
    predicted_patches = torch.tensor(predicted).view(-1, 1, 4, 4)
    reconstructed = combine_patches(predicted_patches, (28, 28), 4, 2)

    return {
        'original': dataset['original_img'][sample_idx][0],
        'resized': dataset['resized_img'][sample_idx][0],
        'predicted': reconstructed
    }

# Add new imports at the top
from skimage.metrics import structural_similarity as ssim
import numpy as np


def calculate_metrics(original, predicted):
    """Calculate image quality metrics"""
    orig_np = original.squeeze().numpy()
    pred_np = predicted.squeeze().numpy()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        original.flatten().float(),
        predicted.flatten().float(),
        dim=0
    ).item()

    # SSIM
    ssim_score = ssim(orig_np, pred_np,
                      data_range=1.0,  # MNIST is normalized [0,1]
                      channel_axis=None)

    # MSE
    mse = torch.mean((original - predicted) ** 2).item()

    return {
        'cosine_similarity': cos_sim,
        'ssim': ssim_score,
        'mse': mse
    }


def show_regression_results(samples, save_path=None):
    """Show/save regression comparisons with difference images"""
    n_samples = len(samples)
    fig = plt.figure(figsize=(15, 8))  # Increased height for 4 rows
    gs = GridSpec(4, n_samples, figure=fig)  # Changed to 4 rows

    metrics = []

    for i, sample in enumerate(samples):
        # Calculate metrics
        sample_metrics = calculate_metrics(sample['original'], sample['predicted'])
        metrics.append(sample_metrics)

        # Original image
        ax1 = fig.add_subplot(gs[0, i])
        ax1.imshow(sample['original'].squeeze(), cmap=CONFIG["global"]["cmap"])
        ax1.set_title(f"Original\nClass {subset_train['original_img'][class_indices[i]][1]}")
        ax1.axis('off')

        # Resized image
        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(sample['resized'].squeeze(), cmap=CONFIG["global"]["cmap"])
        ax2.set_title("Resized")
        ax2.axis('off')

        # Predicted image
        ax3 = fig.add_subplot(gs[2, i])
        im = ax3.imshow(sample['predicted'].squeeze(), cmap=CONFIG["global"]["cmap"])
        ax3.set_title("Predicted")
        ax3.axis('off')

        # Difference image
        ax4 = fig.add_subplot(gs[3, i])
        diff = np.abs(sample['original'].squeeze().numpy() -
                      sample['predicted'].squeeze().numpy())
        diff_im = ax4.imshow(diff, cmap='Reds', vmin=0, vmax=1)
        plt.colorbar(diff_im, ax=ax4, fraction=0.046, pad=0.04)
        ax4.set_title(f"Difference\nSSIM: {sample_metrics['ssim']:.2f}\nMSE: {sample_metrics['mse']:.4f}")
        ax4.axis('off')

    plt.tight_layout()

    # Update metrics file
    if os.path.exists(metrics_fname):
        with open(metrics_fname, 'r') as f:
            existing_metrics = json.load(f)

        existing_metrics.update({
            'average_metrics': {
                'mean_cosine': np.mean([m['cosine_similarity'] for m in metrics]),
                'mean_ssim': np.mean([m['ssim'] for m in metrics]),
                'mean_mse': np.mean([m['mse'] for m in metrics])
            },
            'per_sample_metrics': metrics
        })

        with open(metrics_fname, 'w') as f:
            json.dump(existing_metrics, f, indent=2)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, format='png', transparent=False)
        plt.close()
    else:
        plt.show()


# Update the prediction section to track metrics
test_samples = []
all_metrics = []

for idx in class_indices:
    sample = predict_and_reconstruct(idx)
    metrics = calculate_metrics(sample['original'], sample['predicted'])
    all_metrics.append(metrics)
    test_samples.append(sample)

# Update JSON metrics
if os.path.exists(metrics_fname):
    with open(metrics_fname, 'r') as f:
        metrics = json.load(f)

    metrics.update({
        'image_metrics': {
            'average_cosine': np.mean([m['cosine_similarity'] for m in all_metrics]),
            'average_ssim': np.mean([m['ssim'] for m in all_metrics]),
            'average_mse': np.mean([m['mse'] for m in all_metrics]),
            'per_image_metrics': all_metrics
        }
    })

    with open(metrics_fname, 'w') as f:
        json.dump(metrics, f, indent=2)

# Generate and save results
test_samples = [predict_and_reconstruct(idx) for idx in class_indices]
if CONFIG["global"].get("show_samples", {}).get("show_regression_results", True):
    show_regression_results(test_samples, save_path=image_fname)


def calculate_comprehensive_metrics(dataset_type="train"):
    """Calculate metrics for specified dataset type"""
    # Select correct dataset
    dataset = subset_train if dataset_type == "train" else subset_test
    n_samples = len(dataset['original_img'])
    from tqdm import tqdm

    all_metrics = []
    class_metrics = {cls: [] for cls in CONFIG['data']['classes']}


    for idx in tqdm(range(n_samples), desc=f"Processing {dataset_type}"):
        try:
            sample = predict_and_reconstruct(idx, dataset_type)
            metrics = calculate_metrics(sample['original'], sample['predicted'])
            label = dataset['original_img'][idx][1]

            all_metrics.append(metrics)
            if label in class_metrics:
                class_metrics[label].append(metrics)

        except IndexError as e:
            print(f"Skipping invalid index {idx} in {dataset_type} set: {str(e)}")
            continue

    return {
        'dataset_type': dataset_type,
        'overall': {
            'ssim': np.nanmean([m['ssim'] for m in all_metrics]),
            'mse': np.nanmean([m['mse'] for m in all_metrics]),
            'cosine': np.nanmean([m['cosine_similarity'] for m in all_metrics])
        },
        'classes': {
            cls: {
                'ssim': np.nanmean([m['ssim'] for m in class_metrics[cls]]),
                'mse': np.nanmean([m['mse'] for m in class_metrics[cls]]),
                'cosine': np.nanmean([m['cosine_similarity'] for m in class_metrics[cls]])
            } for cls in CONFIG['data']['classes']
        }
    }


def save_metrics_report(metrics, model_name):
    """Save metrics in organized text format"""
    report = [
        f"Model: {model_name}",
        f"Dataset: {metrics['dataset_type'].upper()}",
        "-" * 40,
        "Overall Metrics:",
        f"SSIM: {metrics['overall']['ssim']:.4f}",
        f"MSE: {metrics['overall']['mse']:.4f}",
        f"Cosine Similarity: {metrics['overall']['cosine']:.4f}",
        "\nClass-wise Metrics:"
    ]

    for cls, values in metrics['classes'].items():
        report.extend([
            f"Class {cls}:",
            f"  SSIM: {values['ssim']:.4f}",
            f"  MSE: {values['mse']:.4f}",
            f"  Cosine: {values['cosine']:.4f}"
        ])

    os.makedirs("metrics_reports", exist_ok=True)
    filename = f"{CONFIG["global"]["base_results"]}/{model_name}_{metrics['dataset_type']}_metrics.txt"

    with open(filename, 'w') as f:
        f.write("\n".join(report))


def compare_all_metrics():
    """Generate comparison plots from saved metrics"""
    metrics_files = [f for f in os.listdir("metrics_reports") if f.endswith(".txt")]
    all_metrics = []

    # Load and parse all metrics
    for fname in metrics_files:
        with open(os.path.join("metrics_reports", fname)) as f:
            content = f.read()
            # Add parsing logic here to extract metrics
            # Store in all_metrics list

    # Generate comparison plots
    plt.figure(figsize=(15, 8))
    # Add visualization code comparing metrics
    plt.savefig("metrics_comparison.png")


# Replace the existing metrics section after model training with:

model_name = CONFIG["global"]["default_model"]

# Calculate metrics
train_metrics = calculate_comprehensive_metrics("train")
test_metrics = calculate_comprehensive_metrics("test")

# Save reports
save_metrics_report(train_metrics, model_name)
save_metrics_report(test_metrics, model_name)