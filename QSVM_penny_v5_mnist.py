import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pennylane as qml
from pennylane import numpy as pnp
from PIL import Image
import warnings
from SVMR_Dataset import ProcessedDataset, PatchExtractor
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pickle
import os
import hashlib
import json
from Logger import Logger
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set log level for detailed progress tracking
Logger.set_log_level("INFO")

def get_best_device(wires):
    """Returns the best available device"""
    Logger.step("Selecting best quantum device")
    devices_priority = [
        ("lightning.gpu", "GPU Lightning"),
        ("lightning.qubit", "CPU Lightning"),
        ("default.qubit", "Default CPU")
    ]
    for device_name, description in devices_priority:
        try:
            test_dev = qml.device(device_name, wires=2)
            Logger.info(f"Using: {description}")
            return device_name
        except:
            continue
    Logger.info("Using: Default CPU")
    return "default.qubit"

# =============================================
# LOAD MNIST DATA
# =============================================

def load_mnist_data(n_samples_per_class=2):
    """Loads and prepares a small subset of MNIST dataset (classes 1 and 8)"""
    Logger.step("Loading MNIST dataset")
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Filter classes 1 and 8, limit to n_samples_per_class per class
    mask_1 = (mnist.targets == 1)
    mask_8 = (mnist.targets == 8)
    indices_1 = np.where(mask_1)[0][:n_samples_per_class]
    indices_8 = np.where(mask_8)[0][:n_samples_per_class]
    indices = np.concatenate([indices_1, indices_8])
    X = mnist.data[indices].numpy().astype(np.float32) / 255.0
    y = mnist.targets[indices].numpy()
    y = np.where(y == 1, -1, 1)  # Convert to -1/+1 for SVM compatibility

    # Reshape images to (n_samples, height*width)
    X = X.reshape(X.shape[0], -1)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    Logger.info(f"MNIST dataset loaded: {len(X)} samples (classes 1 and 8)")
    Logger.info(f"Features per sample: {X.shape[1]}")
    Logger.info(f"Classes: {np.unique(y)} (1=-1, 8=+1)")
    return X, y, scaler

# =============================================
# CLASSICAL KERNELS
# =============================================

def rbf_kernel_classical(x1, x2, gamma=1.0):
    """Classical RBF kernel"""
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

def linear_kernel_classical(x1, x2):
    """Classical linear kernel"""
    return np.dot(x1, x2)

def polynomial_kernel_classical(x1, x2, degree=2, coeff=1.0):
    """Classical polynomial kernel"""
    return (coeff + np.dot(x1, x2)) ** degree

def compute_classical_kernel_matrix(X1, X2, kernel_func, **kwargs):
    """Computes classical kernel matrix"""
    Logger.step("Computing classical kernel matrix")
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_func(X1[i], X2[j], **kwargs)
        Logger.info(f"Computed row {i+1}/{n1} of classical kernel matrix")
    return K

# =============================================
# QUANTUM SVM HYBRID (QUANTUM KERNEL)
# =============================================

n_qubits = 2
device_type = get_best_device(n_qubits)
dev = qml.device(device_type, wires=n_qubits)

def data_encoding_circuit(x):
    """Data encoding circuit - ZZFeatureMap inspired"""
    for i in range(min(len(x), n_qubits)):
        qml.RY(x[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
        if i < len(x) - 1:
            qml.RZ(x[i] * x[i + 1], wires=i + 1)
    for i in range(min(len(x), n_qubits)):
        qml.RY(x[i], wires=i)

@qml.qnode(dev)
def quantum_kernel_circuit(x1, x2):
    """Quantum kernel circuit"""
    data_encoding_circuit(x1)
    qml.adjoint(data_encoding_circuit)(x2)
    return qml.probs(wires=range(n_qubits))

def quantum_kernel_hybrid(x1, x2):
    """Computes hybrid quantum kernel based on fidelity"""
    probs = quantum_kernel_circuit(x1, x2)
    return probs[0]

def compute_quantum_kernel_matrix_hybrid(X1, X2):
    """Computes hybrid quantum kernel matrix"""
    Logger.step("Computing hybrid quantum kernel matrix")
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = quantum_kernel_hybrid(X1[i], X2[j])
        Logger.info(f"Computed row {i+1}/{n1} of hybrid quantum kernel matrix")
    return K

# =============================================
# FULLY QUANTUM SVM
# =============================================

class FullyQuantumSVM:
    """Fully quantum SVM based on Rebentrost et al."""

    def __init__(self, n_qubits=5, C=1.0):
        Logger.step("Initializing FullyQuantumSVM")
        self.n_qubits = n_qubits
        self.C = C
        self.dev = qml.device(device_type, wires=2 * n_qubits + 2)
        self.models = []
        self.X_train = None
        self.y_train = None

    def quantum_state_preparation(self, x):
        """Enhanced quantum state preparation"""
        norm = np.linalg.norm(x)
        if norm > 1e-8:
            x_normalized = x / norm
        else:
            x_normalized = np.ones_like(x) * 1e-8
        for i in range(min(len(x_normalized), self.n_qubits)):
            angle = 2 * np.arctan2(abs(x_normalized[i]), 1)
            if abs(angle) > 1e-8:
                qml.RY(angle, wires=i)

    def quantum_inner_product(self, x1, x2):
        """Fast quantum inner product using SWAP test"""
        dev_temp = qml.device(device_type, wires=5)

        @qml.qnode(dev_temp)
        def inner_product_circuit():
            for i in range(min(2, len(x1))):
                if abs(x1[i]) > 1e-8:
                    qml.RY(2 * np.arctan2(abs(x1[i]), 1), wires=i)
            for i in range(min(2, len(x2))):
                if abs(x2[i]) > 1e-8:
                    qml.RY(2 * np.arctan2(abs(x2[i]), 1), wires=i + 2)
            qml.Hadamard(wires=4)
            qml.CSWAP(wires=[4, 0, 2])
            qml.CSWAP(wires=[4, 1, 3])
            qml.Hadamard(wires=4)
            return qml.probs(wires=4)

        return inner_product_circuit()

    def quantum_kernel_matrix(self, X):
        """Computes quantum kernel matrix"""
        Logger.step("Computing fully quantum kernel matrix")
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    probs = self.quantum_inner_product(X[i], X[j])
                    K[i, j] = probs[0]
                except Exception as e:
                    Logger.error(f"Kernel error ({i},{j})", exc=e)
                    K[i, j] = np.exp(-0.5 * np.linalg.norm(X[i] - X[j]) ** 2)
            Logger.info(f"Computed row {i+1}/{n} of fully quantum kernel matrix")
        return K

    def quantum_least_squares_solver(self, A, b):
        """Simplified quantum linear system solver"""
        Logger.step("Solving quantum linear system")
        try:
            A_reg = A + self.C * np.eye(len(A))
            solution = np.linalg.solve(A_reg, b)
            return solution
        except np.linalg.LinAlgError as e:
            Logger.error("Linear algebra error in solver", exc=e)
            solution = np.linalg.lstsq(A, b, rcond=None)[0]
            return solution

    def fit(self, X, y, cache_path=None, vis_dir=None, vis_samples=None, test_dataset=None, X_test=None, y_test=None):
        """Trains the fully quantum SVM with SVR for each pixel, saving partial images"""
        Logger.step("Training FullyQuantumSVM")
        if cache_path and os.path.exists(cache_path):
            Logger.info(f"Loading cached FullyQuantumSVM model from {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                self.models = data['models']
                self.X_train = data['X_train']
                self.y_train = data['y_train']
                Logger.info("Loaded cached FullyQuantumSVM model successfully")
                return
            except Exception as e:
                Logger.error(f"Error loading cached model", exc=e)

        self.X_train = X.copy()
        self.y_train = y.copy()
        K = self.quantum_kernel_matrix(X)
        self.models = []
        vis_pixels = [4, 8, 12, 16]  # Save images after these pixels
        for pixel_idx in range(y.shape[1]):
            Logger.info(f"Training SVR for pixel {pixel_idx + 1}/{y.shape[1]}")
            pixel_values = y[:, pixel_idx]
            alpha = self.quantum_least_squares_solver(K, pixel_values)
            self.models.append(alpha)

            # Save partial images after specified pixels
            if vis_dir and vis_samples and test_dataset is not None and X_test is not None and y_test is not None and (pixel_idx + 1) in vis_pixels:
                Logger.info(f"Saving partial images after pixel {pixel_idx + 1}")
                partial_preds = self.predict_partial(X_test, pixel_idx + 1)
                for img_idx in vis_samples:
                    pred_patches = partial_preds[img_idx * test_dataset.num_patches_per_image:(img_idx + 1) * test_dataset.num_patches_per_image]
                    pred_patches = pred_patches.reshape(-1, 4, 4)
                    mse = mean_squared_error(
                        test_dataset.reconstruct_high(img_idx).numpy().squeeze(),
                        test_dataset.large_patch_extractor.reconstruct_image(
                            torch.tensor(pred_patches, dtype=torch.float32)
                        ).numpy()
                    )
                    save_path = os.path.join(vis_dir, f"pixel_{pixel_idx + 1}_image_{img_idx}.png")
                    show_full_image_comparison(img_idx, test_dataset, pred_patches, mse, "QSVM Fully", save_path=save_path)

        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'models': self.models,
                        'X_train': self.X_train,
                        'y_train': self.y_train
                    }, f)
                Logger.info(f"Saved FullyQuantumSVM model to {cache_path}")
            except Exception as e:
                Logger.error(f"Error saving FullyQuantumSVM model", exc=e)

    def predict_partial(self, X_test, num_pixels):
        """Predicts up to num_pixels for partial visualization"""
        predictions = []
        for i, x_test in enumerate(X_test):
            pixel_preds = []
            for pixel_idx, alpha in enumerate(self.models[:num_pixels]):
                decision_value = 0
                for j, x_train in enumerate(self.X_train):
                    probs = self.quantum_inner_product(x_test, x_train)
                    kernel_val = probs[0]
                    decision_value += alpha[j] * kernel_val
                pixel_preds.append(np.clip(decision_value, 0, 1))
            # Pad with zeros for remaining pixels
            pixel_preds.extend([0] * (16 - len(pixel_preds)))
            predictions.append(pixel_preds)
        return np.array(predictions)

    def predict(self, X_test):
        """Predicts high-resolution patches"""
        Logger.step("Predicting with FullyQuantumSVM")
        predictions = []
        for i, x_test in enumerate(X_test):
            pixel_preds = []
            for pixel_idx, alpha in enumerate(self.models):
                decision_value = 0
                for j, x_train in enumerate(self.X_train):
                    probs = self.quantum_inner_product(x_test, x_train)
                    kernel_val = probs[0]
                    decision_value += alpha[j] * kernel_val
                pixel_preds.append(np.clip(decision_value, 0, 1))
            predictions.append(pixel_preds)
            Logger.info(f"Predicted patch {i+1}/{len(X_test)}")
        return np.array(predictions)

# =============================================
# QUANTUM VARIATIONAL SVM
# =============================================

class VariationalQuantumSVM:
    """Variational QSVM with parameterized ansatz"""

    def __init__(self, n_qubits=4, n_layers=2):
        Logger.step("Initializing VariationalQuantumSVM")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device_type, wires=n_qubits)
        self.models = []
        self.params = None

        @qml.qnode(self.dev)
        def quantum_classifier_circuit(x, params):
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(x[i], wires=i)

            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(params[layer * self.n_qubits * 3 + i * 3], wires=i)
                    qml.RZ(params[layer * self.n_qubits * 3 + i * 3 + 1], wires=i)
                    qml.RX(params[layer * self.n_qubits * 3 + i * 3 + 2], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.quantum_classifier_circuit = quantum_classifier_circuit

    def cost_function(self, params, X, y):
        """Cost function for variational classifier"""
        predictions = np.array([self.quantum_classifier_circuit(x, params) for x in X])
        mse = np.mean((predictions - y) ** 2)
        return mse

    def fit(self, X, y, cache_path=None, vis_dir=None, vis_samples=None, test_dataset=None, X_test=None, y_test=None):
        """Trains variational SVM for each pixel, saving partial images"""
        Logger.step("Training VariationalQuantumSVM")
        if cache_path and os.path.exists(cache_path):
            Logger.info(f"Loading cached VariationalQuantumSVM model from {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                self.models = data['models']
                Logger.info("Loaded cached VariationalQuantumSVM model successfully")
                return []
            except Exception as e:
                Logger.error(f"Error loading cached model", exc=e)

        n_params = self.n_layers * self.n_qubits * 3
        self.models = []
        costs = []
        vis_epochs = [10, 20, 30, 40, 50]  # Save images at these epochs
        for pixel_idx in range(y.shape[1]):
            Logger.info(f"Training SVR for pixel {pixel_idx + 1}/{y.shape[1]}")
            pixel_values = y[:, pixel_idx]
            self.params = np.random.normal(0, 0.1, n_params)
            pixel_costs = []
            for epoch in range(50):
                cost = self.cost_function(self.params, X, pixel_values)
                pixel_costs.append(cost)
                grad = np.zeros_like(self.params)
                eps = 1e-3
                for i in range(len(self.params)):
                    params_plus = self.params.copy()
                    params_plus[i] += eps
                    params_minus = self.params.copy()
                    params_minus[i] -= eps
                    cost_plus = self.cost_function(params_plus, X, pixel_values)
                    cost_minus = self.cost_function(params_minus, X, pixel_values)
                    grad[i] = (cost_plus - cost_minus) / (2 * eps)
                self.params -= 0.005 * grad
                if epoch % 20 == 0:
                    Logger.info(f"Pixel {pixel_idx + 1}, Epoch {epoch}: Cost = {cost:.4f}")
                # Save partial images at specified epochs
                if vis_dir and vis_samples and test_dataset is not None and X_test is not None and y_test is not None and (epoch + 1) in vis_epochs:
                    Logger.info(f"Saving partial images for pixel {pixel_idx + 1} at epoch {epoch + 1}")
                    partial_pred = self.quantum_classifier_circuit(X_test, self.params)
                    partial_pred = np.clip(partial_pred, 0, 1)
                    for img_idx in vis_samples:
                        # Predict all patches for the image
                        pred_patches = np.zeros((test_dataset.num_patches_per_image, 16))
                        for patch_idx in range(test_dataset.num_patches_per_image):
                            global_idx = img_idx * test_dataset.num_patches_per_image + patch_idx
                            pred_patches[patch_idx, pixel_idx] = partial_pred[global_idx]
                        pred_patches = pred_patches.reshape(-1, 4, 4)
                        mse = mean_squared_error(
                            test_dataset.reconstruct_high(img_idx).numpy().squeeze(),
                            test_dataset.large_patch_extractor.reconstruct_image(
                                torch.tensor(pred_patches, dtype=torch.float32)
                            ).numpy()
                        )
                        save_path = os.path.join(vis_dir, f"epoch_{epoch + 1}_pixel_{pixel_idx + 1}_image_{img_idx}.png")
                        show_full_image_comparison(img_idx, test_dataset, pred_patches, mse, "QSVM Variational", save_path=save_path)
            self.models.append(self.params.copy())
            costs.append(pixel_costs)

        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump({'models': self.models}, f)
                Logger.info(f"Saved VariationalQuantumSVM model to {cache_path}")
            except Exception as e:
                Logger.error(f"Error saving VariationalQuantumSVM model", exc=e)

        return costs

    def predict(self, X):
        """Predicts high-resolution patches"""
        Logger.step("Predicting with VariationalQuantumSVM")
        predictions = []
        for i, x in enumerate(X):
            pixel_preds = []
            for params in self.models:
                pred = self.quantum_classifier_circuit(x, params)
                pixel_preds.append(np.clip(pred, 0, 1))
            predictions.append(pixel_preds)
            Logger.info(f"Predicted patch {i+1}/{len(X)}")
        return np.array(predictions)

# =============================================
# EVALUATION FUNCTIONS
# =============================================

def evaluate_svm(svm_model, X_train, X_test, y_train, y_test, model_name, cache_path=None, vis_dir=None, vis_samples=None, test_dataset=None):
    """Evaluates SVR models for super-resolution, one SVR per pixel"""
    Logger.step(f"Evaluating {model_name}")
    try:
        cache_dir = os.path.dirname(cache_path) if cache_path else "."
        model_paths = [os.path.join(cache_dir, f"{model_name}_pixel_{i}.pkl") for i in range(y_train.shape[1])]
        all_models_exist = all(os.path.exists(p) for p in model_paths)

        if isinstance(svm_model, FullyQuantumSVM):
            # Handle FullyQuantumSVM directly
            if all_models_exist:
                Logger.info(f"Loading cached {model_name} model from {cache_path}")
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                svm_model.models = data['models']
                svm_model.X_train = data['X_train']
                svm_model.y_train = data['y_train']
                Logger.info(f"Loaded cached {model_name} model successfully")
            else:
                Logger.info(f"Training {model_name}")
                svm_model.fit(X_train, y_train, cache_path, vis_dir, vis_samples, test_dataset, X_test, y_test)

            Logger.info(f"Computing training MSE for {model_name}")
            y_train_pred = svm_model.predict(X_train)
            train_mse = mean_squared_error(y_train, y_train_pred)

            Logger.info(f"Computing test MSE for {model_name}")
            y_pred = svm_model.predict(X_test)
            test_mse = mean_squared_error(y_test, y_pred)

            # Save final prediction images
            if vis_dir and vis_samples and test_dataset is not None:
                Logger.info(f"Saving final prediction images for {model_name}")
                for img_idx in vis_samples:
                    pred_patches = y_pred[img_idx * test_dataset.num_patches_per_image:(img_idx + 1) * test_dataset.num_patches_per_image]
                    pred_patches = pred_patches.reshape(-1, 4, 4)
                    mse = mean_squared_error(
                        test_dataset.reconstruct_high(img_idx).numpy().squeeze(),
                        test_dataset.large_patch_extractor.reconstruct_image(
                            torch.tensor(pred_patches, dtype=torch.float32)
                        ).numpy()
                    )
                    save_path = os.path.join(vis_dir, f"final_image_{img_idx}.png")
                    show_full_image_comparison(img_idx, test_dataset, pred_patches, mse, model_name, save_path=save_path)

        else:
            # Handle classical SVR models
            models = []
            if all_models_exist:
                Logger.info(f"Loading cached {model_name} models from {cache_dir}")
                for i, path in enumerate(model_paths):
                    with open(path, 'rb') as f:
                        models.append(pickle.load(f))
                Logger.info(f"Loaded cached {model_name} models successfully")
            else:
                Logger.info(f"Training {model_name}")
                for pixel_idx in range(y_train.shape[1]):
                    Logger.info(f"Training SVR for pixel {pixel_idx + 1}/{y_train.shape[1]}")
                    svm = SVR(kernel=svm_model.kernel, C=svm_model.C, gamma=svm_model.gamma if hasattr(svm_model, 'gamma') else 'scale')
                    svm.fit(X_train, y_train[:, pixel_idx])
                    models.append(svm)
                    if cache_path:
                        try:
                            os.makedirs(cache_dir, exist_ok=True)
                            with open(model_paths[pixel_idx], 'wb') as f:
                                pickle.dump(svm, f)
                            Logger.info(f"Saved {model_name} model for pixel {pixel_idx + 1} to {model_paths[pixel_idx]}")
                        except Exception as e:
                            Logger.error(f"Error saving {model_name} model for pixel {pixel_idx + 1}", exc=e)

            Logger.info(f"Computing training MSE for {model_name}")
            y_train_pred = np.zeros_like(y_train)
            for i in range(X_train.shape[0]):
                for pixel_idx, svm in enumerate(models):
                    y_train_pred[i, pixel_idx] = svm.predict(X_train[i].reshape(1, -1))
            train_mse = mean_squared_error(y_train, y_train_pred)

            Logger.info(f"Computing test MSE for {model_name}")
            y_pred = np.zeros_like(y_test)
            for i in range(X_test.shape[0]):
                for pixel_idx, svm in enumerate(models):
                    y_pred[i, pixel_idx] = svm.predict(X_test[i].reshape(1, -1))
            test_mse = mean_squared_error(y_test, y_pred)

            # Save final prediction images
            if vis_dir and vis_samples and test_dataset is not None:
                Logger.info(f"Saving final prediction images for {model_name}")
                for img_idx in vis_samples:
                    pred_patches = y_pred[img_idx * test_dataset.num_patches_per_image:(img_idx + 1) * test_dataset.num_patches_per_image]
                    pred_patches = pred_patches.reshape(-1, 4, 4)
                    mse = mean_squared_error(
                        test_dataset.reconstruct_high(img_idx).numpy().squeeze(),
                        test_dataset.large_patch_extractor.reconstruct_image(
                            torch.tensor(pred_patches, dtype=torch.float32)
                        ).numpy()
                    )
                    save_path = os.path.join(vis_dir, f"final_image_{img_idx}.png")
                    show_full_image_comparison(img_idx, test_dataset, pred_patches, mse, model_name, save_path=save_path)

        Logger.info(f"{model_name} - Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
        return train_mse, test_mse, y_pred
    except Exception as e:
        Logger.error(f"Error evaluating {model_name}", exc=e)
        return float('inf'), float('inf'), np.zeros_like(y_test)

def evaluate_kernel_svm(K_train, K_test, X_test, y_train, y_test, kernel_name, cache_path=None, vis_dir=None, vis_samples=None, test_dataset=None):
    """Evaluates SVR with precomputed kernel"""
    Logger.step(f"Evaluating {kernel_name}")
    try:
        cache_dir = os.path.dirname(cache_path) if cache_path else "."
        model_paths = [os.path.join(cache_dir, f"{kernel_name}_pixel_{i}.pkl") for i in range(y_train.shape[1])]
        all_models_exist = all(os.path.exists(p) for p in model_paths)

        models = []
        if all_models_exist:
            Logger.info(f"Loading cached {kernel_name} models from {cache_dir}")
            for i, path in enumerate(model_paths):
                with open(path, 'rb') as f:
                    models.append(pickle.load(f))
            Logger.info(f"Loaded cached {kernel_name} models successfully")
        else:
            Logger.info(f"Training {kernel_name} models")
            for pixel_idx in range(y_train.shape[1]):
                svm = SVR(kernel='precomputed', C=1.0)
                svm.fit(K_train, y_train[:, pixel_idx])
                models.append(svm)
                if cache_path:
                    try:
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(model_paths[pixel_idx], 'wb') as f:
                            pickle.dump(svm, f)
                        Logger.info(f"Saved {kernel_name} model for pixel {pixel_idx + 1} to {model_paths[pixel_idx]}")
                    except Exception as e:
                        Logger.error(f"Error saving {kernel_name} model for pixel {pixel_idx + 1}", exc=e)

        Logger.info(f"Computing training MSE for {kernel_name}")
        y_train_pred = np.zeros_like(y_train)
        for i, x_train in enumerate(K_train):
            for pixel_idx, svm in enumerate(models):
                y_train_pred[i, pixel_idx] = svm.predict(x_train.reshape(1, -1))
        train_mse = mean_squared_error(y_train, y_train_pred)

        Logger.info(f"Computing test MSE for {kernel_name}")
        y_pred = np.zeros_like(y_test)
        for i in range(K_test.shape[0]):
            for pixel_idx, svm in enumerate(models):
                y_pred[i, pixel_idx] = svm.predict(K_test[i].reshape(1, -1))
        test_mse = mean_squared_error(y_test, y_pred)

        # Save final prediction images
        if vis_dir and vis_samples and test_dataset is not None:
            Logger.info(f"Saving final prediction images for {kernel_name}")
            for img_idx in vis_samples:
                pred_patches = y_pred[img_idx * test_dataset.num_patches_per_image:(img_idx + 1) * test_dataset.num_patches_per_image]
                pred_patches = pred_patches.reshape(-1, 4, 4)
                mse = mean_squared_error(
                    test_dataset.reconstruct_high(img_idx).numpy().squeeze(),
                    test_dataset.large_patch_extractor.reconstruct_image(
                        torch.tensor(pred_patches, dtype=torch.float32)
                    ).numpy()
                )
                save_path = os.path.join(vis_dir, f"final_image_{img_idx}.png")
                show_full_image_comparison(img_idx, test_dataset, pred_patches, mse, kernel_name, save_path=save_path)

        Logger.info(f"{kernel_name} - Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
        return train_mse, test_mse, y_pred
    except Exception as e:
        Logger.error(f"Error evaluating {kernel_name}", exc=e)
        return float('inf'), float('inf'), np.zeros_like(y_test)

# =============================================
# PATCH DEDUPLICATION (COMMENTED OUT)
# =============================================

# def deduplicate_patches(patches, variance_threshold=1e-4):
#     """Remove patches with low variance (e.g., uniform background)"""
#     Logger.step("Deduplicating patches")
#     patches = patches.to(torch.float32)  # Ensure float32 dtype
#     variances = torch.var(patches, dim=1)
#     mask = variances > variance_threshold
#     Logger.info(f"Kept {mask.sum().item()}/{len(patches)} patches after deduplication")
#     return patches[mask], mask

# =============================================
# MODIFIED SUPERRESPATCHDATASET
# =============================================

class SuperResPatchDataset(torch.utils.data.Dataset):
    """
    Modified to precompute patches during initialization for faster data access.
    """
    def __init__(self, original_ds, low_res_config, high_res_config, small_patch_size, large_patch_size,
                 stride, scale_factor, cache_dir="./cache", cache_rebuild=False):
        Logger.step("Initializing SuperResPatchDataset")
        assert large_patch_size[0] == small_patch_size[0] * scale_factor
        assert large_patch_size[1] == small_patch_size[1] * scale_factor
        assert high_res_config['target_size'][0] == low_res_config['target_size'][0] * scale_factor
        assert high_res_config['target_size'][1] == low_res_config['target_size'][1] * scale_factor

        self.original_ds = original_ds
        self.scale_factor = scale_factor
        self.stride = stride

        # Processed datasets
        low_cache_dir = os.path.join(cache_dir, "low_res")
        high_cache_dir = os.path.join(cache_dir, "high_res")
        os.makedirs(low_cache_dir, exist_ok=True)
        os.makedirs(high_cache_dir, exist_ok=True)

        self.low_res_ds = ProcessedDataset(original_ds, cache_dir=low_cache_dir, cache_rebuild=cache_rebuild, **low_res_config)
        self.high_res_ds = ProcessedDataset(original_ds, cache_dir=high_cache_dir, cache_rebuild=cache_rebuild, **high_res_config)

        # Patch extractors
        low_patch_cache = os.path.join(cache_dir, "low_patches")
        high_patch_cache = os.path.join(cache_dir, "high_patches")
        self.small_patch_extractor = PatchExtractor(small_patch_size, stride, low_patch_cache, low_res_config['target_size'])
        self.large_patch_extractor = PatchExtractor(large_patch_size, stride * scale_factor, high_patch_cache, high_res_config['target_size'])

        assert self.small_patch_extractor.num_patches_per_image == self.large_patch_extractor.num_patches_per_image

        self.num_images = len(self.low_res_ds)
        self.num_patches_per_image = self.small_patch_extractor.num_patches_per_image
        Logger.info(f"Processing {self.num_images} images with {self.num_patches_per_image} patches each")

        # Precompute patches
        Logger.step("Precomputing patches")
        self.small_patches = []
        self.large_patches = []
        self.metadata = []
        for img_idx in tqdm(range(self.num_images), desc="Precomputing patches"):
            low_tensor = self.low_res_ds.data[img_idx]
            high_tensor = self.high_res_ds.data[img_idx]
            low_pil = transforms.ToPILImage()(low_tensor.squeeze())
            high_pil = transforms.ToPILImage()(high_tensor.squeeze())
            small_patches = self.small_patch_extractor.process(low_pil, img_idx)
            large_patches = self.large_patch_extractor.process(high_pil, img_idx)
            label = self.low_res_ds.labels[img_idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            for patch_idx in range(self.num_patches_per_image):
                self.small_patches.append(small_patches[patch_idx])
                self.large_patches.append(large_patches[patch_idx])
                self.metadata.append((label, img_idx, patch_idx))
        self.small_patches = torch.stack(self.small_patches)
        self.large_patches = torch.stack(self.large_patches)
        Logger.info(f"Precomputed {len(self.small_patches)} patches")

    def __len__(self):
        return len(self.small_patches)

    def __getitem__(self, idx):
        return (self.metadata[idx][0], self.metadata[idx][1], self.metadata[idx][2], self.small_patches[idx]), self.large_patches[idx]

    def reconstruct_low(self, img_idx, device=torch.device("cpu")):
        patches = self.small_patches[img_idx * self.num_patches_per_image:(img_idx + 1) * self.num_patches_per_image]
        return self.small_patch_extractor.reconstruct_image(patches, device)

    def reconstruct_high(self, img_idx, device=torch.device("cpu")):
        patches = self.large_patches[img_idx * self.num_patches_per_image:(img_idx + 1) * self.num_patches_per_image]
        return self.large_patch_extractor.reconstruct_image(patches, device)

# =============================================
# VISUALIZATION
# =============================================

def show_full_image_comparison(img_idx, dataset, pred_patches, mse, model_name, save_path=None):
    """Displays and optionally saves low-res, high-res, and reconstructed images"""
    Logger.step(f"Visualizing image {img_idx} for {model_name}")
    low_res = dataset.reconstruct_low(img_idx).numpy().squeeze()
    high_res = dataset.reconstruct_high(img_idx).numpy().squeeze()
    recon_image = dataset.large_patch_extractor.reconstruct_image(
        torch.tensor(pred_patches, dtype=torch.float32)
    ).numpy()
    samples = {
        'Low Resolution (14x14)': torch.tensor(low_res, dtype=torch.float32),
        'High Resolution (28x28)': torch.tensor(high_res, dtype=torch.float32),
        f'Reconstructed ({model_name})': torch.tensor(recon_image, dtype=torch.float32)
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, tensor) in zip(axes, samples.items()):
        ax.imshow(tensor.numpy(), cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(f"Image Index: {img_idx}, MSE: {mse:.4f}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            Logger.info(f"Saved visualization to {save_path}")
        except Exception as e:
            Logger.error(f"Error saving visualization to {save_path}", exc=e)
    plt.close(fig)

# =============================================
# MAIN FUNCTION
# =============================================

def main():
    """Main function for complete execution"""
    Logger.step("Starting super-resolution experiment")
    print("=== COMPARISON: CLASSICAL vs HYBRID vs QUANTUM SVRs for Super-Resolution ===")
    print("Dataset: MNIST (Classes 1 and 8, 2 samples per class for training, 1 per class for testing)")

    # Configuration
    low_res_config = {
        'target_size': (14, 14),
        'resize_alg': Image.BICUBIC,
        'image_format': None,
        'quality': None
    }
    high_res_config = {
        'target_size': (28, 28),
        'resize_alg': Image.BICUBIC,
        'image_format': None,
        'quality': None
    }
    small_patch_size = (2, 2)
    large_patch_size = (4, 4)
    stride = 1  # Low-res stride=1, high-res stride=2 (set in SuperResPatchDataset)
    scale_factor = 2
    cache_dir = "./cache"
    n_samples_per_class = 2  # 2 samples per class for training, 1 for testing

    # Generate unique cache key
    config_str = (f"low_{low_res_config['target_size'][0]}x{low_res_config['target_size'][1]}_"
                  f"high_{high_res_config['target_size'][0]}x{high_res_config['target_size'][1]}_"
                  f"patch_{small_patch_size[0]}x{small_patch_size[1]}_"
                  f"stride_{stride}_classes_1_8_samples_{n_samples_per_class}")
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    model_cache_dir = os.path.join(cache_dir, "models", config_hash)
    benchmark_dir = os.path.join(cache_dir, "benchmarks", config_hash)
    vis_dir = os.path.join(cache_dir, "visualizations", config_hash)
    os.makedirs(model_cache_dir, exist_ok=True)
    os.makedirs(benchmark_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Load MNIST and filter classes 1 and 8
    Logger.step("Preparing MNIST dataset")
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mask_1 = (mnist.targets == 1)
    mask_8 = (mnist.targets == 8)
    train_indices_1 = np.where(mask_1)[0][:n_samples_per_class]
    train_indices_8 = np.where(mask_8)[0][:n_samples_per_class]
    test_indices_1 = np.where(mask_1)[0][n_samples_per_class:n_samples_per_class+1]
    test_indices_8 = np.where(mask_8)[0][n_samples_per_class:n_samples_per_class+1]
    train_indices = np.concatenate([train_indices_1, train_indices_8])
    test_indices = np.concatenate([test_indices_1, test_indices_8])

    # Create training and test datasets
    train_dataset = torch.utils.data.Subset(mnist, train_indices)
    test_dataset = torch.utils.data.Subset(mnist, test_indices)

    # Create SuperResPatchDataset for training
    Logger.step("Creating training SuperResPatchDataset")
    train_dataset = SuperResPatchDataset(
        original_ds=train_dataset,
        low_res_config=low_res_config,
        high_res_config=high_res_config,
        small_patch_size=small_patch_size,
        large_patch_size=large_patch_size,
        stride=stride,
        scale_factor=scale_factor,
        cache_dir=cache_dir,
        cache_rebuild=False
    )

    # Create SuperResPatchDataset for testing
    Logger.step("Creating test SuperResPatchDataset")
    test_dataset = SuperResPatchDataset(
        original_ds=test_dataset,
        low_res_config=low_res_config,
        high_res_config=high_res_config,
        small_patch_size=small_patch_size,
        large_patch_size=large_patch_size,
        stride=stride,
        scale_factor=scale_factor,
        cache_dir=cache_dir,
        cache_rebuild=False
    )

    # Prepare data
    Logger.step("Preparing training and test data")
    X_train, y_train = [], []
    for x, y in tqdm(train_dataset, desc="Processing training patches"):
        X_train.append(x[3])
        y_train.append(y)
    X_train = torch.stack(X_train).flatten(start_dim=1).numpy()
    y_train = torch.stack(y_train).flatten(start_dim=1).numpy()

    X_test, y_test = [], []
    for x, y in tqdm(test_dataset, desc="Processing test patches"):
        X_test.append(x[3])
        y_test.append(y)
    X_test = torch.stack(X_test).flatten(start_dim=1).numpy()
    y_test = torch.stack(y_test).flatten(start_dim=1).numpy()

    # Select test images for visualization (one per class: indices 0 and 1)
    vis_samples = [0, 1]  # Visualize both test images (class 1 and class 8)
    Logger.info(f"Visualization image indices: {vis_samples}")

    # # Deduplicate patches (commented out to avoid errors)
    # deduplicate = True
    # if deduplicate:
    #     X_train, train_mask = deduplicate_patches(torch.tensor(X_train, dtype=torch.float32), variance_threshold=1e-4)
    #     y_train = y_train[train_mask]
    #     Logger.info(f"After deduplication: {len(X_train)} training patches")

    Logger.info(f"Training set: {len(X_train)} patches")
    Logger.info(f"Test set: {len(X_test)} patches")

    # Store results
    results = {}
    predictions = {}

    # Classical SVRs
    Logger.step("Testing Classical SVRs")
    svr_linear = SVR(kernel='linear', C=1.0)
    cache_path_linear = os.path.join(model_cache_dir, "svr_linear")
    vis_dir_linear = os.path.join(vis_dir, "svr_linear")
    train_mse_linear, test_mse_linear, y_pred_linear = evaluate_svm(
        svr_linear, X_train, X_test, y_train, y_test, "SVR Linear", cache_path_linear, vis_dir_linear, vis_samples, test_dataset
    )
    results['SVR Linear'] = {'train_mse': train_mse_linear, 'test_mse': test_mse_linear}
    predictions['SVR Linear'] = y_pred_linear

    svr_rbf = SVR(kernel='rbf', C=1.0, gamma='scale')
    cache_path_rbf = os.path.join(model_cache_dir, "svr_rbf")
    vis_dir_rbf = os.path.join(vis_dir, "svr_rbf")
    train_mse_rbf, test_mse_rbf, y_pred_rbf = evaluate_svm(
        svr_rbf, X_train, X_test, y_train, y_test, "SVR RBF", cache_path_rbf, vis_dir_rbf, vis_samples, test_dataset
    )
    results['SVR RBF'] = {'train_mse': train_mse_rbf, 'test_mse': test_mse_rbf}
    predictions['SVR RBF'] = y_pred_rbf

    # Custom RBF Kernel
    Logger.step("Testing Custom RBF Kernel")
    K_train_rbf = compute_classical_kernel_matrix(X_train, X_train, rbf_kernel_classical, gamma=0.5)
    K_test_rbf = compute_classical_kernel_matrix(X_test, X_train, rbf_kernel_classical, gamma=0.5)
    cache_path_rbf_custom = os.path.join(model_cache_dir, "rbf_custom")
    vis_dir_rbf_custom = os.path.join(vis_dir, "rbf_custom")
    train_mse_rbf_custom, test_mse_rbf_custom, y_pred_rbf_custom = evaluate_kernel_svm(
        K_train_rbf, K_test_rbf, X_test, y_train, y_test, "RBF Custom", cache_path_rbf_custom, vis_dir_rbf_custom, vis_samples, test_dataset
    )
    results['RBF Custom'] = {'train_mse': train_mse_rbf_custom, 'test_mse': test_mse_rbf_custom}
    predictions['RBF Custom'] = y_pred_rbf_custom

    # Hybrid QSVM
    Logger.step("Testing Hybrid QSVM")
    K_train_hybrid = compute_quantum_kernel_matrix_hybrid(X_train, X_train)
    K_test_hybrid = compute_quantum_kernel_matrix_hybrid(X_test, X_train)
    cache_path_hybrid = os.path.join(model_cache_dir, "qsvm_hybrid")
    vis_dir_hybrid = os.path.join(vis_dir, "qsvm_hybrid")
    train_mse_hybrid, test_mse_hybrid, y_pred_hybrid = evaluate_kernel_svm(
        K_train_hybrid, K_test_hybrid, X_test, y_train, y_test, "Quantum Hybrid", cache_path_hybrid, vis_dir_hybrid, vis_samples, test_dataset
    )
    results['QSVM Hybrid'] = {'train_mse': train_mse_hybrid, 'test_mse': test_mse_hybrid}
    predictions['QSVM Hybrid'] = y_pred_hybrid

    # Fully Quantum SVM
    Logger.step("Testing Fully Quantum SVM")
    try:
        fully_quantum_svm = FullyQuantumSVM(n_qubits=n_qubits, C=1.0)
        cache_path_fully = os.path.join(model_cache_dir, "fully_quantum_svm.pkl")
        vis_dir_fully = os.path.join(vis_dir, "qsvm_fully")
        train_mse_fully, test_mse_fully, y_pred_fully = evaluate_svm(
            fully_quantum_svm, X_train, X_test, y_train, y_test, "Fully Quantum SVR",
            cache_path_fully, vis_dir_fully, vis_samples, test_dataset
        )
        results['QSVM Fully'] = {'train_mse': train_mse_fully, 'test_mse': test_mse_fully}
        predictions['QSVM Fully'] = y_pred_fully
    except Exception as e:
        Logger.error("Error in Fully Quantum SVR", exc=e)
        results['QSVM Fully'] = {'train_mse': float('inf'), 'test_mse': float('inf')}
        predictions['QSVM Fully'] = np.zeros_like(y_test)

    # Variational QSVM
    Logger.step("Testing Variational QSVM")
    try:
        variational_svm = VariationalQuantumSVM(n_qubits=n_qubits, n_layers=2)
        cache_path_variational = os.path.join(model_cache_dir, "variational_svm.pkl")
        vis_dir_variational = os.path.join(vis_dir, "qsvm_variational")
        costs = variational_svm.fit(X_train, y_train, cache_path_variational, vis_dir_variational, vis_samples, test_dataset, X_test, y_test)
        train_mse_variational, test_mse_variational, y_pred_variational = evaluate_svm(
            variational_svm, X_train, X_test, y_train, y_test, "Variational QSVM",
            cache_path_variational, vis_dir_variational, vis_samples, test_dataset
        )
        results['QSVM Variational'] = {
            'train_mse': train_mse_variational,
            'test_mse': test_mse_variational,
            'training_costs': costs
        }
        predictions['QSVM Variational'] = y_pred_variational
    except Exception as e:
        Logger.error("Error in Variational QSVM", exc=e)
        results['QSVM Variational'] = {
            'train_mse': float('inf'),
            'test_mse': float('inf'),
            'training_costs': [0.5]
        }
        predictions['QSVM Variational'] = np.zeros_like(y_test)

    # Compute image-level metrics
    Logger.step("Computing image-level metrics")
    benchmark_results = {}
    for model_name in results:
        benchmark_results[model_name] = {
            'train_mse': results[model_name]['train_mse'],
            'test_mse': results[model_name]['test_mse'],
            'test_images': []
        }
        for img_idx in range(len(test_dataset)):
            label = test_dataset[img_idx][0][0]
            if isinstance(label, torch.Tensor):
                label = label.item()
            high_res = test_dataset.reconstruct_high(img_idx).numpy().squeeze()
            pred_patches = predictions[model_name][img_idx * test_dataset.num_patches_per_image:
                                                  (img_idx + 1) * test_dataset.num_patches_per_image]
            pred_patches = pred_patches.reshape(-1, 4, 4)
            recon_image = test_dataset.large_patch_extractor.reconstruct_image(
                torch.tensor(pred_patches, dtype=torch.float32)
            ).numpy()
            high_res = (high_res * 255).round().astype(np.uint8)
            recon_image = (recon_image * 255).round().astype(np.uint8)
            psnr_value = psnr(high_res, recon_image, data_range=255)
            ssim_value = ssim(high_res, recon_image, data_range=255)
            benchmark_results[model_name]['test_images'].append({
                'image_idx': img_idx,
                'label': int(label),
                'psnr': psnr_value,
                'ssim': ssim_value
            })
            Logger.info(f"{model_name} - Image {img_idx} (Label {label}): PSNR={psnr_value:.2f}, SSIM={ssim_value:.4f}")

    # Save benchmark results
    benchmark_path = os.path.join(benchmark_dir, "benchmarks.json")
    try:
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        Logger.info(f"Saved benchmark results to {benchmark_path}")
    except Exception as e:
        Logger.error(f"Error saving benchmark results", exc=e)

    # Visualization
    Logger.step("Visualizing sample predictions")
    img_idx = 0
    pred_patches = predictions['QSVM Hybrid'][img_idx * test_dataset.num_patches_per_image:(img_idx + 1) * test_dataset.num_patches_per_image]
    pred_patches = pred_patches.reshape(-1, 4, 4)
    mse = mean_squared_error(
        test_dataset.reconstruct_high(img_idx).numpy().squeeze(),
        test_dataset.large_patch_extractor.reconstruct_image(
            torch.tensor(pred_patches, dtype=torch.float32)
        ).numpy()
    )
    show_full_image_comparison(
        img_idx,
        test_dataset,
        pred_patches,
        mse,
        "QSVM Hybrid",
        save_path=os.path.join(vis_dir, "qsvm_hybrid", f"final_image_{img_idx}.png")
    )

    # Summary
    Logger.step("Displaying final results")
    print("\n=== FINAL RESULTS SUMMARY ===")
    print("Mean Squared Errors:")
    for model, metrics in results.items():
        print(f"  {model:<20}: Train MSE={metrics['train_mse']:.4f}, Test MSE={metrics['test_mse']:.4f}")
        for img_metrics in benchmark_results[model]['test_images']:
            print(f"    Image {img_metrics['image_idx']} (Label {img_metrics['label']}): "
                  f"PSNR={img_metrics['psnr']:.2f}, SSIM={img_metrics['ssim']:.4f}")

    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    test_mses = [results[model]['test_mse'] for model in models]
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown'][:len(models)]
    bars = plt.bar(models, test_mses, color=colors)
    plt.title('Comparison of Test MSE - MNIST Super-Resolution (Classes 1 and 8)')
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max([mse for mse in test_mses if mse < float('inf')] + [1.0]) * 1.2)
    for bar, mse in zip(bars, test_mses):
        if mse < float('inf'):
            plt.text(bar.get_x() + bar.get_width() / 2, mse + 0.01,
                     f'{mse:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()