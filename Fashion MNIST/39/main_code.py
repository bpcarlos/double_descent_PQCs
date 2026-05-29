import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from mnist1d.data import make_dataset, get_dataset_args
import pennylane as qml
from torch.autograd import grad
from sklearn.preprocessing import MinMaxScaler
import time
from torch.utils.data import DataLoader, TensorDataset
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

def compute_ZS(model, X):
    """
    Compute the Jacobian matrix of function per sample Z_S = [∇θ fθ(x1); ...; ∇θ fθ(xn)]
    for a model fθ and input batch X.

    Args:
        model: torch.nn.Module
        X: input tensor of shape (n_samples, input_dim)

    Returns:
        Z_S: Jacobian matrix of shape (n_samples * output_dim, n_params)
    """
    model.eval()
    X = X.requires_grad_(True)

    # Forward pass
    output = model(X)  # shape: [n_samples, output_dim]
    n_samples, output_dim = output.shape

    # Flatten model parameters into a vector
    params = [p for p in model.parameters() if p.requires_grad]

    # Store gradients row-by-row
    Z_rows = []

    for i in range(n_samples):
        for k in range(output_dim):
            grad_output = torch.zeros_like(output)
            grad_output[i, k] = 1.0  # Select derivative wrt output[i, k]

            grads = torch.autograd.grad(
                outputs=output,
                inputs=params,
                grad_outputs=grad_output,
                retain_graph=True,
                create_graph=False
            )
            grad_vector = torch.cat([g.reshape(-1) for g in grads])  # Flatten
            Z_rows.append(grad_vector)

    Z_S = torch.stack(Z_rows, dim=0)  # Shape: (n_samples * output_dim, n_params)
    return Z_S

def gauss_newton_mse_from_tensor(model, X_train):
    torch.set_default_dtype(torch.float64)
    model.eval()
    device = next(model.parameters()).device
    X_train = X_train.to(device)

    n = X_train.size(0)
    num_params = sum(p.numel() for p in model.parameters())
    gn_matrix = torch.zeros((num_params, num_params), dtype=torch.float64, device=device)

    for i in range(n):
        x_i = X_train[i:i+1].requires_grad_(True)
        f_i = model(x_i)

        # Compute gradients of each output dimension w.r.t. parameters
        grads = []
        for j in range(f_i.shape[1]):  # f_i has shape [1, K]
            grad_output = torch.zeros_like(f_i)
            grad_output[0, j] = 1.0
            g = grad(f_i, model.parameters(), grad_outputs=grad_output,
                     retain_graph=True, create_graph=False)
            g = torch.cat([p.view(-1) for p in g])  # flatten
            grads.append(g)

        # Form Jacobian J_i of shape [K, num_params]
        J_i = torch.stack(grads, dim=0)
        gn_matrix += J_i.T @ J_i

    gn_matrix /= n
    return gn_matrix

# tol is the precision below which eigenvalues are counted as zero
def min_nonzero_eigval(H, tol=1e-10):
    if isinstance(H, torch.Tensor):
        H = H.detach().cpu().numpy()

    eigvals = np.linalg.eigvalsh(H)

    # Filter out (near-)zero eigenvalues
    nonzero_eigvals = eigvals[eigvals > tol]

    if nonzero_eigvals.size == 0:
        return 0.0  # or np.nan or raise an exception
    else:
        return np.min(nonzero_eigvals), eigvals

def compute_avg_grad_mse(model, X, y):
    """
    Compute the average gradient of the MSE loss for a batch of inputs.

    Args:
        model: torch.nn.Module
        X: input tensor of shape (n_samples, input_dim)
        y: target tensor of shape (n_samples, output_dim)

    Returns:
        avg_grad: Gradient vector of shape (n_params,)
    """
    model.eval()
    criterion = nn.MSELoss(reduction='mean')  # average over all elements

    # Ensure parameters are being tracked for gradients
    for param in model.parameters():
        param.requires_grad = True

    # Zero any existing gradients
    model.zero_grad()

    # Forward pass
    preds = model(X)  # shape: (n_samples, output_dim)
    loss = criterion(preds, y)

    # Backward pass
    loss.backward()

    # Collect and flatten gradients
    grads = [p.grad.reshape(-1) for p in model.parameters() if p.requires_grad]
    avg_grad = torch.cat(grads)  # Shape: (n_params,)

    return avg_grad

def compute_functional_hessian(model, X, loss_fn, y):
    """
    Compute the functional Hessian H_f^S(θ) as defined in the paper.

    Args:
        model: torch.nn.Module
        X: input tensor of shape (n_samples, input_dim)
        loss_fn: callable loss function (takes predictions and targets, returns scalar loss)
        y: target tensor of shape (n_samples, output_dim)

    Returns:
        H_f: Functional Hessian of shape (n_params, n_params)
    """
    model.eval()
    X = X.requires_grad_(True)

    # Get model parameters as a list
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    # Forward pass
    outputs = model(X)  # shape: [n_samples, output_dim]
    n_samples, output_dim = outputs.shape

    # Compute gradient of loss w.r.t. model outputs
    outputs.retain_grad()
    loss = loss_fn(outputs, y)
    grad_outputs = torch.autograd.grad(loss, outputs, retain_graph=True, create_graph=True)[0]  # (n_samples, output_dim)
    print('nabla_f loss: ', grad_outputs)
    H_f = torch.zeros((n_params, n_params), device=X.device)

    # Loop over samples and output dimensions
    for i in range(n_samples):
        for k in range(output_dim):
            # Get scalar grad coefficient
            grad_coeff = grad_outputs[i, k]

            # Get Hessian of f_theta^k(x_i) w.r.t. params
            grads = torch.autograd.grad(
                outputs=outputs[i, k],
                inputs=params,
                grad_outputs=torch.ones(()).to(X.device),
                create_graph=True,
                retain_graph=True
            )
            grad_vector = torch.cat([g.reshape(-1) for g in grads])

            # Compute Hessian of this scalar output w.r.t. params
            hess_rows = []
            for g in grad_vector:
                second_grads = torch.autograd.grad(
                    g, params, retain_graph=True, create_graph=False
                )
                hess_row = torch.cat([sg.reshape(-1) for sg in second_grads])
                hess_rows.append(hess_row)

            H_fk = torch.stack(hess_rows, dim=0)  # (n_params, n_params)

            # Weight by grad_coeff and add to H_f
            H_f += grad_coeff * H_fk

    H_f /= n_samples
    return H_f

def normalize_to_minus_pi_pi(X):
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)  # Normalize to [0, 1]
    return X_norm * np.pi - np.pi / 2  # [-pi/2, pi/2]


def load_fashion_mnist_multiclass(
    selected_classes=(0, 1),
    n_train=20,
    n_test=600,
    n_val=50,
    pca_dim=10,
    seed=42,
    root="data",
):
    """
    Load Fashion-MNIST, select subset of classes, and return tensor datasets.
    Labels are one-hot encoded and then mapped from {0,1} to {-1,1}.
    """

    # Load Fashion-MNIST from torchvision
    tfm = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root=root, train=True, download=True, transform=tfm)
    test_ds = datasets.FashionMNIST(root=root, train=False, download=True, transform=tfm)

    def ds_to_numpy(ds):
        xs, ys = [], []
        for img, y in ds:
            xs.append(img.numpy().reshape(-1))  # 28x28 -> 784
            ys.append(int(y))
        return np.stack(xs), np.array(ys, dtype=np.int64)

    x_train_full, y_train_full = ds_to_numpy(train_ds)
    x_test_full, y_test_full = ds_to_numpy(test_ds)

    x = np.concatenate([x_train_full, x_test_full], axis=0)
    y = np.concatenate([y_train_full, y_test_full], axis=0)

    # Filter selected classes
    mask = np.isin(y, selected_classes)
    x_filtered = x[mask]
    y_filtered = y[mask]

    # PCA
    pca = PCA(n_components=pca_dim, random_state=seed)
    x_pca = pca.fit_transform(x_filtered)
    x_filtered = normalize_to_minus_pi_pi(x_pca)

    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False, categories=[list(selected_classes)])
    y_oh = encoder.fit_transform(y_filtered.reshape(-1, 1))

    # Convert 0 -> -1, 1 -> +1
    y_oh = np.where(y_oh == 0, -1, 1)

    # Random subsample
    np.random.seed(seed)
    all_indices = np.random.permutation(len(x_filtered))

    idx_train = all_indices[:n_train]
    idx_test = all_indices[n_train:n_train + n_test]
    idx_val = all_indices[n_train + n_test:n_train + n_test + n_val]

    X_train = torch.tensor(x_filtered[idx_train], dtype=torch.float32)
    y_train = torch.tensor(y_oh[idx_train], dtype=torch.float32)

    X_test = torch.tensor(x_filtered[idx_test], dtype=torch.float32)
    y_test = torch.tensor(y_oh[idx_test], dtype=torch.float32)

    X_val = torch.tensor(x_filtered[idx_val], dtype=torch.float32)
    y_val = torch.tensor(y_oh[idx_val], dtype=torch.float32)

    return X_train, y_train, X_test, y_test, X_val, y_val, len(selected_classes)

def load_mnist1d_multiclass(selected_classes=[0, 1], n_train=20, n_test=600, n_val=50, pca_dim=10, seed=42):
    """
    Load MNIST1D, select subset of classes, and return tensor datasets.
    Labels are one-hot encoded for use with MSE loss.
    """
    # Load full MNIST1D dataset
    defaults = get_dataset_args()
    data = make_dataset(defaults)
    x, y, _ = data['x'], data['y'], data['t']  # x: (70000, 40), y: (70000,)

    # Filter selected classes
    mask = np.isin(y, selected_classes)
    x_filtered = x[mask]
    y_filtered = y[mask]

    # Apply PCA
    pca = PCA(n_components=pca_dim)
    x_pca = pca.fit_transform(x_filtered)
    x_filtered = normalize_to_minus_pi_pi(x_pca)

    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False, categories=[selected_classes])
    y_oh = encoder.fit_transform(y_filtered.reshape(-1, 1))

    # Convert 0 → -1
    y_oh = np.where(y_oh == 0, -1, 1)

    # this was my test to see if downscaling y has the same effect as upscaling f
    #y_oh = y_oh / 100

    # Random subsample
    np.random.seed(seed)
    all_indices = np.random.permutation(len(x_filtered))

    # Define train/test split
    idx_train = all_indices[:n_train]
    idx_test = all_indices[n_train:n_train + n_test]
    idx_val = all_indices[n_train + n_test:n_train+n_test+n_val]

    X_train = torch.tensor(x_filtered[idx_train], dtype=torch.float32)
    y_train = torch.tensor(y_oh[idx_train], dtype=torch.float32)
    X_test = torch.tensor(x_filtered[idx_test], dtype=torch.float32)
    y_test = torch.tensor(y_oh[idx_test], dtype=torch.float32)
    X_val = torch.tensor(x_filtered[idx_val], dtype=torch.float32)
    y_val = torch.tensor(y_oh[idx_val], dtype=torch.float32)

    return X_train, y_train, X_test, y_test, X_val, y_val, len(selected_classes)

def load_synthetic_multiclass_linear(weights, input_dim=5, n_train=100, output_dim=3, n_test=1000, noise_std=0.0, seed=42):
    """
    Generate synthetic data from a multi-dimensional linear function (no bias).
    Outputs are continuous scalar values (regression).
    """
    np.random.seed(seed)

    # Linear weight matrix [input_dim x output_dim]
    weights = weights # np.random.randn(input_dim, output_dim)

    n_total = n_train + n_test

    # Generate inputs
    X = np.random.uniform(-np.pi/2, np.pi/2, size=(n_total, input_dim))

    np.random.seed(seed)
    # Generate continuous multi-output targets
    y = X @ weights + np.random.normal(0, noise_std, size=(n_total, output_dim))

    # Normalize inputs and outputs
    #X = normalize_to_minus_pi_pi(X)
    #y = normalize_to_minus_one_one(y)

    # Convert to PyTorch tensors and split
    X_train = torch.tensor(X[:n_train], dtype=torch.float32)
    y_train = torch.tensor(y[:n_train], dtype=torch.float32)
    X_test = torch.tensor(X[n_train:], dtype=torch.float32)
    y_test = torch.tensor(y[n_train:], dtype=torch.float32)

    return X_train, y_train, X_test, y_test

#%% model
def create_quantum_layer(n_qubits, depth, n_classes):
    weight_shapes = {"weights": (depth, n_qubits, 3)}

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def quantum_circuit(inputs, weights):
        n_layers = weights.shape[0]

        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RX(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
                qml.RY(weights[layer, i, 2], wires=i)

            for i in range(n_qubits):
                qml.CZ(wires=[i, (i + 1) % n_qubits])

            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

        return [qml.expval(qml.PauliX(i)) for i in range(n_classes)]

    # Pass both sets of weights to the TorchLayer
    qnode = qml.QNode(quantum_circuit, dev, interface="torch", diff_method="backprop")
    return qml.qnn.TorchLayer(qnode, weight_shapes)

class QuantumModel(nn.Module):
    def __init__(self, q_layer, sm_temp=1, c=1):
        super().__init__()
        self.q_layer = q_layer
        self.sm_temp = sm_temp
        # when we want to make c a trainable parameter, we use nn.Parameter as below
        self.c = c  # nn.Parameter(torch.tensor(c, dtype=torch.float32))

    def forward(self, x):
        return self.q_layer(x)*self.c  #torch.exp(self.c) # for trainable c I found exp(c) working better

        #x = (self.q_layer(x) + 1) / 2  # map to [0, 1]
        #return x
        #x = self.q_layer(x) / self.sm_temp
        #return torch.nn.functional.softmax(x, dim=1)
        #print('scale: ',(torch.max(torch.max(y_train), torch.abs(torch.min(y_train))))
        
def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test,
                epochs=3500, lr=1, loss=nn.MSELoss(),
                print_every=200, batch_size=16, snapshot_every=50):
    params = sum(p.numel() for p in model.parameters())

    lr=lr/(params)  # learning rate depends on number of parameters (more params need smaller lr)
    print('learning rate: ', lr)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = loss

    # learning rate scheduler as in the Schölkopf paper
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//4, gamma=0.75)

    train_loss = []
    val_loss = []
    val_epochs = []
    last_time = time.time()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        # ---------- Training ----------
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_loss)

        if epoch % print_every == 0 or epoch == 10 or epoch == epochs:
            # ---------- Validation ----------
            model.eval()
            val_epoch_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    val_epoch_loss += loss.item()
            avg_val_loss = val_epoch_loss / len(val_loader)
            val_loss.append(avg_val_loss)
            val_epochs.append(epoch)
            now = time.time()
            elapsed = now - last_time
            last_time = now
            print(f"Epoch {epoch:>3}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Elapsed: {elapsed:.2f}s")

    # ---------- Plot loss ----------
    plt.plot(range(1, len(train_loss)+1), train_loss, label="Train Loss")
    plt.plot(val_epochs, val_loss, "o-", label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # ---------- Final evaluation ----------
    model.eval()
    with torch.no_grad():
        # Test
        test_pred = model(X_test)
        pred_labels = test_pred.argmax(dim=1)
        true_labels = y_test.argmax(dim=1)
        correct = (pred_labels == true_labels).sum().item()
        test_acc = correct / len(y_test)
        test_loss = loss_fn(test_pred, y_test).item()

        # Train
        train_pred = model(X_train)
        pred_labels = train_pred.argmax(dim=1)
        true_labels = y_train.argmax(dim=1)
        correct = (pred_labels == true_labels).sum().item()
        train_acc = correct / len(y_train)
        train_loss_final = loss_fn(train_pred, y_train).item()

        print(f'Train acc.: {train_acc:.4f}, Test acc.: {test_acc:.4f}')

    return test_loss, train_loss_final, test_acc, train_acc

#%% Experiment settings

n_train_init = 10 # this is not the actual number of train data, it's for loading sample data and checking dimensions

# for synthetic data
noise_std = 0.5
weights = np.ones((n_qubits, n_qubits))
input_dim = n_qubits

# for mnist data
pca_dim = n_qubits
selected_classes = [0,1,2,3,4,5,6,7]  #,5,6,7,8,9]

# load sample data
X_train_init, y_train_init, X_test_init, y_test_init, X_val, y_val, K = load_fashion_mnist_multiclass(selected_classes=selected_classes, n_train=n_train_init, n_test=400, n_val=250, pca_dim=pca_dim)
#X_train_init, y_train_init, X_test_init, y_test_init = load_synthetic_multiclass_linear(weights, input_dim=pca_dim, n_train=n_train_init, output_dim=n_qubits, n_test=400, noise_std=noise_std, seed=42)

output_dim = y_train_init.shape[1]
K = y_train_init.shape[1]  # K is output dimension too (not sure why I defined it twice...)

loss = nn.MSELoss() # at the moment it only works for MSE

n_train = 39
peak_pos = n_train*K  # test error peak expected at n_train*K
print(f'Peak expected at n_params = {peak_pos}')

# for testing A2, the array of depths should only contain one entry
# for plotting DD (and for testing A3), it should contain multiple, e.g., depths=[6,8,10,12,14] (for the current setting we expect the peak at 10 layers)
depths = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

reps = 10  # number of repetitions to average over

batch_size = n_train
lr = 1  # actual learning rate is lr / no_params
epochs = 2500

# c is the constant we use to scale the output: f(x, theta)*c
c = 150

total_param_arr = []
for depth in depths:
    qlayer = create_quantum_layer(n_qubits=n_qubits, depth=depth, n_classes=K)
    model = QuantumModel(q_layer=qlayer)
    params = sum(p.numel() for p in model.parameters())
    total_param_arr.append(params)

print(f'Params: {total_param_arr}')

# standard deviation of parameter initialization
std_init = 0.0001

results = [[] for r in range(reps)]
eig_vals_arr = [[]for r in range(reps)]
rank_arr = [[]for r in range(reps)]
min_eig_init_arr = [[]for r in range(reps)]
min_eig_arr = [[]for r in range(reps)]

for r in range(reps):
    print(f"Repetition {r+1}")
    seed = 43 + r
    test_loss_arr = []
    train_loss_arr = []
    test_acc_arr = []
    train_acc_arr = []
    rank_arr = []

    # load train, test and val data (atm val=test, we should change that)
    # X_train, y_train, X_test, y_test = load_synthetic_multiclass_linear(weights, input_dim=pca_dim, n_train=n_train, output_dim=n_qubits, n_test=1000, noise_std=noise_std, seed=seed)
    X_train, y_train, X_test, y_test, X_val, y_val, K = load_fashion_mnist_multiclass(selected_classes=selected_classes, n_train=n_train, n_test=1000, n_val=0, pca_dim=pca_dim, seed=seed)

    print('train shape: ', X_train.shape, 'test shape: ', X_test.shape)

    for depth in depths:
        torch.manual_seed(seed)

        # define model with depth layers, count parameters
        qlayer = create_quantum_layer(n_qubits=n_qubits, depth=depth, n_classes=K)
        model = QuantumModel(q_layer=qlayer, sm_temp=0.1, c=c)
        params = sum(p.numel() for p in model.parameters())

        # parameter initialization
        for name, param in model.named_parameters():
            if "weight" in name or "weights" in name:
                nn.init.normal_(param, mean=0.0, std=std_init)
        print('param init std: ', std_init)

        # H_o at initialization
        H_o_init = gauss_newton_mse_from_tensor(model, X_train)
        min_eig_init, eigvals = min_nonzero_eigval(H_o_init)
        min_eig_init_arr[r].append(min_eig_init)
        print(f'Lambda_min at init {min_eig_init:.2e}')

        # call training loop and store losses in list
        test_loss, train_loss, test_acc, train_acc = train_model(model, X_train, y_train, X_test, y_test, X_test, y_test, loss=loss, batch_size=batch_size,lr=lr,epochs=epochs)

        test_loss_arr.append(test_loss)
        train_loss_arr.append(train_loss)
        test_acc_arr.append(test_acc)
        train_acc_arr.append(train_acc)

        results[r].append((params, test_loss, train_loss))

        # H_o after training (same as C^S_f for MSE)
        H_o = gauss_newton_mse_from_tensor(model, X_train)
        min_eig, eigvals = min_nonzero_eigval(H_o)

        print(f'Max. eigval C^S_f:', np.max(eigvals))
        print(f'Lambda_min after train {min_eig:.2e}')
        print('Rank of H_o: ', np.linalg.matrix_rank(H_o),'/ ', np.min((n_train*K, len(eigvals))))

        rank_arr.append(np.linalg.matrix_rank(H_o))
        min_eig_arr[r].append(min_eig)

        # #H_f after training (this takes very long for large models!)
        # H_f = compute_functional_hessian(model, X_train, loss, y_train).detach().numpy()
        # H_f_rank = np.linalg.matrix_rank(H_f)
        # plt.imshow(H_f)
        # plt.colorbar()
        # plt.show()
        # print(f'Rank of H_f: {H_f_rank} / {params}')

        print(f'Depth: {depth}, Params: {params}, Test Loss: {test_loss:.4f}, Train Loss: {train_loss:.4f}')

        # check convergence to local / global minima via average train gradients
        grads = compute_avg_grad_mse(model, X_train, y_train)

        print('Avg. gradient of train set', torch.mean(grads))

        # see how large / small the largest / smallest test output is after training (typically gets large at interpolation)
        y_pred = model(X_test)
        print(f"min output: {torch.min(y_pred).item():.4f}, max output: {torch.max(y_pred).item():.4f}")

    # plot test loss and smallest non-zero eigenvalues for each repetition
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:red'
    ax1.plot(total_param_arr,test_loss_arr,marker='o',color=color)
    ax1.set_xlabel("Total Parameters")
    ax1.set_ylabel("Test Loss", color=color)
    ax1.tick_params(axis='y')
    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.plot(total_param_arr,min_eig_arr[r],marker='o',color=color)
    ax2.set_ylabel("Min Eigenvalue", color=color)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y')
    #ax2.plot(total_param_arr,min_eig_init_arr[r],marker='o',color='lightblue')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plot some other things (rank of H_o, train loss, test accuracy)
    # plt.plot(total_param_arr, rank_arr, marker='o', color='green', label='Rank of H_o')
    # plt.legend()
    # plt.show()

    # plt.plot(total_param_arr, train_loss_arr, marker='o', color='blue', label='train loss')
    # plt.legend()
    # plt.show()

    # plt.plot(total_param_arr, test_acc_arr, marker='o', color='purple', label='test acc.')
    # plt.legend()
    # plt.show()

results = np.array(results)
params, mean_losses, mean_train_loss = zip(*np.mean(results, axis=0))
_, std_losses, _ = zip(*np.std(results, axis=0))

params = np.array(params)
mean_losses = np.array(mean_losses)
std_losses = np.array(std_losses) / np.sqrt(reps)

mean_eig = np.mean(min_eig_arr, axis=0)
std_eig = np.std(min_eig_arr, axis=0) / np.log(10)
mean_eig_init = np.mean(min_eig_init_arr, axis=0)
std_eig_init = np.std(min_eig_init_arr, axis=0) / np.log(10)

# Plot
fig, ax1 = plt.subplots(figsize=(8, 5))

# Left y-axis: Test Loss
color = 'tab:red'
ax1.set_xlabel("Number of Parameters")
ax1.set_ylabel("Test Loss")
ax1.plot(params, mean_losses, marker='o', color=color, label="Population loss")
ax1.fill_between(params, mean_losses - std_losses, mean_losses + std_losses, alpha=0.2, color=color)
ax1.tick_params(axis='y')
ax1.legend(fontsize=14)

color = 'tab:blue'
ax2 = ax1.twinx()
ax2.plot(params,mean_eig,marker='o',color=color, label='Min. non-zero ev')

ax2.fill_between(params, mean_eig - std_eig, mean_eig + std_eig, alpha=0.2, color=color)
ax2.set_yscale('log')
ax2.tick_params(axis='y')
ax2.plot(total_param_arr,mean_eig_init, marker='o',color='lightblue', label='Min. non-zero ev (init)')
ax2.fill_between(params, mean_eig_init - std_eig_init, mean_eig_init + std_eig_init, alpha=0.2, color=color)
ax2.legend(fontsize=14)

# Interpolation threshold line (optional)
ax1.axvline(x=peak_pos, color='gray', linestyle='--', label=f'Interpolation Threshold (K·N = {K*n_train})')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()

plt.grid(True)
plt.tight_layout()
plt.legend(fontsize=14)
plt.show()

import pickle

# Collect all relevant experiment data in a dictionary
experiment_data = {
    "depths": depths,
    "total_param_arr": total_param_arr,
    "results": results,
    "min_eig_arr": min_eig_arr,
    "min_eig_init_arr": min_eig_init_arr,
    "rank_arr": rank_arr,
    "test_loss_arr": test_loss_arr,
    "train_loss_arr": train_loss_arr,
    "test_acc_arr": test_acc_arr,
    "train_acc_arr": train_acc_arr,
    "params": params,
    "mean_losses": mean_losses,
    "std_losses": std_losses,
    "mean_eig": mean_eig,
    "std_eig": std_eig,
    "mean_eig_init": mean_eig_init,
    "std_eig_init": std_eig_init,
    "peak_pos": peak_pos,
    "K": K,
    "n_train": n_train,
}

# Save to file
with open("experiment_results.pkl", "wb") as f:
    pickle.dump(experiment_data, f)

print("✅ Experiment data saved to 'experiment_results.pkl'")
