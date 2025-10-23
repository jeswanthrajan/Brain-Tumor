import os, random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import pennylane as qml

# ---------------- CONFIG ----------------
DATA_DIR = "dataset"          # dataset folder with 4 class subfolders
VAL_SPLIT = 0.2               # 80/20 split
BATCH_SIZE = 16
IMG_SIZE = 128
NUM_CLASSES = 4
N_QUBITS = 6
N_Q_LAYERS = 2
NUM_EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------- DATA ----------------

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])

# Force fixed class order
class_order = ["glioma", "meningioma", "Normal", "pituitary"]
full_dataset = ImageFolder(DATA_DIR, transform=transform)

# Manually override class attributes
full_dataset.classes = class_order
full_dataset.class_to_idx = {cls: i for i, cls in enumerate(class_order)}

# Reassign sample labels according to our custom order
full_dataset.samples = [
    (path, full_dataset.class_to_idx[os.path.basename(os.path.dirname(path))])
    for path, _ in full_dataset.samples
]
full_dataset.targets = [label for _, label in full_dataset.samples]

num_total = len(full_dataset)
num_val = int(VAL_SPLIT * num_total)
num_train = num_total - num_val

train_ds, val_ds = random_split(full_dataset, [num_train, num_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Found {num_total} images. Train={num_train}, Val={num_val}, Classes={full_dataset.classes}")


# ---------------- SIMPLE CNN FEATURE EXTRACTOR ----------------
class SimpleCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ---------------- QUANTUM LAYER ----------------
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

weight_shapes = {"weights": (N_Q_LAYERS, N_QUBITS, 3)}
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# ---------------- HYBRID MODEL ----------------
class HybridModel(nn.Module):
    def __init__(self, feat_dim=128, n_qubits=N_QUBITS, num_classes=NUM_CLASSES):
        super().__init__()
        self.reduce = nn.Linear(feat_dim, n_qubits)
        self.qlayer = qlayer
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, feats):
        x = self.reduce(feats)
        x = torch.tanh(x) * np.pi  # bound inputs to [-pi, pi]
        q_out = self.qlayer(x)
        out = self.classifier(q_out)
        return out

# ---------------- TRAIN SETUP ----------------
cnn = SimpleCNN(out_dim=128).to(DEVICE)
hybrid = HybridModel(feat_dim=128, n_qubits=N_QUBITS, num_classes=NUM_CLASSES).to(DEVICE)

params = list(cnn.parameters()) + list(hybrid.parameters())
optimizer = optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

# ---------------- EVAL FUNCTION ----------------
def evaluate(cnn, hybrid, loader):
    cnn.eval()
    hybrid.eval()
    total, correct, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            feats = cnn(imgs)
            logits = hybrid(feats)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total

# ---------------- TRAIN LOOP ----------------
best_acc = 0.0
val_accuracies = []
val_losses = []

for epoch in range(1, NUM_EPOCHS + 1):
    cnn.train()
    hybrid.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
    
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        feats = cnn(imgs)
        logits = hybrid(feats)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=running_loss / len(train_loader))
    
    val_loss, val_acc = evaluate(cnn, hybrid, val_loader)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch}: Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'cnn': cnn.state_dict(),
            'hybrid': hybrid.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, "best_cnn_vqc.pt")
        print(f"✅ New best model saved (Acc={val_acc*100:.2f}%)")

print(f"Training done. Best validation accuracy: {best_acc*100:.2f}%")

# ---------------- PLOT RESULTS ----------------
plt.figure(figsize=(10,5))
plt.plot(range(1, NUM_EPOCHS+1), [a*100 for a in val_accuracies], 'b-o', label='Validation Accuracy (%)')
plt.plot(range(1, NUM_EPOCHS+1), val_losses, 'r--o', label='Validation Loss')
plt.title('Validation Accuracy and Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




##import os, random
##import numpy as np
##from tqdm import tqdm
##
##import torch
##import torch.nn as nn
##import torch.optim as optim
##from torch.utils.data import DataLoader, random_split
##import torchvision.transforms as T
##from torchvision.datasets import ImageFolder
##
##import pennylane as qml
##
### ---------------- CONFIG ----------------
##DATA_DIR = "dataset"          # dataset folder with 4 class subfolders
##VAL_SPLIT = 0.2               # 80/20 split
##BATCH_SIZE = 16
##IMG_SIZE = 128
##NUM_CLASSES = 4
##N_QUBITS = 6
##N_Q_LAYERS = 2
##NUM_EPOCHS = 60
##LR = 1e-3
##WEIGHT_DECAY = 1e-5
##SEED = 42
##DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##
### ----------------------------------------
##torch.manual_seed(SEED)
##np.random.seed(SEED)
##random.seed(SEED)
##
### ---------------- DATA ----------------
##transform = T.Compose([
##    T.Resize((IMG_SIZE, IMG_SIZE)),
##    T.RandomHorizontalFlip(),
##    T.RandomRotation(10),
##    T.ToTensor(),
##    T.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
##])
##
##full_dataset = ImageFolder(DATA_DIR, transform=transform)
##num_total = len(full_dataset)
##num_val = int(VAL_SPLIT * num_total)
##num_train = num_total - num_val
##
##train_ds, val_ds = random_split(full_dataset, [num_train, num_val])
##
##train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
##val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
##
##print(f"Found {num_total} images. Train={num_train}, Val={num_val}, Classes={full_dataset.classes}")
##
### ---------------- SIMPLE CNN FEATURE EXTRACTOR ----------------
##class SimpleCNN(nn.Module):
##    def __init__(self, out_dim=128):
##        super().__init__()
##        self.features = nn.Sequential(
##            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
##            nn.BatchNorm2d(32),
##            nn.ReLU(),
##            nn.MaxPool2d(2),
##
##            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
##            nn.BatchNorm2d(64),
##            nn.ReLU(),
##            nn.MaxPool2d(2),
##
##            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
##            nn.BatchNorm2d(128),
##            nn.ReLU(),
##            nn.AdaptiveAvgPool2d((1,1))
##        )
##        self.fc = nn.Linear(128, out_dim)
##
##    def forward(self, x):
##        x = self.features(x)
##        x = x.view(x.size(0), -1)
##        x = self.fc(x)
##        return x
##
### ---------------- QUANTUM LAYER ----------------
##dev = qml.device("default.qubit", wires=N_QUBITS)
##
##@qml.qnode(dev, interface="torch")
##def quantum_circuit(inputs, weights):
##    qml.templates.AngleEmbedding(inputs, wires=range(N_QUBITS))
##    qml.templates.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
##    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
##
##weight_shapes = {"weights": (N_Q_LAYERS, N_QUBITS, 3)}
##qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
##
### ---------------- HYBRID MODEL ----------------
##class HybridModel(nn.Module):
##    def __init__(self, feat_dim=128, n_qubits=N_QUBITS, num_classes=NUM_CLASSES):
##        super().__init__()
##        self.reduce = nn.Linear(feat_dim, n_qubits)
##        self.qlayer = qlayer
##        self.classifier = nn.Sequential(
##            nn.Linear(n_qubits, 32),
##            nn.ReLU(),
##            nn.Dropout(0.2),
##            nn.Linear(32, num_classes)
##        )
##
##    def forward(self, feats):
##        x = self.reduce(feats)
##        x = torch.tanh(x) * np.pi  # bound inputs to [-pi, pi]
##        q_out = self.qlayer(x)
##        out = self.classifier(q_out)
##        return out
##
### ---------------- TRAIN SETUP ----------------
##cnn = SimpleCNN(out_dim=128).to(DEVICE)
##hybrid = HybridModel(feat_dim=128, n_qubits=N_QUBITS, num_classes=NUM_CLASSES).to(DEVICE)
##
##params = list(cnn.parameters()) + list(hybrid.parameters())
##optimizer = optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
##criterion = nn.CrossEntropyLoss()
##
### ---------------- EVAL FUNCTION ----------------
##def evaluate(cnn, hybrid, loader):
##    cnn.eval()
##    hybrid.eval()
##    total, correct, total_loss = 0, 0, 0.0
##    with torch.no_grad():
##        for imgs, labels in loader:
##            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
##            feats = cnn(imgs)
##            logits = hybrid(feats)
##            loss = criterion(logits, labels)
##            total_loss += loss.item() * imgs.size(0)
##            preds = logits.argmax(dim=1)
##            correct += (preds == labels).sum().item()
##            total += imgs.size(0)
##    return total_loss / total, correct / total
##
### ---------------- TRAIN LOOP ----------------
##best_acc = 0.0
##for epoch in range(1, NUM_EPOCHS + 1):
##    cnn.train()
##    hybrid.train()
##    running_loss = 0.0
##    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
##    
##    for imgs, labels in pbar:
##        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
##        feats = cnn(imgs)
##        logits = hybrid(feats)
##        loss = criterion(logits, labels)
##
##        optimizer.zero_grad()
##        loss.backward()
##        optimizer.step()
##
##        running_loss += loss.item()
##        pbar.set_postfix(loss=running_loss / len(train_loader))
##    
##    val_loss, val_acc = evaluate(cnn, hybrid, val_loader)
##    print(f"Epoch {epoch}: Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%")
##
##    if val_acc > best_acc:
##        best_acc = val_acc
##        torch.save({
##            'cnn': cnn.state_dict(),
##            'hybrid': hybrid.state_dict(),
##            'optimizer': optimizer.state_dict(),
##            'epoch': epoch
##        }, "best_cnn_vqc.pt")
##        print(f"✅ New best model saved (Acc={val_acc*100:.2f}%)")
##
##print(f"Training done. Best validation accuracy: {best_acc*100:.2f}%")
