import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog
import pennylane as qml

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
N_QUBITS = 6
N_Q_LAYERS = 2
NUM_CLASSES = 4

# Same order as training
CLASS_NAMES = ["glioma", "meningioma", "Normal", "pituitary"]

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ---------------- CNN FEATURE EXTRACTOR ----------------
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
            nn.AdaptiveAvgPool2d((1, 1))
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
        x = torch.tanh(x) * np.pi
        q_out = self.qlayer(x)
        out = self.classifier(q_out)
        return out

# ---------------- LOAD MODEL ----------------
def load_best_model():
    cnn = SimpleCNN(out_dim=128).to(DEVICE)
    hybrid = HybridModel(feat_dim=128, n_qubits=N_QUBITS, num_classes=NUM_CLASSES).to(DEVICE)
    
    checkpoint = torch.load("best_cnn_vqc.pt", map_location=DEVICE)
    cnn.load_state_dict(checkpoint['cnn'])
    hybrid.load_state_dict(checkpoint['hybrid'])
    
    cnn.eval()
    hybrid.eval()
    print("✅ Loaded best model checkpoint successfully!")
    return cnn, hybrid

# ---------------- PREDICT SINGLE IMAGE ----------------
def predict_image(img_path, cnn, hybrid, threshold=0.7):
    print(f"\n🧠 Analyzing image: {img_path}\n")
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feats = cnn(img_tensor)
        logits = hybrid(feats)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    confidence = conf.item()
    if confidence < threshold:
        label = "❌ Unknown / Out of domain"
    else:
        label = CLASS_NAMES[pred.item()]

    print(f"🧩 Predicted Class: {label}")
    print("-----------------------------------------")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    img_path = filedialog.askopenfilename(
        title="📂 Select an image to test:",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )

    if img_path:
        cnn, hybrid = load_best_model()
        predict_image(img_path, cnn, hybrid)
    else:
        print("❌ No image selected. Exiting...")
