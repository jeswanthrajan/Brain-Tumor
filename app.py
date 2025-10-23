import os
import sqlite3
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import pennylane as qml
import numpy as np
from flask import Flask, render_template, request, session, flash, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ==========================
# Flask Config
# ==========================
app = Flask(__name__)
app.secret_key = "dyuiknbvcxswe678ijc6i"
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================
# SQLite3 Database
# ==========================
DB_NAME = "users.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

# Create table if not exists
with get_db_connection() as conn:
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        phone TEXT,
                        password TEXT NOT NULL
                    )''')
    conn.commit()

# ==========================
# PyTorch + Pennylane Hybrid Model
# ==========================
IMG_SIZE = 128
NUM_CLASSES = 4
N_QUBITS = 6
N_Q_LAYERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['glioma', 'meningioma', 'Normal', 'pituitary']

class SimpleCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
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

# Quantum layer
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

weight_shapes = {"weights": (N_Q_LAYERS, N_QUBITS, 3)}
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# Hybrid Model
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

# Load pretrained weights
cnn = SimpleCNN(out_dim=128).to(DEVICE)
hybrid = HybridModel(feat_dim=128, n_qubits=N_QUBITS, num_classes=NUM_CLASSES).to(DEVICE)

checkpoint = torch.load("best_cnn_vqc.pt", map_location=DEVICE)
cnn.load_state_dict(checkpoint['cnn'])
hybrid.load_state_dict(checkpoint['hybrid'])
cnn.eval()
hybrid.eval()

# Image preprocessing
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])

# ==========================
# Flask Routes
# ==========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password)

        try:
            conn = get_db_connection()
            conn.execute("INSERT INTO users (name, email, phone, password) VALUES (?, ?, ?, ?)",
                         (name, email, phone, hashed_password))
            conn.commit()
            conn.close()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already registered.", "danger")

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user"] = {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"]
            }
            flash("Login successful!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid credentials.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("index"))

@app.route("/about")
def about():
    return render_template("about.html")

# ---------------- IMAGE PREDICTION ROUTE ----------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part", "danger")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file", "danger")
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess image
            img = Image.open(filepath).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                feats = cnn(img_tensor)
                logits = hybrid(feats)
                probs = torch.softmax(logits, dim=1)
                conf, pred_class = torch.max(probs, dim=1)

            confidence = conf.item()
            if confidence < 0.7:  # adjustable threshold
                pred_label = "❌ Unknown / Out of domain"
            else:
                pred_label = CLASS_NAMES[pred_class.item()]

            return render_template("result.html",
                                   filename=filename,
                                   prediction=pred_label,
                                   confidence=f"{confidence*100:.2f}%")

    return render_template("predict.html")


# ==========================
# Run App
# ==========================
if __name__ == "__main__":
    app.run(debug=True)
