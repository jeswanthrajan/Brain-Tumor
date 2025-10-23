# =========================================
# CNN + VQC Brain Tumour Model - TEST SCRIPT
# =========================================

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# =========================
# CONFIG
# =========================
MODEL_PATH = "cnn_vqc_brain_tumour_model.h5"
IMAGE_SIZE = (64, 64)  # must match training size
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']  # 🔹 Change to match your dataset folders

# =========================
# LOAD MODEL
# =========================
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# =========================
# FUNCTION: Predict image
# =========================
def predict_image(image_path):
    try:
        img = load_img(image_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img) / 255.0  # normalize
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

        result = f"Predicted: {CLASS_NAMES[predicted_class]} ({confidence:.2f}% confidence)"
        print(result)
        messagebox.showinfo("Prediction Result", result)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# =========================
# FILE DIALOG UI
# =========================
root = tk.Tk()
root.withdraw()  # hide main window

file_path = filedialog.askopenfilename(
    title="Select an image file for testing",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

if file_path:
    print("✅ Selected file:", file_path)
    predict_image(file_path)
else:
    print("❌ No file selected.")
