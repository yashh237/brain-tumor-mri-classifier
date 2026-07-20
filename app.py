import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Configuration
# ---------------------------
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MODEL_PATH = "brain_tumor_resnet50.pth"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------------------
# Grad-CAM class (same as before)
# ---------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx, output

# ---------------------------
# Load model (cached so it only loads once, not on every interaction)
# ---------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()
grad_cam = GradCAM(model, model.layer4)

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="wide")
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload a brain MRI image to classify it as glioma, meningioma, pituitary tumor, or no tumor, with an explainable AI heatmap showing what the model focused on.")

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_img = Image.open(uploaded_file).convert("RGB")

    input_tensor = eval_transform(original_img).unsqueeze(0).to(device)
    cam, predicted_idx, raw_output = grad_cam.generate(input_tensor)

    probabilities = torch.softmax(raw_output, dim=1)[0].detach().cpu().numpy()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx] * 100

    resized_original = original_img.resize((224, 224))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded MRI")
        st.image(resized_original, use_container_width=True)

    with col2:
        st.subheader("Grad-CAM Explanation")
        fig, ax = plt.subplots()
        ax.imshow(resized_original)
        ax.imshow(cam, cmap="jet", alpha=0.5)
        ax.axis("off")
        st.pyplot(fig)

    st.subheader("Prediction Result")
    if predicted_class == "notumor":
        st.success(f"**No tumor detected** — Confidence: {confidence:.1f}%")
    else:
        st.error(f"**{predicted_class.capitalize()} detected** — Confidence: {confidence:.1f}%")

    st.subheader("Confidence Breakdown")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name.capitalize()}: {probabilities[i]*100:.1f}%")
        st.progress(float(probabilities[i]))

    st.caption("⚠️ This tool is for educational/portfolio demonstration purposes only and is not a substitute for professional medical diagnosis.")
else:
    st.info("Please upload an MRI image to get started.")