import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision import models, transforms
import torch.nn as nn
import os

# -------------------------------
# SETTINGS
# -------------------------------
st.set_page_config(page_title="DFU Detection", layout="wide")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# -------------------------------
# FINAL DARK UI (FIXED)
# -------------------------------
st.markdown("""
<style>

/* FULL DARK BACKGROUND */
[data-testid="stAppViewContainer"],
.block-container {
    background-color: #0D1117;
    color: white;
}

/* TEXT VISIBILITY FIX */
h1, h2, h3, h4, h5, h6, p, span, div {
    color: white !important;
}

/* CENTER CONTENT */
.block-container {
    max-width: 1100px;
    margin: auto;
}

/* REMOVE HEADER WHITE */
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: #0D1117;
}

/* CARDS */
.section {
    background-color: #161B22;
    padding: 25px;
    border-radius: 12px;
    margin-top: 25px;
    border: 1px solid #30363D;
}

/* RESULT BOX */
.result-box {
    background-color: #161B22;
    color: #58A6FF;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 18px;
    border: 1px solid #30363D;
}

/* SUCCESS */
.success-box {
    background-color: #238636;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
}

/* WARNING */
.warning-box {
    background-color: #30363D;
    color: #F2C94C;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background-color: #161B22;
    border-radius: 10px;
    padding: 10px;
}

/* IMAGES */
img {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODELS
# -------------------------------
yolo_model = YOLO("runs/runs/detect/train/weights/best.pt")
yolo_model.to("cpu")

classifier = models.efficientnet_b0(pretrained=False)
classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, 5)
classifier.load_state_dict(torch.load("classifier.pth", map_location="cpu"))
classifier.eval()

classes = [
    "Mild Ulcer (Grade 1)",
    "Moderate Ulcer (Grade 2)",
    "Severe Ulcer (Grade 3)",
    "Critical Ulcer (Grade 4)",
    "Normal (No Ulcer)"
]

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# -------------------------------
# GRAD-CAM
# -------------------------------
def generate_gradcam(model, image_tensor, target_class):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.features[-1]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    loss = output[0, target_class]

    model.zero_grad()
    loss.backward()

    grads = gradients[0].detach().numpy()[0]
    acts = activations[0].detach().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (160, 160))

    if cam.max() != 0:
        cam = cam / cam.max()

    handle_f.remove()
    handle_b.remove()

    return cam

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<h1>Diabetic Foot Ulcer Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p>AI-powered detection, severity analysis, and explainability</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Foot Image", type=["jpg", "png"])

# -------------------------------
# MAIN PIPELINE
# -------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # -------- IMAGE SECTION --------
    st.markdown('<div class="section">', unsafe_allow_html=True)

    col1, col2 = st.columns([1,1], gap="large")

    with col1:
        st.image(image, caption="Original Image")

    results = yolo_model(img_np, device="cpu")
    yolo_img = results[0].plot()

    with col2:
        st.image(yolo_img, caption="Ulcer Detection")

    st.markdown('</div>', unsafe_allow_html=True)

    # -------- DETECTION --------
    boxes = results[0].boxes.xyxy

    if boxes is not None and len(boxes) > 0:
        st.markdown('<div class="success-box">✔ Ulcer Detected</div>', unsafe_allow_html=True)
        x1, y1, x2, y2 = map(int, boxes[0])
        cropped = img_np[y1:y2, x1:x2]
    else:
        st.markdown('<div class="warning-box">No detection → analyzing full image</div>', unsafe_allow_html=True)
        cropped = img_np

    st.markdown("---")

    # -------- CLASSIFICATION --------
    st.markdown('<div class="section">', unsafe_allow_html=True)

    st.markdown("### Severity Analysis")

    crop_pil = Image.fromarray(cropped)
    input_tensor = transform(crop_pil).unsqueeze(0)

    with torch.no_grad():
        outputs = classifier(input_tensor)
        _, pred = torch.max(outputs, 1)

    result = classes[pred.item()]

    st.markdown(
        f'<div class="result-box">Predicted Condition: {result}</div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # -------- GRAD-CAM --------
    st.markdown('<div class="section">', unsafe_allow_html=True)

    st.markdown("### Model Explainability")

    try:
        cam = generate_gradcam(classifier, input_tensor, pred.item())

        cam = (cam * 255).astype(np.uint8)
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

        original = cv2.resize(cropped, (160, 160))
        overlay = cv2.addWeighted(original, 0.6, cam, 0.4, 0)

        st.image(overlay, width=400)

    except:
        st.warning("Grad-CAM could not be generated")
    st.info("Grad-CAM highlights the regions the model focuses on while making predictions. Warmer colors indicate higher importance.")
    st.markdown('</div>', unsafe_allow_html=True)
