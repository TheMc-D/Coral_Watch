import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load('../models/model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# UI
st.title("🪸 CoralWatch AI")
st.write("Upload an underwater coral image to detect bleaching.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = output.max(1)
        confidence = torch.softmax(output, dim=1).max().item()
    
    label = "🔴 Bleached" if predicted.item() == 0 else "🟢 Healthy"
    
    st.markdown(f"### Prediction: {label}")
    st.markdown(f"**Confidence: {confidence * 100:.1f}%**")