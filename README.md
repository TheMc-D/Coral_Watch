# 🪸 CoralWatch AI

A deep learning tool that detects coral bleaching from underwater images, built to support citizen-science reef conservation monitoring.

## Results
- **96% test accuracy** on 257 unseen images
- **98% recall** on bleached coral detection
- Binary classification: Healthy vs Bleached

## How It Works
Fine-tuned a pretrained MobileNetV2 convolutional neural network on 9,662 labeled underwater reef images using transfer learning. Trained on Google Colab GPU.

## Project Structure
Coral_Watch/
├── notebooks/       # Data exploration, training, evaluation
├── models/          # Trained model weights (see below)
├── app/             # Streamlit web app
└── data/            # Dataset (not included, see below)

## Setup

### 1. Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn torch torchvision jupyter streamlit
```

### 2. Get the dataset
Download the [Coral Reef Images dataset](https://www.kaggle.com) and place it in `data/raw/`.

### 3. Get the model
This is Unavailable at the moment.

### 4. Run the app
```bash
cd app
streamlit run app.py
```

## Tech Stack
Python · PyTorch · MobileNetV2 · Transfer Learning · Streamlit · Google Colab