# 🪸 CoralWatch AI

A deep learning tool that detects coral bleaching from underwater images, built to support citizen-science reef conservation monitoring.

## Results
| Metric | Score |
|--------|-------|
| Test Accuracy | 96% |
| Bleached Recall | 98% |
| Healthy Precision | 97% |
| F1 Score (avg) | 0.96 |

## Dataset
- **Source:** Coral Reef Images Dataset (Kaggle)
- **Total images:** 10,382 (RGB, .jpg/.png)
- **Classes:** Bleached (5,355) · Healthy (5,027)
- **Split:** 9,662 train / 463 validation / 257 test
- **Preprocessing:** Resize to 224×224, normalization (ImageNet stats)
- **Augmentation:** Random horizontal flip, random rotation (±15°)

## Model Architecture
- **Base model:** MobileNetV2 pretrained on ImageNet
- **Approach:** Transfer learning — base layers frozen, custom classification head trained
- **Head:** Linear(1280 → 2)
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 5
- **Hardware:** Google Colab T4 GPU

## Training Progress
| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 89.4% | 90.1% |
| 2 | 92.1% | 88.6% |
| 3 | 93.1% | 88.6% |
| 4 | 93.3% | 90.9% |
| 5 | 93.0% | 92.0% |

## How It Works
Fine-tuned a pretrained MobileNetV2 convolutional neural network on 9,662 labeled underwater reef images using transfer learning. Trained on Google Colab GPU.

## Project Structure
Coral_Watch/

├── notebooks/

│   ├── session2_classic_ml.ipynb      # Classic ML workflow

│   ├── session5_data_exploration.ipynb # Dataset exploration

│   └── session7_training.ipynb        # Model training & evaluation

├── models/          # Trained model weights (see below)

├── app/

│   └── app.py       # Streamlit web app

└── data/            # Dataset (not included)

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

## Future Work
- Add severity scoring (mild / moderate / severe bleaching)
- Train on larger dataset with more reef species
- Deploy as public Streamlit web app
