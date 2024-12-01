# Lung Cancer Detection Model

This project implements a deep learning model for detecting lung cancer in histopathological images using transfer learning with the ConvNeXT architecture.

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py`
4. Run inference UI: `python inference.py`

## Model Architecture

We use the ConvNeXT base model fine-tuned on lung cancer histopathological images. The model classifies images into three categories:
- Normal lung tissue (lung_n)
- Lung adenocarcinoma (lung_aca)
- Lung squamous cell carcinoma (lung_scc)

## Dataset

We use the Lung and Colon Cancer Histopathological Images dataset available on HuggingFace.

## Visualization

The inference script includes Grad-CAM visualization to highlight regions that influenced the model's decision.