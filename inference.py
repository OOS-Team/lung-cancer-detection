import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F
import gradio as gr

class LungCancerPredictor:
    def __init__(self, model_path="lung-cancer-model"):
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        self.model.eval()
        
    def predict_with_gradcam(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt")
        
        # Forward pass with gradient computation
        self.model.zero_grad()
        outputs = self.model(**inputs)
        pred_logits = outputs.logits
        pred_probs = F.softmax(pred_logits, dim=1)
        
        # Get prediction
        pred_class = pred_probs.argmax(1).item()
        pred_conf = pred_probs[0, pred_class].item()
        
        # Generate Grad-CAM
        pred_logits[0, pred_class].backward()
        gradients = self.model.classifier.get_gradients()[0]
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Get activations
        activations = self.model.classifier.activations
        for i in range(activations.size()[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap = heatmap / torch.max(heatmap)
        
        # Create visualization
        plt.figure(figsize=(12, 4))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        # Heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap.detach().numpy(), cmap='jet')
        plt.title("Activation Map")
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        heatmap = heatmap.detach().numpy()
        heatmap = np.uint8(255 * heatmap)
        heatmap = Image.fromarray(heatmap).resize(image.size)
        overlay = Image.blend(image, Image.fromarray(heatmap), 0.5)
        plt.imshow(overlay)
        plt.title(f"Prediction: {self.model.config.id2label[pred_class]}\nConfidence: {pred_conf:.2%}")
        plt.axis('off')
        
        return plt.gcf()

def create_gradio_interface():
    predictor = LungCancerPredictor()
    
    def predict(image):
        return predictor.predict_with_gradcam(image)
    
    iface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="filepath"),
        outputs=gr.Plot(),
        title="Lung Cancer Detection with Visualization",
        description="Upload a lung histopathology image to get prediction and attention visualization"
    )
    
    return iface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch()