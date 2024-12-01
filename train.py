import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import TrainingArguments, Trainer
import evaluate
import torchvision.transforms as T

# Constants
MODEL_CHECKPOINT = "facebook/convnext-base-224"
DATASET_NAME = "keremberke/lung-and-colon-cancer-histopathological-images"
OUTPUT_DIR = "lung-cancer-model"

def load_and_prepare_dataset():
    # Load dataset
    dataset = load_dataset(DATASET_NAME)
    
    # Create label mappings
    labels = dataset["train"].features["label"].names
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # Filter for lung cancer images only
    dataset = dataset.filter(lambda x: x["label"] in ["lung_n", "lung_aca", "lung_scc"])
    
    return dataset, label2id, id2label

def preprocess_data(dataset, processor):
    # Define augmentations
    train_transforms = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
    ])

    def transform_train(example_batch):
        images = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        inputs = processor(images, return_tensors="pt")
        return inputs
    
    def transform_val(example_batch):
        images = [image.convert("RGB").resize((224, 224)) for image in example_batch["image"]]
        inputs = processor(images, return_tensors="pt")
        return inputs

    # Apply transforms
    dataset["train"] = dataset["train"].with_transform(transform_train)
    dataset["test"] = dataset["test"].with_transform(transform_val)
    
    return dataset

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = predictions.argmax(-1)
    return accuracy.compute(predictions=predictions, references=labels)

def main():
    # Load and prepare dataset
    dataset, label2id, id2label = load_and_prepare_dataset()
    
    # Load processor and model
    processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )
    
    # Preprocess dataset
    processed_dataset = preprocess_data(dataset, processor)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        compute_metrics=compute_metrics,
    )
    
    # Train model
    trainer.train()
    
    # Save the final model
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()