import torch
from torch import nn
import torchvision
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import List, Union, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
import sys


# Configuration
NUM_CLASSES = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = Path("./cat_dog")
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42


class MushroomDataset(torch.utils.data.Dataset):
    """Custom dataset for mushroom classification."""
    
    def __init__(self, image_paths: List[str], image_classes: List[str], 
                 transform=None, classes: List[str] = None):
        self.image_paths = image_paths
        self.image_classes = image_classes
        self.transform = transform
        self.classes = classes or []
        
        # Create class to index mapping
        indexes = list(range(len(self.classes)))
        self.class_to_index = dict(zip(self.classes, indexes))
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.class_to_index[self.image_classes[idx]]
        
        # Load and transform image
        image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB)
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_model() -> nn.Module:
    """Create and configure the EfficientNet model."""
    # Load pretrained model
    efficient_net_weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    model = torchvision.models.efficientnet_v2_s(weights=efficient_net_weights)
    
    # Freeze feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace classifier head
    model.classifier[1] = nn.Linear(in_features=1280, out_features=NUM_CLASSES, bias=True)
    
    return model.to(DEVICE), efficient_net_weights.transforms()


def load_data(only_load_classes: bool = False) -> Tuple[List[str], List[str], List[str]]:
    """Load image paths and labels from the data directory."""
    assert DATA_PATH.is_dir(), f"Data path {DATA_PATH} does not exist"
    
    file_paths = []
    path_labels = []
    classes = []
    
    for image_folder in sorted(os.listdir(DATA_PATH)):
        folder_path = DATA_PATH / image_folder
        if not folder_path.is_dir():
            continue
            
        classes.append(image_folder)
        
        if not only_load_classes:
            for image_file in os.listdir(folder_path):
                full_path = folder_path / image_file
                
                # Check if image is valid
                try:
                    torchvision.io.read_image(str(full_path), mode=torchvision.io.ImageReadMode.RGB)
                    file_paths.append(str(full_path))
                    path_labels.append(image_folder)
                except Exception as e:
                    print(f"Skipping corrupted image: {full_path}")
                    continue
    
    if not only_load_classes:
        print(f"Loaded {len(file_paths)} images from {len(classes)} classes")
    return file_paths, path_labels, classes


def create_dataloaders(file_paths: List[str], path_labels: List[str], 
                      classes: List[str], transforms) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and test dataloaders."""
    # Split data
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        file_paths, path_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Create datasets
    train_dataset = MushroomDataset(X_train_paths, y_train, transforms, classes)
    test_dataset = MushroomDataset(X_test_paths, y_test, transforms, classes)
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=min(os.cpu_count(), 4)  # Limit num_workers to prevent issues
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=min(os.cpu_count(), 4)
    )
    
    return train_dataloader, test_dataloader


def load_trained_model(model_path: str) -> nn.Module:
    """Load a trained model from saved state dict."""
    model, _ = create_model()
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def predict_single_image(model: nn.Module, image_path: str, transforms, classes: List[str]) -> dict:
    """Predict the class of a single image."""
    model.eval()
    
    # Load and preprocess image
    image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB)
    image = transforms(image).unsqueeze(0).to(DEVICE)  # Add batch dimension
    
    with torch.inference_mode():
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    return {
        'predicted_class': classes[predicted_class],
        'confidence': confidence,
        'all_probabilities': {classes[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    }


def main():
    """Main function to handle command line arguments and run prediction."""
    parser = argparse.ArgumentParser(description='Image Classification Tool')
    parser.add_argument('-m', '--model', required=True, help='Path to the trained model file')
    parser.add_argument('-i', '--image', required=True, help='Path to the image to classify')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        sys.exit(1)
    
    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        sys.exit(1)
    
    try:
        # Load model and transforms
        model, transforms = create_model()
        model = load_trained_model(args.model)
        
        # Load classes (need to get class names from data directory)
        _, _, classes = load_data(only_load_classes=True)
        
        # Predict
        result = predict_single_image(model, args.image, transforms, classes)
        
        # Output results
        result_all_probabilities_dict = result['all_probabilities']
        sorted_results = sorted(result_all_probabilities_dict.items(), key=lambda x: x[1], reverse=True)
        
        for class_name, probability in sorted_results:
            print(f"{class_name}: {probability * 100:.2f}%")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
(.venv) /home/roka/Documents/App/mushroom/.venv/bin/python /home/roka/Documents/App/mushroom/mec.py -m mushroom_master2.pt -i example_shroom_f.jpg
poisonous sporocarp: 77.89%
edible sporocarp: 13.49%
edible mushroom sporocarp: 4.32%
poisonous mushroom sporocarp: 4.30%
"""