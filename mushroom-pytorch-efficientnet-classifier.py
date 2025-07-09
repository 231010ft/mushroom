import torch
from torch import nn
import torchvision
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import List, Union, Tuple
from pathlib import Path
from torchinfo import summary
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy


# Configuration
NUM_CLASSES = 176
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = Path("./merged_dataset")  # Adjust this path to your dataset location
MODEL_SAVE_PATH = "mushroom_master_a.pt"
ACCURACY_SAVE_PATH = "accuracy.txt"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 40
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
    efficient_net_weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    model = torchvision.models.efficientnet_v2_s(weights=efficient_net_weights)
    
    for param in model.features.parameters():
        param.requires_grad = False
    
    model.classifier[1] = nn.Linear(in_features=1280, out_features=NUM_CLASSES, bias=True)
    
    # 既存のtransformsにデータ拡張を追加
    # efficient_net_weights.transforms()は検証用として使い、学習用にはより強力な拡張を定義
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.TrivialAugmentWide(num_magnitude_bins=31), # 最新のデータ拡張手法
        efficient_net_weights.transforms(),
    ])
    
    # 検証用には既存のtransformsを使用
    val_transforms = efficient_net_weights.transforms()
    
    return model.to(DEVICE), train_transforms, val_transforms # train_transformsとval_transformsを返す


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
    
    print(f"Loaded {len(file_paths)} images from {len(classes)} classes")
    return file_paths, path_labels, classes


def create_dataloaders(file_paths: List[str], path_labels: List[str], 
                       classes: List[str], train_transforms, val_transforms) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: # 引数を変更
    """Create train and test dataloaders."""
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        file_paths, path_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # 学習データにはtrain_transformsを、テストデータにはval_transformsを適用
    train_dataset = MushroomDataset(X_train_paths, y_train, train_transforms, classes)
    test_dataset = MushroomDataset(X_test_paths, y_test, val_transforms, classes) # ここでval_transformsを使用
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=min(os.cpu_count(), 4)
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=min(os.cpu_count(), 4)
    )
    
    return train_dataloader, test_dataloader


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate accuracy manually to avoid torchmetrics issues."""
    correct = (predictions.argmax(dim=1) == targets).float()
    return correct.mean().item()


def train_model(model: nn.Module, train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader) -> float:
    """Train the model and return the best test accuracy."""
    # Setup training components
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)
    
    max_test_acc = -1
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_acc = 0
        
        for X, y in train_dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            # Forward pass
            logits = model(X)
            loss = loss_fn(logits, y)
            
            # Accumulate metrics
            train_loss += loss.item()
            train_acc += calculate_accuracy(logits, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Average training metrics
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        test_acc = 0
        
        with torch.inference_mode():
            for X, y in test_dataloader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                
                test_logits = model(X)
                test_loss += loss_fn(test_logits, y).item()
                test_acc += calculate_accuracy(test_logits, y)
        
        # Average test metrics
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.3f} | "
              f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.3f}")
        
        # Save best model
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            with open(ACCURACY_SAVE_PATH, "w") as f:
                f.write(str(max_test_acc))
            print(f"New best model saved with accuracy: {max_test_acc:.3f}")
    
    return max_test_acc


def load_trained_model(model_path: str = MODEL_SAVE_PATH) -> nn.Module:
    """Load a trained model from saved state dict."""
    model, _ = create_model()
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model


def test_model(model: nn.Module, test_dataloader: torch.utils.data.DataLoader,
               classes: List[str]) -> dict:
    """Test the model and return detailed metrics."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    correct_predictions = 0
    total_predictions = 0
    
    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            # Get predictions
            logits = model(X)
            predictions = torch.argmax(logits, dim=1)
            
            # Store for analysis
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            # Count correct predictions
            correct_predictions += (predictions == y).sum().item()
            total_predictions += y.size(0)
    
    # Calculate overall accuracy
    accuracy = correct_predictions / total_predictions
    
    # Calculate per-class accuracy
    class_accuracies = {}
    for i, class_name in enumerate(classes):
        class_mask = np.array(all_labels) == i
        if class_mask.sum() > 0:
            class_correct = np.array(all_predictions)[class_mask] == i
            class_accuracies[class_name] = class_correct.sum() / class_mask.sum()
        else:
            class_accuracies[class_name] = 0.0
    
    return {
        'overall_accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'predictions': all_predictions,
        'labels': all_labels
    }


def main():
    """Main training function."""
    print(f"Using device: {DEVICE}")
    
    file_paths, path_labels, classes = load_data()
    
    model, train_transforms, val_transforms = create_model() # 戻り値を変更
    train_dataloader, test_dataloader = create_dataloaders(file_paths, path_labels, classes, train_transforms, val_transforms) # 引数を変更
    
    print("\nModel Summary:")
    print(summary(model, input_size=(BATCH_SIZE, 3, 224, 224)))
    
    best_accuracy = train_model(model, train_dataloader, test_dataloader)
    print(f"\nTraining completed! Best test accuracy: {best_accuracy:.3f}")

def test_saved_model():
    """Test a saved model."""
    print(f"Testing saved model from {MODEL_SAVE_PATH}")
    
    file_paths, path_labels, classes = load_data()
    _, _, transforms = create_model() # 予測時にはval_transformsを使用
    _, test_dataloader = create_dataloaders(file_paths, path_labels, classes, transforms, transforms) # ここもtransforms, transformsにしないとエラーが出ます。あるいは、テスト用にval_transformsを渡す。
    
    model = load_trained_model()
    results = test_model(model, test_dataloader, classes)
    
    print(f"\nTest Results:")
    print(f"Overall Accuracy: {results['overall_accuracy']:.3f}")
    print("\nPer-class Accuracies:")
    for class_name, accuracy in results['class_accuracies'].items():
        print(f"  {class_name}: {accuracy:.3f}")
    
    return results

def predict_single_image(model: nn.Module, image_path: str, transforms, classes: List[str]) -> dict:
    """Predict the class of a single image."""
    model.eval()
    
    image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB)
    image = transforms(image).unsqueeze(0).to(DEVICE) # transformsはval_transformsを想定
    
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


if __name__ == "__main__":
    main()
    # test_saved_model()
    # predict on a single image
    # model, transforms = create_model()
    # model = load_trained_model()
    # file_paths, path_labels, classes = load_data(only_load_classes=True)
    # result = predict_single_image(model, "example_shroom_f.jpg", transforms, classes) # edible mushroom sporocarp ce (254) (276)
    # print(f"Predicted: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
    # print(f"All Possiblities: {result['all_probabilities']}")


"""
Output:
Model loaded from mushroom_master.pt
Loaded 0 images from 4 classes
Predicted: edible mushroom sporocarp (confidence: 0.490)
All Possiblities: {'edible sporocarp': 0.05334497615695, 'poisonous mushroom sporocarp': 0.03803054615855217, 'poisonous sporocarp': 0.41831693053245544, 'edible mushroom sporocarp': 0.49030759930610657}
"""