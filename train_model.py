import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from PIL import Image
import os

# Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Define transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets using your specific folder structure
data_dir = "/Users/ruthv/Downloads/WaterPollution/waterpoldataset"  # Your dataset path
image_datasets = {
    'train': datasets.ImageFolder(f'{data_dir}/train', transform=data_transforms),
    'val': datasets.ImageFolder(f'{data_dir}/validation', transform=data_transforms),
    'test': datasets.ImageFolder(f'{data_dir}/test', transform=data_transforms)
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=32, shuffle=(x == 'train'), num_workers=2)
    for x in ['train', 'val', 'test']
}

# Get class names (should be 'natural' and 'trash')
class_names = image_datasets['train'].classes
print(f"Classes: {class_names}")

# Modified CNN model for CAM visualization
class WaterPollutionCNN(nn.Module):
    def __init__(self):
        super(WaterPollutionCNN, self).__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.fc = nn.Linear(128, 2)  # 2 classes: natural and trash
        
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Save the feature maps for CAM visualization
        self.feature_maps = features
        
        # Global Average Pooling
        x = self.gap(features)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc(x)
        return x
    
    def get_cam(self, idx):
        # Get the weights of the output layer
        params = list(self.fc.parameters())
        weights = params[0].data
        
        # Get the weights for the predicted class
        weight_class = weights[idx].cpu()
        
        # Get the feature maps from the last convolutional layer
        feature_maps = self.feature_maps.cpu()
        
        # Create the class activation map
        batch_size, num_channels, height, width = feature_maps.shape
        cam = torch.zeros((height, width), dtype=torch.float32)
        
        # Weight the feature maps with the class weights
        for i in range(num_channels):
            cam += weight_class[i] * feature_maps[0, i, :, :]
        
        # Apply ReLU to focus on the features that have a positive influence on the class
        cam = F.relu(cam)
        
        # Normalize the CAM
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().numpy()

# Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Save best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Best val Acc: {best_acc:.4f}')
    return model, history

# Evaluate model on test set
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    return cm, report

# Plot history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Visualize prediction with CAM
def visualize_prediction_with_cam(model, image_path, predicted_class, confidence):
    """
    Visualizes the input image and its class activation map
    
    Args:
        model: The trained model
        image_path: Path to the input image
        predicted_class: Predicted class name
        confidence: Prediction confidence
    """
    # Load original image for display
    original_img = Image.open(image_path).convert('RGB')
    img_display = np.array(original_img.resize((224, 224)))
    
    # Load and preprocess image for prediction
    img_tensor = data_transforms(original_img).unsqueeze(0).to(device)
    
    # Forward pass to get CAM
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
    
    # Generate Class Activation Map
    cam = model.get_cam(pred.item())
    
    # Resize CAM to match the size of the image
    cam_resized = cv2.resize(cam, (224, 224))
    
    # Convert CAM to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Create a blended image (original + heatmap)
    alpha = 0.6  # Transparency factor
    blended = cv2.addWeighted(img_display, 1 - alpha, heatmap, alpha, 0)
    
    # Create figure for visualization
    plt.figure(figsize=(12, 5))
    
    # Display original image
    plt.subplot(1, 3, 1)
    plt.imshow(img_display)
    plt.title('Original Image')
    plt.axis('off')
    
    # Display CAM heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Class Activation Map')
    plt.axis('off')
    
    # Display blended image
    plt.subplot(1, 3, 3)
    plt.imshow(blended)
    plt.title(f'Prediction: {predicted_class} ({confidence:.2f}%)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_cam.png')
    plt.show()

# Predict on a test image with CAM visualization
def predict_test_image(image_name):
    """
    Predicts whether a specific image from the test folder contains trash or not,
    and visualizes the class activation map.
    
    Args:
        image_name (str): The filename of the image in the test folder (e.g., "image1.jpg")
                         Can be from either 'natural' or 'trash' subfolders
    
    Returns:
        str: The predicted class ('natural' or 'trash')
    """
    # Try to find the image in either the 'natural' or 'trash' subfolder
    for class_folder in ['natural', 'trash']:
        image_path = f"{data_dir}/test/{class_folder}/{image_name}"
        print(f"The image path is: {image_path}")
    
        try:
            if os.path.exists(image_path):
                print("Found the image path")
                img = Image.open(image_path).convert('RGB')
                img_tensor = data_transforms(img).unsqueeze(0).to(device)
                
                model.eval()
                with torch.no_grad():
                    output = model(img_tensor)
                    _, pred = torch.max(output, 1)
                    
                prediction = class_names[pred.item()]
                confidence = F.softmax(output, dim=1)[0][pred.item()].item() * 100
                
                print(f"Image: {image_name}")
                print(f"Prediction: {prediction}")
                print(f"Confidence: {confidence:.2f}%")
                print(f"Actual class: {class_folder}")
                
                # Visualize with CAM
                visualize_prediction_with_cam(model, image_path, prediction, confidence)
                
                return prediction
        except Exception as e:
            print(f"Error processing image: {e}")
            continue
    
    print(f"Could not find image '{image_name}' in test folder")
    return None

# Main function
def main():
    global model
    # Create model
    model = WaterPollutionCNN().to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    model, history = train_model(
        model, dataloaders, criterion, optimizer, num_epochs=10
    )
    
    # Plot training history
    plot_history(history)
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate on test set
    print("Evaluating on test set:")
    evaluate_model(model, dataloaders['test'])
    
    # Example: Predict on a specific test image with CAM visualization
    predict_test_image("image9.jpg")  # Replace with an actual image name from your test folder

if __name__ == "__main__":
    main()