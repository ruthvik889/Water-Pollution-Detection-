import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

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

# Define the CNN model
class WaterPollutionCNN(nn.Module):
    def __init__(self):
        super(WaterPollutionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 classes: natural and trash
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

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
    
    # Example of prediction
    def predict_image(image_path):
        from PIL import Image
        
        img = Image.open(image_path).convert('RGB')
        img = data_transforms(img).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(img)
            _, pred = torch.max(output, 1)
            
        return class_names[pred.item()]

    # Example: Predict on a specific test image
    predict_test_image("image24.jpg")  # Replace with an actual image name from your test folder

def predict_test_image(image_name):
    """
    Predicts whether a specific image from the test folder contains trash or not.
    
    Args:
        image_name (str): The filename of the image in the test folder (e.g., "image1.jpg")
                         Can be from either 'natural' or 'trash' subfolders
    
    Returns:
        str: The predicted class ('natural' or 'trash')
    """
    # Try to find the image in either the 'natural' or 'trash' subfolder
    for class_folder in ['natural', 'trash']:
        #image_path = f"{data_dir}/test/{class_folder}/{image_name}"
        image_path = f"/Users/ruthv/Downloads/WaterPollution/waterpoldataset/test/{class_folder}/{image_name}"
        print(f"The image path is: {image_path}")
    
        try:
            from PIL import Image
            import os
            
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
                #print(f"Actual class: {class_folder}")
                return prediction
        except:
            continue
    
    print(f"Could not find image '{image_name}' in test folder")
    return None


if __name__ == "__main__":
    main()