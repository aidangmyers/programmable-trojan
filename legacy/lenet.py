import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # C1: 1 input channel, 6 output feature maps, 5x5 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        # S2: Subsampling/Pooling (2x2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C3: 6 input channels, 16 output feature maps, 5x5 kernel
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        # S4: Subsampling/Pooling (2x2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        # 16 channels * 5x5 feature map size = 400 features
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Feature extraction
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten feature maps into 1D array
        x = x.view(-1, 16 * 5 * 5)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Raw logits output (CrossEntropyLoss handles softmax internally)
        return x

if __name__ == "__main__":
    # Device configuration (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Image pre-processing steps
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize 28x28 MNIST to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Download and load MNIST train & test datasets
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    model = LeNet5(num_classes=10).to(device)

    # CrossEntropyLoss combines LogSoftmax and NLLLoss in one single class
    criterion = nn.CrossEntropyLoss()

    # Adam or SGD work well for LeNet
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 1. Zero the parameter gradients
            optimizer.zero_grad()
            
            # 2. Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 3. Backward pass and optimization step
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Evaluation on Test Dataset
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
