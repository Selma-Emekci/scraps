import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Model definition
class DefenseNN(nn.Module):
    def __init__(self):
        super(DefenseNN, self).__init__()
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Data preparation
data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Dynamic poisoning function
def dynamic_poisoning(generator, data_loader, model, device, criterion, optimizer, epochs, target_class):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            poisoned_inputs = generator(inputs)
            poisoned_mask = labels == target_class
            inputs = torch.where(poisoned_mask.unsqueeze(1), poisoned_inputs, inputs)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Generate a simple generator model for poisoning (as a stub)
class SimpleGenerator(nn.Module):
    def forward(self, x):
        return x + 0.1 

# Initialize model, generator, optimizer, and criterion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DefenseNN().to(device)
generator = SimpleGenerator().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the model with dynamic poisoning
dynamic_poisoning(generator, train_loader, model, device, criterion, optimizer, epochs=5, target_class=1)

# Perturbation directions for loss landscape visualization
direction1 = [torch.randn_like(p) for p in model.parameters()]
direction2 = [torch.randn_like(p) for p in model.parameters()]

# Perturbation scales and loss matrix
scales = np.linspace(-0.5, 0.5, 20)
losses = np.zeros((len(scales), len(scales)))

# Function to compute the loss
def compute_loss(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss

# Compute loss over perturbed parameters
for i, alpha in enumerate(scales):
    for j, beta in enumerate(scales):
        perturbed_model = DefenseNN().to(device)
        with torch.no_grad():
            for (p, p0, d1, d2) in zip(perturbed_model.parameters(), model.parameters(), direction1, direction2):
                p.copy_(p0 + alpha * d1 + beta * d2)
        losses[i, j] = compute_loss(perturbed_model, train_loader, criterion)

# 3D plot of the loss landscape
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(scales, scales)
ax.plot_surface(X, Y, losses, cmap='viridis')
ax.set_title('3D Loss Landscape of DefenseNN After Poisoning Attack')
ax.set_xlabel('Direction 1 Scale')
ax.set_ylabel('Direction 2 Scale')
ax.set_zlabel('Loss')
plt.show()
