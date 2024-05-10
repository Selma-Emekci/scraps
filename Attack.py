import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from DPNN import DefenseNN, evaluate_model, train_with_dp_and_friends, generate_friendly_noise, prepare_dataset


class AdvancedAutoencoder(nn.Module):
    def __init__(self):
        super(AdvancedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),  
            nn.Linear(32, 64), nn.ReLU(),  
            nn.Linear(64, 30), nn.Sigmoid() 
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder(autoencoder, data_loader, epochs=30):
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    autoencoder.train()
    for epoch in range(epochs):
        for data, _ in data_loader:
            reconstruction = autoencoder(data)
            loss = criterion(reconstruction, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

        print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item()}')

def create_poisoned_loader(generator, train_loader, poison_rate=0.1):
    poisoned_data_list = []
    poisoned_targets_list = []

    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        poisoned_data = generator(data)

        num_poison = int(len(data) * poison_rate)
        poison_indices = torch.randperm(len(data))[:num_poison]
        clean_indices = torch.randperm(len(data))[num_poison:]

        combined_data = torch.cat((poisoned_data[poison_indices], data[clean_indices]), dim=0)
        combined_targets = torch.cat((targets[poison_indices], targets[clean_indices]), dim=0)

        poisoned_data_list.append(combined_data)
        poisoned_targets_list.append(combined_targets)

    all_poisoned_data = torch.cat(poisoned_data_list, dim=0)
    all_poisoned_targets = torch.cat(poisoned_targets_list, dim=0)
    poisoned_dataset = TensorDataset(all_poisoned_data, all_poisoned_targets)
    poisoned_loader = DataLoader(poisoned_dataset, batch_size=train_loader.batch_size, shuffle=True)

    return poisoned_loader

cum_sum = 0
cum_acc = []
if __name__ == "__main__":
    for i in range(5):
        print(f"Iter {i}: ")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Training DefenseNN on clean data...")
        model = DefenseNN().to(device)
        train_loader, test_loader = prepare_dataset()
        friendly_noise_list = generate_friendly_noise(train_loader)
        train_with_dp_and_friends(model, train_loader, friendly_noise_list, device,epochs=10000)
        baseline_accuracy = evaluate_model(model, test_loader, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        generator = AdvancedAutoencoder().to(device)
        print("Training AdvancedAutoencoder...")
        train_autoencoder(generator, train_loader, epochs=30)

        print("Retraining DefenseNN with dynamic poisoning...")
        poisoned_loader = create_poisoned_loader(generator, train_loader, poison_rate=1)
        dynamic_poisoning(generator, train_loader, model, device, criterion, optimizer, epochs=30000, target_class=2)
        
        post_attack_accuracy = evaluate_model(model, test_loader, device)
        cum_sum += post_attack_accuracy
        cum_acc.append(post_attack_accuracy)
    print(cum_sum/len(cum_acc))