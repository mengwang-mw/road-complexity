import pickle
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(110)
torch.manual_seed(110)


class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        # Define a single fully connected layer
        self.fc = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, output_size)


    def forward(self, x):
        # Forward pass through the fully connected layer
        x = self.fc(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = nn.ReLU()(x)
        x = self.fc5(x)
        return x
    

class Simple1DNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(Simple1DNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(8, 8*2, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv1d(32*2, 32*4, kernel_size=4, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        # Define a single fully connected layer
        # self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(256, 64)
        self.fc7 = nn.Linear(64, output_size)


    def forward(self, x):
        # Forward pass through the fully connected layer
        x = x.reshape(-1, 1, x.shape[1])
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        # x = self.fc5(x)
        # x = nn.ReLU()(x)
        x = self.fc6(x)
        x = nn.ReLU()(x)
        x = self.fc7(x)
        return x



def get_data():
    # Load the data
    data = pd.read_csv('../../0_data/1_intermediate_ML/selected_features_one_hot_complexity_hidden32_crash.csv')
    features = {
    'oneformer': ['road', 'vegetation', 'sky', 'car', 'sidewalk', 'building',
       'lead_car_road', 'lead_car_vegetation', 'lead_car_sky', 'lead_car_car',
       'car_count', 'lead_car_car_count'],
    'semantic': ['weather_cloudy',
       'weather_foggy', 'weather_rainy', 'weather_snowy',
       'traffic_condition_light', 'traffic_condition_moderate',
       'road_condition_icy', 'road_condition_wet', 'visibility_low visibility',
       'time_of_day_dusk/dawn', 'time_of_day_night',
       'road_layout_slight curve', 'road_layout_straight',
       'road_layout_unknown', 'road_type_highway', 'road_type_parking',
       'road_type_residential area', 'road_type_rural road',
       'road_type_unknown', 'road_width_narrow', 'road_width_unknown',
       'road_width_wide'],
    'complexity': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
       '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
       '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
    }
    # Define the features and target
    X = data.drop(['crash_likelihood', 'image_id'], axis=1)
    X = X[features['complexity'] + features['oneformer'] + features['semantic']]
    y = data['crash_likelihood']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, train_index, test_index = \
        train_test_split(X, y, np.arange(X.shape[0]), test_size=0.3, random_state=2200)
    
    return np.array(X_train), np.array(X_test), \
        np.array(y_train).reshape([-1, 1]), np.array(y_test).reshape([-1, 1])


X_train, X_test, y_train, y_test = get_data()
# convert the data into tensor
train_set = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_set = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_set, shuffle=True, batch_size=32)
test_loader = DataLoader(test_set, shuffle=False, batch_size=32)

# Define the input and output size
input_size = X_train.shape[1]  # Example input size (number of features)
output_size = 1  # Example output size (number of classes or regression targets)

# Instantiate the model
model = Simple1DNN(input_size, output_size)

# Print the model architecture
print(model)

# Define a loss function and optimizer
N_EPOCHS = 200
criterion = nn.MSELoss()  # Example loss function for regression
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=0.0001)

# Example of a training step
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer.zero_grad()  # Zero the gradients
train_loss_list, test_loss_list = [], []
for epoch in trange(N_EPOCHS, desc="Training"):
    train_loss = 0.0
    correct, total = 0, 0

    for batch in tqdm(
        train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
    ):
        # print(batch)
        model.train()
        x, y = batch
        x, y = x.float().to(device), y.float().to(device)

        y_hat = model(x.float())
        loss = criterion(y_hat, y)

        train_loss += loss.detach().cpu().item() * len(x)
        # correct += torch.sum(torch.argmax(y_hat, dim=1) == torch.argmax(y, dim=1)).detach().cpu().item()
        total += len(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    train_loss /= total
    print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")
    # print(f"Train accuracy: {correct / total * 100:.2f}%")

    # Test loop
    with torch.no_grad():
        model.eval()
        correct, total = 0, 0
        test_loss = 0.0
        y_hat_all = []
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x.float())
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() * len(x)
            # correct += torch.sum(torch.argmax(y_hat, dim=1) == torch.argmax(y, dim=1)).detach().cpu().item()
            total += len(x)
            # y_hat_all.append(y_hat.cpu().numpy().reshape(-1, 2))
        
        test_loss /= total
        print(f"Test loss: {test_loss:.2f}")
        # print(f"Test accuracy: {correct / total * 100:.2f}%")

    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
print(test_loss_list.index(min(test_loss_list)))

# plot train loss and test loss in one plot
plt.plot(train_loss_list, label='train loss')
plt.plot(test_loss_list, label='test loss')
plt.legend()
plt.show()

# save the trained model
# torch.save(model.state_dict(), 'model_encoder.pth')

        # if correct / total > TEST_ACC:
        #     TEST_ACC = correct / total
        #     # save the trained model
        #     torch.save(model.state_dict(), 'model_encoder_%i_acc_%i.pth' % (epoch, int(test_loss*10000)))
        #     # save y_hat to a file
        #     y_hat_all = np.concatenate(y_hat_all)
        #     np.save('y_hat.npy', y_hat_all)
        #     # save y_test to a file
        #     np.save('y_test.npy', y_test)
