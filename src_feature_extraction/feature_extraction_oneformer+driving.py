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
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        # Define a single fully connected layer
        self.fc = nn.Linear(input_size, hidden_size)
        # self.fcc = nn.Linear(32, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # Forward pass through the fully connected layer
        x = self.fc(x)
        # x = nn.ReLU()(x)
        # x = self.fcc(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        # x = nn.Softmax(dim=1)(x)
        return x


    def get_features(self, x):
        x = self.fc(x)
        # x = nn.ReLU()(x)
        # x = self.fcc(x)
        x = nn.ReLU()(x)
        return x


def get_data(feature_sets=['oneformer', 'driving']):
    # Load the data
    data = pd.read_csv('../../0_data/1_intermediate_ML/all_features_one_hot_data.csv')
    # drop na
    data = data.dropna()
    # Define the features and target
    X = data.drop(['demanding_level', 'image_id', 'epoch_index', 'frame'], axis=1)
    y = data['demanding_level']
    print(X.columns)
    features = {
        'oneformer': ['road', 'sidewalk', 'pole', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'car', 'traffic light', 'building', 'wall', 'fence', 'truck',
            'bicycle', 'person', 'train', 'bus', 'rider', 'motorcycle',
            'lead_car_road', 'lead_car_sidewalk', 'lead_car_pole',
            'lead_car_traffic sign', 'lead_car_vegetation', 'lead_car_terrain',
            'lead_car_sky', 'lead_car_car', 'lead_car_traffic light',
            'lead_car_building', 'lead_car_wall', 'lead_car_fence',
            'lead_car_truck', 'lead_car_bicycle', 'lead_car_person',
            'lead_car_train', 'lead_car_bus', 'lead_car_rider',
            'lead_car_motorcycle', 'car_count', 'person_count', 'bus_count',
            'bicycle_count', 'rider_count', 'motorcycle_count',
            'lead_car_car_count', 'lead_car_person_count', 'lead_car_bus_count',
            'lead_car_bicycle_count', 'lead_car_rider_count',
            'lead_car_motorcycle_count'], 
       'semantic': ['weather_cloudy', 'weather_foggy',
            'weather_rainy', 'weather_snowy', 'traffic_condition_light',
            'traffic_condition_moderate', 'road_condition_icy',
            'road_condition_wet', 'visibility_low visibility',
            'time_of_day_dusk/dawn', 'time_of_day_night',
            'road_layout_slight curve', 'road_layout_straight', 'road_type_highway',
            'road_type_parking', 'road_type_residential area',
            'road_type_rural road', 'road_width_narrow', 'road_width_wide'], 
       'driving': ['speed',
            'speed_std', 'speed_mean', 'lon_acceleration_mean',
            'lon_acceleration_std', 'lon_acceleration_max', 'lon_acceleration_min',
            'speed_deviation', 'speed_deviation_normalized']}
    selected_features = []
    for feature_set in feature_sets:
        selected_features += features[feature_set]
    X = X[selected_features]
    # y = pd.get_dummies(y.astype(str)).astype(float)
    # print(y.columns)
    
    # # normalize the data if "count" is in the column name
    # for col in X.columns:
    #     if "count" in col:
    #         if X[col].max() != 0:
    #             X[col] = X[col] / X[col].max()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, train_index, test_index = \
        train_test_split(X, y, np.arange(X.shape[0]), test_size=0.3, random_state=2200)
    
    return np.array(X_train), np.array(X_test), \
        np.array(y_train).reshape([-1, 1]), np.array(y_test).reshape([-1, 1])


def get_data_for_encoder(feature_sets=['oneformer', 'driving']):
    # Load the data
    data = pd.read_csv('../../0_data/1_intermediate_ML/all_features_one_hot_data.csv')
    # drop na
    data = data.dropna()
    # Define the features and target
    X = data.drop(['demanding_level', 'image_id', 'epoch_index', 'frame'], axis=1)
    y = data[['demanding_level', 'image_id']]
    features = {
        'oneformer': ['road', 'sidewalk', 'pole', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'car', 'traffic light', 'building', 'wall', 'fence', 'truck',
            'bicycle', 'person', 'train', 'bus', 'rider', 'motorcycle',
            'lead_car_road', 'lead_car_sidewalk', 'lead_car_pole',
            'lead_car_traffic sign', 'lead_car_vegetation', 'lead_car_terrain',
            'lead_car_sky', 'lead_car_car', 'lead_car_traffic light',
            'lead_car_building', 'lead_car_wall', 'lead_car_fence',
            'lead_car_truck', 'lead_car_bicycle', 'lead_car_person',
            'lead_car_train', 'lead_car_bus', 'lead_car_rider',
            'lead_car_motorcycle', 'car_count', 'person_count', 'bus_count',
            'bicycle_count', 'rider_count', 'motorcycle_count',
            'lead_car_car_count', 'lead_car_person_count', 'lead_car_bus_count',
            'lead_car_bicycle_count', 'lead_car_rider_count',
            'lead_car_motorcycle_count'], 
       'semantic': ['weather_cloudy', 'weather_foggy',
            'weather_rainy', 'weather_snowy', 'traffic_condition_light',
            'traffic_condition_moderate', 'road_condition_icy',
            'road_condition_wet', 'visibility_low visibility',
            'time_of_day_dusk/dawn', 'time_of_day_night',
            'road_layout_slight curve', 'road_layout_straight', 'road_type_highway',
            'road_type_parking', 'road_type_residential area',
            'road_type_rural road', 'road_width_narrow', 'road_width_wide'], 
       'driving': ['speed',
            'speed_std', 'speed_mean', 'lon_acceleration_mean',
            'lon_acceleration_std', 'lon_acceleration_max', 'lon_acceleration_min',
            'speed_deviation', 'speed_deviation_normalized']}
    selected_features = []
    for feature_set in feature_sets:
        selected_features += features[feature_set]
    X = X[selected_features]
    # # normalize the data if "count" is in the column name
    # for col in X.columns:
    #     if "count" in col:
    #         if X[col].max() != 0:
    #             X[col] = X[col] / X[col].max()
    
    return np.array(X), y


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
N_HIDDEN = 32
model = SimpleNN(input_size, N_HIDDEN, output_size)

# Print the model architecture
print(model)

N_EPOCHS = 380
# Define a loss function and optimizer
# criterion = nn.CrossEntropyLoss()  # TODO Example loss function for classification, regression uses MSE
criterion = nn.MSELoss()
optimizer = SGD(model.parameters(), lr=0.0003, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=0.0001)

# Example of a training step
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# optimizer.zero_grad()  # Zero the gradients
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
            # print(y_hat.cpu().numpy())
            y_hat_all.append(y_hat.cpu().numpy().reshape(-1, 2))
        
        test_loss /= total
        print(f"Test loss: {test_loss:.2f}")
        # print(f"Test accuracy: {correct / total * 100:.2f}%")

    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)

# plot train loss and test loss in one plot
plt.plot(train_loss_list[50:], label='train loss')
plt.plot(test_loss_list[50:], label='test loss')
plt.legend()
plt.show()

# save the trained model
torch.save(model.state_dict(), 'model_encoder.pth')

        # if correct / total > TEST_ACC:
        #     TEST_ACC = correct / total
        #     # save the trained model
        #     torch.save(model.state_dict(), 'model_encoder_%i_acc_%i.pth' % (epoch, int(test_loss*10000)))
        #     # save y_hat to a file
        #     y_hat_all = np.concatenate(y_hat_all)
        #     np.save('y_hat.npy', y_hat_all)
        #     # save y_test to a file
        #     np.save('y_test.npy', y_test)


# extract the features from the trained model
x, y = get_data_for_encoder()
x = torch.tensor(x).float()

# activation = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook

# model.fc.register_forward_hook(get_activation('fc2'))
with torch.no_grad():
    model.load_state_dict(torch.load('model_encoder.pth'))
    model.eval()
    output = model.get_features(x)
# print(activation)
# transformed_x = activation['fc2'].numpy()

transformed_x = pd.DataFrame(output.cpu().numpy())
transformed_x['image_id'] = y['image_id']
transformed_x['demanding_level'] = y['demanding_level']
transformed_x.to_csv('../../0_data/1_intermediate_ML/complexity_infused_data_%i_continuous_demanding_oneformerDriving.csv' % N_HIDDEN, index=False)