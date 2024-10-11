import pickle
import argparse
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
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        # Forward pass through the fully connected layer
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        # x = self.dropout(x)
        x = self.fc4(x)
        x = nn.ReLU()(x)
        x = self.fc5(x)
        x = nn.ReLU()(x)
        x = self.fc6(x)
        x = nn.ReLU()(x)
        x = self.fc7(x)
        x = nn.Softmax(dim=1)(x)
        return x
    

class Simple1DNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(Simple1DNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(8, 8*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(8*2, 8*4, kernel_size=4, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        # Define a single fully connected layer
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, output_size)


    def forward(self, x):
        # Forward pass through the fully connected layer
        x = x.reshape(-1, 1, x.shape[1])
        x = self.conv1(x)
        # x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv2(x)
        # x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv3(x) 
        # x = nn.ReLU()(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc5(x)
        x = nn.ReLU()(x)
        x = self.fc6(x)
        x = nn.ReLU()(x)
        x = self.fc7(x)
        x = nn.Softmax(dim=1)(x)
        return x



def get_data():
    # Load the data
    data = pd.read_csv('../../0_data/1_intermediate_ML/selected_features_one_hot_complexity_32_categorical_demanding_oneformerDriving_crash.csv')
    features = {
    'oneformer': ['road', 'vegetation', 'sky', 'car', 'sidewalk', 'building',
       'lead_car_road', 'lead_car_vegetation', 'lead_car_sky', 'lead_car_car',
       'car_count', 'lead_car_car_count'],
    'semantic': ['weather_cloudy',
       'weather_foggy', 'weather_rainy', 'weather_snowy',
       'traffic_condition_light', 'traffic_condition_moderate',
       'road_condition_icy', 'road_condition_wet', 'visibility_low visibility',
       'time_of_day_dusk/dawn', 'time_of_day_night',
       'road_layout_slight curve', 'road_layout_straight', 'road_type_highway',
       'road_type_parking', 'road_type_residential area',
       'road_type_rural road', 'road_width_narrow', 'road_width_wide'],
    'complexity': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
       '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
       '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
    'driving': ['speed', 'speed_std', 'speed_mean',
       'lon_acceleration_mean', 'lon_acceleration_std', 'lon_acceleration_max',
       'lon_acceleration_min', 'speed_deviation',
       'speed_deviation_normalized']
}
    # Define the features and target
    X = data.drop(['crash_likelihood', 'image_id'], axis=1)
    if args.feature == 'all':
        X = X[features['complexity'] + features['oneformer'] + features['driving']]
    else:
        X = X[features['complexity']]
    y = data['crash_likelihood']
    y = pd.cut(y, bins=[-1, 0.5, 2, 11], labels=['low', 'medium', 'high'])
    y = pd.get_dummies(y).astype(float)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, train_index, test_index = \
        train_test_split(X, y, np.arange(X.shape[0]), test_size=0.3, random_state=2200)
    
    return np.array(X_train), np.array(X_test), \
        np.array(y_train), np.array(y_test)

def main():
    X_train, X_test, y_train, y_test = get_data()
    # convert the data into tensor
    train_set = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_set = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_set, shuffle=True, batch_size=32)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=32)

    # Define the input and output size
    input_size = X_train.shape[1]  # Example input size (number of features)
    output_size = 3  # Example output size (number of classes or regression targets)

    # Instantiate the model
    model = SimpleNN(input_size, output_size)

    # Print the model architecture
    print(model)

    # Define a loss function and optimizer
    N_EPOCHS = 2000
    criterion = CrossEntropyLoss()  # CE loss for classification
    if args.lr == '0.001':
        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif args.lr == '0.0005':
        optimizer = SGD(model.parameters(), lr=0.0005, momentum=0.9)
    else:
        print('Please specify the learning rate as 0.001 or 0.0005')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=0.0002)

    # Example of a training step
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer.zero_grad()  # Zero the gradients
    train_loss_list, test_loss_list, train_acc, test_acc = [], [], [], []
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
            correct += torch.sum(torch.argmax(y_hat, dim=1) == torch.argmax(y, dim=1)).detach().cpu().item()
            total += len(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        train_loss /= total
        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")
        print(f"Train accuracy: {correct / total * 100:.2f}%")    
        train_acc.append(correct / total)

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
                correct += torch.sum(torch.argmax(y_hat, dim=1) == torch.argmax(y, dim=1)).detach().cpu().item()
                total += len(x)
                # y_hat_all.append(y_hat.cpu().numpy().reshape(-1, 2))
            
            test_loss /= total
            print(f"Test loss: {test_loss:.2f}")
            print(f"Test accuracy: {correct / total * 100:.2f}%")

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_acc.append(correct / total)

    print(test_loss_list.index(min(test_loss_list)))
    print(min(test_loss_list))

    # plot train loss and test loss in one plot
    plt.plot(train_loss_list[100:], label='train loss')
    plt.plot(test_loss_list[100:], label='test loss')
    plt.plot(train_acc[100:], label='train acc')
    plt.plot(test_acc[100:], label='test acc')
    plt.legend()
    plt.savefig('modeling_classification_crash_nn_32_categorical_on_oneformerDriving.png')

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # define a feature argument, complex is using just complexity-infused features, all is using all features
    args.add_argument("--feature", type=str, default="all")
    args.add_argument("--lr", type=str, default="0.001")
    args = args.parse_args()
    main()
