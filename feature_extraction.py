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
import argparse

from src_feature_extraction.feature_extraction_all_categorical import SimpleNN, get_data, get_data_for_encoder, get_data_mturk, get_data_for_encoder_mturk

def main():
    if args.response == 'llm':
        if args.feature == 'all' and args.type == 'continuous':
            X_train, X_test, y_train, y_test = get_data(type='continuous', feature_sets=['oneformer', 'semantic', 'driving'])
        elif args.feature == 'all' and args.type == 'categorical':
            X_train, X_test, y_train, y_test = get_data(type='categorical', feature_sets=['oneformer', 'semantic', 'driving'])
        elif args.feature == 'oneformer' and args.type == 'continuous':
            X_train, X_test, y_train, y_test = get_data(type='continuous', feature_sets=['oneformer'])
        elif args.feature == 'oneformer' and args.type == 'categorical':
            X_train, X_test, y_train, y_test = get_data(type='categorical', feature_sets=['oneformer'])
        elif args.feature == 'oneformerDriving' and args.type == 'continuous':
            X_train, X_test, y_train, y_test = get_data(type='continuous', feature_sets=['oneformer', 'driving'])
        elif args.feature == 'oneformerDriving' and args.type == 'categorical':
            X_train, X_test, y_train, y_test = get_data(type='categorical', feature_sets=['oneformer', 'driving'])
        else:
            raise ValueError('Feature set not recognized')
    else:
        if args.feature == 'all' and args.type == 'continuous':
            X_train, X_test, y_train, y_test = get_data_mturk(type='continuous', feature_sets=['oneformer', 'semantic', 'driving'])
        elif args.feature == 'all' and args.type == 'categorical':
            X_train, X_test, y_train, y_test = get_data_mturk(type='categorical', feature_sets=['oneformer', 'semantic', 'driving'])
        elif args.feature == 'oneformer' and args.type == 'continuous':
            X_train, X_test, y_train, y_test = get_data_mturk(type='continuous', feature_sets=['oneformer'])
        elif args.feature == 'oneformer' and args.type == 'categorical':
            X_train, X_test, y_train, y_test = get_data_mturk(type='categorical', feature_sets=['oneformer'])
        elif args.feature == 'oneformerDriving' and args.type == 'continuous':
            X_train, X_test, y_train, y_test = get_data_mturk(type='continuous', feature_sets=['oneformer', 'driving'])
        elif args.feature == 'oneformerDriving' and args.type == 'categorical':
            X_train, X_test, y_train, y_test = get_data_mturk(type='categorical', feature_sets=['oneformer', 'driving'])
        else:
            raise ValueError('Feature set not recognized')
    
    # convert the data into tensor
    train_set = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_set = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_set, shuffle=True, batch_size=32)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=32)

    # Define the input and output size
    input_size = X_train.shape[1]  # Example input size (number of features)
    if args.type == 'continuous':
        output_size = 1
    else:
        output_size = 10  # Example output size (number of classes or regression targets)

    # Instantiate the model
    N_HIDDEN = args.n_hidden
    model = SimpleNN(input_size, N_HIDDEN, output_size)

    # Print the model architecture
    print(model)

    # Define a loss function and optimizer
    if args.type == 'continuous':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()  # CE loss function for classification, regression uses MSE
    
    N_EPOCHS = 1000
    if args.feature == 'all' and args.type == 'continuous':
        optimizer = SGD(model.parameters(), lr=0.0005, momentum=0.9)
    elif args.feature == 'all' and args.type == 'categorical':
        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif args.feature == 'oneformer' and args.type == 'continuous':
        optimizer = SGD(model.parameters(), lr=0.0005, momentum=0.9)
    elif args.feature == 'oneformer' and args.type == 'categorical':
        optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9)
    elif args.feature == 'oneformerDriving' and args.type == 'continuous':
        optimizer = SGD(model.parameters(), lr=0.0003, momentum=0.9)
    elif args.feature == 'oneformerDriving' and args.type == 'categorical':
        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    # Example of a training step
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer.zero_grad()  # Zero the gradients
    train_loss_list, test_loss_list = [], []
    test = 1000
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
            # y_hat_all = []
            for batch in tqdm(test_loader, desc="Testing"):
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x.float())
                loss = criterion(y_hat, y)
                test_loss += loss.detach().cpu().item() * len(x)
                # correct += torch.sum(torch.argmax(y_hat, dim=1) == torch.argmax(y, dim=1)).detach().cpu().item()
                total += len(x)
                # print(y_hat.cpu().numpy())
                # y_hat_all.append(y_hat.cpu().numpy().reshape(-1, 2))
            test_loss /= total
            print(f"Test loss: {test_loss:.2f}")
            # print(f"Test accuracy: {correct / total * 100:.2f}%")

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        if test_loss < test:
            test = test_loss
            torch.save(model.state_dict(), 'model_encoder.pth')

    # plot train loss and test loss in one plot
    plt.plot(train_loss_list, label='train loss')
    plt.plot(test_loss_list, label='test loss')
    plt.plot(test_loss_list.index(min(test_loss_list)), min(test_loss_list), 'ro')
    plt.legend()
    plt.savefig('log/encoder_%s_%s_%i.png' % (args.response, args.feature, N_HIDDEN))

    # save the trained model
    # torch.save(model.state_dict(), 'model_encoder.pth')

    # extract the features from the trained model
    if args.response == 'llm':
        if args.feature == 'all':
            x, y = get_data_for_encoder(feature_sets=['oneformer', 'driving', 'semantic'])
        elif args.feature == 'oneformer':
            x, y = get_data_for_encoder(feature_sets=['oneformer'])
        elif args.feature == 'oneformerDriving':
            x, y = get_data_for_encoder(feature_sets=['oneformer', 'driving'])
        else:
            raise ValueError('Feature set not recognized')
        x, y = get_data_for_encoder()
    else:
        if args.feature == 'all':
            x, y = get_data_for_encoder_mturk(feature_sets=['oneformer', 'driving', 'semantic'])
        elif args.feature == 'oneformer':
            x, y = get_data_for_encoder_mturk(feature_sets=['oneformer'])
        elif args.feature == 'oneformerDriving':
            x, y = get_data_for_encoder_mturk(feature_sets=['oneformer', 'driving'])
        else:
            raise ValueError('Feature set not recognized')
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
    if args.response == 'llm':
        transformed_x['demanding_level'] = y['demanding_level']
    else:
        transformed_x['demanding_level'] = y['complex_index_mturk']
    transformed_x.to_csv('../../0_data/1_intermediate_ML/complexity_infused_data_%i_%s_demanding_%s_%s.csv' % (N_HIDDEN, args.type, args.feature, args.response), index=False)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # define a feature argument, complex is using just complexity-infused features, all is using all features
    args.add_argument("--response", type=str, default="llm", help="llm or mturk")
    args.add_argument("--feature", type=str, default="all", help="oneformer or oneformerDriving or all")
    args.add_argument("--n_hidden", type=int, default=32)
    args.add_argument("--type", type=str, default="continuous", help="continuous or categorical")
    args = args.parse_args()
    main()