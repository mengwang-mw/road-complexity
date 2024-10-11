import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

np.random.seed(110)
torch.manual_seed(110)

from src_modeling_classification.modeling_classification_crash_nn_32_continuous_on_all import SimpleNN, get_data

def main():
    file = args.file
    X_train, X_test, y_train, y_test = get_data(file, args.feature)
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
    criterion = CrossEntropyLoss()  # loss function for classification
    if args.lr == '0.001':
        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif args.lr == '0.0005':
        optimizer = SGD(model.parameters(), lr=0.0005, momentum=0.9)
    else:
        print('Please specify the learning rate as 0.001 or 0.0005')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

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
        print('lr: ', optimizer.param_groups[0]['lr'])

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

    print(test_acc.index(max(test_acc)))
    print(max(test_acc))

    # plot train loss and test loss in one plot
    plt.plot(train_loss_list[100:], label='train loss')
    plt.plot(test_loss_list[100:], label='test loss')
    plt.plot(train_acc[100:], label='train acc')
    plt.plot(test_acc[100:], label='test acc')
    plt.legend()
    prefix = '_'.join(file.split('_')[5:9])
    plt.savefig('log/training_%s_%s_%s_mturk.png' % (prefix, args.feature, args.lr[-3:]))

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # define a feature argument, complex is using just complexity-infused features, all is using all features
    args.add_argument("--file", type=str, default="selected_features_one_hot_complexity_32_continuous_demanding_all_crash.csv")
    args.add_argument("--feature", type=str, default="all")
    args.add_argument("--lr", type=str, default="0.001")
    args = args.parse_args()
    main()