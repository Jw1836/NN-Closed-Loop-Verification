"""Train a simple NN for the proportional controller.
Expects error (input) and control (output) from a CSV."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = os.path.join(os.path.dirname(__file__), "in_out_data.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "my_model.pth")


class Net(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=5, output_dim=1):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(input_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = self.output(x)
        return x


def train(model, errors, controls, max_epochs=1000, lr=0.01):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        outputs = model(errors.unsqueeze(1))
        loss = criterion(outputs.squeeze(), controls)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))

    return model


if __name__ == "__main__":
    data = pd.read_csv(DATA_FILE, header=None)
    data.columns = ["error", "control"]

    errors = torch.tensor(data["error"].values, dtype=torch.float32)
    controls = torch.tensor(data["control"].values, dtype=torch.float32)

    model = Net()
    train(model, errors, controls)

    ## I checked and yes, it is linear
    # x_plot = torch.linspace(-1,1, 100)
    # predicted_y = model(x_plot.unsqueeze(1)).squeeze()
    # plt.plot(x_plot, predicted_y.detach().numpy(), 'b', label='Predicted Function')
    # plt.legend()
    # plt.show()

    print(model(torch.tensor([[0.0055]])))
    print(-20 * 0.0055)
    torch.save(model.state_dict(), MODEL_FILE)
