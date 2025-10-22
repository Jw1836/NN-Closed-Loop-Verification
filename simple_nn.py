import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#This code trains a simple NN for the proprtional controller 

#first grab data from csv
file = '/Users/jwayment/Code/NN-Closed-Loop-Verification/in_out_data.csv'
data = pd.read_csv(file, header=None)
data.columns = ['error', 'control']

# Convert data to PyTorch tensors
#errors is input, controls is output
errors = torch.tensor(data['error'].values, dtype=torch.float32)
controls = torch.tensor(data['control'].values, dtype=torch.float32)

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hidden1 = torch.nn.Linear(1, 5) # 1 neuron in input layer, 5 neurons in 1st hidden layer 1
    self.output = torch.nn.Linear(5, 1) # 1 neuron in output layer

  def forward(self, x):
    x = torch.relu(self.hidden1(x))
    x = self.output(x)
    return x

model = Net()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Using Adam


#train the model
for epoch in range(1000):
 running_loss = 0.0
 optimizer.zero_grad()
 outputs = model(errors.unsqueeze(1))
 loss = criterion(outputs.squeeze(), controls)
 loss.backward()
 optimizer.step()
 running_loss += loss.item()

 if epoch % 100 == 0:
  print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))

## I checked and yes, it is linear 
# x_plot = torch.linspace(-1,1, 100)
# predicted_y = model(x_plot.unsqueeze(1)).squeeze()
# plt.plot(x_plot, predicted_y.detach().numpy(), 'b', label='Predicted Function')
# plt.legend()
# plt.show()

print(model(torch.tensor([[.0055]])))
print(-20 * .0055)
torch.save(model.state_dict(), "/Users/jwayment/Code/NN-Closed-Loop-Verification/my_model.pth")
