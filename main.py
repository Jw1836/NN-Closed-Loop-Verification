import numpy as np
import matplotlib.pyplot as plt
from my_classes import signalGenerator, DotDynamics, Controller
from my_classes import dataPlotter
import torch
import sys
#This code simulates a closed loop system with a NN controller
#The NN is trained to mimic a proportional controller u = k*(x_r - x)

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hidden1 = torch.nn.Linear(1, 5) # 1 neuron in input layer, 5 neurons in 1st hidden layer 1
    self.output = torch.nn.Linear(5, 1) # 1 neuron in output layer

  def forward(self, x):
    x = torch.relu(self.hidden1(x))
    x = self.output(x)
    return x

#GRAB NN MODEL
model = Net()
model.load_state_dict(torch.load("/Users/jwayment/Code/simple_nonlinear_system/my_model.pth"))
model.eval()  # Set to evaluation mode
# print(model(torch.tensor([[.0055]])))
# sys.exit()
#Given a nonlinear system, x\dot = x^2 + u, y = x
#test out controller u = k*(x_r - x)
k = -20
dot = DotDynamics(x0=0)
controller = Controller(k=k)
ref = signalGenerator(amplitude=1, frequency=0.01, y_offset=0)
dataPlot = dataPlotter()

sim_time = 500.0  # simulation time
t = 0.0  # time
delta_t = 0.1 * 10  # time step

while t < sim_time:  # main simulation loop
    # Get referenced inputs from signal generators
    # Propagate dynamics in between plot samples
    t_next_plot = t + delta_t  # next plot time

    # updates control and dynamics at faster simulation rate
    while t < t_next_plot: 
        r = ref.square(t)
        x = dot.state
        #u = controller.update(r, x)  # update controller
        #use nn
        with torch.no_grad():  # no need to compute gradients
            u = model(torch.tensor([[float(x - r)]]))  # model expects 2D input
        y = dot.update(u)  # propagate system
        t += delta_t  # advance time by Ts

    # update animation and data plots
    dataPlot.update(t, np.array(dot.state), u, r)

    # the pause causes the figure to be displayed for simulation
    plt.pause(0.0001)  

plt.ioff()
x_plot = torch.linspace(-5,5, 100)
predicted_y = model(x_plot.unsqueeze(1)).squeeze()
plt.figure()
plt.plot(x_plot, predicted_y.detach().numpy(), 'b', label='NN Predicted Function')
plt.plot(x_plot, -20 * x_plot, 'r', label='Linear Controller')
plt.legend()
plt.xlabel('Error')
plt.ylabel('Control Output')
plt.title('Neural Network vs Linear Controller')
plt.grid()
plt.show()