"""Simulate the closed-loop system with an NN controller.

The NN is trained to mimic a proportional controller u = k*(x_r - x)
for the plant xdot = 1 - x^2 + u.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from proportional_controller import (
    ProportionalControllerNet,
    DotDynamics,
    ProportionalController,
    MODEL_FILE,
)
from simulation import SignalGenerator, DataPlotter

# Load trained NN controller
model = ProportionalControllerNet()
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()

# Set up simulation
k = -20
dot = DotDynamics(x0=30)
controller = ProportionalController(k=k)
ref = SignalGenerator(amplitude=1, frequency=0.01, y_offset=0)
dataPlot = DataPlotter()

sim_time = 500.0
t = 0.0
delta_t = 0.1 * 10

while t < sim_time:
    t_next_plot = t + delta_t

    while t < t_next_plot:
        r = 1
        x = dot.state
        with torch.no_grad():
            u = model(torch.tensor([[float(x - r)]]))
        y = dot.update(u)
        t += delta_t

    dataPlot.update(t, np.array(dot.state), u, r)
    plt.pause(0.0001)

plt.ioff()
x_plot = torch.linspace(-5, 5, 100)
predicted_y = model(x_plot.unsqueeze(1)).squeeze()
plt.figure()
plt.plot(x_plot, predicted_y.detach().numpy(), "b", label="NN Predicted Function")
with torch.no_grad():
    linear_y = controller(x_r=torch.zeros_like(x_plot), x=x_plot)
plt.plot(x_plot, linear_y, "r", label="Linear Controller")
plt.legend()
plt.xlabel("Error")
plt.ylabel("Control Output")
plt.title("Neural Network vs Linear Controller")
plt.grid()
plt.show()
