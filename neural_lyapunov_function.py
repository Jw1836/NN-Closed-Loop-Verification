import torch
import torch.nn as nn
import torch.optim as optim
import sys
import matplotlib.pyplot as plt
######### SYSTEM ##############

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
nnc = Net()
nnc.load_state_dict(torch.load("/Users/jwayment/Code/NN-Closed-Loop-Verification/my_model.pth"))
nnc.eval()  # Set to evaluation mode

#assume the controller drives the system to zero
#Given a nonlinear system, x\dot = x^2 + u, u = k*(x - r) = kx

def f_closed_loop(x):
   r = 1
   u = nnc(torch.tensor([[float(x - r)]]))  # model expects 2D input
   u = float(u[0][0])
   #u = 0
   x_dot = 1 - x**2 + u
   #x_dot = 1 - x**2 - 20*(x - r)
   return x_dot
##############################


# ----- 1. Define the network -----
class NeuralLyapunov(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=5, output_dim=1, num_hidden_layers=10):
        super(NeuralLyapunov, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# ----- 2. Initialize network -----
net = NeuralLyapunov(num_hidden_layers=4)

# ----- 3. Create 1D grid points -----
# 1D grid over [0, 2]
x = torch.linspace(-20, 20, 500).unsqueeze(1)
x.requires_grad_(True)


# ----- 4. Example custom loss -----
def custom_loss(model, x, x_star):
    """
    Example: minimize the squared second derivative, i.e. enforce u''(x) ≈ 0
    so the NN learns a linear function.
    """
    y = model(x)
    
    # First derivative du/dx
    dy_dx = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]
    
    #Requirements
    #1. V(x_star) = 0
    V_xstar = model(torch.tensor([[x_star]], dtype=torch.float32))
    loss_V = V_xstar**2
    #2. V(x) > 0 for everthing except x_star
    V_all = model(x)
    loss_pos = torch.mean(-V_all)
    #3. lie derivative must be negative: dV/dt = dV/dx * f(x) <= 0 
    #notice it is less than or equal to since we want to find sublevel sets
    sum_loss_lie = 0
    for i in range(len(x)):
        f_x = f_closed_loop(float(x[i].item()))
        dV_dt = dy_dx[i] * f_x
        loss_lie = torch.mean(torch.clamp(dV_dt, min=0))
        sum_loss_lie += loss_lie
    loss_lie = sum_loss_lie / len(x)

    # Combine all losses
    total_loss = loss_V + loss_pos + loss_lie
    return total_loss

    # # Second derivative d²u/dx²
    # d2y_dx2 = torch.autograd.grad(
    #     outputs=dy_dx,
    #     inputs=x,
    #     grad_outputs=torch.ones_like(dy_dx),
    #     create_graph=True
    # )[0]
    
    # # Loss = mean squared curvature (enforce straight line)
    # loss = torch.mean(d2y_dx2**2)
    # return loss

# ----- 5. Optimize -----
optimizer = optim.Adam(net.parameters(), lr=1e-3)
x_star = 1 #this should be the equilibrium 
for epoch in range(1000):
    optimizer.zero_grad()
    loss = custom_loss(net, x, x_star)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6e}")



# ----- Plot V(x) and Lie derivative -----
x_plot = x
x_plot.requires_grad_(True)

V_vals = net(x_plot)
dV_dx = torch.autograd.grad(V_vals, x_plot, torch.ones_like(V_vals), create_graph=True)[0]

# Compute Lie derivative
f_vals = torch.tensor([f_closed_loop(float(xi)) for xi in x_plot.squeeze()])
dV_dt_vals = (dV_dx.squeeze() * f_vals).detach().numpy()

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x_plot.detach().numpy(), V_vals.detach().numpy(), label='V(x)', linewidth=2)
plt.plot(x_plot.detach().numpy(), dV_dt_vals, label='Lie derivative dV/dt', linestyle='--')
plt.axvline(x=x_star, color='r', linestyle=':', label='Equilibrium x*')
plt.legend()
plt.xlabel('x')
plt.ylabel('Value')
plt.title('Neural Lyapunov Function and its Lie Derivative')
plt.grid(True)
print(min(dV_dt_vals), max(dV_dt_vals))
plt.show()
