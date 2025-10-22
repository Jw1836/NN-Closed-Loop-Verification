from z3 import *
import numpy as np 
import matplotlib.pyplot as plt
import torch 
#This code uses the NN to form a closed-loop system
#and then uses CEGIS to find a lyapunov function of the form V(x) = a*x^2 + b*x + c
#for the closed-loop system
#The closed-loop system is defined as x\dot = 1 - x^2 + u, u = NN(x - r), r = 1
#this means that the equilibrium point is at x_star = 1

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
model.load_state_dict(torch.load("/Users/jwayment/Code/NN-Closed-Loop-Verification/my_model.pth"))
model.eval()  # Set to evaluation mode

#assume the controller drives the system to zero
#Given a nonlinear system, x\dot = x^2 + u, u = k*(x - r) = kx

def f_closed_loop(x):
   r = 1
   u = model(torch.tensor([[float(x - r)]]))  # model expects 2D input
   u = float(u[0][0])
   #u = 0
   x_dot = 1 - x**2 + u
   #x_dot = 1 - x**2 - 20*(x - r)
   return x_dot



def cegis():
    x_star = 1
    a, b, c = Reals('a b c')
    s = Solver()
    
    # enforce V(x_star)=0
    s.add(a*x_star**2 + b*x_star + c == 0)
    
    # sample domain [0,5]
    #samples = [i/2 for i in range(11)]  # 0,0.5,1,...,5
    #computational domain
    B = np.linspace(-2, 5, 100)
    for x_val in B:
        if x_val == x_star:
            continue
        V = a*x_val**2 + b*x_val + c
        dV = (2*a*x_val + b)* f_closed_loop(x_val)
        s.add(V > 0)
        s.add(dV <= 0)
    
    if s.check() == sat:
        m = s.model()
        print(f"Lyapunov candidate: V(x) = {m[a]}*x^2 + {m[b]}*x + {m[c]}")
        a_num = m[a].numerator_as_long()
        b_num = m[b].numerator_as_long()
        c_num = m[c].numerator_as_long()
        a_den = m[a].denominator_as_long()
        b_den = m[b].denominator_as_long()
        c_den = m[c].denominator_as_long()
        a_final = a_num / a_den
        b_final = b_num / b_den
        c_final = c_num / c_den
        return a_final, b_final, c_final
    else:
        print("No Lyapunov function found.")
        return None, None, None


a, b, c = cegis()
if a is not None:
    #print the lyapunov function
    x_axis = np.linspace(-5, 5, 400)
    V_vals = a*x_axis**2 + b*x_axis + c
    V_d_vals = (2*a*x_axis + b) * np.vectorize(f_closed_loop)(x_axis)
    plt.plot(x_axis, V_vals, label='Lyapunov Function V(x)')
    plt.plot(x_axis, V_d_vals, label='dV/dt')
    plt.legend()
    plt.show()

#