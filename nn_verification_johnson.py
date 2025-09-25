import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
#This code verifies the NN robustness by perturbing the inputs
#and then calculating the reachable sets
#This method is the open loop verification method from Dr. Johnson's paper
def purelin(x):
    return x
def nn_function(x, W1, theta1, W2, theta2):
    # First layer
    z1 = W1 @ x + theta1.flatten()
    a1 = np.tanh(z1)

    # Second layer
    z2 = W2 @ a1 + theta2.flatten()
    a2 = purelin(z2)

    return a2

def relu(x):
  return np.maximum(0, x)

#n is dimension of x
def find_beta_max_min(w, x, theta, delta):
    n = x.shape[0]
    # Decision variables
    beta = cp.Variable()         # scalar
    dx = cp.Variable(n)          # Δx^[ℓ]
    
    # Constraints
    constraints = [
        beta == w @ (x + dx) + theta,
        cp.norm_inf(dx) <= delta
    ]

    # --- Maximize beta ---
    prob_max = cp.Problem(cp.Maximize(beta), constraints)
    beta_max = prob_max.solve()
    dx_max = dx.value.copy()   # save optimizer result

    # --- Minimize beta ---
    prob_min = cp.Problem(cp.Minimize(beta), constraints)
    beta_min = prob_min.solve()
    dx_min = dx.value.copy()

    #print("Max beta:", beta_max, "with Δx:", dx_max)
    #print("Min beta:", beta_min, "with Δx:", dx_min)

    return beta_max, beta_min

def find_max_gamma(beta_max, beta_min, x, w, theta, function):
    if(function == "tanh"):
        gamma_1 = abs(np.tanh(beta_max) - np.tanh(w.T @ x + theta))
        gamma_2 = abs(np.tanh(beta_min) - np.tanh(w.T @ x + theta))
    elif(function == "purelin"):
        gamma_1 = abs(purelin(beta_max) - purelin(w.T @ x + theta))
        gamma_2 = abs(purelin(beta_min) - purelin(w.T @ x + theta))

    return max(gamma_1, gamma_2)

def find_max_sensitivity(x_val, delta):
    #iterate through layers
    for i in range(L-1):
        # print("delta for layer", i+1, ":", delta)
        # print("input for layer", i+1, ":", x_val)
        #print(f"================ Layer {i+1} ====================")
        #iterate through neurosns in that layer
        if(i == 0):
            neurons = W1.shape[0]
            w = W1
            theta = theta1
            function = "tanh"
        if(i == 1):
            neurons = W2.shape[0]
            w = W2
            theta = theta2
            function = "purelin"
        gamma_list = []
        for j in range(neurons):
            #print(f"Layer {i+1}, Neuron {j+1}")
            beta_max, beta_min = find_beta_max_min(w[j,:], x_val, theta[j], delta)
            gamma = find_max_gamma(beta_max, beta_min, x_val, w[j,:], theta[j], function)
            gamma_list.append(gamma)
        #the max gamma for that layer is the max sensitivity for that layer
        max_gamma = max(gamma_list)
        #update delta and input for next layer
        delta = max_gamma
        if(function == "tanh"):
            x_val = np.tanh(w @ x_val + theta.flatten())
        elif(function == "purelin"):
            x_val = purelin(w @ x_val + theta.flatten())
        else:
            print("Error: unknown activation function")
            sys.exit(1)
        # print("delta for next layer:", delta)
        # print("input for next layer:", x_val)
        # print("--------------------------------------------------")
    print("X final:", x_val, delta)
    return delta
# n = 5
# w = np.random.randn(n)
# x = np.random.randn(n)
# theta = np.random.randn()
# delta = 1.0

# beta_max, beta_min = find_beta_max_min(n, w, x, theta, delta)
##################################################################
## PARAMETERS FROM TRAINED NN MODEL , 2 n input, 5 n hidden, 2 n output
##################################################################
# W^[1]
L = 3 #number of layers 
W1 = np.array([
    [-0.9507, -0.7680],
    [ 0.9707,  0.0270],
    [-0.6876, -0.0626],
    [ 0.4301,  0.1724],
    [ 0.7408, -0.7948]
])

# θ^[1]
theta1 = np.array([
    [ 1.1836],
    [-0.9087],
    [-0.3463],
    [ 0.2626],
    [-0.6768]
])

# W^[2]
W2 = np.array([
    [0.8280, 0.6839, 1.0645, -0.0302,  1.7372],
    [1.4436, 0.0824, 0.8721,  0.1490, -1.9154]
])

# θ^[2]
theta2 = np.array([
    [-1.4048],
    [-0.4827]
])
##################################################################


#plot 1000 outputs from random inputs in [0,1]x[0,1]
num_samples = 1000
x1_samples = np.random.uniform(0, 1, num_samples)
x2_samples = np.random.uniform(0, 1, num_samples)
for i in range(num_samples):
    x_val = np.array([x1_samples[i], x2_samples[i]])
    output = nn_function(x_val, W1, theta1, W2, theta2)
    plt.scatter(output[0], output[1], color='blue', s=10)
#plt.show()

delta = .1
num_rectangles = (1 / (delta * 2))**2
x_1_centers = np.linspace(delta, 1-delta, int(np.sqrt(num_rectangles)))
x_2_centers = np.linspace(delta, 1-delta, int(np.sqrt(num_rectangles)))
print("num_rectangles:", num_rectangles)
print("x_1_centers:", x_1_centers)
x_list = []
for i in range(len(x_1_centers)):
    for j in range(len(x_2_centers)):
        x_val = np.array([x_1_centers[i], x_2_centers[j]])
        x_list.append(x_val)
        output = nn_function(x_val, W1, theta1, W2, theta2)
        max_sen = find_max_sensitivity(x_val, delta)[0]
        # print("Max sensitivity for input", x_val, "is:", max_sen)
        rect = Rectangle((output[0]-max_sen, output[1]-max_sen), 2*max_sen, 2*max_sen, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        #plt.scatter(output[0], output[1], color='black', s=50) #plot center of rectangle

plt.title(f"delta ={delta} NN reachability sets from inputs in [0,1]x[0,1]")
plt.show()


# plt.scatter([x[0] for x in x_list], [x[1] for x in x_list], color='green', s=50, label='Input Centers')
# plt.scatter(x1_samples, x2_samples, color='blue', s=10, label='Random Inputs')
# plt.title(f"Input centers for delta ={delta} in [0,1]x[0,1]")
# plt.xlim([-0.1, 1.1])
# plt.ylim([-0.1, 1.1])
# plt.xlabel("Input 1")
# plt.ylabel("Input 2")
# plt.axis('equal')
# plt.grid(True)
# plt.legend()
# plt.show()
# plt.title("NN Outputs from 1000 Random Inputs in [0,1]x[0,1]")
# plt.xlim([-4.5, -1.5])
# plt.ylim([-1.5, 2])
# plt.xlabel("Output 1")
# plt.ylabel("Output 2")
# plt.axis('equal')
# plt.grid(True)
# plt.show()