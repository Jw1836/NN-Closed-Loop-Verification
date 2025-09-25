import numpy as np
import csv
#This file contains the signal generator, dynamics, controller, and data plotter classes'
class DotDynamics:
    def __init__(self, x0):
        self.state = x0  # initial state
        self.Ts = 0.01  # sample time

    def update(self, u):
        # This is the external method that takes the input u at time
        # t and returns the output y at time t.
        self.rk4_step(u)  # propagate the state by one time sample
        y = self.state
        return y

    def f(self, state, u): #this is x_dot = f(x, u)
        # nonlinear dynamics
        return state**2 + u
    
    def rk4_step(self, u):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        F1 = self.f(self.state, u)
        F2 = self.f(self.state + self.Ts / 2 * F1, u)
        F3 = self.f(self.state + self.Ts / 2 * F2, u)
        F4 = self.f(self.state + self.Ts * F3, u)
        self.state = self.state + self.Ts / 6 * (F1 + 2*F2 + 2*F3 + F4)

class Controller: 
    def __init__(self, k):
        self.k = k  # controller gain

    def update(self, x_r, x):
        # This is the external method that takes the output y at time
        # t and returns the input u at time t.
        e = x - x_r  # tracking error
        u = self.k * e  # control law
        in_out_pair = [e, u]
        #self.write_to_file(in_out_pair)
        return u
    
    def write_to_file(self, data):
        file = '/Users/jwayment/Code/simple_nonlinear_system/in_out_data.csv'
        try:
            with open(file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data)
        except IOError:
            print("Could not open or write to the file.")
    


class signalGenerator:
    def __init__(self, amplitude=1.0, frequency=0.001, y_offset=0):
        self.amplitude = amplitude  # signal amplitude
        self.frequency = frequency  # signal frequency
        self.y_offset = y_offset  # signal y-offset

    def square(self, t):
        if t % (1.0/self.frequency) <= 0.5/self.frequency:
            out = self.amplitude + self.y_offset
        else:
            out = - self.amplitude + self.y_offset
        return out

    def sawtooth(self, t):
        tmp = t % (0.5/self.frequency)
        out = 4 * self.amplitude * self.frequency*tmp \
              - self.amplitude + self.y_offset
        return out

    def step(self, t):
        if t >= 0.0:
            out = self.amplitude + self.y_offset
        else:
            out = self.y_offset
        return out

    def random(self, t):
        out = np.random.normal(self.y_offset, self.amplitude)
        return out

    def sin(self, t):
        out = self.amplitude * np.sin(2*np.pi*self.frequency*t) \
              + self.y_offset
        return out
    



from matplotlib import get_backend
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import numpy as np

plt.ion()  # enable interactive drawing


class dataPlotter:
    def __init__(self):
        # Number of subplots = num_of_rows*num_of_cols
        self.num_rows = 1    # Number of subplot rows
        self.num_cols = 1    # Number of subplot columns
        # Crete figure and axes handles
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=True)
        # move_figure(self.fig, 500, 500)
        # Instantiate lists to hold the time and data histories
        self.time_history = []  # time
        self.theta_ref_history = []  # reference angle
        self.theta_history = []  # angle theta
        self.torque_history = []  # control torque
        # create a handle for every subplot.
        self.handle = []
        self.handle.append(myPlot(self.ax, xlabel = 'time', ylabel='x_pos', title='Dot Data'))
        #self.handle.append(myPlot(self.ax[1], xlabel='t(s)', ylabel='torqe(N-m)'))


    def update(self, t: float, states: np.ndarray, ctrl: float, reference: float = 0.):
        # update the time history of all plot variables
        self.time_history.append(t)  # time
        self.theta_ref_history.append(reference)  # reference base position
        self.theta_history.append(states.item(0))  # rod angle (converted to degrees)
        self.torque_history.append(ctrl)  # force on the base
        # update the plots with associated histories
        self.handle[0].update(self.time_history, [self.theta_history, self.theta_ref_history])
        #self.handle[1].update(self.time_history, [self.torque_history])

    def write_data_file(self):
        with open('io_data.npy', 'wb') as f:
            np.save(f, self.time_history)
            np.save(f, self.theta_history)
            #np.save(f, self.torque_history)



class myPlot:
    ''' 
        Create each individual subplot.
    '''
    def __init__(self, ax,
                 xlabel='',
                 ylabel='',
                 title='',
                 legend=None):
        ''' 
            ax - This is a handle to the  axes of the figure
            xlable - Label of the x-axis
            ylable - Label of the y-axis
            title - Plot title
            legend - A tuple of strings that identify the data. 
                     EX: ("data1","data2", ... , "dataN")
        '''
        self.legend = legend
        self.ax = ax                  # Axes handle
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b']
        # A list of colors. The first color in the list corresponds
        # to the first line object, etc.
        # 'b' - blue, 'g' - green, 'r' - red, 'c' - cyan, 'm' - magenta
        # 'y' - yellow, 'k' - black
        self.line_styles = ['-', '-', '--', '-.', ':']
        # A list of line styles.  The first line style in the list
        # corresponds to the first line object.
        # '-' solid, '--' dashed, '-.' dash_dot, ':' dotted

        self.line = []

        # Configure the axes
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        self.ax.set_title(title)
        self.ax.grid(True)

        # Keeps track of initialization
        self.init = True   

    def update(self, time, data):
        ''' 
            Adds data to the plot.  
            time is a list, 
            data is a list of lists, each list corresponding to a line on the plot
        '''
        if self.init == True:  # Initialize the plot the first time routine is called
            for i in range(len(data)):
                # Instantiate line object and add it to the axes
                self.line.append(Line2D(time,
                                        data[i],
                                        color=self.colors[np.mod(i, len(self.colors) - 1)],
                                        ls=self.line_styles[np.mod(i, len(self.line_styles) - 1)],
                                        label=self.legend if self.legend != None else None))
                self.ax.add_line(self.line[i])
            self.init = False
            # add legend if one is specified
            if self.legend != None:
                plt.legend(handles=self.line)
        else: # Add new data to the plot
            # Updates the x and y data of each line.
            for i in range(len(self.line)):
                self.line[i].set_xdata(time)
                self.line[i].set_ydata(data[i])

        # Adjusts the axis to fit all of the data
        self.ax.relim()
        self.ax.autoscale()
        plt.draw()
           
def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    figmgr = plt.get_current_fig_manager()
    figmgr.canvas.manager.window.raise_()
    geom = figmgr.window.geometry()
    x,y,dx,dy = geom.getRect()
    figmgr.window.setGeometry(10, 10, dx, dy)
    # backend = get_backend()
    # if backend == 'TkAgg':
    #     f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    # elif backend == 'WXAgg':
    #     f.canvas.manager.window.SetPosition((x, y))
    # else:
    #     # This works for QT and GTK
    #     # You can also use window.setGeometry
    #     #f.canvas.manager.window.move(x, y)
    #     f.canvas.manager.setGeometry(x, y)

# f, ax = plt.subplots()
# move_figure(f, 500, 500)
# plt.show()