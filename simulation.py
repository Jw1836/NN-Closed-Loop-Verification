"""Generic simulation utilities: signal generation and real-time plotting."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.ion()  # enable interactive drawing


class SignalGenerator:
    def __init__(self, amplitude=1.0, frequency=0.001, y_offset=0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.y_offset = y_offset

    def square(self, t):
        if t % (1.0 / self.frequency) <= 0.5 / self.frequency:
            return self.amplitude + self.y_offset
        else:
            return -self.amplitude + self.y_offset

    def sawtooth(self, t):
        tmp = t % (0.5 / self.frequency)
        return (
            4 * self.amplitude * self.frequency * tmp - self.amplitude + self.y_offset
        )

    def step(self, t):
        if t >= 0.0:
            return self.amplitude + self.y_offset
        else:
            return self.y_offset

    def random(self, t):
        return np.random.normal(self.y_offset, self.amplitude)

    def sin(self, t):
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t) + self.y_offset


class DataPlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1, sharex=True)
        self.time_history = []
        self.state_history = []
        self.ref_history = []
        self.ctrl_history = []
        self.handle = [
            SubPlot(self.ax, xlabel="time", ylabel="x_pos", title="Dot Data")
        ]

    def update(self, t, states, ctrl, reference=0.0):
        self.time_history.append(t)
        self.ref_history.append(reference)
        self.state_history.append(np.atleast_1d(states).item(0))
        self.ctrl_history.append(ctrl)
        self.handle[0].update(
            self.time_history,
            [self.state_history, self.ref_history],
        )

    def write_data_file(self, filename="io_data.npy"):
        with open(filename, "wb") as f:
            np.save(f, self.time_history)
            np.save(f, self.state_history)


class SubPlot:
    """Single subplot with auto-scaling line management."""

    def __init__(self, ax, xlabel="", ylabel="", title="", legend=None):
        self.legend = legend
        self.ax = ax
        self.colors = ["b", "g", "r", "c", "m", "y", "b"]
        self.line_styles = ["-", "-", "--", "-.", ":"]
        self.line = []
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        self.ax.set_title(title)
        self.ax.grid(True)
        self.init = True

    def update(self, time, data):
        if self.init:
            for i in range(len(data)):
                self.line.append(
                    Line2D(
                        time,
                        data[i],
                        color=self.colors[np.mod(i, len(self.colors) - 1)],
                        ls=self.line_styles[np.mod(i, len(self.line_styles) - 1)],
                        label=self.legend if self.legend is not None else None,
                    )
                )
                self.ax.add_line(self.line[i])
            self.init = False
            if self.legend is not None:
                plt.legend(handles=self.line)
        else:
            for i in range(len(self.line)):
                self.line[i].set_xdata(time)
                self.line[i].set_ydata(data[i])

        self.ax.relim()
        self.ax.autoscale()
        plt.draw()
