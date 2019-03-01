import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import MinMaxScaler

cwd = "/Users/rohan/downloads/EMG_Data_Assignment1.xlsx"
dataFrame = pd.read_excel(cwd, delimiter=',',)


class SignalProcess(object):

    def __init__(self, dataFrame):
        dataFrame.drop(dataFrame.columns[[8, 9, 10, 11, 12]], axis=1, inplace=True)
        self.dataFrame = dataFrame

    def get_metrics(self):
        return self.dataFrame.describe()

    def scaled_vals(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaled_values = scaler.fit_transform(self.dataFrame)
        x = rescaled_values.reshape(8, 2048)
        return x

    def filter_signal(self):
        # set order and threshold of butterworth signal
        b, a = signal.butter(4, 0.2, 'low')

        # create the filtered signal
        filtered_signal = []
        for i in self.scaled_vals():
            output_signal = signal.filtfilt(b, a, i)
            filtered_signal.append(output_signal)

        filtered_signal = np.asarray(filtered_signal)
        print(filtered_signal.shape)
        return filtered_signal

    def signal_plotter(self):
        x_axis = []
        x_vals = 0.029
        for n in range(0, 2048):
            x_axis.append(x_vals)
            x_vals = x_vals + 0.029

        for i in self.filter_signal():
            plt.plot(x_axis, i)
            plt.show()

x = SignalProcess(dataFrame)
x.signal_plotter()