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

    # def freq_dom(self):
    #     fourier_list = []
    #     for i in self.matched_linear_filter():
    #         fourier_list.append(np.fft.fft(i))
    #     fourier_list = np.asarray(fourier_list)
    #     return fourier_list

    def rms_val(self):
        filtered_vals = []
        for i in self.differential_emg():
            rms = np.sqrt(np.mean(i)**2)
            filtered_vals.append(rms)
        return filtered_vals

    def white_noise(self):
        # mu_list = []
        # for i in self.scaled_vals():
        #     mu, sigma = np.mean(i), np.std(i)
        #     mu_list.append(np.random.normal(mu, sigma, 20))
        # # mu, sigma = np.mean(self.scaled_vals()), np.std(self.scaled_vals())
        # return mu_list
        return np.random.normal(2*self.scaled_vals() + 2, 15)
        white_noise_list = []
        for i in self.scaled_vals():
            mu = np.mean(i)
            sigma = np.std(i)
            print(mu, sigma)
            white_noise_list.append(np.random.normal(mu, sigma, 2048))
        return white_noise_list

    def filter_signal(self):
        # set order and threshold of butterworth signal
        b, a = signal.butter(4, 0.2, 'low')

        # create the filtered signal
        filtered_signal = []
        for i in self.white_noise():
            output_signal = signal.filtfilt(b, a, i)
            filtered_signal.append(output_signal)

        filtered_signal = np.asarray(filtered_signal)
        # print(filtered_signal.shape) # (8, 2048)
        return filtered_signal

    def matched_linear_filter(self):
        linear_filter_vals = []
        for i in self.filter_signal():
            corr = signal.correlate(i, np.ones(128), mode='same') / 128
            linear_filter_vals.append(corr)
        return linear_filter_vals


    def signal_plotter(self):
        x_axis = []
        x_vals = 0.029
        for n in range(0, 2048):
            x_axis.append(x_vals)
            x_vals = x_vals + 0.029

        for i, k in enumerate(self.matched_linear_filter()):
            plt.subplot(8, 1, i+1)
            plt.plot(x_axis, k, label=i)
            plt.legend()
            plt.xlabel('time (seconds)')
            # plt.xlabel('frequency (hertz)')
            plt.ylabel('voltage (millivolts)')
        # plt.tight_layout()
        plt.show()


    def differential_emg(self):
        return -np.diff(self.matched_linear_filter(), axis=0)


    def sqrt_vals(self):
        rms_list = []
        for i in self.rms_val():
            rms_list.append(np.sqrt(i))
        return rms_list


x = SignalProcess(dataFrame)
x.signal_plotter()
print(x.sqrt_vals())