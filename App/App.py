import pywt
import numpy as np
from App.Dwt import *
import scipy.fftpack
from Drawings.wavelets_visualizations import *
from entropy import *
from sampen import sampen2

class FrameSettings:
    def __init__(self, dt, step , width_of_frame):
        self.dt=dt # sampling density
        self.f=1/self.dt #sampling frequency
        self.step = step # in seconds
        self.width = width_of_frame # in seconds , width of frame, sometimes called 'epoch'
        self.samples_in_frame = int(self.f * self.width)

class Wavelet:
    def __init__(self, names , example_signal = None):
        self.names = names
        self.example_signal = example_signal

    def set_example_signal(self, signal):
        self.example_signal = signal


    def get_decomposition_info(self,freq):
        if (self.example_signal is None):
            print('Brak przykłądowego syganłu')
        else:
            get_dec_info(self.example_signal, self.names, freq)


    def decomposition(self, data, details_levels, approximation = True):
        print('dekompozycja poziomu'+str(details_levels))
        final_data =[]

        for wavelet in self.names:
            print('falka: '+wavelet+'...')
            prefinal_data =[]
            d_data = decomposition(data, wavelet)
            data_details_levels = np.arange(1,len(d_data),1)
            levels_len = len(data_details_levels)
            for idx, det in enumerate(data_details_levels):
                if np.array(details_levels).__contains__(det):
                    f_data = np.array(d_data[det][0])
                    prefinal_data.append(f_data)

                    if approximation and idx == levels_len-1:
                        f_data = d_data[len(d_data)-1][0]
                        prefinal_data.append(f_data)

            final_data.append(prefinal_data)

        return final_data


def feature_dist(data,labels):
    for dec in data:
        dec_data = energy(dec)
        final_data = [None]*len(np.unique(labels))
        std_dev = [None] * len(np.unique(labels))

        for l_idx, l in enumerate(np.unique(labels)):
            sum_of_energy = [0]*len(dec_data[0])
            sum = 0

            for idx, sample in enumerate(dec_data):
                if labels[idx] == l :
                    #max_val = np.max(sample)

                    for i,e in enumerate(sample):
                        sum_of_energy[i] =  sum_of_energy[i] + e
                    #print(np.sum(sum_of_energy))
                    sum = sum +1


            for s in range(0,len(sum_of_energy)):
                sum_of_energy[s] =sum_of_energy[s]/sum
            final_data[l_idx] =  sum_of_energy

            std_devs = [0] * len(dec_data[0])
            for idx, sample in enumerate(dec_data):
                if labels[idx] == l :
                    #max_val = np.max(sample)

                    for i,e in enumerate(sample):
                        std_devs[i] =  std_devs[i] + (e - sum_of_energy[i] )**2
                    #print(np.sum(sum_of_energy))
                    sum = sum +1
            for s in range(0, len(std_devs)):
                std_devs[s] = np.sqrt(std_devs[s] / sum)
            std_dev[l_idx] = std_devs
        plot_feature_distribution(final_data,std_dev)
            # print(sum_of_energy)
            # print(np.sum(sum_of_energy))



def avg_energy(data,labels):
    for dec in data:
        dec_data = energy(dec)
        final_data = [None]*len(np.unique(labels))
        for l_idx, l in enumerate(np.unique(labels)):
            sum_of_energy = [0]*len(dec_data[0])
            sum = 0
            for idx, sample in enumerate(dec_data):
                if labels[idx] == l :
                    #max_val = np.max(sample)
                    for i,e in enumerate(sample):
                        sum_of_energy[i] =  sum_of_energy[i] + e/np.sum(sample)
                    #print(np.sum(sum_of_energy))
                    sum = sum +1


            for s in range(0,len(sum_of_energy)):
                sum_of_energy[s] =sum_of_energy[s]/sum
            final_data[l_idx] =  sum_of_energy

        plot_two_classes_energy(final_data)
            # print(sum_of_energy)
            # print(np.sum(sum_of_energy))

def energy(data):  # (1976, 2, 82)
    final_data = None
    # for lvl in data:
    lvl = data[0]
    print(np.shape(data[0]))
    #print(np.shape(data[1]))
    for lvl in data:
        prefinal_data = []
        for f in lvl:
            std_array = []
            for ch in f:
                # Y = scipy.fftpack.fft(ch)
                # # # #print(len(Y))
                # mY = np.abs(Y)  # Find magnitude
                #std_array.append(max_freq(ch,frame))
                #std_array.append(np.sum(ch ** 2))
                #std_array.append(sampen(ch, 3, 0.4 * np.std(ch)))
                #std_array.append( ApEn(ch,2,3))
                #std_array.append(feature(ch, 3))
                #std_array.append(app_entropy(x, order=2, metric='chebyshev'))
                #std_array.append(np.std(ch))
                #std_array.append(np.var(ch))
                #std_array.append(np.mean(ch))

                std_array.append(np.min(ch))
            prefinal_data.append(std_array)
        if final_data is None:

            final_data = prefinal_data
        else:
            final_data = np.concatenate((final_data, prefinal_data), axis=1)
    return final_data


def feature_extraction(data): # (1976, 2, 82)
    final_data = None
    # for lvl in data:
    lvl =data[0]
    print(np.shape(data[0]))
    #print(np.shape(data[1]))
    for lvl in data:
        prefinal_data = []
        for f in lvl:
            std_array = []
            for ch in f:
                Y = scipy.fftpack.fft(ch)
                # #print(len(Y))
                mY = np.abs(Y)  # Find magnitude
                std_array.append(np.max(mY ))
                std_array.append(np.sum(ch ** 2))
                #std_array.append( sampen(ch, 2, 0.4*np.std(ch)))
                std_array.append(ApEn(ch, 2, 3))
                std_array.append(feature(ch,3))
                #std_array.append(feature(ch,4))
                #std_array.append(np.std(ch))
                # std_array.append(np.var(ch))
                # std_array.append(np.mean(ch))
                std_array.append(np.max(ch))

                std_array.append(np.min(ch))
            prefinal_data.append(std_array)
        if final_data is None:
            final_data = prefinal_data
        else:
            final_data = np.concatenate((final_data,prefinal_data), axis = 1)
    return final_data


def sampen(L, m, r):
    N = len(L)
    B = 0.0
    A = 0.0
    # Split time series and save all templates of length m
    xmi = np.array([L[i: i + m] for i in range(N - m)])
    xmj = np.array([L[i: i + m] for i in range(N - m + 1)])
    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
    # Similar for computing A
    m += 1
    xm = np.array([L[i: i + m] for i in range(N - m + 1)])
    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
    return -np.log(A / B)


def ApEn(U, m, r):
    U = np.array(U)
    N = U.shape[0]

    def _phi(m):
        z = N - m + 1.0
        x = np.array([U[i:i + m] for i in range(int(z))])
        X = np.repeat(x[:, np.newaxis], 1, axis=2)
        C = np.sum(np.absolute(x - X).max(axis=2) <= r, axis=0) / z
        return np.log(C).sum() / z

    return abs(_phi(m + 1) - _phi(m))

def feature(data, power):
    n = len(data)
    mean = np.mean(data)
    var = pow(np.var(data), power)
    return sum( pow((x - mean),power)/((n -1)*var) for x in data)

def max_freq(data,frame):
    Fs = 1 / frame.dt
    n = len(data)
    k = np.arange(n)
    Ts = n / Fs
    frq = k / Ts  # Frequency range two sided
    Y = scipy.fftpack.fft(data) / n
    mY = np.abs(Y)  # Find magnitude
    peakY = np.max(mY)  # Find max peak
    locY = np.argmax(mY)  # Find its location
    frqY = frq[locY]  # Get the actual frequency value

    return frqY