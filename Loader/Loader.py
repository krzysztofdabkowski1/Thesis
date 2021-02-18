import pyedflib
import numpy as np
from Loader.Divider import Divide_signals_on_frames, Divide_one_participant_signal_on_frames
from os import listdir
import os
from os.path import isfile, join

def load_data_from_NEDC(frame, epilepsy_catalog_path, no_epilepsy_catalog_path, channels):
    e_path = epilepsy_catalog_path
    ne_path = no_epilepsy_catalog_path
    paths =[(e_path,1),(ne_path,0)]
    data = []
    labels = []
    for path_f in paths:
        files = [f for f in listdir(path_f[0]) if isfile(join(path_f[0], f))]

        for filename in files:
            input_file = pyedflib.EdfReader(path_f[0]+'/'+filename)

            n = input_file.signals_in_file
            n = n-3
            sigbufs = np.zeros((len(channels), input_file.getNSamples()[0]))

            idx = 0
            for i in np.arange(n):
                if np.array(channels).__contains__(i):
                    sigbufs[idx , :] = input_file.readSignal(i)
                    idx = idx + 1

            f_data, f_labels = Divide_one_participant_signal_on_frames(sigbufs, path_f[1], frame)
            for d in f_data:
                #data.append(f_data[:][:][:][0])
                data.append(d)
            for f in f_labels:
                #labels.append(f_labels[:][0])
                labels.append(f)
    return data, labels
def load_data_from_Bonn(frame, catalog_path, divide_on_two):
    data = []
    labels = []
    for subdir, dirs, files in os.walk(catalog_path):
        label = 0
        for dir in dirs:
            files = [f for f in listdir(catalog_path+'/'+dir) if isfile(join(catalog_path+'/'+dir, f))]

            for filename in files:
                input_file = open(catalog_path+'/'+dir+'/'+filename, "r")

                one_signal_array = []
                for line in input_file.readlines():
                    one_signal_array.append(float(line))


                if(divide_on_two):
                    if(label == 0 or label == 1):
                        data.append([one_signal_array])
                        labels.append(0)
                    else:
                        data.append([one_signal_array])
                        labels.append(1)

                else:
                    data.append([one_signal_array])
                    labels.append(label)

            print(np.shape(data))
            label = label +1
    print('p')
    print(np.shape(data))
    data, labels = Divide_signals_on_frames(data,labels, frame)
    print('p')
    print(np.shape(data))
    return data, labels

def load_data_motor_movement(frame, catalog_path, channels, amount_of_frames):

    data = []
    labels = []

    for subdir, dirs, files in os.walk(catalog_path):
        for dir in dirs:
            print(dir)
            if(len(data)>= amount_of_frames):
                break
            files = [f for f in listdir(catalog_path + '/' + dir) if isfile(join(catalog_path + '/' + dir, f))]

            for filename in files:
                if(filename.endswith(tuple(['5.edf','9.edf','13.edf']))):
                    input_file = pyedflib.EdfReader(catalog_path + '/' + dir + '/' + filename)
                    ann = input_file.read_annotation()
                    labels_f = [int(chr(i[2][1])) for i in ann]
                    intervals = [int((i[0]/10000000)*frame.f) for i in ann]
                    for index, inte in enumerate(intervals):
                        if(labels_f[index] != 0):
                            sigbufs = np.zeros((len(channels), frame.samples_in_frame))
                            idx = 0
                            for i in np.arange(input_file.signals_in_file):
                                if np.array(channels).__contains__(i):
                                    sigbufs[idx , :] = input_file.readSignal(9)[inte:inte+frame.samples_in_frame] #- input_file.readSignal(i-7)[inte:inte+frame.samples_in_frame]
                                    idx = idx + 1
                            data.append(sigbufs)
                            labels.append(labels_f[index])
                            if (len(data) >= amount_of_frames):
                                break
                    #print(np.shape(data))



    # for path_f in paths:
    #     files = [f for f in listdir(path_f[0]) if isfile(join(path_f[0], f))]
    #
    #     for filename in files:
    #         input_file = pyedflib.EdfReader(path_f[0]+'/'+filename)
    #
    #         n = input_file.signals_in_file
    #         n = n-3
    #         sigbufs = np.zeros((len(channels), input_file.getNSamples()[0]))
    #
    #         idx = 0
    #         for i in np.arange(n):
    #             if np.array(channels).__contains__(i):
    #                 sigbufs[idx , :] = input_file.readSignal(i)
    #                 idx = idx + 1
    #
    #         f_data, f_labels = Divide_one_participant_signal_on_frames(sigbufs, path_f[1], frame)
    #         for d in f_data:
    #             #data.append(f_data[:][:][:][0])
    #             data.append(d)
    #         for f in f_labels:
    #             #labels.append(f_labels[:][0])
    #             labels.append(f)
    return data, labels

