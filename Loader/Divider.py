
import numpy as np
import math as mt
import matplotlib.pyplot as plt



def Divide_signals_on_frames(data,labels,frame):
    final_frames = []
    final_labels =[]
    for num in range(0,len(data)):
        frames_tmp, labels_tmp = Divide_one_participant_signal_on_frames(data[num],labels[num], frame)
        for i in range(0,len(frames_tmp)):
            final_frames.append(frames_tmp[i])
        for i in range(0,len(labels_tmp)):
            final_labels.append(labels_tmp[i])
    return final_frames, final_labels



# def Divide_one_participant_signal_on_frames(data,label, frame):
#     num_of_frames = mt.floor(len(data[0])/(frame.width * frame.f))
#     labels = np.full((num_of_frames,1),label)
#     frames =[]
#     for i in np.arange(0,num_of_frames,1):
#         frames.append(data[:,int(i*frame.width * frame.f):int((i+1)*frame.width * frame.f)])
#     return frames, labels

def Divide_one_participant_signal_on_frames(data,label, frame):
    chanel = []
    frames = []
    frame_begin = 0
    it = 0
    while(int(frame_begin+frame.width * frame.f)<=len(data[0])):
        frames.append([data[0][int(frame_begin):int(frame_begin + frame.width * frame.f)]])
        frame_begin = frame_begin + frame.step * frame.f
        it = it + 1

    labels = np.full(it,label)
    return frames, labels

def Schuffle_data(data, labels):
    id = np.array((np.random.permutation(len(data))))
    s_data, s_labels = np.array(data)[id.astype(int)], np.array(labels)[id.astype(int)]
    return s_data, s_labels

def Divide_data_set(data, labels, test_coef=0.2):
    len_of_data = len(data)
    border = mt.floor(len_of_data*test_coef)
    test_data, test_labels = data[:border], labels[:border]
    trainning_data, training_labels = data[border:], labels[border:]
    return trainning_data, training_labels, test_data, test_labels

