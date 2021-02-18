
from sklearn import svm

from Loader.Loader import load_data_from_Bonn, load_data_motor_movement
from App.App import FrameSettings, Wavelet, feature_extraction, avg_energy, feature_dist
from Loader.Divider import Divide_signals_on_frames, Schuffle_data, Divide_data_set
from App.Dwt import decomposition
from Drawings.wavelets_visualizations import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler

channels = [9] #np.arange(0,20,1) # [2,9,16,6,13,20] #
frame = FrameSettings(dt = 1/173.61,
                     step = 22,
                     width_of_frame =22)
# data, labels =load_data_motor_movement(frame = frame,
#                                    catalog_path = 'C://Users/Asus/Desktop/database_EEG/motor_imagery/files' ,
#                                   channels = channels,
#                                    amount_of_frames=60)
data, labels = load_data_from_Bonn(frame = frame,
                                  catalog_path = 'C://Users/Asus/Desktop/database_EEG/epilepsy_Bonn',
                                   divide_on_two = 0)


# plot_raw_signal(f_data[0][0],frame.f)
#
#data, labels = Schuffle_data(data, labels )

wavelets = pywt.wavelist(kind='discrete')  #['db6'] #
wave_tmp = []
for i,w in enumerate(wavelets):
    if(('sym' in wavelets[i]) or ('coif' in wavelets[i]) or ('db' in wavelets[i]) or ('haar' in wavelets[i])):
        wave_tmp.append(wavelets[i])
wavelets = wave_tmp
print(wavelets)
wavelet = Wavelet(names = wavelets,
                  example_signal = data[0][0])
#plot_dwt(data[0][0],wavelets[0],frame.f)

#plot_dwt_decomposition(data[0][0],wavelets[0],frame.f)
#plot_cwt(data[0][0],np.arange(1,100),wavelets[0],frame.f)

#plot_dwt_decomposition(data[0][0],'db4', int(frame.f))
wavelet.get_decomposition_info(frame.f)

details_levels = [1,2,3,4,5,6,7,8]
d_data= wavelet.decomposition(data,details_levels = details_levels, approximation =1)

print(np.shape(d_data[0][0]))
print(len(d_data[0][0][1]))
#avg_energy(d_data,labels)
#feature_dist(d_data,labels)
#print(np.shape(d_data[0]))
print(len(d_data))
#print(np.shape(d_data[2][1]))
f = open("4_klasy.csv", "a")
for i, wave in enumerate(wavelets):

    t_data = feature_extraction(d_data[i])
    print(np.shape(t_data))
    # scaler = MinMaxScaler()
    # t_data= scaler.fit_transform(t_data)
    train_data,test_data, train_labels , test_labels = train_test_split(t_data, labels, test_size=0.1, random_state=183, stratify= labels)#Divide_data_set(f_data, f_labels, test_coef=0.9)

    # clf = svm.SVC(kernel = 'rbf', gamma='scale')
    #clf = make_pipeline(StandardScaler(), svm.SVC(kernel = 'rbf', gamma='auto')) # clf =svm.SVC(probability=True, gamma= 'auto')
    clf = make_pipeline(StandardScaler(), GaussianNB())
    #clf = KNeighborsClassifier(n_neighbors=4)
#
    #clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
    #clf = make_pipeline(StandardScaler(), LDA())
#clf =LDA()
# xx=[]
# yy=[]
# i=0
# for epoch in x:
#     if( y[i]==5 or y[i]==6):
#         xx.append(epoch[5])
#         yy.append(y[i])
#     i+=1
#print(standard_deviation(train_data)[0])
# print(np.shape(train_data))
# #train_data = feature_extraction(train_data)
# print(train_data[0])
# print(train_labels)
#print(train_labels)
#train_data = np.reshape(train_data,(len(train_data),-1))
# test_data = np.reshape(test_data,(len(test_data),-1))
#print(np.shape(train_data))
    print('trenowanie modelu')
    #plot_3d(train_data,train_labels)
    #plot_3d_svm(clf, np.array(train_data),np.array(train_labels))
    # plot_example_svm(arrows= False,seed = 775, margins = [1.1, 1.1])
    # plot_example_svm(True)
    #plot_decision_region(np.array(train_data),np.array(train_labels),clf)
    clf.fit(train_data, train_labels)
# print(train_labels)


##############################################################################
    print('testowanie modelu :'+ wave)
    pred_labels = clf.predict(test_data)
    #print(pred_labels)
    print(accuracy_score(test_labels, pred_labels))

    print(wave)
    scores = cross_val_score(clf, train_data, train_labels, cv=5)
    cv_5=scores.mean()
    #print('#################')

    scores = cross_val_score(clf, train_data, train_labels, cv=10)
    cv_10=scores.mean()
    #print('#################')
    scores = cross_val_score(clf, train_data, train_labels, cv=15)
    cv_15=scores.mean()
    #print('#################')


    f.write(wave+' '+str(round(cv_5, 3))+' '+str(round(cv_10, 3))+ ' '+ str(round(cv_15, 3))+'\n')
f.close()
    # cm =confusion_matrix(test_labels,pred_labels)
    # print((cm[1][0]+cm[0][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))

