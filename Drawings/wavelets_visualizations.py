import math
import pywt
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from matplotlib.gridspec import GridSpec
from mlxtend.plotting import plot_decision_regions
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm, datasets
from matplotlib import colors as c
from sklearn.neighbors import KNeighborsClassifier


def plot_cwt(signal,scales,waveletname,sampling_freq):
    dt=1/sampling_freq
    t = np.arange(0.0, len(signal))*dt

    cwtmatr, freqs = pywt.cwt(signal, scales, waveletname,sampling_period=dt)
    gs=GridSpec(2,20)
    fig=plt.figure()
    ax0=fig.add_subplot(gs[0,0:16])
    ax1=fig.add_subplot(gs[1,:])
    ax0.plot(t, signal)
    ax0.margins(0.00)
    ax0.set( ylabel='oryginalny sygnał',title='')
    ax0.grid()
    ax0.set_xticklabels([])
    ax1.set_xlabel('czas(s)')
    ax1.set_ylabel('skala')
    im=ax1.imshow(cwtmatr, aspect='auto',vmin=cwtmatr.min(),vmax=cwtmatr.max(),origin='lower',cmap='Greys',extent=[0,len(signal)*dt,0,len(scales)])
    fig.colorbar(im,ax=ax1)
    plt.show()

def plot_dwt(signal,waveletname,sampling_freq):
    dt=1/sampling_freq
    dwtmatr = pywt.wavedec(signal, waveletname)
    list_of_len=[]
    for i,row in enumerate(dwtmatr):
        x=math.ceil(len(dwtmatr[len(dwtmatr)-1])/len(row))
        dwtmatr[i]=np.repeat(row,x)
        list_of_len.append(len(row)*x)
    print(signal)
    cut_len=min(list_of_len)
    for i in range(len(dwtmatr)-1,0,-1):
        dwtmatr[i]=dwtmatr[i][0:cut_len]
    fig, ax=plt.subplots(figsize = (15,2))
    plt.imshow(dwtmatr[1::1], aspect='auto',origin='upper',cmap='Greys',extent=[1,len(signal)*dt, 0, pywt.dwt_max_level(len(signal), pywt.Wavelet(waveletname).dec_len) ])

    ax.set_xlabel('czas(s)')
    ax.set_ylabel('poziom')
    plt.colorbar()
    plt.show()


def plot_raw_signal(signal,sampling_freq):
	dt=1/sampling_freq
	t = np.arange(0.0, len(signal))*dt
	fig, ax = plt.subplots(figsize=(7,2))
	ax.plot(t, signal)

	ax.set(xlabel='czas (s)', ylabel='oryginalny sygnał',title='')
	ax.grid()
	plt.show()

def plot_dwt_decomposition(signal,waveletname,sampling_freq):
	dt=1/sampling_freq
	t = np.arange(0.0, len(signal))
	level=pywt.dwt_max_level(len(signal),waveletname)
	print(len(signal))
	data = pywt.wavedec(signal, waveletname)
	fig, axarr = plt.subplots(nrows=level+2, ncols=1, figsize=(9,7))
	axarr[0].plot(t, signal, 'b')
	axarr[0].set_ylabel("Oryginalny sygnał", fontsize=5, rotation=90)
	for ii in range(1,level+1):
		t = np.arange(0.0, len(data[level+1-ii]))
		axarr[ii].plot(t,data[level+1-ii],'r')
		axarr[ii].set_ylabel("D {}".format(ii ), fontsize=7, rotation=90)
		axarr[ii].set_yticklabels([])
	t = np.arange(0.0, len(data[0]))
	axarr[level+1].plot(t,data[0],'r')
	axarr[level+1].set_ylabel("A {}".format((level) ), fontsize=7, rotation=90)
	axarr[level+1].set_yticklabels([])
	plt.tight_layout()
	plt.show()

def draw_psi_an_phi(name):
	wavelet = pywt.Wavelet(name)
	phi, psi, x = wavelet.wavefun(level=5)
	fig, ax = plt.subplots(1, 2,figsize=(10,3))

	ax[0].plot(x,psi)
	ax[0].set_title('Falka '+name)
	ax[1].plot(x, phi,"green")
	ax[1].set_title('Funkcja skalująca '+name)
	plt.show()


def plot_feature_distribution(data, std):
    levels = ['d1','d2','d3','d4','d5','a5']
    labels = ['grupa D+C', 'grupa A+B']

    x = np.arange(len(data[0]))
    rects = [None] * len(data)
    fig, ax = plt.subplots()
    width = 0.2
    for idx,d in enumerate(data):
        rects[idx] = ax.bar(x-width*idx+0.1, d, yerr=std[idx], alpha=0.8, ecolor='black', capsize=5, width = width, label=labels[idx])
    ax.set_ylabel('')
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_title('Maksymalna wartość w próbce — dekompozycja dla falki sym14')
    #ax.yaxis.grid(True)

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_two_classes_energy(data):
    labels = ['grupa D', 'grupa C', 'grupa B', 'grupa A']
    levels = ['d1','d2','d3','d4','d5','a5']

    x = np.arange(len(data[0]))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    rects = [None]*len(data)
    for idx,d in enumerate(data):
        rects[idx] = ax.bar(x- width / len(data)+width*idx-0.2, data[idx], width, label=labels[idx])

    ax.set_ylabel('Procentowy udział')
    ax.set_title('Średni udział energii sygnału na kilku poziomach dekompozycji')
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.legend()


    fig.tight_layout()

    plt.show()

# sampling_freq=200
# x = np.linspace(0, 1, num=2048)
# chirp_signal = np.sin(250 * np.pi * x**2)
# plot_raw_signal(chirp_signal,sampling_freq=sampling_freq)
# plot_dwt(chirp_signal,waveletname='db3',sampling_freq=sampling_freq)
# plot_cwt(chirp_signal,scales=np.arange(1,30),waveletname='mexh',sampling_freq=sampling_freq)
# plot_dwt_decomposition(chirp_signal,'sym5',sampling_freq=sampling_freq)





def draw_fft(signal,sampling_freq):
	N=len(signal)
	T=1/sampling_freq
	yf = scipy.fftpack.fft(signal)
	xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
	fig, ax = plt.subplots()
	ax.plot(xf[:150], 2.0/N * np.abs(yf[:150]))
	plt.show()




def plot_decision_region(X, y, clf):
	# Training a classifier
	clf.fit(X, y)


	# Plotting decision regions
	axx = plot_decision_regions(X, y, clf=clf, legend=0,colors = 'r,b', markers= 'ox')
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.show()


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin


def plot_3d(data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = 100

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    print(np.shape(labels))
    for i,d in enumerate(data):
        if(labels[i] == 0):
            c = 'g'
        else:
            c = 'r'
        ax.scatter(d[0], d[1], d[2], c= c, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def plot_3d_svm(clf,X, Y ):

    clf = clf.fit(X, Y)

    # The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
    # Solve for w3 (z)
    z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]

    tmp = np.linspace(-4, 4, 10)
    x, y = np.meshgrid(tmp, tmp)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i,d in enumerate(X):
        if(Y[i] == 0):
            c = 'b'
            m = 'x'
        else:
            c = 'r'
            m = 'o'

        ax.scatter(d[0], d[1], d[2], c= c, marker= m)
    ax.scatter(0, 0, 0,  marker='x', c= 'b', label ='1')
    ax.scatter(0, 0, 0, marker='o', c='r', label='-1')
    ax.plot_surface(x, y, z(x, y), alpha =0.6)
    ax.view_init(3, 6)
    plt.legend(loc='best')
    plt.show()

def plot_example_svm(arrows = False, seed = 0, margins = [2,2]):
    np.random.seed(seed)
    X = np.r_[np.random.randn(20, 2) - margins, np.random.randn(20, 2) + margins]
    Y = [0] * 20 + [1] * 20
    _, ax = plt.subplots()
    # figure number
    fignum = 1

    # fit the model
    for name, penalty in (('unreg', 1), ('reg', 0.05)):
        clf = svm.SVC(kernel='linear', C=penalty)
        clf.fit(X, Y)

        # get the separating hyperplane
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-5, 5)
        yy = a * xx - (clf.intercept_[0]) / w[1]

        # plot the parallels to the separating hyperplane that pass through the
        # support vectors (margin away from hyperplane in direction
        # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
        # 2-d.
        margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
        yy_down = yy - np.sqrt(1 + a ** 2) * margin
        yy_up = yy + np.sqrt(1 + a ** 2) * margin

        # plot the line, the points, and the nearest vectors to the plane
        plt.figure(fignum, figsize=(10, 5),)
        plt.clf()
        plt.plot(xx, yy, 'k-')
        plt.plot(xx, yy_down, 'k--')
        plt.plot(xx, yy_up, 'k--')


        #
        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                    facecolors='none', zorder=10, edgecolors='k')
        print(Y)
        plt.scatter(X[:20, 0], X[:20, 1],marker = 'x', c=Y[:20], zorder=10, cmap=c.ListedColormap(['b', 'darkgrey']))
        plt.scatter(X[20:, 0], X[20:, 1],marker = 'o',  c=Y[20:], zorder=10, cmap=c.ListedColormap(['r', 'darkgrey']))

        plt.axis('tight')
        x_min = -4.8
        x_max = 4.2
        y_min = -6
        y_max = 6

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.figure(fignum, figsize=(4, 3))
        plt.pcolormesh(XX, YY, Z ,cmap=c.ListedColormap(['lightgrey', 'None']), shading='gouraud')


        fignum = fignum + 1
        if(arrows):
            plt.annotate('granica decyzyjna', xy=(xx[10],yy[10]), xytext=(xx[10]-2.5,yy[10]-2),
                        arrowprops=dict(facecolor='black', shrink=0.05, width = 0.3, headwidth =4))
            plt.annotate('płaszczyzna "negatywna"', xy=(xx[10],yy_down[10]), xytext=(xx[10]-2.5,yy_down[10]-2),
                         arrowprops=dict(facecolor='black', shrink=0.05, width = 0.3, headwidth =4))
            plt.annotate('płaszczyzna "pozytywna"', xy=(xx[12],yy_up[12]), xytext=(xx[12]+1,yy_up[12]-0.2),
                        arrowprops=dict(facecolor='black', shrink=0.05, width = 0.3, headwidth =4))

            plt.annotate(s='' ,xy=(xx[36], yy_down[36]), xytext=(xx[38], yy_up[38]), arrowprops=dict(arrowstyle='<->'))
            plt.text(xx[35],yy[38], 'margines', horizontalalignment='center', verticalalignment ='center', c = 'r')

            for i in np.arange(0, len(clf.support_vectors_), 1):
                plt.annotate('', xy=(clf.support_vectors_[i, 0], clf.support_vectors_[i, 1]), xytext=(3, 3),
                             arrowprops=dict(facecolor='black', shrink=0.05, width=0.3, headwidth=4))
            plt.text(3.1,3.4, 'wektory nośne', horizontalalignment='center', verticalalignment ='center')

            plt.annotate(s='', xy = (xx[17]-(clf.intercept_[0]) / w[1], yy_up[17]-(clf.intercept_[0]) / w[1]), xytext = (xx[17], yy[17]), arrowprops=dict(facecolor='black'))
            plt.text(xx[16], yy[17]+0.9, 'w', horizontalalignment='center', verticalalignment ='center', fontsize=15)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel('x1')
        plt.ylabel('x2')
        ax = plt.gca()
        #ax.spines['left'].set_position('left')
        #ax.spines['right'].set_color('none')
        #ax.spines['bottom'].set_position('bottom')
        #ax.spines['top'].set_color('none')
        ax.plot(1, -6, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(-4.8,1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        plt.xticks(())
        plt.yticks(())
    plt.show()

def xor_example():
    np.random.seed(1)
    X_xor = np.random.randn(200,3)
    y_xor = np.logical_xor(X_xor[:,0]>0, X_xor[:,1]>0)
    Y_xor = np.where(y_xor, 1 ,-1)

    print(X_xor[0])

    X_xor[:,2] = X_xor[:,1] * X_xor[:,0]
    print(np.shape(X_xor[:,0:2]))
    plt.scatter(X_xor[Y_xor == 1,0], X_xor[Y_xor ==1, 1], c='b', marker = 'x', label ='l')
    plt.scatter(X_xor[Y_xor == -1,0], X_xor[Y_xor ==-1, 1], c='r', marker = 'o', label ='-l')
    plt.ylim([-3,3])
    plt.xlim([-3,3])
    plt.legend(loc = 'best')
    plt.show()
    plot_3d_svm(svm.SVC(kernel='linear'), X_xor, y_xor)
    plot_decision_region( X_xor[:,0:2], np.array(y_xor.astype(np.integer)), KNeighborsClassifier(n_neighbors=1,  metric='minkowski'))

# xor_example()