import numpy as np
import matplotlib.pyplot as plt


# setting the axes at the centre
fig = plt.figure(figsize=(6, 6), dpi=80)
ax = fig.add_subplot(1, 1, 1)
#ax.spines['left'].set_position('left')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xlim(left=0)




def Haar(start, end, scale, amp = 1,full =False):
    amplitude = amp
    amp=scale/amp
    half = start+(end-start)/2
    half_val = (end-start)/2
    t1 = np.arange(start, half, half_val/10000)
    t2 = np.arange(half,end, half_val/10000)
    first = np.linspace(scale, scale, 10000)
    second = np.linspace(-scale, -scale, 10000)
    plt.plot(t1, first, 'b', zorder=1)
    plt.plot(t2, second, 'b', zorder=1)

    #plt.legend(loc='upper left')
    plt.xlim(-0.1, 1.1)
    x_values = np.array(np.arange(0,1+half_val, half_val))
    y_values = np.array(np.arange(-amplitude, amplitude+1, 1))
    plt.xticks(x_values)
    plt.yticks(y_values)

    # show the plot
    if full:
        plt.axvline(start, 0.5, 0.5 + 0.45*amp, ls='--', lw=0.9, c='b', zorder=1)
        plt.axvline(half, 0.5-0.45*amp,  0.5 + 0.45*amp, ls='--', lw=0.9, c='b', zorder=1)
        plt.axvline(end, 0.5-0.45*amp, 0.5, ls='--', lw=0.9, c='b', zorder=1)
        plt.scatter([half], -scale, c='b')
        plt.scatter([start], scale, c='b')
        plt.scatter([end], 0, c='b')
        plt.scatter([half], scale, facecolors='#ffffff', edgecolors='b', zorder=2)
        plt.scatter([start], 0, facecolors='#ffffff', edgecolors='b', zorder=2)
        plt.scatter([end], -scale, facecolors='#ffffff', edgecolors='b', zorder=2)
        t3 = np.arange(-0.1, start,(start+0.1) / 10000)
        t4 = np.arange(end, 1.1, (1.1-end) / 10000)
        third = np.linspace(0, 0, 10000)
        fourth = np.linspace(0, 0, 10000)
        plt.plot(t3, third, 'b', zorder=1)
        plt.plot(t4, fourth, 'b', zorder=1)
    else:
        plt.axvline(start,0.5, 0.5 + 0.45*amp, ls='--', lw=0.9, c='b')
        plt.axvline(half,0.5-0.45*amp,  0.5 + 0.45*amp, ls='--', lw=0.9, c='b')
        plt.axvline(end, 0.5-0.45*amp, 0.5, ls='--', lw=0.9, c='b')


def HaarScalling(start, end, scale,amp =1, full =False):
    amplitude = np.abs(amp)
    amp=scale/np.abs(amp)
    t1 = np.arange(start, end, (end-start)/10000)
    first = np.linspace(scale, scale, 10000)
    plt.plot(t1, first, 'b', zorder=1)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-scale-0.1*scale, scale+0.1*scale)
    x_values = np.array(np.arange(0,1+(end-start), (end-start)))
    y_values = np.array(np.arange(-amplitude, amplitude+1, 1))
    plt.xticks(x_values)
    plt.yticks(y_values)

    # show the plot
    if full:
        plt.axvline(start, 0.5, 0.5 +0.45*amp, ls='--', lw=0.9, c='b', zorder=1)
        plt.axvline(end,  0.5, 0.5 +0.45*amp, ls='--', lw=0.9, c='b', zorder=1)
        plt.scatter(start, scale, c='b', zorder=1)
        plt.scatter(end, 0, c='b', zorder=1)
        plt.scatter(end, scale, facecolors='#ffffff', edgecolors='b', zorder=2)
        plt.scatter(start, 0, facecolors='#ffffff', edgecolors='b', zorder=2)
        t3 = np.arange(-0.1, start,(start+0.1) / 10000)
        t4 = np.arange(end, 1.1, (1.1-end) / 10000)
        third = np.linspace(0, 0, 10000)
        fourth = np.linspace(0, 0, 10000)
        plt.plot(t3, third, 'b', zorder=1)
        plt.plot(t4, fourth, 'b', zorder=1)
    else:
        plt.axvline(start, 0.5, 0.5 +0.5*amp, ls='--', lw=0.9, c='b')
        plt.axvline(end, 0.5, 0.5 + 0.5*amp, ls='--', lw=0.9, c='b')


amp=4
#Haar(0,1,1, True)
# Haar(0.25,0.5,2,amp, False)
# Haar(0.5,0.75,1,amp, False)

HaarScalling(0.5,0.75,-1,amp, False)
HaarScalling(0.25,0.5,2,amp, False)

HaarScalling(0.0,0.25,-3,amp, False)
HaarScalling(0.75,1,1,amp, False)
plt.show()