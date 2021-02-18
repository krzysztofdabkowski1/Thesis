import pywt
import numpy as np


def decomposition(data, waveletname, level=None):
	epochs=len(data)
	chanels = len(data[0])
	samples=len(data[0][0])
	level=pywt.dwt_max_level(samples,waveletname)
	ret=[]
	for dec_lvl in range(0,level+1):
		size=len(pywt.wavedec(data[0][0],waveletname, mode = 0, level =level)[dec_lvl])
		one_level_dec = np.zeros( (epochs, chanels,size) )
		for epoch in range(0,epochs):
			for chanel in range(0,chanels):
				one_level_dec[epoch][chanel]=pywt.wavedec(data[epoch][chanel],waveletname,mode = 0, level=level)[dec_lvl]
				#print(type(pywt.wavedec(data[epoch,chanel],waveletname,level)))
		ret.append([one_level_dec])
	return ret[::-1]

def get_dec_info(data, waveletnames,max_freq_in_signal=100,level=None):
	samples=len(data)
	for name in waveletnames:
		max_level=pywt.dwt_max_level(samples,name)
		print(name+':')
		if level==None or level>max_level:
			level=max_level
		freq=max_freq_in_signal
		for i in range(0,level):
			freq_half=freq/2
			print("Poziom {} - detale       - {}Hz - {}Hz".format(i+1,freq,freq_half))
			freq=freq_half
		print("Poziom {} - aproksymacja - {}Hz - {}Hz".format(level,freq_half,0))
