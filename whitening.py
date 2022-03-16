# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:53:03 2022

@author: Abhinav Rijal
"""
import h5py
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import readligo as rl
import pywt
import random
from math import *
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import csv
#import ipdb
import numpy as np
import pandas as pd
import math
import cmath
from scipy import interpolate, fftpack, signal



def autocorr(amp):
    N=len(amp)
    M=4096 #order
    r=[]
    for k in range(0,M+1):
        s=0
        for n in range(k,N):
            s=s+(1/N)*amp[n]*amp[n-k]
        r.append(s)
    return r

def linpred(r):
    M=4096
    W=np.zeros([M,M])
    for i in range(0,M):
        for j in range(0,M):
            if i+j<M:
                W[i,j+i]=r[j]
    for i in range(0,M):
        for j in range(0,M):
            if j<i:
                W[i][j]=W[j][i]
    y = np.linalg.inv(W)
    rp=[]
    for i in range(1,M+1):
        rp.append(r[i])
    rp=np.array(rp)
    c=np.matmul(y,rp)
    return(c)

def err(c,x):
    #M=4096
    N=len(x)
    c=list(c)
    c.insert(0,0)
    x_predicted=[]
    for n in range(0,N):
        s=0
        for m in range(0,M+1):
            if n>=m:
                s=s+c[m]*x[n-m]
        x_predicted.append(s)
    e=np.array(x)-np.array(x_predicted)
    return e

    



#Waveform input
time4=[]
amplitude3=[]
file1=input('Enter the file-name string (or path string) with signal waveform time series data at 10 Kpc in .xlsx format.')
df5=pd.read_excel(file1)

print('The length of pure signal data is; ',  len(df5.Time))


time4=df5.Time
amplitude3=df5.Amplitude


plt.figure(figsize = [15,10])


plt.subplot(2,2,3)
plt.title('Signal time series')
plt.plot(time4,amplitude3)
plt.xlabel('Time(secs)')
plt.ylabel('GW wave (strain)')
plt.show()




Amplitude3=np.array(amplitude3)
Time3=np.array(time4)


#Frequency domain
ts3=Time3[1]-Time3[0]
Amp3=fftpack.fft(Amplitude3)
f3 = fftpack.fftfreq(len(Time3),ts3)
abs_Amp3=abs(Amp3)
f3 = fftpack.fftshift(f3)
abs_Amp3 = fftpack.fftshift(abs_Amp3)
n3 = len(f3)
f3 = f3[int(n3/2)+1:n3-1]
abs_Amp3 = ts3*2.0*abs_Amp3[int(n3/2)+1:n3-1]


plt.figure(figsize = [15,10])
plt.title('GW signals in frequency domain')
plt.loglog(f3,abs_Amp3,label='M13')
plt.xlabel('Frequency')
plt.ylabel('GW wave (strain)')
plt.legend()
plt.show()




#---------------------
# Read in noise strain data
#---------------------
fileName = input('Enter the file-name string (or path string) with noise time series data in .hdf5 format.')
strain, time, channel_dict = rl.loaddata(fileName, 'H1')
ts = time[1] - time[0] #-- Time between samples
fs = int(1.0 / ts)          #-- Sampling frequency

#---------------------------------------------------------
# Find a good segment, get first 16 seconds of data
#---------------------------------------------------------
segList = rl.dq_channel_to_seglist(channel_dict['DEFAULT'], fs)
length = 16  # seconds
strain_seg = strain[segList[0]][0:(length*fs)]
time_seg = time[segList[0]][0:(length*fs)]

#---------------------
# Plot the time series
#----------------------
fig = plt.figure(figsize=(15,15))
fig.subplots_adjust(wspace=0.3, hspace=0.3)
plt.subplot(321)
plt.plot(time_seg - time_seg[0], strain_seg)
plt.xlabel('Time since GPS ' + str(time_seg[0]))
plt.ylabel('Strain')

#------------------------------------------
# Apply a Blackman Window, and plot the FFT
#------------------------------------------
window = np.blackman(strain_seg.size)
windowed_strain = strain_seg*window
freq_domain = np.fft.rfft(windowed_strain) / fs
freq = np.fft.rfftfreq(len(windowed_strain))*fs

plt.subplot(322)
plt.loglog( freq, abs(freq_domain) )
plt.axis([10, fs/2.0, 1e-24, 1e-18])
plt.grid('on')
plt.xlabel('Freq (Hz)')
plt.ylabel('Strain / Hz')

#----------------------------------
# Make PSD for first chunk of data
#----------------------------------
plt.subplot(323)
Pxx, freqs = mlab.psd(strain_seg, Fs = fs, NFFT=fs)
plt.loglog(freqs, Pxx)
plt.axis([.010, 100, 1e-46, 1e-36])
plt.grid('on')
plt.ylabel('PSD')
plt.xlabel('Freq (Hz)')

#-------------------------
# Plot the ASD
#-------------------------------
plt.subplot(324)
plt.loglog(freqs, np.sqrt(Pxx))
plt.axis([.010, 100, 1e-24, 1e-18])
plt.grid('on')
plt.xlabel('Freq (Hz)')
plt.ylabel('ASD [Strain / Hz$^{1/2}$]')

#--------------------
# Make a spectrogram
#-------------------
NFFT = 1024
window = np.blackman(NFFT)
plt.subplot(325)
spec_power, freqs, bins, im = plt.specgram(strain_seg, NFFT=NFFT, Fs=fs, 
                                    window=window)
plt.xlabel('Time (s)')
plt.ylabel('Freq (Hz)')

#------------------------------------------
# Renormalize by average power in freq. bin
#-----------------------------------------
med_power = np.zeros(freqs.shape)
norm_spec_power = np.zeros(spec_power.shape)
index = 0
for row in spec_power:
    med_power[index] = np.median(row)
    norm_spec_power[index] = row / med_power[index]
    index += 1

ax = plt.subplot(326)
ax.pcolormesh(bins, freqs, np.log10(norm_spec_power))
plt.xlabel('Time (s)')
plt.ylabel('Freq (Hz)')

plt.show()


#Resampling via spline interpolation of signal to noise frequency
tck3 = interpolate.splrep(time4-time4[0],amplitude3) #spline interpolation
amplitude3 = interpolate.splev(time_seg-time_seg[0],tck3,der=0) #resampling 
amplitude3=np.array(amplitude3)
noisy_amp_M13=amplitude3+strain_seg

plt.figure(figsize = [15,10])
plt.title('M13')
plt.plot(time_seg-time_seg[0],amplitude3)
plt.xlabel('Time(secs)')
plt.ylabel('Resampled Noise')
plt.show()


def M13_plots(noisy_amp,dist,which,start,end):
    noisy_amp1=noisy_amp[:end]
    time_seg1=time_seg[start:end]-time_seg[0]
    fig = plt.figure(figsize=(15,15))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.subplot(321)
    plt.title('M13')
    plt.plot(time_seg1, noisy_amp1)
    plt.xlabel('Time since GPS ' + str(time_seg[0]))
    plt.ylabel('Strain')
#------------------------------------------
# Apply a Blackman Window, and plot the FFT
#------------------------------------------
    window = np.blackman(noisy_amp1.size)
    windowed_strain = noisy_amp1*window
    freq_domain = np.fft.rfft(windowed_strain) / fs
    freq = np.fft.rfftfreq(len(windowed_strain))*fs

    plt.subplot(322)
    plt.loglog( freq, abs(freq_domain) )
    plt.axis([10, fs/2.0, 1e-27, 1e-18])
    plt.grid('on')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Strain / Hz')

#----------------------------------
# Make PSD for first chunk of data
#----------------------------------
    plt.subplot(323)
    Pxx, freqs = mlab.psd(noisy_amp1, Fs = fs, NFFT=fs)
    plt.loglog(freqs, Pxx)
    plt.axis([0.01, 100, 1e-54, 1e-36])
    plt.grid('on')
    plt.ylabel('PSD')
    plt.xlabel('Freq (Hz)')

#-------------------------
# Plot the ASD
#-------------------------------
    plt.subplot(324)
    plt.loglog(freqs, np.sqrt(Pxx))
    plt.axis([0.01, 100, 1e-27, 1e-18])
    plt.grid('on')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('ASD [Strain / Hz$^{1/2}$]')

#--------------------
# Make a spectrogram
#-------------------
    NFFT = 1024
    start_sec=start/16384
    window = np.blackman(NFFT)
    plt.subplot(325)
    spec_power, freqs, bins, im = plt.specgram(noisy_amp1, NFFT=NFFT, Fs=fs, window=window)
    print(min(freqs),max(freqs),min(bins),max(bins))
    print(freqs[1]-freqs[0],bins[1]-bins[0])
    plt.ylim(0.01, 100)
    plt.xlabel('Time (s)'+'+'+str(start_sec)+'s')
    plt.ylabel('Freq (Hz)')

#------------------------------------------
# Renormalize by average power in freq. bin
#-----------------------------------------
    med_power = np.zeros(freqs.shape)
    norm_spec_power = np.zeros(spec_power.shape)
    index = 0
    for row in spec_power:
        med_power[index] = np.median(row)
        norm_spec_power[index] = row / med_power[index]
        index += 1

    ax = plt.subplot(326)
    ax.pcolormesh(bins, freqs, np.log10(norm_spec_power))
    plt.ylim(0.01, 100)
    plt.xlabel('Time (s)'+'+'+str(start_sec)+'s')
    plt.ylabel('Freq (Hz)')
    plt.savefig('Plots_'+str(dist)+which+'.png')
    plt.show()


    
def SNR13(abs_Amp):
    asd=np.array(asd3)
    psd = asd**2
    

    snrsq = integral(abs_Amp**2/psd,df3)
    snr = np.sqrt(snrsq)
    return snr

def fft_M13(amplitude):
    Amplitude=np.array(amplitude)
    Time=np.array(time4)

    ts=Time[1]-Time[0]
    Amp=fftpack.fft(Amplitude)
    f = fftpack.fftfreq(len(Time),ts)
    abs_Amp=abs(Amp)
    f = fftpack.fftshift(f)
    abs_Amp = fftpack.fftshift(abs_Amp)
    n = len(f)
    f = f[int(n/2)+1:n-1]
    abs_Amp = ts*2.0*abs_Amp[int(n/2)+1:n-1]
    return abs_Amp


#Bandpass Filter and whitening
numtaps = 16385
f1, f2 = 0.010, 100
fir_filter= signal.firwin(numtaps, [f1, f2], pass_zero=False, fs=16384.0)

plt.figure(figsize = [10,5])
plt.title('Band pass filter (0.010-100 Hz passband) impulse response')
plt.stem(fir_filter)

plt.xlabel('n')
plt.ylabel('h[n]')

plt.show()
plt.savefig("BPF_h[n]")




freq_domain = np.fft.rfft(fir_filter) 
freq = np.fft.rfftfreq(len(fir_filter))*fs



plt.figure(figsize = [10,5])
plt.loglog( freq, abs(freq_domain) )
plt.title('Band pass filter (0.010-100 Hz passband) transfer function')
plt.axis([1, fs/2.0, 1e-8, 2])
plt.grid('on')
plt.xlabel('Freq (Hz)')
#plt.ylabel('Strain / Hz')



plt.ylabel('H(f)')

plt.show()
plt.savefig("BPF_H(f)")

x= strain_seg[32768:131072]
lcf_x = signal.lfilter(fir_filter, 1, x)

#plt.plot(lcf_x )


freq_domain = np.fft.rfft(lcf_x ) /fs
freq = np.fft.rfftfreq(len(lcf_x ))*fs


M=4096
#x= strain_seg[32768:131072]
r=autocorr(lcf_x[32768:131072])
N=len(lcf_x[32768:131072])
c=linpred(r)


distance=float(input('Enter the distance in Kpc of the simulated signal source.'))

x=(10/distance)*amplitude3[32768:131072]+strain_seg[32768:131072]
M13_plots((10/distance)*amplitude3+strain_seg,str(distance)+'Kpc','_not_whitened',0,len(strain_seg))
e=err(c,x)
M13_plots(e[4097:],str(distance)+'Kpc','_bandpassed_noise_whitened',36865,131072)
lcf_x = signal.lfilter(fir_filter, 1, e)
M13_plots(lcf_x[16384:],str(distance)+'Kpc','_bandpassed_noise_whitened_bandpassed',49152,131072)
e=err(c,lcf_x)
M13_plots(e[16384:],str(distance)+'Kpc','_bandpassed_noise_whitened_bandpassed_whitened',49152,131072)
