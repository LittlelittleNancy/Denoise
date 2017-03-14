# http://stackoverflow.com/questions/3694918/how-to-extract-frequency-associated-with-fft-values-in-python
import pylab as plt
import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq

time   = np.linspace(0,10,2000)
signal = np.cos(5*np.pi*time) + np.cos(7*np.pi*time)

W = fftfreq(signal.size, d=time[1]-time[0])
f_signal = rfft(signal)

# If our original signal time was in seconds, this is now in Hz    
cut_f_signal = f_signal.copy()
cut_f_signal[(W<6)] = 0

cut_signal = irfft(cut_f_signal)


plt.subplot(221)
plt.xlabel('time[%]')
plt.ylabel('sample value [-]')
plt.plot(time,signal)

plt.subplot(222)
plt.xlabel('?')
plt.ylabel('amplitude')
plt.plot(W, f_signal)
plt.xlim(-10, 10)

plt.subplot(223)
plt.xlabel('?')
plt.ylabel('amplitude')
plt.plot(W, cut_f_signal)
plt.xlim(-10, 10)

plt.subplot(224)
plt.xlabel('time[%]')
plt.ylabel('sample value [-]')
plt.plot(time,cut_signal)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.savefig('data/output/wav_bandpass_demo.png')