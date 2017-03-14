# https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftfreq.html
# http://stackoverflow.com/questions/3694918/how-to-extract-frequency-associated-with-fft-values-in-python

import pylab as plt
import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq
import wave
import sys

filename = sys.argv[1] if len(sys.argv) == 2 else 'rich1'
inpath = 'data/input/%s.wav' % filename
outpath = 'data/output/%s_denoise.wav' % filename
figpath = 'data/output/%s_figure.png' % filename

desc = 'will denoise file %s -> %s, %s' % (inpath, outpath, figpath)
if len(sys.argv) != 2:
    print('usage: %s %s' % (sys.argv[0], filename) )
    print(desc)
    sys.exit(0)
print(desc)



wr = wave.open(inpath)
print(wr.getparams())
dtype = {16:np.int16, 8:np.int8, 32:np.int32}[wr.getsampwidth() * 8]
framerate = int(wr.getframerate())
nchannels = wr.getnchannels()
nframes = wr.getnframes()
sampWidth = wr.getsampwidth()
seconds = nframes / nchannels / framerate # in seconds
compressname = wr.getcompname() # not compress
compresstype = wr.getcomptype() # NONE

signal = np.fromstring(wr.readframes(-1), dtype=dtype) #mono
time   = np.linspace(0, 100, signal.size)


# freq_hz = abs(freq[idx] * framerate)

W = fftfreq(signal.size)#, d=1.0/wr.getframerate())
f_signal = rfft(signal)
f_signal_hz = np.array([abs(W[i] * framerate) for i in range(0, len(f_signal))])

# If our original signal time was in seconds, this is now in Hz    
# cut_f_signal = []
# for i in range(0, len(f_signal)):
#     freq_hz = abs(W[i] * framerate)
#     # man 123~493  woman 160~1200
#     if 120 < freq_hz < 1200:
#         cut_f_signal.append(f_signal[i])        
#     else:
#         cut_f_signal.append(0.0)
# cut_f_signal = np.array(cut_f_signal)

cut_f_signal = f_signal.copy()
cut_f_signal[((W * framerate) > 2000)] = 0 # filter > 1200Hz
cut_f_signal[((W * framerate) < 500)]  = 0
cutsignal_hz = np.array([abs(W[i] * framerate) for i in range(0, len(cut_f_signal))])



cut_signal = dtype(irfft(cut_f_signal))

########### TEST time domain filter
# cov_n = 200
# cut_signal = dtype(np.convolve(signal.copy(), [1/cov_n for i in range(cov_n)], 'same'))

# dB = 20 * log10(amplitude)
# amplitude = 14731 / 32767 = 0.44
# dB = 20 * log10(0.44) = -7.13

plt.figure(figsize=(20,15)) 

plt.subplot(221)
plt.plot(time,signal)
plt.xlabel('time[%]')
plt.ylabel('sample value [-]')
plt.title('original audio time domain')

plt.subplot(222)
plt.xlabel('freq [Hz]')
plt.ylabel('|amplitude|')
plt.title('original audio freq domain')
plt.plot(f_signal_hz, abs(f_signal)) # (W,abs(f_signal))
# plt.xlim(0,10)

plt.subplot(223)
plt.xlabel('freq [Hz]')
plt.ylabel('|amplitude|')
plt.title('denoised audio freq domain')
plt.plot(cutsignal_hz, abs(cut_f_signal)) # (W, cut_f_signal)
# plt.xlim(0,10)

plt.subplot(224)
plt.xlabel('time[%]')
plt.ylabel('sample value [-]')
plt.title('denoised audio time domain')
plt.plot(time,cut_signal)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.savefig(figpath)
# plt.show()

wav_file = wave.open(outpath, "w")
wav_file.setparams((1, sampWidth, framerate, nframes, compresstype, compressname))
wav_file.writeframes(cut_signal.tobytes('C'))
wav_file.close()

