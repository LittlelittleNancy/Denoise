# -*- coding: utf-8 -*-
import subprocess
import wave
import struct
import numpy
import csv
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

class Audio:
    def __init__(self, audio):
        self.audio = wave.open(audio,'r')
        self.signal = self.audio.readframes(-1) # bytes
        self.signal = np.fromstring(self.signal, 'Int16') # (184434,)
        self.fr = self.audio.getframerate() #16000kHz
        self.time = np.linspace(0, 100, num=(len(self.signal))) # map to 0~100
        self.fft = np.fft.fft(self.signal) # (184434,)
        print('audio:\t%s\nparams:\t%s' % (audio, self.audio.getparams()))

    def plot(self):
        plt.figure(1)
        plt.title("Audio waveforms")
        plt.plot(self.time, self.signal, '.')
        
        plt.figure(2)
        plt.title("Audio fft")
        plt.plot(self.fft, '.')

gana = Audio('data/input/local.wav')
gana.plot()

# humm.plot()
# humm = Audio('humm.wav')

plt.show()
