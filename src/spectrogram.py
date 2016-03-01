import numpy as np

# Simple short-time Fourier transform
def fourier_spectrogram(signal, window, N_FFT, FR_LEN, SHIFT):
    n = signal.shape[0]
    signal = np.pad(signal, (0, FR_LEN - n % FR_LEN), mode='constant')
    windows = np.array([signal[i:i+FR_LEN] * window
                        for i in range(0, n - n % FR_LEN + 1, SHIFT)])
    complex_spectrogram = np.fft.rfft(windows, n=N_FFT, axis=1).T
    return complex_spectrogram

# Converts the STFT to a Mel-scaled version
# MIN_F and MAX_F are not 0 and fs/2 here. A bit more and a bit less than
# those...
def mel_spectrogram(signal, window, N_MEL, MIN_F, MAX_F, MAX_F_ACTUAL,
                    N_FFT, FR_LEN, SHIFT):
    def mel(f):
        return 1125.0 * np.log(1.0 + f/700.0)
    def melinv(m):
        return 700.0 * (np.exp(m/1125.0) - 1.0)

    fourier_sp = fourier_spectrogram(signal, window, N_FFT, FR_LEN, SHIFT)
    fourier_mag_sp = np.absolute(fourier_sp)

    mel_freqs = np.linspace(mel(MIN_F), mel(MAX_F), num=N_MEL)
    freqs = np.linspace(0, MAX_F, fourier_sp.shape[0])
    filterbank = np.zeros((N_MEL, fourier_sp.shape[0]))

    for i, mf in enumerate(mel_freqs):
        f0 = melinv(mf)
        if i == N_MEL - 1:
            fR = MAX_F_ACTUAL
        else:
            fR = melinv(mel_freqs[i+1])

        if i == 0:
            fL = 0
        else:
            fL = melinv(mel_freqs[i-1])

        height = 2.0 / (fR - fL)

        for j, f in enumerate(freqs):
            if f < fL:
                continue
            elif f >= fL and f < f0:
                filterbank[i,j] = height * (f - fL) / (f0 - fL)
            elif f >= f0 and f < fR:
                filterbank[i,j] = height * (fR - f) / (fR - f0)
            else:
                continue

    mel_sp = np.dot(filterbank, fourier_mag_sp)
    return mel_sp
