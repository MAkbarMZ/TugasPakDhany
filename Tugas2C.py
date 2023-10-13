# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:31:33 2023

@author: ASUS
"""

import math


def hz_to_mel(hz):
    return 2595 * math.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

#Mendefinisikan Filter yang akan digunakan untuk mendapatkan MFCC, bisa diubah untuk lebih akurat, 
def get_filter_banks(num_filters, num_points, sample_rate):
    min_hz = 0
    max_hz = sample_rate / 2
    min_mel = hz_to_mel(min_hz)
    max_mel = hz_to_mel(max_hz)
    
    mel_points = [min_mel + i * (max_mel - min_mel) / (num_filters + 1) for i in range(num_filters + 2)]
    hz_points = [mel_to_hz(mel) for mel in mel_points]
    bin_points = [int((num_points + 1) * hz / sample_rate) for hz in hz_points]
    
    filter_banks = []
    for i in range(num_filters):
        filter_bank = []
        for j in range(num_points):
            if j < bin_points[i]:
                value = 0
            elif bin_points[i] <= j and j < bin_points[i + 1]:
                value = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
            elif bin_points[i + 1] <= j and j < bin_points[i + 2]:
                value = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1])
            else:
                value = 0
            filter_bank.append(value)
        filter_banks.append(filter_bank)
    return filter_banks

#Pendifinisikan dari Langkah-langkah mendapatkan MFCC, Untuk jelas dan mempermodah ngeplot,  bagi jadi 2, MFCC didefinisikan 2 kali, satu untuk di print, satu untuk diplot.. 
def mfcc(x, sample_rate=16000, num_filters=26, num_mfcc=13):
    # Step 1: Sebelum dijalankan ditambahin pre-emphasis untuk meningkatkan yang berfrekuensi/bernilai tinggi
    pre_emphasis = 0.97
    x = [x[0]] + [x[i] - pre_emphasis * x[i-1] for i in range(1, len(x))]
    
    # Step 2: Hamming Window untuk mengurangi nilai anomali
    windowed_x = [0.54 - 0.46 * math.cos(2 * math.pi * i / (len(x) - 1)) * x[i] for i in range(len(x))]
    
    # Step 3: Mendapatkan nilai untuk power spectrum
    power_spectrum = [abs(x) ** 2 for x in windowed_x]
    
    # Step 4: Mengimposi Filter yang didefinisikan sebelumnya ke power spectrum
    num_points = len(power_spectrum)
    filter_banks = get_filter_banks(num_filters, num_points, sample_rate)
    mel_energy = [sum(power_spectrum[i] * filter_banks[j][i] for i in range(num_points)) for j in range(num_filters)]
    
    # Step 5: Dalam hal ini diambil dalam bentuk log untuk mempermudah pengplotan dan hasil, ditambahin nilai epsilon ini untuk memastikan tidak ada yang bernilai 10 tapi menghasilkan nilai yang besar 
    epsilon = 1e-10
    log_mel_energy = [math.log(m + epsilon) for m in mel_energy]
    
    # Step 6: Apply Discrete Cosine Transform (DCT)
    mfccs = []
    for i in range(num_mfcc):
        mfcc_i = 0
        for j in range(num_filters):
            mfcc_i += log_mel_energy[j] * math.cos(math.pi * i / num_filters * (j + 0.5))
        mfccs.append(mfcc_i)
    
    return mfccs

# Input data yang akan dijalankan melalui MFCC filter
x = [0, 6, 10, 12, 5, 2, 1, 5, 2,6,2,1,3,4]

# Print MFCC
mfcc_result = mfcc(x)
print(mfcc_result)
print("Nama: M. Akbar MIftahuzaman")
print("NRP:5009211004")

import matplotlib.pyplot as plt 

def plot_mfcc(x, sample_rate=16000, num_filters=26, num_mfcc=13):
   
 
    pre_emphasis = 0.97
    x = [x[0]] + [x[i] - pre_emphasis * x[i-1] for i in range(1, len(x))]
    
    
    windowed_x = [0.54 - 0.46 * math.cos(2 * math.pi * i / (len(x) - 1)) * x[i] for i in range(len(x))]
    
    
    power_spectrum = [abs(x) ** 2 for x in windowed_x]
    
   
    num_points = len(power_spectrum)
    filter_banks = get_filter_banks(num_filters, num_points, sample_rate)
    mel_energy = [sum(power_spectrum[i] * filter_banks[j][i] for i in range(num_points)) for j in range(num_filters)]

   
    epsilon = 1e-10
    log_mel_energy = [math.log(m + epsilon) for m in mel_energy]

    # Signal asli di ploting
    plt.figure(figsize=(10, 6))
    plt.subplot(4, 1, 1)
    t_values = [i for i in range(len(x))]
    plt.plot(t_values, x)
    plt.title('Original Signal x(t)')
    plt.grid()

    # Plot power spectrum
    plt.subplot(4, 1, 2)
    plt.plot(power_spectrum)
    plt.title('Power Spectrum')
    plt.grid()

    # Yang ini sebenarnya tidak usah diplot tidak apa-apa karena bersifat sama untuk setiap ploting, dibuat untuk kepastian saja
    plt.subplot(4, 1, 3)
    for i, energy in enumerate(log_mel_energy):
        plt.plot(energy, label=f'Filter {i+1}')
    plt.title('Mel Filter Bank Energies')
    plt.legend()
    plt.grid()

    #Hasil 
    plt.subplot(4, 1, 4)
    mfccs = mfcc(x, sample_rate, num_filters, num_mfcc)
    plt.plot(mfccs)
    plt.title('MFCC Coefficients')
    plt.grid()

    plt.tight_layout()
    plt.show()

# Input data untuk Plot Point, penting untuk sama dengan input data pertama (The real solusi manual)
x = [0, 6, 10, 12, 5, 2, 1, 5, 2, 6, 2, 1, 3, 4]

