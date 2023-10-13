# TugasPakDhany
# Berikut merupakan Dokuemntasi untuk Kode yang digunakan, apabila tidak mengakses dalam file python 1C (Untuk 1A dan 1B), 2B (Untuk 2A) dan 2C terdapat dokumeentasi

# 1A, 1B 1C

import math

# Mendefinisikan fungsi yang dijalankan, karena nilai A merupakan varaible random, yang membedakan 1A, 1B dan 1C terletak disini
	def f(t, A):
	    if -3*A  <= t <= 3*A:
	        return 1
	    else:
	        return 0

# Mendefinisikan fourier transform 
	def fourier_transform(f, A, w):
	    integral_real = 0
	    integral_imag = 0
	    delta_t = 0.001  # Step size for numerical integration
	    t_values = [i * delta_t for i in range(int(-A/delta_t), int(A/delta_t)+1)]

    for t in t_values:
        integral_real += f(t, A) * math.cos(-w * t)
        integral_imag += f(t, A) * math.sin(-w * t)

    return integral_real, integral_imag

# Nilai A disini arbitrari/terserah, Bisa diatur dengan kemauan, yang penting rentang yang dijelaskan di bagian awal, W nilai frequency, juga terserah dan digunakan untuk jumlah iterasi 
	A = 1
	w_values = [i * 0.01 for i in range(-1000, 1001)]  # Frequency values

# Digunakan untuk ngebuat definisi baru dimana operasi fourier transform dijalankan untuk setiap nilai frequency
	ft_values = [fourier_transform(f, A, w) for w in w_values]

# Digunakan untuk memploting hasil dan memastikan bahwa hasil benar
	import matplotlib.pyplot as plt


# Buat ngeekstract bagian integral real karena gunain fungsi sin/cos bukan exponensial
	real_part = [ft[0] for ft in ft_values]

# Plot for the original function
	plt.subplot(4, 1, 1)
	t_values = [i * 0.01 for i in range(int(-A/0.01), int(A/0.01)+1)]
	f_values = [f(t, A) for t in t_values]
	plt.plot(t_values, f_values)
	plt.title('Original Function f(t)')
	plt.grid()

# Plot for Transformed Fourier Function
	plt.figure(figsize=(10, 6))
	plt.subplot(2, 1, 1)
	plt.plot(w_values, real_part)
	plt.title('Real Part of Fourier Transform')
	plt.grid()
	
	plt.tight_layout()
	plt.show()

# Dokumentasi 2A

# Math digunakan untuk operasi (menggunakan sin dan cos) dan matplot lib untuk mengplot hasil akhir
	import math
	import matplotlib.pyplot as plt

#Sample data yang akan digunakan, bisa disesuaikan dengan , maksimal 10 data ntah kenapa, mungkin batasan iNterger?
	x = [0,6,10,12,5,2,1,5,2]

#Mendefinisikan bagaimana fast fourier transform akan berjalan, Karena tidak menggunakan numpy, Harus dibuat dua kali perkalian untuk ganjil genap. Langkah selanjutnya mungkin cari algoritma yang bisa ngekomputasi ganjil genap terpisah biar lebih cepat
	def fft(x):
	    N = len(x)

    if N <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    
	#This is really id**t**, Need to make something more compact- Akbar 
	    T = [complex(math.cos(2 * math.pi * k / N), -math.sin(2 * math.pi * k / N)) * odd[k] for k in range(N // 2)]
	    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

    #Terpaksa untuk emnggunakan fungsi xk.conjugate karena fungsi algoritma yang pakai sin cos yang menghasilkan nilai real dan imajiner mungkin jika bisa buat pakai fungsi exp bisa di inverse secaara otomatis karena nilainya berosi?
	def ifft(x):
	    N = len(x)
	    return [xk / N for xk in fft([xk.conjugate() for xk in x])]

# Compute FFT
	result = fft(x)
	print("FFT:", result)

# Inverse FFT
	reconstructed = ifft(result)
	print("Inverse FFT:", reconstructed)

# Plot hasil-hasilnya
	plt.figure(figsize=(12, 10))

	plt.subplot(121)
	plt.title("Original Signal")
	plt.plot(x, marker='o')
	
	plt.subplot(122)
	plt.title("FFT")
	plt.plot([abs(xk) for xk in result], marker='o')

	
	plt.subplot(123)
	plt.title("Inverse FFT")
	plt.plot(reconstructed, marker='o')
	
	plt.tight_layout()
	plt.show()

# Dokumentasi 2B

# Import math digunakan untuk operasi matematika, matplot untuk plotting
import math


        # BACKBURNER, COBA LAGI NANTI, EXPERIMENTAL UNTUK BISA NGATUR JUMLAH ROWS AND COLUMN MATRIX, SAAT INI CUMA 2x2
        #values = [1, 2, 3, 4,]
        #def generate_matrix(rows, cols, values):
        #matrix = [values[i:i+cols] for i in range(0, len(values), cols)]
        # return matrix
        #matrix = generate_matrix(2,2,len(values))

# FFT1D untuk operasi yang digunakan untuk setiap nilai, 
	def fft1d(x):
	    N = len(x)
	    if N <= 1:
	        return x
	    even = fft1d(x[0::2])
	    odd = fft1d(x[1::2])
	    T = [complex(math.cos(2 * math.pi * k / N), -math.sin(2 * math.pi * k / N)) * odd[k] for k in range(N // 2)]
	    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

# This is not in anyway good, Ini secara manual ngulangin operasi 1D FFT untuk setiap row dan column - Akbar
	def fft2d(x):
	    M, N = len(x), len(x[0])
	    y = [[0j for _ in range(N)] for _ in range(M)]
	    for i in range(M):
	        y[i] = fft1d(x[i])
	    for j in range(N):
	        col = [y[i][j] for i in range(M)]
	        col = fft1d(col)
	        for i in range(M):
	            y[i][j] = col[i]
	    return y
	
	def ifft1d(x):
	    N = len(x)
	    return [xk / N for xk in fft1d([xk.conjugate() for xk in x])]
	
	def ifft2d(x):
	    M, N = len(x), len(x[0])

    for i in range(M):
        x[i] = ifft1d(x[i])

    for j in range(N):
        column = [x[i][j] for i in range(M)]
        column = ifft1d(column)
        for i in range(M):
            x[i][j] = column[i]

    return x

# matrix yang akan digunakan , nilainya bisa disesuaikan tapi hanya bisa 2x2 matriks?
	x = [[0, 1,],
	     [1, 1,],]


	X = fft2d(x)

#N geploting matrix yang akan digunakan, menggunakan matplotlib Bisanya

import matplotlib.pyplot as plt
# Display the FFT

	plt.imshow([[abs(element) for element in row] for row in X], cmap='gray')
	plt.colorbar()
	plt.title('2D FFT Magnitude')
	plt.show()
	
	for row in X:
	    print([abs(element) for element in row])
	    
	#Karena nilai disini dalam bentuk grid, satu-satunya cara agar bisa jadi 1 graph ya digabungin wkwkwk
	x_flat = [element for row in x for element in row]
	
	#
	X_flat = [abs(element) for row in X for element in row]
	
	fig, axs = plt.subplots(1, 2, figsize=(10, 4))
	
	axs[0].plot(x_flat)
	axs[0].set_title('Original Signal')
	
	axs[1].plot(X_flat)
	axs[1].set_title('FFT')
	
	plt.show()

# Dokumentasi 2 C 


import math

# 2 Hal ini biar bisa ngubah dari frekuensi ke power spectrum dan sebaliknya
	def hz_to_mel(hz):
    return 2595 * math.log10(1 + hz / 700)

	def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

# Mendefinisikan Filter yang akan digunakan untuk mendapatkan MFCC, bisa diubah untuk lebih akurat, 
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

# Pendifinisikan dari Langkah-langkah mendapatkan MFCC, Untuk jelas dan mempermodah ngeplot,  bagi jadi 2, MFCC didefinisikan 2 kali, satu untuk di print, satu untuk diplot.. 
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

   #Hasil ploting 
    plt.subplot(4, 1, 4)
    mfccs = mfcc(x, sample_rate, num_filters, num_mfcc)
    plt.plot(mfccs)
    plt.title('MFCC Coefficients')
    plt.grid()

    plt.tight_layout()
    plt.show()

# Input data untuk Plot Point, penting untuk sama dengan input data pertama (The real solusi manual)
	x = [0, 6, 10, 12, 5, 2, 1, 5, 2, 6, 2, 1, 3, 4]
