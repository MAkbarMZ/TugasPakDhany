# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 23:45:46 2023

@author: ASUS
"""

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

# Plot
plt.figure(figsize=(12, 10))

plt.subplot(121)
plt.title("Original Signal")
plt.plot(x, marker='o')

plt.subplot(122)
plt.title("FFT")
plt.plot([abs(xk) for xk in result], marker='o')

# Inverse FFT
plt.subplot(123)
plt.title("Inverse FFT")
plt.plot(reconstructed, marker='o')

plt.tight_layout()
plt.show()