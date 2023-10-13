# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 23:36:42 2023

@author: ASUS
"""

import math

# Define the function f(t)
def f(t, A):
    if -3*A  <= t <= 3*A:
        return 1
    else:
        return 0

# Define the Fourier Transform function
def fourier_transform(f, A, w):
    integral_real = 0
    integral_imag = 0
    delta_t = 0.001  # Step size for numerical integration
    t_values = [i * delta_t for i in range(int(-A/delta_t), int(A/delta_t)+1)]

    for t in t_values:
        integral_real += f(t, A) * math.cos(-w * t)
        integral_imag += f(t, A) * math.sin(-w * t)

    return integral_real, integral_imag

#Nilai A disini arbitrari/terserah, Bisa diatur dengan kemauan, yang penting rentang, W nilai frequency, juga terserah dan digunakan untuk jumlah iterasi 
A = 1
w_values = [i * 0.01 for i in range(-1000, 1001)]  # Frequency values

#Digunakan untuk ngebuat definisi baru dimana operasi fourier transform dijalankan untuk setiap nilai frequency
ft_values = [fourier_transform(f, A, w) for w in w_values]

import matplotlib.pyplot as plt


#Buat ngeekstract bagian integral real karena gunain fungsi sin/cos bukan exponensial
real_part = [ft[0] for ft in ft_values]

#Plot for the original function
plt.subplot(4, 1, 1)
t_values = [i * 0.01 for i in range(int(-A/0.01), int(A/0.01)+1)]
f_values = [f(t, A) for t in t_values]
plt.plot(t_values, f_values)
plt.title('Original Function f(t)')
plt.grid()

#Plot for Transformed Fourier Function
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(w_values, real_part)
plt.title('Real Part of Fourier Transform')
plt.grid()

plt.tight_layout()
plt.show()