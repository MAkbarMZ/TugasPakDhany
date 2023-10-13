import math


        # BACKBURNER, COBA LAGI NANTI, EXPERIMENTAL UNTUK BISA NGATUR JUMLAH ROWS AND COLUMN MATRIX, SAAT INI CUMA 2x2
        #values = [1, 2, 3, 4,]
        #def generate_matrix(rows, cols, values):
        #matrix = [values[i:i+cols] for i in range(0, len(values), cols)]
        # return matrix
        #matrix = generate_matrix(2,2,len(values))
        
def fft1d(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft1d(x[0::2])
    odd = fft1d(x[1::2])
    T = [complex(math.cos(2 * math.pi * k / N), -math.sin(2 * math.pi * k / N)) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

#This is not in anyway good, Ini secara manual ngulangin operasi 1D FFT untuk setiap row dan column - Akbar
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

#matrix yang akan digunakan 
x = [[0, 1,],
     [1, 1,],]


X = fft2d(x)

#Ngeploting matrix yang akan digunakan, menggunakan matplotlib Bisanya

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
    


