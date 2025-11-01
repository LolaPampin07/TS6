import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Hacemos un filtro pasa bajos

# Plantilla de diseño
wp = 1 # Frecuencia de corte/paso (rad/seg)
ws = 5 # Frecuencia de stop/detenida (rad/seg)

alpha_p = 1 # Atenuacion maxima a la wp, alfa_max, perdidas en banda de paso
alpha_s = 40 # Atenuacion minima a la ws, alfa_min, minima atenuacion requerida en banda de paso

# Si quiero exigirle mas a la plantilla, disminuyo ws y aumento alpha_s
# Eso va a hacer que aumente el grado de a, y que la matriz tenga mas filas y columnas

# Aproximaciones de modulo
f_aprox_butter = 'butter'
# f_aprox_cheby1 = 'cheby1'
# f_aprox_cheby2 = 'cheby2'
# f_aprox_cauer = 'cauer' # eliptica

# Aproximaciones de fase 
# f_aprox = 'bessel'

# Diseño del filtro analogico
b, a = signal.iirdesign(wp = wp, ws = ws, gpass = alpha_p, gstop = alpha_s, analog = True, ftype = f_aprox_butter, output = 'ba')

# Respuesta en frecuencia 
# w, h = signal.freqs(b, a, worN = np.logspace(1, 6, 1000)) # logspace me contruye un espacio logaritmicamente equiespaciado, de 10^1 a 10^6, con 1000 puntos en el medio 
w, h = signal.freqs(b, a) # Calcula la respuesta en frecuencia del filtro

# Cálculo de fase y retardo de grupo 
phase = np.unwrap(np.angle(h))

# Retardo de grupo = -dφ/dω
gd = -np.diff(phase) / np.diff(w)

# Polos y ceros
z, p, k = signal.tf2zpk(b, a) # Nos devuelve la localizacion de los ceros 

# Gráficos
# plt.figure(figsize = (12,10))

# Magnitud
plt.subplot(2,2,1)
plt.semilogx(w, 20*np.log10(abs(h)))
plt.title('Respuesta en Magnitud')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which = 'both', ls = ':')

# Fase
plt.subplot(2,2,2)
plt.semilogx(w, np.degrees(phase))
plt.title('Fase')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which = 'both', ls = ':')

# Retardo de grupo
plt.subplot(2,2,3)
plt.semilogx(w[:-1], gd)
plt.title('Retardo de Grupo')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('τg [s]')
plt.grid(True, which = 'both', ls = ':')

# Diagrama de polos y ceros
plt.subplot(2,2,4)
plt.plot(np.real(p), np.imag(p), 'x', markersize = 10, label = 'Polos')
if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize = 10, fillstyle = 'none', label = 'Ceros')
plt.axhline(0, color = 'k', lw = 0.5)
plt.axvline(0, color='k', lw = 0.5)
plt.title('Diagrama de Polos y Ceros (plano s)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %% Matriz SOS
sos = signal.tf2sos(b, a, analog = True) # Ya me lo da en forma monica
# La raiz del ultimo valor del array me tiene que dar el radio 

print("\nMatriz SOS\n", sos); # Compruebo que es una seccion bicuadratica 

# %% PUNTO 3
# Defino las tres funciones de trasferencia que me da la consigna
# T1(s) = (s^2 + 9) / (s^2 + 2s + 1)
num1 = [1, 0, 9]
den1 = [1, np.sqrt(2), 1]
T1 = signal.TransferFunction(num1, den1)

# T2(s) = (s^2 + 1/9) / (s^2 + (1/5)s + 1)
num2 = [1, 0, 1/9]
den2 = [1, 1/5, 1]
T2 = signal.TransferFunction(num2, den2)

# T3(s) = (s^2 + s + 1) / (s^2 + 3s + 1)
num3 = [1, 1/5, 1]
den3 = [1, np.sqrt(2), 1]
T3 = signal.TransferFunction(num3, den3)

w1, h1 = signal.freqs(num1, den1) # Calcula la respuesta en frecuencia del filtro
w2, h2 = signal.freqs(num2, den2) # Calcula la respuesta en frecuencia del filtro
w3, h3 = signal.freqs(num3, den3) # Calcula la respuesta en frecuencia del filtro

# Cálculo de fase y retardo de grupo 
phase1 = np.unwrap(np.angle(h1))
phase2 = np.unwrap(np.angle(h2))
phase3 = np.unwrap(np.angle(h3))

# Retardo de grupo = -dφ/dω
gd1 = -np.diff(phase1) / np.diff(w1)
gd2 = -np.diff(phase2) / np.diff(w2)
gd3 = -np.diff(phase3) / np.diff(w3)

# Polos y ceros
z1, p1, k1 = signal.tf2zpk(num1, den1) # Nos devuelve la localizacion de los ceros 
z2, p2, k2 = signal.tf2zpk(num2, den2) # Nos devuelve la localizacion de los ceros 
z3, p3, k3 = signal.tf2zpk(num3, den3) # Nos devuelve la localizacion de los ceros 

# Graficos
plt.figure() 

# Magnitud
plt.subplot(2,2,1)
plt.semilogx(w1, 20*np.log10(abs(h1)), label = 'T1(s)')
plt.semilogx(w2, 20*np.log10(abs(h2)), label = 'T2(s)')
plt.semilogx(w3, 20*np.log10(abs(h3)), label = 'T3(s)')
plt.title('Respuesta en Magnitud')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which = 'both', ls = ':')
plt.legend()

# Fase
plt.subplot(2,2,2)
plt.semilogx(w1, np.degrees(phase1), label = 'T1(s)')
plt.semilogx(w2, np.degrees(phase2), label = 'T2(s)')
plt.semilogx(w3, np.degrees(phase3), label = 'T3(s)')
plt.title('Fase')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which = 'both', ls = ':')
plt.legend()

# Retardo de grupo
plt.subplot(2,2,3)
plt.semilogx(w1[:-1], gd1, label = 'T1(s)')
plt.semilogx(w2[:-1], gd2, label = 'T2(s)')
plt.semilogx(w3[:-1], gd3, label = 'T3(s)')
plt.title('Retardo de Grupo')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('τg [s]')
plt.grid(True, which = 'both', ls = ':')
plt.legend()

# Diagrama de polos y ceros
plt.subplot(2,2,4)
plt.plot(np.real(p1), np.imag(p1), 'x', markersize = 10, label = 'Polos T1(s)')
plt.plot(np.real(p2), np.imag(p2), 'x', markersize = 10, label = 'Polos T2(s)')
plt.plot(np.real(p3), np.imag(p3), 'x', markersize = 10, label = 'Polos T3(s)')
if len(z1) > 0:
    plt.plot(np.real(z1), np.imag(z1), 'o', markersize = 10, fillstyle = 'none', label = 'Ceros T1(s)')
if len(z2) > 0:
    plt.plot(np.real(z2), np.imag(z2), 'o', markersize = 10, fillstyle = 'none', label = 'Ceros T2(s)')
if len(z3) > 0:
    plt.plot(np.real(z3), np.imag(z3), 'o', markersize = 10, fillstyle = 'none', label = 'Ceros T3(s)')
plt.axhline(0, color = 'k', lw = 0.5)
plt.axvline(0, color='k', lw = 0.5)
plt.title('Diagrama de Polos y Ceros (plano s)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


