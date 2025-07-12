from oct2py import Oct2Py
import numpy as np
from scipy.io import savemat, loadmat
import os
import time
import argparse
import random
import pandas as pd
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import pickle
import keyboard
import board
import busio
import gpiozero
from scipy.fft import fft
from scipy.stats import kurtosis, skew
from scipy.signal import welch

def caracteristicas(datos):
	# Parámetros
	n = len(datos)
	fs = 500  # Frecuencia de muestreo (en Hz)

	# Transformada rápida de Fourier (FFT)
	F = fft(datos)

	# ENTROPÍA (en el dominio del tiempo)
	counts, _ = np.histogram(datos, bins=20)
	P = counts / (np.sum(counts) + np.finfo(float).eps)  # Normalizar con EPS para evitar divisiones por cero
	ent = -np.sum(P * np.log2(P + np.finfo(float).eps))  # Entropía de Shannon

	# FRECUENCIA MEDIA
	P2 = np.abs(F) / n
	P1 = P2[:n//2 + 1]
	P1[1:-1] = 2 * P1[1:-1]
	fmed = np.mean(P1)

	# FRECUENCIA PREDOMINANTE
	fpred = 1  # Este valor debe ser calculado correctamente según el espectro
	f, psd_welch = welch(datos, fs, nperseg=n)
	idx_max = np.argmax(psd_welch)
	fpred = f[idx_max]  # Frecuencia correspondiente al máximo de la densidad espectral

	# AGUDEZA (Kurtosis)
	ag = kurtosis(datos)

	# ASIMETRÍA (Skewness)
	asim = skew(datos)

	# RMS (Root Mean Square)
	rms = np.sqrt(np.mean(datos**2))

	# POTENCIA MEDIA
	pwmed = np.mean(psd_welch)

	# VALOR ABSOLUTO MEDIO
	mav = np.mean(np.abs(datos))

	# Retornar todas las características calculadas
	salida = [ent, fmed, fpred, ag, asim, rms, pwmed, mav]
	return salida


fs=500
tpred= 5
fp= 0.1*fs
N = tpred*fs  # Ventana de análisis
x_df = ['ent', 'fmed', 'fpred', 'ag', 'asim', 'rms', 'pwmed', 'gen', 'edad']
señal =np.array([])
temblor =np.array([])
feature_weights = np.array([1.0, 1.0, 1.0, 0.6, 0.6, 0.8, 0.8, 0.3, 0.7])

def apply_feature_weights(X):
    return X * feature_weights


# Cargar modelo
with open("datos/modelos/svm_temblor_3s_com_extensor.pkl", "rb") as f:
    model = pickle.load(f)


def guardar_valor(valor, nombre, gen, edad, ruta = "datos/data/"):
    
    FICHERO = ruta + nombre + ".mat"
    
    if os.path.exists(FICHERO):
        datos = loadmat(FICHERO)
        señal = datos.get('data', np.empty((0, 1)))
        temblor = datos.get('temblor', np.empty((0, 1)))
    else:
        señal = np.empty((0, 1))
        temblor = np.empty((0, 1))
	
    # Asegurar que señal y temblor sean vectores columna
    señal = np.vstack((señal, [[valor]]))

    print(f"[Guardar] Valor guardado: {valor:.4f}")

    if boton.is_pressed:
        led_verde.on()
        temblor = np.vstack((temblor, [[1]]))
    else:
        led_verde.off()
        temblor = np.vstack((temblor, [[0]]))
	
    savemat(FICHERO, {'senal': señal, 'temblor':temblor , 'genero': gen, 'edad':edad, 'unidades':"mV"})
    print(f"[Guardar] Valor guardado: {valor*1000:.4f}")

def analizar(eng,led, valor, gen, edad):

    global x_df, señal, model
    
    señal = np.append(señal, valor)

    if len(señal) < N:
        print(f"[Analizar] No hay suficientes datos (actual: {len(señal)}/{N})")
        return

    ventana = señal[-N:]
    features = caracteristicas(ventana,gen,edad)
    X= pd.DataFrame([features], columns=x_df)
    pred = model.predict(X)
    print(int(pred[0]))    

    
    if int(pred[0])==1:
        led.on()
    else:
        led.off()

def main():
    
    global led
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--modo', choices=['guardar', 'analizar'], required=True)
    parser.add_argument('--genero', choices=['F', 'M'], required=True)
    parser.add_argument('--edad', required=True)
    parser.add_argument('--nombre', required=True)
    args = parser.parse_args()
    
    dict_gen = {"M":0,"F":1}
    gen =dict_gen[args.genero]
    edad = int(args.edad)
    
    i2c=busio.I2C(3,2)

    ads= ADS.ADS1115(i2c)

    chan= AnalogIn(ads,ADS.P0)
    chan.gain =  16
    
    if args.modo == "analizar":
        
        eng = None
        eng= Oct2Py()
        eng.eval("pkg load signal")
            
        led.on()
        time.sleep(1)
    
    cont=0
    while not keyboard.is_pressed('o'):  # Simular iteraciones
    
        valor =  chan.voltage
        if args.modo == 'guardar':
            
            guardar_valor(valor, args.nombre,gen, edad, ruta)
            
        elif args.modo == 'analizar':

            analizar(eng, led,valor, gen, edad)

        time.sleep(0.01)
        cont+=1

if __name__ == "__main__":
    
    try: main()
    finally: led.close()
        
        
