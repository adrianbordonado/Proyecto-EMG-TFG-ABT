#ESTA VERSIÓN NO SE EMPLEA EN EL DISPOSITIVO FINAL, SÓLO PARA EL TESTEO DE SU FUNCIONAMIENTO
#LA VERSIÓN DEL DISPOSITIVO FINAL SE ENCUENTRA EN LA RAMA RASPBIAN

import matlab.engine
import numpy as np
from scipy.io import savemat, loadmat
import os
import time
import argparse
import random
import pandas as pd
import pickle
import scipy.io
from scipy.fft import fft
from scipy.signal import welch, butter, filtfilt
from scipy.stats import kurtosis, skew
import keyboard

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
    
def caracteristicas(datos, gen=1, edad=48):
    n = len(datos)
    fs = 500
    
    #Transformada discreta de Fourier
    
    F = fft(datos)
    
    #Entropía de Shannon
    counts, _ = np.histogram(datos, bins=20)
    P = counts / (np.sum(counts) + np.finfo(float).eps)
    ent = -np.sum(P * np.log2(P + np.finfo(float).eps))
    
    #Frecuencia media
    P2 = np.abs(F) / n
    P1 = P2[:n//2 + 1]
    P1[1:-1] = 2 * P1[1:-1]
    fmed = np.mean(P1)
    
    #Potencia media y Frecuencia PM
    f, psd_welch = welch(datos, fs, nperseg=n)
    pwmed = np.mean(psd_welch)
    idx_max = np.argmax(psd_welch)
    fpm = f[idx_max]
    
    #Agudeza y Asimetría
    
    ag = kurtosis(datos)
    asim = skew(datos)
    
    #RMS
    rms = np.sqrt(np.mean(datos**2))
    
    salida = [ent, fmed, fpm, ag, asim, rms, pwmed, gen , edad]
    return salida


def guardar_valor(valor, nombre, gen, edad, ruta="datos/data/"):
    
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
    temblor = np.vstack((temblor, [[1]]))
    
    savemat(FICHERO, {'data': señal, 'temblor': temblor, 'genero': gen, 'edad': edad, 'unidades': "mV"})

    print(f"[Guardar] Valor guardado: {valor:.4f}")


def analizar(eng, valor, gen, edad):

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
    
    """
    # Convertir a tipo MATLAB y analizar
    matlab_array = matlab.double(ventana)
    resultado = eng.interfaz(matlab_array)
    x_df = pd.concat([x_df, pd.DataFrame([resultado])], ignore_index=True)
    """

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modo', choices=['guardar', 'analizar'], required=True)
    parser.add_argument('--genero', choices=['F', 'M'], required=True)
    parser.add_argument('--edad', required=True)
    parser.add_argument('--nombre', required=True)
    args = parser.parse_args()
    
    dict_gen = {"M":0,"F":1}
    gen =dict_gen[args.genero]
    edad = int(args.edad)

    eng = None
    if args.modo == 'analizar':
        
        print("[Sistema] Iniciando motor de MATLAB...")
        # Buscar engines activos
        sesiones = matlab.engine.find_matlab()
        
        if sesiones:
            
            # Conectar al primero disponible
            print(f"[PYTHON] Conectando al engine de MATLAB existente: {sesiones[0]}")
            eng = matlab.engine.connect_matlab(sesiones[0])
            
        else:
            
            # Iniciar uno nuevo
            print("[PYTHON] No se encontró ningún engine activo. Iniciando uno nuevo...")
            eng = matlab.engine.start_matlab()


    while not keyboard.is_pressed('o'):  # Simular iteraciones
    
        valor = random.uniform(0, 0.1)
        if args.modo == 'guardar':
            
            guardar_valor(valor, args.nombre, gen, edad)
            
        elif args.modo == 'analizar':

            analizar(eng, valor,gen, edad)

    if eng:
        eng.quit()

if __name__ == "__main__":
    main()
