import os
import numpy as np
import pandas as pd
import scipy.io
from scipy.fft import fft
from scipy.signal import welch, butter, filtfilt
from scipy.stats import kurtosis, skew

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

def filtrar_emg(signal, fs=500):
    low, high = 3, 220
    b, a = butter(4, [low / (fs/2), high / (fs/2)], btype='band')
    return filtfilt(b, a, signal)

def remove_outliers_iqr(X, factor=1.45):
    # Calcula el rango intercuartílico para cada columna
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    # Devuelve máscara booleana para filas sin outliers en ninguna columna
    mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
    return mask

"""
def procesar_archivos(emg_dir='emg', temblor_dir='temblor', salida='dataset', tiempo=5 , fs=500, solape=0.95):
    
    ventana = int(2*tiempo*fs)
    paso = int(ventana*(1-solape))
    
    os.makedirs(salida, exist_ok=True)
    emg_files = {f for f in os.listdir(emg_dir) if f.endswith('.mat')}
    temblor_files = {f for f in os.listdir(temblor_dir) if f.endswith('.mat')}
    comunes = emg_files & temblor_files
    antelacion = ventana * 1
    X, y = [], []

    for archivo in comunes:
        try:
            emg_path = os.path.join(emg_dir, archivo)
            temblor_path = os.path.join(temblor_dir, archivo)

            emg = scipy.io.loadmat(emg_path, variable_names=['data'])['data'].squeeze()
            temblor = scipy.io.loadmat(temblor_path, variable_names=['data'])['data'].squeeze()

            emg = filtrar_emg(emg)

            factor = len(temblor) // len(emg)
            if factor < 1:
                emg = emg[::(len(emg) // len(temblor))]
                factor = 1
            temblor_alineado = temblor[::factor][:len(emg)]
            emg = emg[:len(temblor_alineado)]

            for i in range(0, len(emg) - ventana + 1, paso):
                segmento_emg = emg[i:i+ventana]
                segmento_temblor = temblor_alineado[i:i+ventana]

                feats = caracteristicas(segmento_emg)
                target = int(np.round(np.mean(temblor_alineado[i + ventana : i + ventana + antelacion])))

                X.append(feats)
                y.append(target)

        except Exception as e:
            print(f"Error con {archivo}: {e}")

    X, y = np.array(X), np.array(y)

    # Aplicar filtro para quitar outliers
    mask = remove_outliers_iqr(X)
    X_filtered = X[mask]
    y_filtered = y[mask]

    columnas = ['ent', 'fmed', 'fpred', 'ag', 'asim', 'rms', 'pwmed', 'gen', 'edad', 'temblor']
    data_final = np.hstack([X_filtered, y_filtered.reshape(-1, 1)])
    df = pd.DataFrame(data_final, columns=columnas)

    df.to_csv(os.path.join(salida, 'dataset.csv'), index=False)
    print(f"Proceso completo. {len(X_filtered)} ventanas generadas tras eliminar outliers. Dataset guardado en CSV.")
"""

def procesar_archivos(directorio='datos', salida='dataset', tiempo=5, fs=500, solape=0.95):
    ventana = int(2 * tiempo * fs)
    paso = int(ventana * (1 - solape))
    antelacion = ventana * 1  # 1 ventana hacia adelante para target

    os.makedirs(salida, exist_ok=True)
    archivos = [f for f in os.listdir(directorio) if f.endswith('.mat')]

    X, y = [], []

    for archivo in archivos:
        try:
            ruta = os.path.join(directorio, archivo)
            datos = scipy.io.loadmat(ruta)

            if 'senal' not in datos or 'temblor' not in datos:
                print(f"Saltando {archivo}: faltan variables 'senal' o 'temblor'")
                continue

            emg = datos['senal'].squeeze()
            temblor = datos['temblor'].squeeze()
            gen = datos['genero']
            edad= datos['edad']

            emg = filtrar_emg(emg)

            factor = len(temblor) // len(emg)
            if factor < 1:
                emg = emg[::(len(emg) // len(temblor))]
                factor = 1
            temblor_alineado = temblor[::factor][:len(emg)]
            emg = emg[:len(temblor_alineado)]

            for i in range(0, len(emg) - ventana + 1, paso):
                segmento_emg = emg[i:i+ventana]
                segmento_temblor = temblor_alineado[i:i+ventana]

                feats = caracteristicas(segmento_emg)
                target = int(np.round(np.mean(temblor_alineado[i + ventana : i + ventana + antelacion])))

                X.append(feats)
                y.append(target)

        except Exception as e:
            print(f"Error procesando {archivo}: {e}")

    X, y = np.array(X), np.array(y)

    # Filtro de outliers
    mask = remove_outliers_iqr(X)
    X_filtered = X[mask]
    y_filtered = y[mask]

    columnas = ['ent', 'fmed', 'fpred', 'ag', 'asim', 'rms', 'pwmed', 'gen', 'edad', 'temblor']
    data_final = np.hstack([X_filtered, y_filtered.reshape(-1, 1)])
    df = pd.DataFrame(data_final, columns=columnas)

    df.to_csv(os.path.join(salida, 'dataset.csv'), index=False)
    print(f"Proceso completo. {len(X_filtered)} ventanas generadas tras eliminar outliers. Dataset guardado en CSV.")




procesar_archivos(directorio='data', tiempo=3)

