import matlab.engine
import numpy as np
from scipy.io import savemat, loadmat
import os
import time
import argparse
import random
import pandas as pd

N = 10  # Ventana de análisis
x_df = pd.DataFrame(columns=["entropia","frecuencia media", "frecuencia predominante","agudeza","asimetria", "rms", "potencia media", "mav"])
señal =[]

def guardar_valor(valor, nombre):
    
    FICHERO = nombre + ".mat"    
    if os.path.exists(FICHERO):
        datos = loadmat(FICHERO)
        señal = datos.get('senal', []).flatten()
    else:
        señal = np.array([])

    señal = np.append(señal, valor)
    savemat(FICHERO, {'senal': señal})
    print(f"[Guardar] Valor guardado: {valor:.4f}")

def analizar(eng, valor):

    global x_df
    
    señal.append(valor)

    if len(señal) < N:
        print(f"[Analizar] No hay suficientes datos (actual: {len(señal)}/{N})")
        return

    ventana = señal[-N:]

    # Convertir a tipo MATLAB y analizar
    matlab_array = matlab.double(ventana)
    resultado = eng.interfaz(matlab_array)
    x_df = pd.concat([x_df, pd.DataFrame([resultado])], ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modo', choices=['guardar', 'analizar'], required=True)
    parser.add_argument('--nombre', required=True)
    args = parser.parse_args()

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


    while True:  # Simular iteraciones
    
        valor = random.uniform(0, 0.1)
        if args.modo == 'guardar':
            
            guardar_valor(valor, args.nombre)
            
        elif args.modo == 'analizar':

            analizar(eng, valor)

        time.sleep(0.01)

    if eng:
        eng.quit()

if __name__ == "__main__":
    main()
