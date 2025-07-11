import time
import numpy as np
import keyboard
from scipy.io import savemat

# Parámetros
freq = 500  # Hz
interval = 1 / freq
data = []
data_array = np.array(data)
nombre = input("Inserta nombre")
print("Grabando... Mantén presionada la tecla 'z' para registrar 1.")
print("Presiona 'o' para terminar la grabación.")

try:
    while not keyboard.is_pressed('o'):
        value = 1 if keyboard.is_pressed('z') else 0
        data.append(value)
        time.sleep(interval)
except KeyboardInterrupt:
    print("Interrumpido por el usuario.")

# Guardar en .mat

savemat(nombre+'.mat', {'data': data})
print(f"Grabación terminada. {len(data)} muestras guardadas en"+nombre+" '.mat'.")
