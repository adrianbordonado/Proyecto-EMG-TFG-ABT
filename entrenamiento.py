import matlab.engine
import numpy as np
from scipy.io import savemat, loadmat
import os
import time
import argparse
import random
import pandas as pd

def matlab_dict_to_python_dict(mat_dict):
    python_dict = {}
    keys = mat_dict.keys()
    for k in keys:
        # Convertir clave y valor a tipo de Python
        key = str(k)
        value = mat_dict[k]

        # Si el valor es tipo MATLAB array con un solo elemento, lo convertimos
        if isinstance(value, matlab.double) and len(value) == 1:
            value = float(value[0])
        elif isinstance(value, matlab.double):
            value = [float(v) for v in value]
        
        python_dict[key] = value

    return python_dict

eng = matlab.engine.start_matlab()
lista = eng.diccionarizamientoewe("datos.mat",100)


lista_diccionarios_python = [
    matlab_dict_to_python_dict(d) for d in lista
]

x_df = pd.DataFrame(lista_diccionarios_python)
print(x_df)

eng.quit()