import kagglehub
import os
import json
import requests
import shutil
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm.auto import tqdm 
from model_functions import create_model, train_model, test_model 

aidapearson_ocr_data_path = kagglehub.dataset_download('aidapearson/ocr-data')
print(f'Data source import complete. Path: {aidapearson_ocr_data_path}') 

# --- Configuraciones ---
BASE_PATH = 'C:/Users/oscal/.cache/kagglehub/datasets/aidapearson/ocr-data/versions/36'
BATCH_DIR = 'batch_1'
IMAGE_SIZE = 600
NUM_CLASSES = 91 # Número de tokens de LaTeX posibles
MAX_SEQ_LENGTH = 100 # Longitud máxima de la secuencia de salida (ajustar según el dataset)
RNN_UNITS = 128 # Unidades en la capa LSTM

# Variables globales para almacenar los conjuntos de datos después de cargarlos
GLOBAL_DATA_LIST = None
GLOBAL_TRAIN_DATA = None
GLOBAL_TEST_DATA = None

# --- Función Principal ---

def main():
    # Inicializar la barra de progreso de tqdm
    global tqdm
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, desc: x # Placeholder si tqdm no está disponible
        
    # Inicializar el modelo
    model = create_model(IMAGE_SIZE, RNN_UNITS, MAX_SEQ_LENGTH, NUM_CLASSES)
    model.summary()
    
    # Entrenar el modelo (Guarda el conjunto de prueba globalmente)
    trained_model = train_model(model, BASE_PATH, BATCH_DIR, MAX_SEQ_LENGTH, IMAGE_SIZE)
    
    print("\nEntrenamiento completado.")
    
    # Evaluar el modelo con el conjunto de prueba
    test_model(trained_model, BASE_PATH, BATCH_DIR, IMAGE_SIZE, MAX_SEQ_LENGTH)
    
    print("\nEvaluación completada.")

main()
