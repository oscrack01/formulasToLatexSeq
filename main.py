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
from data_functions import load_token_map, load_existing_model

aidapearson_ocr_data_path = kagglehub.dataset_download('aidapearson/ocr-data')
print(f'Data source import complete. Path: {aidapearson_ocr_data_path}') 

# --- Configuraciones ---
BASE_PATH = 'C:/Users/oscal/.cache/kagglehub/datasets/aidapearson/ocr-data/versions/36'
BATCH_DIR_PREFIX = 'batch_' # Prefijo para iterar del 1 al 10
BATCH_DIR = 'batch_1'
EXTRAS_PATH = os.path.join(BASE_PATH, 'extras')
IMAGE_SIZE = 600
NUM_CLASSES = 91 # Número de tokens de LaTeX posibles
MAX_SEQ_LENGTH = 100 # Longitud máxima de la secuencia de salida (ajustar según el dataset)
RNN_UNITS = 128 # Unidades en la capa LSTM
MODEL_SAVE_PATH = '../output/math_seq2seq_lstm.h5' # Ruta del modelo por defecto

# Variables globales para almacenar los conjuntos de datos después de cargarlos
GLOBAL_DATA_LIST = None
GLOBAL_TRAIN_DATA = None
GLOBAL_TEST_DATA = None

# --- Función Principal ---

def main():
    # Inicializar la barra de progreso de tqdm
    global tqdm

    # Cargar y asegurar que el mapa de tokens exista antes de cualquier operación
    load_token_map(EXTRAS_PATH)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, desc: x # Placeholder si tqdm no está disponible

    # --- MENÚ DE INTERACCIÓN ---
    print("\n--- INICIO DEL SISTEMA DE RECONOCIMIENTO DE MATEMÁTICAS ---")
    action = input("¿Desea entrenar un nuevo modelo (T) o cargar un modelo existente (C)? [T/C]: ").strip().upper()
    
    if action == 'T':
        # --- FLUJO DE ENTRENAMIENTO ---
        print("Modo: ENTRENAMIENTO seleccionado. Se entrenará en los 10 batches.")
        # Inicializar el modelo
        model = create_model(IMAGE_SIZE, RNN_UNITS, MAX_SEQ_LENGTH, NUM_CLASSES)
        model.summary()
        trained_model = train_model(model)
        
    elif action == 'C':
        # --- FLUJO DE CARGA ---
        model_path = input(f"Ingrese la ruta del archivo .h5 a cargar (ej: {MODEL_SAVE_PATH}): ").strip()
        if not model_path:
            model_path = MODEL_SAVE_PATH
        
        trained_model = load_existing_model(model_path)
        
        if trained_model is None:
            print("El sistema no puede continuar sin un modelo. Saliendo.")
            return

        # Para evaluar o predecir, necesitamos cargar el conjunto de prueba
        print("Cargando datos de prueba para evaluación/predicción...")
        test_model(trained_model) # Llama a test_model para cargar GLOBAL_TEST_DATA

    else:
        print("Opción no válida. Saliendo.")
        return


    
    # --- EVALUACIÓN Y PREDICCIÓN (Común a ambos flujos) ---
    if action == 'T':
        print("\nEntrenamiento completado.")
        # La evaluación ya se hizo parcialmente dentro de load_existing_model para cargar GLOBAL_TEST_DATA si fue necesario
        test_model(trained_model) 
        print("\nEvaluación completada.")
    
    # --- PREDICCIÓN DE EJEMPLO ---
    if GLOBAL_TEST_DATA:
        print("\n--- FUNCIÓN DE PREDICCIÓN ---")
        prediction_action = input("¿Desea realizar una predicción en una imagen de prueba (P) o una imagen individual (I)? [P/I]: ").strip().upper()
        
        if prediction_action == 'P':
            # Predicción en una muestra del conjunto de prueba (la primera muestra)
            data_sample = GLOBAL_TEST_DATA[0]
            batch_dir = BATCH_DIR_PREFIX + str(data_sample['batch_num'])
            example_path = os.path.join(BASE_PATH, batch_dir, 'background_images', data_sample['filename'])
        
        elif prediction_action == 'I':
            # Predicción en una imagen individual
            example_path = input("Ingrese la ruta completa de la imagen a predecir (ej: /path/to/my/image.png): ").strip()
            if not os.path.exists(example_path):
                print(f"ERROR: La ruta de imagen '{example_path}' no existe. Saliendo de la predicción.")
                return

        else:
            print("Opción de predicción no válida.")
            return

        print(' tengo que implementar predict_single_image(trained_model, example_path)')

    print("\nFin del programa.")


main()
