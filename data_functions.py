import os
import json
import requests
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm.auto import tqdm # Importamos tqdm directamente


# --- Funciones de Carga y Preprocesamiento ---

def download_file(url, save):

    re = requests.get(url, stream=True)

    file_size = int(re.headers.get('Content-Length', 0))
    with tqdm.wrapattr(re.raw, "read", total=file_size) as r:
        with open(save, 'wb') as f:
            shutil.copyfileobj(r, f)




def create_data_frame(raw_data, MAX_SEQ_LENGTH):
    # Procesa los datos crudos del JSON en una lista de diccionarios
    data = []
    for i in range(len(raw_data)):
        image_data = raw_data[i]['image_data']
        # Nos aseguramos de que haya tokens válidos y que no sean demasiado largos
        if image_data['visible_char_map'] and len(image_data['visible_char_map']) <= MAX_SEQ_LENGTH:
            data.append({
                'filename': raw_data[i]['filename'],
                'latex': raw_data[i]['latex'],
                # Usaremos visible_char_map para la secuencia de salida
                'latex_tokens': image_data['visible_char_map']
            })
    return data

def normalizedData(data, BASE_PATH, BATCH_DIR, IMAGE_SIZE, MAX_SEQ_LENGTH):
    """Carga, decodifica y normaliza una sola muestra de imagen y etiqueta."""

    image_path = os.path.join(BASE_PATH, BATCH_DIR, 'background_images', data['filename'])

    # 1. Cargar y decodificar la imagen (Input X)
    image_file = tf.io.read_file(image_path)
    # Establecer la forma (shape) es crucial para tf.io.decode_image
    image = tf.io.decode_image(image_file, channels=3, expand_animations=False)

    # Preprocesar imagen
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

    # Añadir la dimensión de lote (Batch Dimension) para el modelo: (1, 600, 600, 3)
    image = tf.expand_dims(image, axis=0)

    # 2. Obtener y preparar la secuencia de tokens (Output Y)

    # Aseguramos que la secuencia no exceda la longitud máxima.
    token_sequence = data['latex_tokens']

    # El modelo Seq2Seq predice CADA paso. Necesitamos padding.
    # Keras requiere que todas las secuencias tengan la misma longitud.
    # El padding se hará con 0s, asumiendo que el token 0 es un token de padding
    padded_sequence = pad_sequences([token_sequence], maxlen=MAX_SEQ_LENGTH, padding='post', value=0)

    # Convertir a tensor y asegurar la forma (1, MAX_SEQ_LENGTH)
    token = tf.constant(padded_sequence, dtype=tf.int32)

    return image, token

def loadAll(BASE_PATH, BATCH_DIR, MAX_SEQ_LENGTH):
    """Carga el JSON y devuelve la lista de datos."""
    json_path = os.path.join(BASE_PATH, BATCH_DIR, 'JSON', 'kaggle_data_1.json')

    if os.path.exists(json_path):
        with open(json_path) as f:
            raw_data = json.load(f)
            print('Archivo JSON cargado.')
            return create_data_frame(raw_data, MAX_SEQ_LENGTH)
    else:
        print(f'Error: Archivo no encontrado en {json_path}')
        return []


def load_existing_model(model_path):
    """Carga un modelo Keras (.h5) desde una ruta específica."""
    try:
        # Usamos custom_objects={'Model': Model} si hay problemas, pero Keras lo maneja
        model = tf.keras.models.load_model(model_path)
        print(f"\nModelo cargado exitosamente desde: {model_path}")
        return model
    except Exception as e:
        print(f"\nERROR al cargar el modelo desde {model_path}: {e}")
        return None