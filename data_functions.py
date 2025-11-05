import os
import json
import requests
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm.auto import tqdm # Importamos tqdm directamente

GLOBAL_TOKEN_MAP = None

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
    json_dir = os.path.join(BASE_PATH, BATCH_DIR, 'JSON')
    
    # Busca el archivo JSON dentro del directorio 'JSON'
    json_filename = None
    
    if os.path.exists(json_dir):
        # Itera sobre los archivos en el directorio 'JSON'
        for filename in os.listdir(json_dir):
            # Asume que el archivo de datos tiene 'data' en su nombre y termina en '.json'
            if 'data' in filename and filename.endswith('.json'):
                json_filename = filename
                break # Encontramos el archivo, lo usamos

    if json_filename:
        json_path = os.path.join(json_dir, json_filename)
        with open(json_path) as f:
            raw_data = json.load(f)
            # Imprime el nombre real del archivo cargado para depuración
            print(f'Archivo JSON "{json_filename}" cargado.') 
            return create_data_frame(raw_data, MAX_SEQ_LENGTH)
    else:
        # Aquí también mostramos la ruta para facilitar la depuración
        print(f'Error: No se encontró ningún archivo JSON que contenga "data" en {json_dir}')
        return []


def load_existing_model(model_path):
    """Carga un modelo Keras (.h5) desde una ruta específica."""
    try:
        # Añade todas las operaciones internas de TF que Keras podría haber serializado
        custom_objects = {
            # Se ha visto que estas son problemáticas en la serialización HDF5:
            'NotEqual': tf.math.not_equal,
            'ZerosLike': tf.zeros_like,
            'ExpandDims': tf.expand_dims,
            'LogicalOr': tf.math.logical_or,
            'OnesLike': tf.ones_like,
            'Any': tf.reduce_any, # Usar reduce_any en lugar de experimental.numpy.any
            'Concatenate': tf.keras.layers.Concatenate, # A veces Concatenate es el problema
            # Si se usó una función Lambda, también debe registrarse.
        }
        
        # Usamos custom_object_scope para registrar estas funciones al cargar
        with tf.keras.utils.custom_object_scope(custom_objects):
            # Cargar el modelo
            model = tf.keras.models.load_model(model_path)
            
        # Re-compilar el modelo es crucial, usando la función de pérdida correcta.
        # Asumimos que quieres usar la versión con ignore_index=0 para corregir el 100% accuracy.
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(ignore_index=0),
                      metrics=['accuracy'])
        
        print(f"\nModelo cargado exitosamente desde: {model_path}")
        return model
    except Exception as e:
        print(f"\nERROR al cargar el modelo desde {model_path}: {e}")
        return None


def load_token_map(EXTRAS_PATH):
    """Carga e invierte el visible_char_map.json para mapear índice a LaTeX."""
    global GLOBAL_TOKEN_MAP
    
    map_path = os.path.join(EXTRAS_PATH, 'visible_char_map.json')
    
    if os.path.exists(map_path):
        with open(map_path) as f:
            char_to_index = json.load(f)
            print("Mapeo de tokens cargado de visible_char_map.json.")
            
            # Invertir el diccionario: de LaTeX a Índice a Índice a LaTeX
            # Ignoramos el token de padding (asumido como 0)
            index_to_char = {v: k for k, v in char_to_index.items() if v != 0}
            
            # El token 0 es padding
            index_to_char[0] = '' 
            
            GLOBAL_TOKEN_MAP = index_to_char
            return GLOBAL_TOKEN_MAP
    else:
        print(f"Error: No se encontró visible_char_map.json en {map_path}. Usando mapeo temporal.")
        # Usar el mapeo temporal si el archivo no se encuentra
        return {
            0: '', 1: '\\limit', 2: '\\infty', 3: 'x', 4: 'y', 5: '=', 6: '2', 7: '{', 8: '}', 
            9: '\\frac', 10: '(', 11: ')', 12: '+', 13: '-', 14: '1', 15: '0', 16: 'a', 17: 'b', 
            18: '^', 19: '_', 20: '3', 21: '4', 22: '5', 23: '6', 24: '7', 25: '8', 26: '9', 
            27: 'z', 28: '\\pi', 29: 'e', 30: 'c', 31: 'd', 32: 'f', 33: 'g', 34: 'h', 35: 'i',
        }

def map_tokens_to_latex(token_sequence, EXTRAS_PATH):
    """Convierte una secuencia de índices de token a una cadena LaTeX usando el mapa cargado."""
    
    # Asegurar que el mapa esté cargado
    if GLOBAL_TOKEN_MAP is None:
        load_token_map(EXTRAS_PATH)
        
    latex_output = []
    
    if hasattr(token_sequence, 'numpy'):
        token_sequence = token_sequence.numpy().flatten()

    for token_index in token_sequence:
        # El padding (token 0) es mapeado a una cadena vacía
        latex_char = GLOBAL_TOKEN_MAP.get(token_index, f'?({token_index})')
        
        # Ignorar padding explícitamente y caracteres desconocidos
        if latex_char == '' or latex_char.startswith('?'): 
            continue
        
        latex_output.append(latex_char)

    return ' '.join(latex_output).strip()

