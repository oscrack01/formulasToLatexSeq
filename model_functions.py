import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm.auto import tqdm # Importamos tqdm directamente
from data_functions import loadAll, normalizedData, map_tokens_to_latex
import datetime # Importamos datetime para el timestamp

# --- Definición del Modelo (Seq2Seq: Encoder CNN + Decoder LSTM) ---
GLOBAL_DATA_LIST = None 
GLOBAL_TRAIN_DATA = None
GLOBAL_TEST_DATA = None

def create_model(IMAGE_SIZE, RNN_UNITS, MAX_SEQ_LENGTH, NUM_CLASSES):

    # --- ENCODER (VISIÓN) ---
    encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='image_input')

    x = layers.Conv2D(32, (3, 3), activation='relu')(encoder_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    context_vector = layers.Dense(RNN_UNITS, activation='relu', name='context_vector')(x)

    # --- DECODER (SECUENCIA) ---
    # 1. Entrada de tokens (la secuencia de tokens de etiqueta para Teacher Forcing)
    decoder_input = layers.Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='token_input')

    # 2. Embedding de los tokens de entrada
    x_tokens = layers.Embedding(NUM_CLASSES, RNN_UNITS, mask_zero=True)(decoder_input)

    # 3. Concatenar el vector de contexto a CADA paso de la secuencia de tokens
    repeated_context = layers.RepeatVector(MAX_SEQ_LENGTH)(context_vector)
    
    # Concatenar el embedding de tokens y el contexto
    x = layers.Concatenate(axis=-1)([x_tokens, repeated_context]) 

    # 4. Capa LSTM: Ahora usa la información del token anterior + el contexto de la imagen
    decoder_lstm = layers.LSTM(RNN_UNITS, return_sequences=True, name='decoder_lstm')(x)

    # 5. Salida TimeDistributed
    decoder_output = layers.TimeDistributed(
        layers.Dense(NUM_CLASSES, activation='softmax'),
        name='token_output'
    )(decoder_lstm)

    # --- MODELO FINAL ---
    # El modelo ahora requiere DOS entradas: [Imagen, Secuencia de Tokens de Entrada]
    model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output, name='Math_Seq2Seq_Model')

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Funciones de Entrenamiento y Prueba ---

def train_model(model, BASE_PATH, BATCH_DIR, MAX_SEQ_LENGTH, IMAGE_SIZE, MODEL_SAVE_PATH):
    global GLOBAL_TRAIN_DATA, GLOBAL_TEST_DATA
    
    print("Iniciando carga y preprocesamiento de datos...")
    data_list = loadAll(BASE_PATH, BATCH_DIR, MAX_SEQ_LENGTH)
    
    if not data_list:
        return model

    # Separar datos en entrenamiento y prueba (80/20) y guardar globalmente
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    GLOBAL_TRAIN_DATA = train_data
    GLOBAL_TEST_DATA = test_data
    
    print(f"Comenzando entrenamiento en {len(train_data)} muestras. Esto tomará MUCHO tiempo.")
    
    # Loop de entrenamiento (iteración por muestra)
    for i in tqdm(range(len(train_data)), desc="Entrenando muestras (Quick Data)"):
        image, token = normalizedData(train_data[i], BASE_PATH, BATCH_DIR, IMAGE_SIZE, MAX_SEQ_LENGTH)
        sample_weight = np.not_equal(token, 0).astype(np.float32)
        try:
            # CAMBIO CLAVE: Dos entradas para model.fit
            model.fit([image, token], token, epochs=1, verbose=0, sample_weight=sample_weight)
        except Exception as e:
            print(f"\nError al entrenar la muestra {i}: {e}")
            continue

    # Guardar el modelo 
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH) or '.', exist_ok=True)
    
    final_save_path = MODEL_SAVE_PATH
    
    # Lógica para no sobrescribir el archivo
    if os.path.exists(MODEL_SAVE_PATH):
        # Si el archivo existe, añade un timestamp
        timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        
        # Separar el nombre base del sufijo (.h5)
        base, ext = os.path.splitext(MODEL_SAVE_PATH)
        final_save_path = base + timestamp + ext

    model.save(final_save_path)
    print(f"\nModelo guardado en: {final_save_path}")
    return model

def test_model(model, BASE_PATH, BATCH_DIR_DUMMY, IMAGE_SIZE, MAX_SEQ_LENGTH):
    global GLOBAL_TEST_DATA
    
    if GLOBAL_TEST_DATA is None:
        print("El conjunto de prueba no ha sido inicializado. Ejecutando train_model primero...")
        # Llama a loadAll para asegurar que los datos estén cargados, aunque solo para obtener la lista completa
        _ = loadAll(BASE_PATH, BATCH_DIR_DUMMY, MAX_SEQ_LENGTH)
        if GLOBAL_TEST_DATA is None:
             # Si GLOBAL_TEST_DATA sigue siendo None (porque train_model no se llamó), forzamos el split
            data_list = loadAll(BASE_PATH, BATCH_DIR_DUMMY, MAX_SEQ_LENGTH)
            if not data_list: return 
            _, GLOBAL_TEST_DATA = train_test_split(data_list, test_size=0.2, random_state=42)


    test_data = GLOBAL_TEST_DATA
    
    print(f"\nIniciando evaluación en {len(test_data)} muestras de prueba.")
    
    # Variables para acumular los resultados de la evaluación
    all_loss = []
    all_accuracy = []
    
    for i in tqdm(range(len(test_data)), desc="Evaluando muestras"):
        data_sample = test_data[i]
        actual_batch_dir = data_sample.get('BATCH_DIR', BATCH_DIR_DUMMY)
        image, token = normalizedData(data_sample, BASE_PATH, actual_batch_dir, IMAGE_SIZE, MAX_SEQ_LENGTH)
        sample_weight = np.not_equal(token, 0).astype(np.float32)
        try:
            # CAMBIO CLAVE: Dos entradas para model.evaluate
            loss, accuracy = model.evaluate([image, token], token, verbose=0, sample_weight=sample_weight) 
            all_loss.append(loss)
            all_accuracy.append(accuracy)
        except Exception as e:
            print(f"\nError al evaluar la muestra {i}: {e}")
            continue
            
    # Calcular promedios
    if all_loss:
        avg_loss = np.mean(all_loss)
        avg_accuracy = np.mean(all_accuracy)
        print("\n--- Resultados de la Evaluación en el Conjunto de Prueba ---")
        print(f"Pérdida (Loss) Promedio: {avg_loss:.4f}")
        print(f"Precisión (Accuracy) Promedio: {avg_accuracy:.4f}")
    else:
        print("No se pudo evaluar ninguna muestra.")
        
    return model

def predict_single_image(model, image_path, IMAGE_SIZE, EXTRAS_PATH, MAX_SEQ_LENGTH):
    """
    Realiza una predicción de la fórmula LaTeX para una sola imagen.
    """
    print(f"\nRealizando predicción para: {image_path}")
    
    # 1. Cargar y preprocesar la imagen
    try:
        image_file = tf.io.read_file(image_path)
        image = tf.io.decode_image(image_file, channels=3, expand_animations=False)
    except Exception as e:
        return f"Error al cargar o decodificar la imagen: {e}"

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    
    # Añadir la dimensión de lote (1, 600, 600, 3)
    image = tf.expand_dims(image, axis=0)

    # 2. Predecir la secuencia completa
    # Crea una entrada de tokens vacía con la longitud máxima
    token_input = np.zeros((1, MAX_SEQ_LENGTH), dtype=np.int32)
    # La predicción será de forma (1, MAX_SEQ_LENGTH, NUM_CLASSES)
    predictions = model.predict([image, token_input], verbose=0)

    # 3. Decodificar la secuencia
    # En cada paso de la secuencia (dim 1), tomamos el token con mayor probabilidad (argmax en dim 2)
    predicted_tokens = np.argmax(predictions[0], axis=1) # Forma: (MAX_SEQ_LENGTH,)

    # 4. Mapear los tokens a la cadena LaTeX
    latex_formula = map_tokens_to_latex(predicted_tokens, EXTRAS_PATH)
    print("--- Debug: Predicción de Tokens Crudos (Indices) ---")
    # Imprimimos los tokens predichos
    print(predicted_tokens) 
    
    print("--- Predicción LaTeX ---")
    print(latex_formula)
    
    return latex_formula


# model_functions.py (Añadir esta nueva función después de create_model)

def train_model_full(model, BASE_PATH, MAX_SEQ_LENGTH, IMAGE_SIZE, MODEL_SAVE_PATH):
    global GLOBAL_TRAIN_DATA, GLOBAL_TEST_DATA

    all_batches_data = []
    
    # --- 1. Cargar y Combinar Datos de los 10 Batches ---
    print("Iniciando carga de datos de los 10 batches...")
    for i in range(1, 11):
        BATCH_DIR = f'batch_{i}'
        print(f"  -> Cargando {BATCH_DIR}...")
        data_list = loadAll(BASE_PATH, BATCH_DIR, MAX_SEQ_LENGTH)
        for item in data_list:
            item['BATCH_DIR'] = BATCH_DIR
        all_batches_data.extend(data_list)
        
    if not all_batches_data:
        print("No se encontraron datos en ningún batch. Terminando entrenamiento.")
        return model

    # --- 2. Split en Entrenamiento/Prueba Global ---
    # Usaremos todo para el split
    print(f"\nDatos totales cargados: {len(all_batches_data)} muestras. Split en entrenamiento/prueba (80/20).")
    train_data, test_data = train_test_split(all_batches_data, test_size=0.2, random_state=42)
    GLOBAL_TRAIN_DATA = train_data
    GLOBAL_TEST_DATA = test_data
    
    print(f"Comenzando entrenamiento en {len(train_data)} muestras. Esto tomará MUCHO tiempo.")
    
    # --- 3. Loop de Entrenamiento (Batch por Batch) ---
    # Creamos un diccionario para agrupar las muestras de entrenamiento por su batch_dir
    train_data_by_batch = {}
    for data_sample in train_data:
        batch_key = data_sample['BATCH_DIR']
        if batch_key not in train_data_by_batch:
            train_data_by_batch[batch_key] = []
        train_data_by_batch[batch_key].append(data_sample)

    
    # Iteramos sobre los batches que tienen datos de entrenamiento
    for BATCH_DIR, train_batch in train_data_by_batch.items():
        
        print(f"\nEntrenando en {len(train_batch)} muestras de {BATCH_DIR}...")
        
        # Iteramos sobre las muestras de entrenamiento para ESE batch
        for i in tqdm(range(len(train_batch)), desc=f"Muestras de {BATCH_DIR}"):
            data_sample = train_batch[i]
            
            # OJO: normalizedData usa el BATCH_DIR correcto (que ahora es el key del diccionario)
            image, token = normalizedData(data_sample, BASE_PATH, BATCH_DIR, IMAGE_SIZE, MAX_SEQ_LENGTH)
            sample_weight = np.not_equal(token, 0).astype(np.float32)
            try:
                # El modelo necesita [image, token] como entradas y token como etiqueta (salida)
                model.fit([image, token], token, epochs=1, verbose=0, sample_weight=sample_weight) 
            except Exception as e:
                print(f"\nError al entrenar la muestra {data_sample['filename']} en {BATCH_DIR}: {e}")
                continue

    # 4. Guardar el modelo (mismo código que train_model)
    # ... (código para guardar con timestamp) ...
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH) or '.', exist_ok=True)
    final_save_path = MODEL_SAVE_PATH
    if os.path.exists(MODEL_SAVE_PATH):
        timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(MODEL_SAVE_PATH)
        final_save_path = base + timestamp + ext

    model.save(final_save_path)
    print(f"\nModelo guardado en: {final_save_path}")
    
    # Después de entrenar, cargamos el GLOBAL_TEST_DATA final para que la evaluación funcione.
    # Usaremos el último split hecho por la función test_model la primera vez que se ejecute.
    # Por ahora, simplemente retornamos el modelo.
    
    return model

# Nota: GLOBAL_TRAIN_DATA ya no se usa, ya que la iteración por batch simplifica el manejo de rutas.