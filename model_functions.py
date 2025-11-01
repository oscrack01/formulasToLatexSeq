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
    # Captura las características de la imagen (similar a la CNN previa)
    encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='image_input')

    x = layers.Conv2D(32, (3, 3), activation='relu')(encoder_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # La salida del Encoder debe tener la forma adecuada para el Decoder
    # Flatten y luego Dense para crear un vector de contexto (context vector)
    x = layers.Flatten()(x)

    # Vector de contexto: Reducimos las características a un vector de tamaño RNN_UNITS
    context_vector = layers.Dense(RNN_UNITS, activation='relu', name='context_vector')(x)

    # --- DECODER (SECUENCIA) ---
    # 1. Entrada de tokens (la secuencia de tokens en el paso de tiempo t)
    # NOTA: En este diseño, la entrada de tokens se convierte en la etiqueta para el entrenamiento
    # Para inferencia, aún se requiere un bucle auto-regresivo fuera del modelo Keras.
    decoder_input = layers.Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='token_input')

    # 2. Embedding de los tokens de entrada
    x = layers.Embedding(NUM_CLASSES, RNN_UNITS, mask_zero=True)(decoder_input)

    # 3. Concatenar el vector de contexto a CADA paso de la secuencia de tokens
    # Forma del contexto: (Batch_size, 1, RNN_UNITS). Lo repetimos para que coincida con la secuencia
    repeated_context = layers.RepeatVector(MAX_SEQ_LENGTH)(context_vector)
    
    # Concatenar el contexto y el embedding de tokens
    # Forma final: (Batch_size, MAX_SEQ_LENGTH, 2 * RNN_UNITS)
    x = layers.Concatenate(axis=-1)([x, repeated_context]) # Concatenamos el embedding con el contexto

    # 4. Capa LSTM: Ahora usa la información del token anterior (a través de la secuencia de entrada)
    decoder_lstm = layers.LSTM(RNN_UNITS, return_sequences=True, name='decoder_lstm')(x)

    # 5. Salida TimeDistributed
    decoder_output = layers.TimeDistributed(
        layers.Dense(NUM_CLASSES, activation='softmax'),
        name='token_output'
    )(decoder_lstm)

    # --- MODELO FINAL ---
    model = Model(inputs=encoder_input, outputs=decoder_output, name='Math_Seq2Seq_Model')

    # Compilación: Usamos 'sparse_categorical_crossentropy' con return_sequences=True en el output
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
    for i in tqdm(range(len(train_data)), desc="Entrenando muestras"):
        # image forma: (1, 600, 600, 3)
        # token forma: (1, MAX_SEQ_LENGTH)
        image, token = normalizedData(train_data[i], BASE_PATH, BATCH_DIR, IMAGE_SIZE, MAX_SEQ_LENGTH)
        
        try:
            # epochs=1 (entrenamos cada muestra 1 vez), verbose=0 (silencioso)
            model.fit(image, token, epochs=1, verbose=0) 
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

def test_model(model, BASE_PATH, BATCH_DIR, IMAGE_SIZE, MAX_SEQ_LENGTH):
    global GLOBAL_TEST_DATA
    
    if GLOBAL_TEST_DATA is None:
        print("El conjunto de prueba no ha sido inicializado. Ejecutando train_model primero...")
        # Llama a loadAll para asegurar que los datos estén cargados, aunque solo para obtener la lista completa
        _ = loadAll(BASE_PATH, BATCH_DIR, MAX_SEQ_LENGTH)
        if GLOBAL_TEST_DATA is None:
             # Si GLOBAL_TEST_DATA sigue siendo None (porque train_model no se llamó), forzamos el split
            data_list = loadAll(BASE_PATH, BATCH_DIR, MAX_SEQ_LENGTH)
            if not data_list: return 
            _, GLOBAL_TEST_DATA = train_test_split(data_list, test_size=0.2, random_state=42)


    test_data = GLOBAL_TEST_DATA
    
    print(f"\nIniciando evaluación en {len(test_data)} muestras de prueba.")
    
    # Variables para acumular los resultados de la evaluación
    all_loss = []
    all_accuracy = []
    
    for i in tqdm(range(len(test_data)), desc="Evaluando muestras"):
        image, token = normalizedData(test_data[i], BASE_PATH, BATCH_DIR, IMAGE_SIZE, MAX_SEQ_LENGTH)
        
        try:
            # Evaluar el modelo en la muestra individual
            loss, accuracy = model.evaluate(image, token, verbose=0)
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

def predict_single_image(model, image_path, IMAGE_SIZE, EXTRAS_PATH):
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
    # La predicción será de forma (1, MAX_SEQ_LENGTH, NUM_CLASSES)
    predictions = model.predict(image, verbose=0)

    # 3. Decodificar la secuencia
    # En cada paso de la secuencia (dim 1), tomamos el token con mayor probabilidad (argmax en dim 2)
    predicted_tokens = np.argmax(predictions[0], axis=1) # Forma: (MAX_SEQ_LENGTH,)

    # 4. Mapear los tokens a la cadena LaTeX
    latex_formula = map_tokens_to_latex(predicted_tokens, EXTRAS_PATH)
    
    print("--- Predicción LaTeX ---")
    print(latex_formula)
    
    return latex_formula