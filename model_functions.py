import os
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm.auto import tqdm # Importamos tqdm directamente
from data_functions import loadAll, normalizedData
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

    # El Decoder debe tomar el vector de contexto del Encoder y generar una secuencia.
    # Para esto, el vector de contexto se repite MAX_SEQ_LENGTH veces.
    # Forma: (Batch_size, RNN_UNITS) -> (Batch_size, MAX_SEQ_LENGTH, RNN_UNITS)
    repeated_context = layers.RepeatVector(MAX_SEQ_LENGTH)(context_vector)

    # Capa LSTM: Toma el vector de contexto repetido y lo procesa secuencialmente
    # Retorna la secuencia completa (return_sequences=True)
    decoder_lstm = layers.LSTM(RNN_UNITS, return_sequences=True, name='decoder_lstm')(repeated_context)

    # Capa TimeDistributed: Aplica una capa Dense (clasificador) en CADA paso de la secuencia
    # Salida final: (Batch_size, MAX_SEQ_LENGTH, NUM_CLASSES)
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
