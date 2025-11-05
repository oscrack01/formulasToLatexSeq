import kagglehub
import os
from tqdm.auto import tqdm 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Desactiva las optimizaciones oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # (Opcional) Oculta otros mensajes informativos de TF
import tensorflow as tf
from model_functions import create_model, train_model, train_model_full, test_model, predict_single_image
import model_functions 
from data_functions import load_existing_model
import warnings 

aidapearson_ocr_data_path = kagglehub.dataset_download('aidapearson/ocr-data')
print(f'Data source import complete. Path: {aidapearson_ocr_data_path}') 

# --- Warnings ---
warnings.filterwarnings('ignore', category=Warning, module='tensorflow')

# --- Configuraciones ---
BASE_PATH = 'C:/Users/oscal/.cache/kagglehub/datasets/aidapearson/ocr-data/versions/36'
EXTRAS_PATH = os.path.join(BASE_PATH, 'extras')
BATCH_DIR = 'batch_1'
IMAGE_SIZE = 600
NUM_CLASSES = 91 # Número de tokens de LaTeX posibles
MAX_SEQ_LENGTH = 100 # Longitud máxima de la secuencia de salida (ajustar según el dataset)
RNN_UNITS = 128 # Unidades en la capa LSTM
MODEL_SAVE_PATH = '../output/math_seq2seq_lstm.keras' # Ruta del modelo por defecto

# --- Función Principal ---

def main():
    # Inicializar la barra de progreso de tqdm
    global tqdm
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, desc: x # Placeholder si tqdm no está disponible

    # --- MENÚ DE INTERACCIÓN ---
    print("\n--- INICIO DEL SISTEMA DE RECONOCIMIENTO DE MATEMÁTICAS ---")
    action = input("¿Desea entrenar un nuevo modelo (T) o cargar un modelo existente (C)? [T/C]: ").strip().upper()
    
    if action == 'T':
        # --- FLUJO DE ENTRENAMIENTO ---
        
        # 1. Preguntar por la opción de entrenamiento
        train_option = input("Seleccione el modo de entrenamiento:\n"
                             "  (R) Rápido: Usar solo batch_1 para entrenamiento.\n"
                             "  (P) Pesado: Iterar y entrenar con los 10 batches.\n"
                             "[R/P]: ").strip().upper()

        if train_option not in ('R', 'P'):
            print("Opción de entrenamiento no válida. Saliendo.")
            return

        print("Inicializando el modelo con arquitectura Seq2Seq de doble entrada...")
        
        # 2. Inicializar el modelo con la arquitectura de doble entrada
        model = create_model(IMAGE_SIZE, RNN_UNITS, MAX_SEQ_LENGTH, NUM_CLASSES)
        model.summary()
        
        # 3. Llamar a la función de entrenamiento correcta
        if train_option == 'R':
            print(f"Modo: ENTRENAMIENTO RÁPIDO seleccionado. Se entrenará en {BATCH_DIR}.")
            # Se pasan todos los parámetros requeridos
            trained_model = train_model(model, BASE_PATH, BATCH_DIR, MAX_SEQ_LENGTH, IMAGE_SIZE, MODEL_SAVE_PATH) 
        
        elif train_option == 'P':
            print("Modo: ENTRENAMIENTO PESADO seleccionado. Se entrenará en los 10 batches.")
            # Se usa la nueva función (BATCH_DIR no es necesario ya que itera internamente)
            trained_model = train_model_full(model, BASE_PATH, MAX_SEQ_LENGTH, IMAGE_SIZE, MODEL_SAVE_PATH)
            
        else: # (Esto ya se maneja arriba, pero como fallback)
            return
        
    elif action == 'C':
        # --- FLUJO DE CARGA ---
        model_path = input(f"Ingrese la ruta del archivo .h5 a cargar (ej: {MODEL_SAVE_PATH}): ").strip()
        if not model_path:
            model_path = MODEL_SAVE_PATH
        
        trained_model = load_existing_model(model_path)
        
        if trained_model is None:
            print("El sistema no puede continuar sin un modelo. Saliendo.")
            return

        print("Modelo cargado. Listo para predicciones individuales (I).")

    else:
        print("Opción no válida. Saliendo.")
        return


    
# --- EVALUACIÓN Y PREDICCIÓN (Común a ambos flujos) ---
    if action == 'T':
        if train_option == 'R':
            print("\nEntrenamiento rápido completado. Evaluando en el 20% de 'batch_1'.")
            test_model(trained_model, BASE_PATH, BATCH_DIR, IMAGE_SIZE, MAX_SEQ_LENGTH)
        elif train_option == 'P':
             # En el modo pesado, GLOBAL_TEST_DATA será el 20% de todos los 10 batches
            print("\nEntrenamiento pesado completado. Evaluando en el 20% de los 10 batches.")
            # El BATCH_DIR no es relevante aquí si GLOBAL_TEST_DATA ya está lleno, 
            # pero lo necesitamos para que `test_model` cargue los datos si es necesario.
            # Usaremos 'batch_1' como dummy, ya que test_model busca GLOBAL_TEST_DATA
            test_model(trained_model, BASE_PATH, 'batch_1', IMAGE_SIZE, MAX_SEQ_LENGTH) 
        
        print("\nEvaluación completada.")

    
    # --- PREDICCIÓN DE EJEMPLO ---
    # Usamos mf.GLOBAL_TEST_DATA para acceder al valor actual
    print("\n--- FUNCIÓN DE PREDICCIÓN ---")
    prediction_action = input("¿Desea realizar una predicción en una imagen de prueba (P), una imagen individual (I), o Salir (S)? [P/I/S]: ").strip().upper()
    
    if prediction_action == 'P':
        # --- NUEVA LÓGICA DE CARGA CONDICIONAL ---
        if model_functions.GLOBAL_TEST_DATA is None:
            print("Cargando datos de prueba. Esto puede tomar tiempo...")
            # Forzamos la carga y el split llamando a test_model
            test_model(trained_model, BASE_PATH, BATCH_DIR, IMAGE_SIZE, MAX_SEQ_LENGTH) 
        
        if model_functions.GLOBAL_TEST_DATA:
            # Predicción en una muestra del conjunto de prueba (la primera muestra)
            data_sample = model_functions.GLOBAL_TEST_DATA[0]
            batch_dir = BATCH_DIR
            example_path = os.path.join(BASE_PATH, batch_dir, 'background_images', data_sample['filename'])
        else:
            print("ERROR: No se pudo cargar el conjunto de prueba. Saliendo de la predicción.")
            return
            
    elif prediction_action == 'I':
        # Predicción en una imagen individual (no requiere GLOBAL_TEST_DATA)
        example_path = input("Ingrese la ruta completa de la imagen a predecir (ej: /path/to/my/image.png): ").strip()
        if not os.path.exists(example_path):
            print(f"ERROR: La ruta de imagen '{example_path}' no existe. Saliendo de la predicción.")
            return

    elif prediction_action == 'S':
        print("\nSaliendo del programa.")
        return

    else:
        print("Opción de predicción no válida. Saliendo.")
        return

    # Aquí se ejecuta la predicción si la opción fue 'P' o 'I'
    predict_single_image(trained_model, example_path, IMAGE_SIZE, EXTRAS_PATH, MAX_SEQ_LENGTH)

    print("\nFin del programa.")

main()
