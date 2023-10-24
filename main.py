import pickle
import sys
import numpy as np

def load_model_and_scaler(model_path, scaler_path):
    """Carga el modelo y el normalizador desde archivos .pkl."""
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    return model, scaler

def predict_from_args(args, model, scaler):
    """Toma una lista de argumentos (datos), los transforma y realiza predicciones."""
    data = np.array(args).reshape(1, -1)  # Convertir la lista de argumentos en una matriz 2D
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    
    return prediction

if __name__ == "__main__":
    # Las rutas a los archivos .pkl (puedes ajustar estas rutas según tu configuración)
    MODEL_PATH = 'optimal_rf_model.pkl'
    SCALER_PATH = 'scaler.pkl'
    
    # Cargar el modelo y el normalizador
    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    print("Argumentos recibidos:", sys.argv)
    # Asegurarse de que se han proporcionado los argumentos adecuados
    if len(sys.argv) < 9:  # Ajusta este número según el número de características que tu modelo espera
        print("Por favor, proporciona los argumentos adecuados para realizar predicciones.")
        sys.exit()
    
    input_args = [float(arg) for arg in sys.argv[1:]]
    prediction = predict_from_args(input_args, model, scaler)
    
    print("Predicción:", prediction[0])