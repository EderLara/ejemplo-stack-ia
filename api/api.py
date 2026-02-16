import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
import io

app = Flask(__name__)

# Cargar el modelo al iniciar la app
MODEL_PATH = './model/best_mnist_model.h5'
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Modelo cargado correctamente.")
else:
    print("ADVERTENCIA: No se encontró el modelo. Asegúrate de entrenarlo y guardarlo.")
    model = None

def prepare_image(image, target_size=(28, 28)):
    # Convertir a escala de grises si el modelo es MNIST (1 canal)
    # Si es para paneles solares (RGB), usar .convert('RGB') y asegurar shape (28,28,3)
    if image.mode != "L":
        image = image.convert("L") # Cambiar a "RGB" para paneles a color
    
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalización igual que en el entrenamiento
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró el archivo'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
    try:
        # Procesar imagen
        image = Image.open(io.BytesIO(file.read()))
        processed_image = prepare_image(image)
        
        # Predecir
        if model:
            prediction = model.predict(processed_image)
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            confidence = float(np.max(prediction))
            
            return jsonify({
                'class': predicted_class,
                'confidence': confidence,
                'probabilities': prediction.tolist()[0]
            })
        else:
            return jsonify({'error': 'Modelo no cargado'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
