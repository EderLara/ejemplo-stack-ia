"""
Aplicación de Clasificación de Imágenes con IA usando Streamlit.

Esta aplicación web permite a los usuarios subir imágenes y obtener predicciones
de un modelo de inteligencia artificial a través de una API Flask.

Funcionalidades principales:
- Carga de imágenes (JPG, PNG, JPEG)
- Visualización de la imagen subida
- Envío de la imagen a la API para clasificación
- Muestra de resultados con clase predicha y nivel de confianza

Resultados posibles:
- Éxito: Muestra la clase predicha, barra de progreso y porcentaje de confianza
- Error de API: Mensaje de error si la API no puede procesar la imagen
- Error de conexión: Mensaje si no se puede conectar con la API
"""

import streamlit as st
import requests
from PIL import Image

# Configuración de la página de Streamlit
# Establece el título de la pestaña del navegador y el diseño centrado
st.set_page_config(page_title="Clasificador IA", layout="centered")

# Título principal y descripción de la aplicación
st.title("🔍 Clasificación de Imágenes con IA")
st.write("Sube una imagen para que el modelo la analice.")

# URL de la API Flask donde se enviarán las imágenes para clasificación
# Nota: Cambia a "http://api:5000/predict" si usas docker-compose
API_URL = "http://127.0.0.1:5000/predict"

# Widget de carga de archivos
# Permite al usuario seleccionar una imagen desde su dispositivo
# Formatos aceptados: JPG, PNG, JPEG
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png", "jpeg"])

# Verificar si el usuario ha subido un archivo
if uploaded_file is not None:
    # Abrir y mostrar la imagen subida
    # Esto permite al usuario confirmar que subió la imagen correcta
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_column_width=True)
    
    # Botón para iniciar el proceso de clasificación
    # Solo aparece cuando hay una imagen cargada
    if st.button('Clasificar'):
        # Mostrar un spinner mientras se procesa la imagen
        # Esto mejora la experiencia del usuario durante la espera
        with st.spinner('Analizando...'):
            try:
                # Preparar el archivo para enviarlo a la API
                # Se convierte a bytes para la transmisión HTTP
                files = {'file': uploaded_file.getvalue()}
                
                # Enviar la imagen a la API Flask mediante una petición POST
                # La API procesará la imagen y devolverá la predicción
                response = requests.post(API_URL, files=files)
                
                # Verificar si la API respondió exitosamente (código 200)
                if response.status_code == 200:
                    # Extraer los resultados de la respuesta JSON
                    result = response.json()
                    
                    # Mostrar la clase predicha (ej: dígito 0-9 para MNIST)
                    # Resultado: Número de clase identificado por el modelo
                    st.success(f"Predicción: Clase {result['class']}")
                    
                    # Barra de progreso visual que representa la confianza
                    # Resultado: Valor entre 0 y 1 mostrado como barra
                    st.progress(result['confidence'])
                    
                    # Mostrar el porcentaje de confianza de la predicción
                    # Resultado: Porcentaje que indica qué tan seguro está el modelo
                    st.info(f"Confianza: {result['confidence']*100:.2f}%")
                else:
                    # La API respondió pero con un error (código diferente de 200)
                    # Resultado: Mensaje de error y detalles de la respuesta
                    st.error("Error en la predicción de la API.")
                    st.write(response.text)
            except Exception as e:
                # Capturar cualquier error de conexión o excepción inesperada
                # Resultado: Mensaje de error indicando problema de conexión con la API
                st.error(f"No se pudo conectar con la API. Asegúrate de que esté corriendo. Error: {e}")
