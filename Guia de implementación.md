# Guía de Implementación del Sistema de Clasificación

## Estructura del Proyecto

```
proyecto_ia/
├── api/                              # Servicio de API Flask
│   ├── model/                        # Modelos entrenados
│   │   └── best_mnist_model.h5      # Modelo de clasificación
│   ├── api.py                        # Código principal de la API
│   ├── Dockerfile.api                # Configuración Docker para API
│   └── requirements.txt              # Dependencias de la API
├── app/                              # Aplicación web Streamlit
│   ├── app.py                        # Interfaz de usuario
│   ├── Dockerfile.streamlit          # Configuración Docker para Streamlit
│   └── requirements.txt              # Dependencias de Streamlit
├── docker-compose.yml                # Orquestación de servicios
└── Guia de implementación.md         # Este archivo
```

## Paso 1: Preparar el Modelo

1. Abre el cuaderno `Red_CNN_Optimizada.ipynb` en Google Colab o Jupyter.
2. Ejecuta todas las celdas para entrenar la red neuronal.
3. Al final, ejecuta la función `guardar_modelo()` para exportar el modelo.
4. Descarga el archivo generado y colócalo en `api/model/best_mnist_model.h5`.

## Paso 2: Verificar la Estructura de Archivos

Asegúrate de que tu proyecto tenga la siguiente estructura:

### Carpeta `api/`
- `api.py` - API Flask que recibe imágenes y devuelve predicciones
- `Dockerfile.api` - Configuración del contenedor para la API
- `requirements.txt` - Librerías necesarias (Flask, TensorFlow, Pillow, etc.)
- `model/best_mnist_model.h5` - Modelo entrenado

### Carpeta `app/`
- `app.py` - Interfaz web con Streamlit
- `Dockerfile.streamlit` - Configuración del contenedor para Streamlit
- `requirements.txt` - Librerías necesarias (Streamlit, Requests, Pillow)

### Raíz del proyecto
- `docker-compose.yml` - Orquestación de los servicios API y Streamlit

## Paso 3: Despliegue con Docker

### Opción A: Usando Docker Compose (Recomendado)

1. Abre una terminal en la raíz del proyecto.
2. Asegúrate de tener Docker instalado y corriendo.
3. Ejecuta el comando:

```bash
docker-compose up --build
```

Docker descargará las librerías, construirá los contenedores y conectará la red interna entre servicios.

### Opción B: Ejecución Local (Sin Docker)

Si prefieres ejecutar sin Docker:

1. **Iniciar la API:**
```bash
cd api
pip install -r requirements.txt
python api.py
```

2. **Iniciar Streamlit (en otra terminal):**
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

Nota: Asegúrate de que la URL de la API en `app.py` sea `http://127.0.0.1:5000/predict`

## Paso 4: Pruebas

### Vía Web (Streamlit)

1. Abre tu navegador y ve a `http://localhost:8501`
2. Verás la interfaz de Streamlit con el título "🔍 Clasificación de Imágenes con IA"
3. Sube una imagen usando el botón "Elige una imagen..."
4. Haz clic en "Clasificar"
5. Verás:
   - La clase predicha (ej: "Predicción: Clase 7")
   - Una barra de progreso visual
   - El porcentaje de confianza (ej: "Confianza: 98.45%")

### Vía API (Postman o cURL)

#### Usando Postman:
- **Método:** POST
- **URL:** `http://localhost:5000/predict`
- **Body:** selecciona `form-data`
  - **Key:** `file` (tipo File)
  - **Value:** (selecciona tu imagen)
- Dale a "Send" y recibirás un JSON con la clasificación

#### Usando cURL:
```bash
curl -X POST -F "file=@ruta/a/tu/imagen.png" http://localhost:5000/predict
```

#### Respuesta esperada:
```json
{
  "class": 7,
  "confidence": 0.9845,
  "probabilities": [0.001, 0.002, ..., 0.9845, ...]
}
```

## Paso 5: Detener los Servicios

Para detener los contenedores de Docker:

```bash
docker-compose down
```

## Solución de Problemas

### Error: "No se pudo conectar con la API"
- Verifica que el contenedor de la API esté corriendo: `docker ps`
- Revisa los logs: `docker-compose logs api`
- Asegúrate de que el puerto 5000 no esté ocupado

### Error: "Modelo no cargado"
- Verifica que el archivo `api/model/best_mnist_model.h5` exista
- Revisa que el modelo se haya entrenado correctamente

### Error: "No se encontró el archivo" o "No se seleccionó ningún archivo"
- Asegúrate de enviar el archivo con la key `file` en el form-data
- Verifica que el formato de imagen sea JPG, PNG o JPEG

## Notas Adicionales

- La API corre en el puerto 5000
- Streamlit corre en el puerto 8501
- Los servicios se comunican a través de una red interna de Docker
- Las imágenes se normalizan automáticamente a 28x28 píxeles en escala de grises