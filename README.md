# 🤖 Sistema de Clasificación de Imágenes con IA

Sistema completo de clasificación de imágenes utilizando Deep Learning, implementado con una arquitectura de microservicios que incluye una API REST (Flask) y una interfaz web interactiva (Streamlit).

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema de clasificación de imágenes basado en redes neuronales convolucionales (CNN). El sistema está diseñado con una arquitectura de microservicios que separa la lógica de predicción (API) de la interfaz de usuario (aplicación web), permitiendo escalabilidad y mantenimiento independiente de cada componente.

### Características principales

- 🧠 Modelo de Deep Learning entrenado para clasificación de imágenes (MNIST)
- 🔌 API REST con Flask para predicciones
- 🎨 Interfaz web interactiva con Streamlit
- 🐳 Despliegue con Docker y Docker Compose
- 📊 Visualización de resultados con niveles de confianza
- 🔄 Arquitectura de microservicios escalable

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────┐
│                    Usuario                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           Streamlit App (Puerto 8501)                    │
│  - Interfaz de usuario                                   │
│  - Carga de imágenes                                     │
│  - Visualización de resultados                           │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP POST
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Flask API (Puerto 5000)                       │
│  - Recepción de imágenes                                 │
│  - Preprocesamiento                                      │
│  - Predicción con modelo CNN                             │
│  - Respuesta JSON                                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Modelo TensorFlow/Keras                          │
│  - Red neuronal convolucional                            │
│  - Clasificación de imágenes                             │
└─────────────────────────────────────────────────────────┘
```

## 📁 Estructura del Proyecto

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
├── README.md                         # Este archivo
├── Guia de implementación.md         # Guía paso a paso
└── Manual de streamlit.md            # Manual completo de Streamlit
```

## 🚀 Inicio Rápido

### Prerrequisitos

- Docker y Docker Compose instalados
- Modelo entrenado (`best_mnist_model.h5`) en la carpeta `api/model/`

### Instalación y Ejecución

1. **Clonar el repositorio**
```bash
git clone <url-del-repositorio>
cd proyecto_ia
```

2. **Asegurarse de tener el modelo entrenado**
```bash
# El modelo debe estar en: api/model/best_mnist_model.h5
```

3. **Iniciar los servicios con Docker Compose**
```bash
docker-compose up --build
```

4. **Acceder a la aplicación**
- Interfaz web: http://localhost:8501
- API: http://localhost:5000

## 📖 Documentación

### 📘 [Guía de Implementación](Guia%20de%20implementación.md)

Guía completa paso a paso para implementar el sistema:
- Preparación del modelo
- Verificación de la estructura de archivos
- Despliegue con Docker y local
- Pruebas del sistema (Web y API)
- Solución de problemas comunes

### 📗 [Manual de Streamlit](Manual%20de%20streamlit.md)

Manual completo de Streamlit para ciencia de datos:
- Componentes básicos y avanzados
- Visualización de datos
- Conexión a bases de datos (SQL y NoSQL)
- Integración con APIs
- Creación de dashboards
- Machine Learning interactivo
- Estilos y UX
- Despliegue en producción

## 🔧 Uso del Sistema

### Interfaz Web (Streamlit)

1. Abre tu navegador en http://localhost:8501
2. Haz clic en "Elige una imagen..." para cargar una imagen
3. Haz clic en el botón "Clasificar"
4. Visualiza los resultados:
   - Clase predicha
   - Nivel de confianza (barra de progreso)
   - Porcentaje de confianza

### API REST (Flask)

#### Endpoint de predicción

**URL:** `POST http://localhost:5000/predict`

**Parámetros:**
- `file`: Archivo de imagen (form-data)

**Ejemplo con cURL:**
```bash
curl -X POST -F "file=@imagen.png" http://localhost:5000/predict
```

**Ejemplo con Python:**
```python
import requests

url = "http://localhost:5000/predict"
files = {'file': open('imagen.png', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

**Respuesta exitosa:**
```json
{
  "class": 7,
  "confidence": 0.9845,
  "probabilities": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.9845, 0.008, 0.009]
}
```

**Respuestas de error:**
```json
{"error": "No se encontró el archivo"}
{"error": "No se seleccionó ningún archivo"}
{"error": "Modelo no cargado"}
```

## 🛠️ Tecnologías Utilizadas

### Backend (API)
- **Python 3.9**
- **Flask** - Framework web para la API REST
- **TensorFlow/Keras** - Deep Learning y modelo CNN
- **Pillow** - Procesamiento de imágenes
- **NumPy** - Operaciones numéricas

### Frontend (Aplicación Web)
- **Streamlit** - Framework para aplicaciones web de datos
- **Requests** - Cliente HTTP para comunicación con la API
- **Pillow** - Manejo de imágenes

### Infraestructura
- **Docker** - Contenedorización de servicios
- **Docker Compose** - Orquestación de contenedores

## 🔍 Detalles Técnicos

### Modelo de Machine Learning

- **Arquitectura:** Red Neuronal Convolucional (CNN)
- **Dataset:** MNIST (dígitos escritos a mano)
- **Input:** Imágenes 28x28 píxeles en escala de grises
- **Output:** 10 clases (dígitos 0-9)
- **Formato:** HDF5 (.h5)

### Preprocesamiento de Imágenes

1. Conversión a escala de grises
2. Redimensionamiento a 28x28 píxeles
3. Normalización (valores entre 0 y 1)
4. Expansión de dimensiones para el modelo

### Comunicación entre Servicios

- **Protocolo:** HTTP
- **Formato:** JSON para respuestas, multipart/form-data para imágenes
- **Red:** Red interna de Docker (docker-compose)

## 📊 Métricas y Monitoreo

El sistema proporciona:
- Clase predicha (0-9)
- Nivel de confianza de la predicción
- Probabilidades para todas las clases
- Mensajes de error descriptivos en español

## 🐛 Solución de Problemas

### Error: "No se pudo conectar con la API"
- Verifica que el contenedor de la API esté corriendo: `docker ps`
- Revisa los logs: `docker-compose logs api`
- Asegúrate de que el puerto 5000 no esté ocupado

### Error: "Modelo no cargado"
- Verifica que el archivo `api/model/best_mnist_model.h5` exista
- Revisa que el modelo se haya entrenado correctamente
- Comprueba los permisos del archivo

### Error: "No se encontró el archivo"
- Asegúrate de enviar el archivo con la key `file` en el form-data
- Verifica que el formato de imagen sea JPG, PNG o JPEG

## 🔒 Seguridad

- Validación de tipos de archivo permitidos
- Manejo de errores y excepciones
- Mensajes de error sin información sensible
- Límites de tamaño de archivo (configurable)

## 🚦 Estado del Proyecto

✅ Funcionalidades implementadas:
- API REST funcional
- Interfaz web interactiva
- Despliegue con Docker
- Documentación completa

🔄 Mejoras futuras:
- Autenticación de usuarios
- Historial de predicciones
- Soporte para múltiples modelos
- Métricas de uso y monitoreo
- Tests automatizados

## 📝 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 👥 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Haz fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📧 Contacto

Para preguntas o sugerencias, por favor abre un issue en el repositorio.

---

**Desarrollado con ❤️ para la comunidad de Data Science**
