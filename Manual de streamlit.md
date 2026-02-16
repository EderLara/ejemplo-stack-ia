# Manual Completo de Streamlit para Ciencia de Datos

## Tabla de Contenidos

1. [Introducción a Streamlit](#1-introducción-a-streamlit)
2. [Instalación y Configuración](#2-instalación-y-configuración)
3. [Componentes Básicos](#3-componentes-básicos)
4. [Visualización de Datos](#4-visualización-de-datos)
5. [Conexión a Bases de Datos](#5-conexión-a-bases-de-datos)
6. [Integración con APIs](#6-integración-con-apis)
7. [Creación de Dashboards](#7-creación-de-dashboards)
8. [Manejo de Estado y Sesiones](#8-manejo-de-estado-y-sesiones)
9. [Carga y Procesamiento de Archivos](#9-carga-y-procesamiento-de-archivos)
10. [Machine Learning en Streamlit](#10-machine-learning-en-streamlit)
11. [Estilos y UX](#11-estilos-y-ux)
12. [Despliegue y Producción](#12-despliegue-y-producción)

---

## 1. Introducción a Streamlit

Streamlit es un framework de Python de código abierto que permite crear aplicaciones web interactivas para ciencia de datos y machine learning de forma rápida y sencilla, sin necesidad de conocimientos profundos de desarrollo web.

### Ventajas de Streamlit

- Sintaxis simple y pythónica
- Actualización automática al guardar cambios
- Componentes interactivos integrados
- Excelente integración con librerías de ciencia de datos (Pandas, NumPy, Matplotlib, Plotly)
- Despliegue gratuito en Streamlit Cloud
- Comunidad activa y amplia documentación

### Casos de uso

- Dashboards de análisis de datos
- Aplicaciones de machine learning
- Herramientas de visualización interactiva
- Prototipos rápidos de aplicaciones
- Reportes dinámicos
- Herramientas internas de equipos de datos

---

## 2. Instalación y Configuración

### Instalación básica

```bash
pip install streamlit
```

### Instalación con dependencias adicionales

```bash
pip install streamlit pandas numpy matplotlib plotly scikit-learn
```


### Primera aplicación

Crea un archivo `app.py`:

```python
import streamlit as st

st.title("Mi primera app con Streamlit")
st.write("¡Hola, mundo!")
```

Ejecuta la aplicación:

```bash
streamlit run app.py
```

### Configuración de la página

```python
import streamlit as st

# Configuración debe ser la primera llamada de Streamlit
st.set_page_config(
    page_title="Mi Dashboard",
    page_icon="📊",
    layout="wide",  # "centered" o "wide"
    initial_sidebar_state="expanded",  # "auto", "expanded", "collapsed"
    menu_items={
        'Get Help': 'https://www.streamlit.io',
        'Report a bug': "https://github.com/tu-repo/issues",
        'About': "# Mi aplicación de análisis de datos"
    }
)
```

---

## 3. Componentes Básicos

### Texto y Markdown

```python
import streamlit as st

# Títulos y encabezados
st.title("Título principal")
st.header("Encabezado")
st.subheader("Subencabezado")

# Texto simple
st.text("Texto simple sin formato")
st.write("Texto con formato automático")

# Markdown
st.markdown("**Texto en negrita** y *cursiva*")
st.markdown("---")  # Línea horizontal

# Código
st.code("print('Hola mundo')", language="python")

# LaTeX
st.latex(r"\sum_{i=1}^{n} x_i^2")
```


### Widgets de entrada

```python
import streamlit as st

# Botón
if st.button("Haz clic aquí"):
    st.write("¡Botón presionado!")

# Checkbox
agree = st.checkbox("Acepto los términos")
if agree:
    st.write("Términos aceptados")

# Radio buttons
option = st.radio(
    "Selecciona una opción:",
    ["Opción 1", "Opción 2", "Opción 3"]
)

# Selectbox (dropdown)
choice = st.selectbox(
    "Elige un elemento:",
    ["Python", "JavaScript", "Java", "C++"]
)

# Multiselect
options = st.multiselect(
    "Selecciona múltiples opciones:",
    ["A", "B", "C", "D"],
    default=["A", "B"]
)

# Slider
age = st.slider("Selecciona tu edad:", 0, 100, 25)

# Slider de rango
values = st.slider(
    "Selecciona un rango:",
    0.0, 100.0, (25.0, 75.0)
)

# Text input
name = st.text_input("Ingresa tu nombre:")

# Text area
message = st.text_area("Escribe un mensaje:")

# Number input
number = st.number_input("Ingresa un número:", min_value=0, max_value=100, value=50)

# Date input
import datetime
date = st.date_input("Selecciona una fecha:", datetime.date.today())

# Time input
time = st.time_input("Selecciona una hora:", datetime.time(8, 45))

# File uploader
uploaded_file = st.file_uploader("Sube un archivo", type=["csv", "xlsx", "txt"])

# Color picker
color = st.color_picker("Elige un color:", "#00f900")
```

### Mensajes y alertas

```python
import streamlit as st

# Mensajes de información
st.info("Esto es un mensaje informativo")
st.success("¡Operación exitosa!")
st.warning("Advertencia: Revisa los datos")
st.error("Error: Algo salió mal")

# Excepciones
try:
    result = 10 / 0
except Exception as e:
    st.exception(e)
```


### Contenedores y layout

```python
import streamlit as st

# Columnas
col1, col2, col3 = st.columns(3)
with col1:
    st.header("Columna 1")
    st.write("Contenido 1")
with col2:
    st.header("Columna 2")
    st.write("Contenido 2")
with col3:
    st.header("Columna 3")
    st.write("Contenido 3")

# Columnas con proporciones diferentes
col1, col2 = st.columns([2, 1])  # col1 es el doble de ancho que col2

# Expander (contenido colapsable)
with st.expander("Ver más detalles"):
    st.write("Contenido oculto que se puede expandir")

# Container
with st.container():
    st.write("Esto está dentro de un contenedor")
    st.write("Puedes agrupar elementos")

# Sidebar
st.sidebar.title("Panel lateral")
st.sidebar.write("Contenido del sidebar")
option = st.sidebar.selectbox("Selecciona:", ["A", "B", "C"])

# Tabs
tab1, tab2, tab3 = st.tabs(["Datos", "Gráficos", "Análisis"])
with tab1:
    st.write("Contenido de la pestaña Datos")
with tab2:
    st.write("Contenido de la pestaña Gráficos")
with tab3:
    st.write("Contenido de la pestaña Análisis")
```

---

## 4. Visualización de Datos

### Mostrar DataFrames

```python
import streamlit as st
import pandas as pd
import numpy as np

# Crear datos de ejemplo
df = pd.DataFrame({
    'Columna 1': [1, 2, 3, 4],
    'Columna 2': [10, 20, 30, 40]
})

# Mostrar DataFrame
st.dataframe(df)

# DataFrame con estilo
st.dataframe(df.style.highlight_max(axis=0))

# Tabla estática
st.table(df)

# Métricas
st.metric(label="Temperatura", value="25°C", delta="1.2°C")

col1, col2, col3 = st.columns(3)
col1.metric("Ventas", "1,234", "+12%")
col2.metric("Usuarios", "5,678", "-3%")
col3.metric("Ingresos", "$45K", "+8%")
```


### Gráficos nativos de Streamlit

```python
import streamlit as st
import pandas as pd
import numpy as np

# Datos de ejemplo
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C']
)

# Line chart
st.line_chart(chart_data)

# Area chart
st.area_chart(chart_data)

# Bar chart
st.bar_chart(chart_data)

# Mapa
map_data = pd.DataFrame(
    np.random.randn(100, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon']
)
st.map(map_data)
```

### Gráficos con Matplotlib

```python
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Crear figura
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, np.cos(x), label='cos(x)')
ax.legend()
ax.set_title('Funciones trigonométricas')

# Mostrar en Streamlit
st.pyplot(fig)
```

### Gráficos con Plotly

```python
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Datos de ejemplo
df = px.data.iris()

# Scatter plot
fig = px.scatter(df, x='sepal_width', y='sepal_length', 
                 color='species', size='petal_length',
                 title='Iris Dataset')
st.plotly_chart(fig, use_container_width=True)

# Gráfico de barras interactivo
df_sales = pd.DataFrame({
    'Mes': ['Ene', 'Feb', 'Mar', 'Abr', 'May'],
    'Ventas': [100, 150, 120, 180, 200]
})
fig = px.bar(df_sales, x='Mes', y='Ventas', title='Ventas Mensuales')
st.plotly_chart(fig)

# Gráfico 3D
fig = go.Figure(data=[go.Scatter3d(
    x=df['sepal_length'],
    y=df['sepal_width'],
    z=df['petal_length'],
    mode='markers',
    marker=dict(size=5, color=df['petal_width'], colorscale='Viridis')
)])
fig.update_layout(title='Visualización 3D')
st.plotly_chart(fig)
```


### Gráficos con Altair

```python
import streamlit as st
import altair as alt
import pandas as pd

# Datos de ejemplo
df = pd.DataFrame({
    'x': range(100),
    'y': np.random.randn(100).cumsum()
})

# Gráfico de línea con Altair
chart = alt.Chart(df).mark_line().encode(
    x='x',
    y='y'
).properties(
    width=700,
    height=400,
    title='Serie temporal'
)
st.altair_chart(chart, use_container_width=True)

# Gráfico interactivo con selección
brush = alt.selection_interval()
chart = alt.Chart(df).mark_point().encode(
    x='x',
    y='y',
    color=alt.condition(brush, 'x:Q', alt.value('lightgray'))
).add_selection(brush)
st.altair_chart(chart)
```

---

## 5. Conexión a Bases de Datos

### SQLite

```python
import streamlit as st
import sqlite3
import pandas as pd

@st.cache_resource
def init_connection():
    return sqlite3.connect('database.db', check_same_thread=False)

conn = init_connection()

@st.cache_data(ttl=600)
def run_query(query):
    return pd.read_sql_query(query, conn)

# Ejecutar consulta
df = run_query("SELECT * FROM usuarios LIMIT 10")
st.dataframe(df)

# Insertar datos
def insert_data(name, email):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO usuarios (name, email) VALUES (?, ?)", (name, email))
    conn.commit()

name = st.text_input("Nombre:")
email = st.text_input("Email:")
if st.button("Guardar"):
    insert_data(name, email)
    st.success("Datos guardados correctamente")
```


### PostgreSQL

```python
import streamlit as st
import psycopg2
import pandas as pd

@st.cache_resource
def init_connection():
    return psycopg2.connect(
        host="localhost",
        database="mi_base_datos",
        user="usuario",
        password="contraseña",
        port=5432
    )

conn = init_connection()

@st.cache_data(ttl=600)
def load_data(query):
    return pd.read_sql_query(query, conn)

# Cargar datos
df = load_data("SELECT * FROM ventas WHERE fecha >= '2024-01-01'")
st.dataframe(df)

# Análisis agregado
query = """
    SELECT 
        DATE_TRUNC('month', fecha) as mes,
        SUM(monto) as total_ventas
    FROM ventas
    GROUP BY mes
    ORDER BY mes
"""
df_monthly = load_data(query)
st.line_chart(df_monthly.set_index('mes'))
```

### MySQL

```python
import streamlit as st
import mysql.connector
import pandas as pd

@st.cache_resource
def init_connection():
    return mysql.connector.connect(
        host="localhost",
        user="usuario",
        password="contraseña",
        database="mi_base_datos"
    )

conn = init_connection()

@st.cache_data(ttl=600)
def run_query(query):
    cursor = conn.cursor()
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    return pd.DataFrame(data, columns=columns)

df = run_query("SELECT * FROM productos")
st.dataframe(df)
```

### MongoDB

```python
import streamlit as st
from pymongo import MongoClient
import pandas as pd

@st.cache_resource
def init_connection():
    return MongoClient("mongodb://localhost:27017/")

client = init_connection()
db = client["mi_base_datos"]
collection = db["usuarios"]

@st.cache_data(ttl=600)
def load_data():
    items = collection.find()
    return pd.DataFrame(list(items))

df = load_data()
st.dataframe(df)

# Insertar documento
def insert_document(data):
    collection.insert_one(data)

if st.button("Agregar usuario"):
    new_user = {
        "nombre": "Juan",
        "edad": 30,
        "ciudad": "Madrid"
    }
    insert_document(new_user)
    st.success("Usuario agregado")
```


### Conexión con Streamlit Secrets

Para mayor seguridad, usa secrets en lugar de hardcodear credenciales:

Crea `.streamlit/secrets.toml`:

```toml
[postgres]
host = "localhost"
port = 5432
database = "mi_base_datos"
user = "usuario"
password = "contraseña"

[mongodb]
connection_string = "mongodb://localhost:27017/"
```

Úsalo en tu código:

```python
import streamlit as st
import psycopg2

@st.cache_resource
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])

conn = init_connection()
```

---

## 6. Integración con APIs

### Consumir API REST

```python
import streamlit as st
import requests
import pandas as pd

st.title("Consumo de API REST")

# API pública de ejemplo
@st.cache_data(ttl=3600)
def fetch_data(endpoint):
    response = requests.get(f"https://api.example.com/{endpoint}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code}")
        return None

# Obtener datos
data = fetch_data("users")
if data:
    df = pd.DataFrame(data)
    st.dataframe(df)

# POST request
def create_user(name, email):
    url = "https://api.example.com/users"
    payload = {"name": name, "email": email}
    response = requests.post(url, json=payload)
    return response.json()

with st.form("user_form"):
    name = st.text_input("Nombre")
    email = st.text_input("Email")
    submitted = st.form_submit_button("Crear usuario")
    
    if submitted:
        result = create_user(name, email)
        st.success(f"Usuario creado: {result}")
```


### API con autenticación

```python
import streamlit as st
import requests

# API con token
API_KEY = st.secrets["api_key"]  # Guardar en secrets.toml

@st.cache_data(ttl=600)
def fetch_protected_data():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.get("https://api.example.com/protected", headers=headers)
    return response.json()

data = fetch_protected_data()
st.json(data)

# OAuth 2.0 (ejemplo simplificado)
def get_access_token(client_id, client_secret):
    url = "https://oauth.example.com/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    response = requests.post(url, data=data)
    return response.json()["access_token"]
```

### Crear tu propia API con Streamlit

```python
import streamlit as st
from streamlit.components.v1 import html

# Aunque Streamlit no es para crear APIs, puedes exponer datos
# Mejor usar FastAPI o Flask para APIs y Streamlit para frontend

# Ejemplo de compartir datos mediante query params
query_params = st.experimental_get_query_params()
user_id = query_params.get("user_id", [""])[0]

if user_id:
    st.write(f"Mostrando datos para usuario: {user_id}")
    # Cargar datos específicos del usuario
```

---

## 7. Creación de Dashboards

### Dashboard completo de análisis de ventas

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuración de página
st.set_page_config(
    page_title="Dashboard de Ventas",
    page_icon="📊",
    layout="wide"
)

# Título
st.title("📊 Dashboard de Análisis de Ventas")

# Sidebar con filtros
st.sidebar.header("Filtros")
date_range = st.sidebar.date_input(
    "Rango de fechas:",
    value=(datetime.now() - timedelta(days=30), datetime.now())
)
region = st.sidebar.multiselect(
    "Región:",
    ["Norte", "Sur", "Este", "Oeste"],
    default=["Norte", "Sur", "Este", "Oeste"]
)
producto = st.sidebar.selectbox(
    "Producto:",
    ["Todos", "Producto A", "Producto B", "Producto C"]
)

# Cargar datos (simulados)
@st.cache_data
def load_sales_data():
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    df = pd.DataFrame({
        'fecha': dates,
        'ventas': np.random.randint(1000, 5000, len(dates)),
        'region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], len(dates)),
        'producto': np.random.choice(['Producto A', 'Producto B', 'Producto C'], len(dates))
    })
    return df

df = load_sales_data()

# Filtrar datos
df_filtered = df[
    (df['fecha'] >= pd.Timestamp(date_range[0])) &
    (df['fecha'] <= pd.Timestamp(date_range[1])) &
    (df['region'].isin(region))
]

if producto != "Todos":
    df_filtered = df_filtered[df_filtered['producto'] == producto]
```


# KPIs principales
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_ventas = df_filtered['ventas'].sum()
    st.metric("Ventas Totales", f"${total_ventas:,.0f}", "+12%")

with col2:
    promedio_ventas = df_filtered['ventas'].mean()
    st.metric("Promedio Diario", f"${promedio_ventas:,.0f}", "+5%")

with col3:
    max_ventas = df_filtered['ventas'].max()
    st.metric("Máximo", f"${max_ventas:,.0f}", "+8%")

with col4:
    num_transacciones = len(df_filtered)
    st.metric("Transacciones", f"{num_transacciones:,}", "+15%")

st.markdown("---")

# Gráficos
col1, col2 = st.columns(2)

with col1:
    st.subheader("Tendencia de Ventas")
    fig = px.line(df_filtered, x='fecha', y='ventas', 
                  title='Ventas en el tiempo')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Ventas por Región")
    ventas_region = df_filtered.groupby('region')['ventas'].sum().reset_index()
    fig = px.pie(ventas_region, values='ventas', names='region',
                 title='Distribución por región')
    st.plotly_chart(fig, use_container_width=True)

# Gráfico de barras
st.subheader("Ventas por Producto")
ventas_producto = df_filtered.groupby('producto')['ventas'].sum().reset_index()
fig = px.bar(ventas_producto, x='producto', y='ventas',
             title='Comparación de productos')
st.plotly_chart(fig, use_container_width=True)

# Tabla de datos
with st.expander("Ver datos detallados"):
    st.dataframe(df_filtered, use_container_width=True)
    
    # Botón de descarga
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name='ventas.csv',
        mime='text/csv'
    )
```

### Dashboard con múltiples páginas

Estructura de carpetas:
```
mi_dashboard/
├── Home.py
├── pages/
│   ├── 1_📊_Análisis.py
│   ├── 2_📈_Predicciones.py
│   └── 3_⚙️_Configuración.py
```

`Home.py`:
```python
import streamlit as st

st.set_page_config(page_title="Dashboard Principal", page_icon="🏠")

st.title("🏠 Dashboard Principal")
st.write("Bienvenido al sistema de análisis de datos")

st.markdown("""
### Secciones disponibles:
- 📊 Análisis: Visualización de datos históricos
- 📈 Predicciones: Modelos de machine learning
- ⚙️ Configuración: Ajustes del sistema
""")
```


`pages/1_📊_Análisis.py`:
```python
import streamlit as st
import pandas as pd

st.title("📊 Análisis de Datos")
st.write("Análisis detallado de métricas")

# Tu código de análisis aquí
```

---

## 8. Manejo de Estado y Sesiones

### Session State

```python
import streamlit as st

# Inicializar estado
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# Incrementar contador
if st.button("Incrementar"):
    st.session_state.counter += 1

st.write(f"Contador: {st.session_state.counter}")

# Guardar datos del usuario
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

name = st.text_input("Nombre:")
if st.button("Guardar nombre"):
    st.session_state.user_data['name'] = name
    st.success(f"Nombre guardado: {name}")

# Mostrar datos guardados
if st.session_state.user_data:
    st.write("Datos guardados:", st.session_state.user_data)
```

### Callbacks

```python
import streamlit as st

def increment_counter():
    st.session_state.counter += 1

def decrement_counter():
    st.session_state.counter -= 1

if 'counter' not in st.session_state:
    st.session_state.counter = 0

col1, col2 = st.columns(2)
with col1:
    st.button("➕ Incrementar", on_click=increment_counter)
with col2:
    st.button("➖ Decrementar", on_click=decrement_counter)

st.write(f"Valor: {st.session_state.counter}")
```

### Formularios con estado

```python
import streamlit as st

if 'submitted_data' not in st.session_state:
    st.session_state.submitted_data = []

with st.form("my_form"):
    name = st.text_input("Nombre")
    age = st.number_input("Edad", min_value=0, max_value=120)
    email = st.text_input("Email")
    
    submitted = st.form_submit_button("Enviar")
    
    if submitted:
        data = {"name": name, "age": age, "email": email}
        st.session_state.submitted_data.append(data)
        st.success("Datos enviados correctamente")

# Mostrar historial
if st.session_state.submitted_data:
    st.subheader("Datos enviados:")
    st.dataframe(pd.DataFrame(st.session_state.submitted_data))
```


---

## 9. Carga y Procesamiento de Archivos

### Cargar archivos CSV

```python
import streamlit as st
import pandas as pd

st.title("Carga y análisis de archivos CSV")

uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")

if uploaded_file is not None:
    # Leer CSV
    df = pd.read_csv(uploaded_file)
    
    st.success(f"Archivo cargado: {uploaded_file.name}")
    st.write(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    # Mostrar primeras filas
    st.subheader("Vista previa")
    st.dataframe(df.head())
    
    # Información del dataset
    st.subheader("Información del dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Tipos de datos:")
        st.write(df.dtypes)
    
    with col2:
        st.write("Valores nulos:")
        st.write(df.isnull().sum())
    
    # Estadísticas descriptivas
    st.subheader("Estadísticas descriptivas")
    st.dataframe(df.describe())
    
    # Filtrar columnas
    columns = st.multiselect("Selecciona columnas:", df.columns.tolist())
    if columns:
        st.dataframe(df[columns])
```

### Cargar archivos Excel

```python
import streamlit as st
import pandas as pd

uploaded_file = st.file_uploader("Sube un archivo Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Leer Excel
    excel_file = pd.ExcelFile(uploaded_file)
    
    # Seleccionar hoja
    sheet_name = st.selectbox("Selecciona una hoja:", excel_file.sheet_names)
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    
    st.dataframe(df)
```

### Cargar imágenes

```python
import streamlit as st
from PIL import Image
import numpy as np

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Abrir imagen
    image = Image.open(uploaded_file)
    
    # Mostrar imagen
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    # Información de la imagen
    st.write(f"Dimensiones: {image.size}")
    st.write(f"Formato: {image.format}")
    st.write(f"Modo: {image.mode}")
    
    # Convertir a array
    img_array = np.array(image)
    st.write(f"Shape del array: {img_array.shape}")
    
    # Aplicar filtros
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image)
    with col2:
        st.subheader("Escala de grises")
        gray_image = image.convert('L')
        st.image(gray_image)
```


### Cargar múltiples archivos

```python
import streamlit as st
import pandas as pd

uploaded_files = st.file_uploader(
    "Sube múltiples archivos CSV",
    type="csv",
    accept_multiple_files=True
)

if uploaded_files:
    all_dfs = []
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        df['source_file'] = uploaded_file.name
        all_dfs.append(df)
    
    # Combinar todos los DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    st.write(f"Total de archivos: {len(uploaded_files)}")
    st.write(f"Total de filas: {len(combined_df)}")
    st.dataframe(combined_df)
```

### Descargar archivos

```python
import streamlit as st
import pandas as pd

# Crear datos de ejemplo
df = pd.DataFrame({
    'Columna1': [1, 2, 3, 4, 5],
    'Columna2': ['A', 'B', 'C', 'D', 'E']
})

# Descargar CSV
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Descargar CSV",
    data=csv,
    file_name='datos.csv',
    mime='text/csv'
)

# Descargar Excel
import io
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Datos', index=False)
    
st.download_button(
    label="Descargar Excel",
    data=buffer.getvalue(),
    file_name='datos.xlsx',
    mime='application/vnd.ms-excel'
)

# Descargar JSON
json_str = df.to_json(orient='records', indent=2)
st.download_button(
    label="Descargar JSON",
    data=json_str,
    file_name='datos.json',
    mime='application/json'
)
```

---

## 10. Machine Learning en Streamlit

### Aplicación de clasificación

```python
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px

st.title("🤖 Clasificador de Iris")

# Cargar datos
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df, iris

df, iris = load_data()

# Sidebar - Parámetros del modelo
st.sidebar.header("Parámetros del modelo")
test_size = st.sidebar.slider("Tamaño del conjunto de prueba", 0.1, 0.5, 0.2)
n_estimators = st.sidebar.slider("Número de árboles", 10, 200, 100)
max_depth = st.sidebar.slider("Profundidad máxima", 1, 20, 5)

# Entrenar modelo
@st.cache_resource
def train_model(test_size, n_estimators, max_depth):
    X = df[iris.feature_names]
    y = df['species']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X_test, y_test, y_pred

model, accuracy, X_test, y_test, y_pred = train_model(test_size, n_estimators, max_depth)

# Mostrar métricas
st.subheader("Rendimiento del modelo")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Muestras de entrenamiento", len(df) - len(X_test))
col3.metric("Muestras de prueba", len(X_test))
```


# Predicción interactiva
st.subheader("Hacer una predicción")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Hacer predicción
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]

species_names = ['setosa', 'versicolor', 'virginica']
predicted_species = species_names[prediction]

st.success(f"Predicción: {predicted_species}")

# Mostrar probabilidades
prob_df = pd.DataFrame({
    'Especie': species_names,
    'Probabilidad': prediction_proba
})
fig = px.bar(prob_df, x='Especie', y='Probabilidad', 
             title='Probabilidades de predicción')
st.plotly_chart(fig)

# Importancia de características
st.subheader("Importancia de características")
feature_importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

fig = px.bar(feature_importance, x='Importance', y='Feature', 
             orientation='h', title='Importancia de características')
st.plotly_chart(fig)
```

### Aplicación de regresión

```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go

st.title("📈 Regresión Polinomial")

# Generar datos
@st.cache_data
def generate_data(n_points, noise):
    np.random.seed(42)
    X = np.linspace(0, 10, n_points)
    y = 2 * X + 3 + np.random.normal(0, noise, n_points)
    return X.reshape(-1, 1), y

# Parámetros
n_points = st.sidebar.slider("Número de puntos", 10, 200, 50)
noise = st.sidebar.slider("Ruido", 0.0, 5.0, 1.0)
degree = st.sidebar.slider("Grado del polinomio", 1, 10, 2)

X, y = generate_data(n_points, noise)

# Entrenar modelo
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# Predicciones
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
X_test_poly = poly.transform(X_test)
y_pred = model.predict(X_test_poly)

# Visualizar
fig = go.Figure()
fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', 
                         name='Datos', marker=dict(size=8)))
fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_pred, mode='lines',
                         name=f'Regresión (grado {degree})', 
                         line=dict(width=3)))
fig.update_layout(title='Regresión Polinomial', 
                  xaxis_title='X', yaxis_title='y')
st.plotly_chart(fig, use_container_width=True)

# Métricas
from sklearn.metrics import r2_score, mean_squared_error
y_train_pred = model.predict(X_poly)
r2 = r2_score(y, y_train_pred)
mse = mean_squared_error(y, y_train_pred)

col1, col2 = st.columns(2)
col1.metric("R² Score", f"{r2:.4f}")
col2.metric("MSE", f"{mse:.4f}")
```


### Clustering interactivo

```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import plotly.express as px

st.title("🎯 Clustering con K-Means")

# Generar datos
@st.cache_data
def generate_clusters(n_samples, n_features, centers, cluster_std):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                      centers=centers, cluster_std=cluster_std, random_state=42)
    return X, y

# Parámetros
st.sidebar.header("Parámetros de datos")
n_samples = st.sidebar.slider("Número de muestras", 100, 1000, 300)
n_features = st.sidebar.slider("Número de características", 2, 5, 2)
true_centers = st.sidebar.slider("Centros reales", 2, 10, 3)
cluster_std = st.sidebar.slider("Desviación estándar", 0.5, 3.0, 1.0)

X, y_true = generate_clusters(n_samples, n_features, true_centers, cluster_std)

# K-Means
st.sidebar.header("Parámetros de K-Means")
n_clusters = st.sidebar.slider("Número de clusters (k)", 2, 10, 3)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_pred = kmeans.fit_predict(X)

# Visualización (solo para 2D)
if n_features == 2:
    df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
    df['Cluster'] = y_pred.astype(str)
    df['True Label'] = y_true.astype(str)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Clusters predichos")
        fig = px.scatter(df, x='Feature 1', y='Feature 2', color='Cluster',
                        title='K-Means Clustering')
        # Agregar centroides
        centers = kmeans.cluster_centers_
        fig.add_scatter(x=centers[:, 0], y=centers[:, 1], 
                       mode='markers', marker=dict(size=15, symbol='x', color='black'),
                       name='Centroides')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Etiquetas reales")
        fig = px.scatter(df, x='Feature 1', y='Feature 2', color='True Label',
                        title='Distribución real')
        st.plotly_chart(fig, use_container_width=True)

# Método del codo
st.subheader("Método del codo")
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    kmeans_temp.fit(X)
    inertias.append(kmeans_temp.inertia_)

elbow_df = pd.DataFrame({'K': list(K_range), 'Inertia': inertias})
fig = px.line(elbow_df, x='K', y='Inertia', markers=True,
              title='Método del codo para determinar K óptimo')
st.plotly_chart(fig, use_container_width=True)

# Métricas
from sklearn.metrics import silhouette_score
silhouette = silhouette_score(X, y_pred)
st.metric("Silhouette Score", f"{silhouette:.4f}")
```

---

## 11. Estilos y UX

### Personalización con CSS

```python
import streamlit as st

# CSS personalizado
st.markdown("""
<style>
    /* Cambiar color de fondo */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* Estilo para títulos */
    h1 {
        color: #1f77b4;
        font-family: 'Arial', sans-serif;
        text-align: center;
    }
    
    /* Estilo para métricas */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #2e7d32;
    }
    
    /* Botones personalizados */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-size: 16px;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #145a8c;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #e8eaf6;
    }
</style>
""", unsafe_allow_html=True)

st.title("Aplicación con estilos personalizados")
```


### Temas personalizados

Crea `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### HTML personalizado

```python
import streamlit as st

# Tarjetas personalizadas
st.markdown("""
<div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; margin: 10px 0;">
    <h3 style="color: #1976d2; margin: 0;">📊 Tarjeta informativa</h3>
    <p style="margin: 10px 0 0 0;">Contenido de la tarjeta con estilo personalizado</p>
</div>
""", unsafe_allow_html=True)

# Badges
st.markdown("""
<span style="background-color: #4caf50; color: white; padding: 5px 10px; 
             border-radius: 5px; font-size: 12px;">ACTIVO</span>
<span style="background-color: #f44336; color: white; padding: 5px 10px; 
             border-radius: 5px; font-size: 12px; margin-left: 5px;">CRÍTICO</span>
""", unsafe_allow_html=True)

# Alertas personalizadas
def custom_alert(message, alert_type="info"):
    colors = {
        "info": "#2196f3",
        "success": "#4caf50",
        "warning": "#ff9800",
        "error": "#f44336"
    }
    st.markdown(f"""
    <div style="background-color: {colors[alert_type]}; color: white; 
                padding: 15px; border-radius: 5px; margin: 10px 0;">
        {message}
    </div>
    """, unsafe_allow_html=True)

custom_alert("Esta es una alerta informativa", "info")
custom_alert("¡Operación exitosa!", "success")
custom_alert("Advertencia: Revisa los datos", "warning")
custom_alert("Error: Algo salió mal", "error")
```

### Componentes personalizados

```python
import streamlit as st
import streamlit.components.v1 as components

# Componente HTML personalizado
def progress_bar(value, max_value, label):
    percentage = (value / max_value) * 100
    html = f"""
    <div style="margin: 20px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span>{label}</span>
            <span>{value}/{max_value}</span>
        </div>
        <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px;">
            <div style="background-color: #4caf50; width: {percentage}%; 
                        height: 100%; border-radius: 10px; transition: width 0.3s;">
            </div>
        </div>
    </div>
    """
    components.html(html, height=80)

progress_bar(75, 100, "Progreso del proyecto")
progress_bar(45, 60, "Tareas completadas")
```

### Animaciones y transiciones

```python
import streamlit as st
import time

# Spinner personalizado
with st.spinner('Procesando datos...'):
    time.sleep(2)
st.success('¡Completado!')

# Progress bar
progress_text = "Operación en progreso..."
my_bar = st.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(0.01)
    my_bar.progress(percent_complete + 1, text=progress_text)

my_bar.empty()
st.success("¡Proceso completado!")

# Animación de carga
placeholder = st.empty()
for i in range(5):
    placeholder.info(f"Cargando... {i+1}/5")
    time.sleep(0.5)
placeholder.success("¡Datos cargados!")
```


### Mejores prácticas de UX

```python
import streamlit as st

# 1. Feedback inmediato
if st.button("Guardar datos"):
    with st.spinner("Guardando..."):
        time.sleep(1)  # Simular operación
    st.success("✅ Datos guardados correctamente")
    st.balloons()  # Animación de celebración

# 2. Validación de entrada
email = st.text_input("Email:")
if email and "@" not in email:
    st.error("❌ Por favor ingresa un email válido")

# 3. Tooltips informativos
st.text_input("Nombre de usuario:", 
              help="El nombre de usuario debe tener entre 3 y 20 caracteres")

# 4. Estados de carga
@st.cache_data
def load_heavy_data():
    time.sleep(2)
    return pd.DataFrame({'data': range(1000)})

with st.spinner("Cargando datos..."):
    data = load_heavy_data()
st.success("Datos cargados")

# 5. Confirmación de acciones críticas
if st.button("Eliminar todos los datos", type="primary"):
    if 'confirm_delete' not in st.session_state:
        st.session_state.confirm_delete = True
        st.warning("⚠️ ¿Estás seguro? Haz clic nuevamente para confirmar")
    else:
        st.error("Datos eliminados")
        del st.session_state.confirm_delete

# 6. Navegación clara
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a:", ["Inicio", "Análisis", "Configuración"])

# 7. Mensajes de error amigables
try:
    result = some_function()
except Exception as e:
    st.error("😕 Algo salió mal. Por favor intenta nuevamente.")
    with st.expander("Ver detalles técnicos"):
        st.code(str(e))

# 8. Indicadores visuales
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Usuarios activos", "1,234", "+12%", delta_color="normal")
with col2:
    st.metric("Tasa de error", "2.3%", "-0.5%", delta_color="inverse")
with col3:
    st.metric("Tiempo de respuesta", "145ms", "+5ms", delta_color="off")
```

### Layout responsivo

```python
import streamlit as st

# Adaptar layout según el contenido
def responsive_layout(items):
    if len(items) <= 2:
        cols = st.columns(len(items))
    elif len(items) <= 4:
        cols = st.columns(2)
    else:
        cols = st.columns(3)
    
    for idx, item in enumerate(items):
        with cols[idx % len(cols)]:
            st.write(item)

items = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"]
responsive_layout(items)

# Container con scroll
st.markdown("""
<div style="height: 300px; overflow-y: scroll; border: 1px solid #ddd; 
            padding: 10px; border-radius: 5px;">
""" + "<br>".join([f"Línea {i}" for i in range(50)]) + """
</div>
""", unsafe_allow_html=True)
```

### Iconos y emojis

```python
import streamlit as st

# Emojis en títulos
st.title("📊 Dashboard de Análisis")
st.header("🔍 Exploración de Datos")
st.subheader("📈 Tendencias")

# Iconos en botones y mensajes
if st.button("🚀 Iniciar análisis"):
    st.success("✅ Análisis completado")

# Tabs con iconos
tab1, tab2, tab3 = st.tabs(["📊 Datos", "📈 Gráficos", "⚙️ Config"])

# Métricas con iconos
col1, col2, col3 = st.columns(3)
col1.metric("💰 Ingresos", "$45,231")
col2.metric("👥 Usuarios", "1,234")
col3.metric("⭐ Rating", "4.8/5")
```

---

## 12. Despliegue y Producción

### Despliegue en Streamlit Cloud

1. Sube tu código a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. Selecciona el archivo principal (app.py)
5. Haz clic en "Deploy"

Estructura recomendada:
```
mi_app/
├── app.py
├── requirements.txt
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml (no subir a GitHub)
└── README.md
```


### Despliegue con Docker

`Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

`docker-compose.yml`:
```yaml
version: '3.8'

services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - STREAMLIT_SERVER_PORT=8501
```

Comandos:
```bash
docker build -t mi-app-streamlit .
docker run -p 8501:8501 mi-app-streamlit
```

### Despliegue en Heroku

`Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

`setup.sh`:
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

Comandos:
```bash
heroku login
heroku create mi-app-streamlit
git push heroku main
```

### Optimización de rendimiento

```python
import streamlit as st

# 1. Usar cache para datos
@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_data():
    # Operación costosa
    return pd.read_csv("large_file.csv")

# 2. Cache para recursos
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

# 3. Lazy loading
if st.button("Cargar datos"):
    data = load_data()
    st.dataframe(data)

# 4. Pagination para grandes datasets
def paginate_dataframe(df, page_size=50):
    total_pages = len(df) // page_size + 1
    page = st.number_input("Página", 1, total_pages, 1)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    return df.iloc[start_idx:end_idx]

# 5. Usar session state eficientemente
if 'data' not in st.session_state:
    st.session_state.data = load_data()

# 6. Evitar re-renders innecesarios
with st.form("my_form"):
    input1 = st.text_input("Input 1")
    input2 = st.text_input("Input 2")
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        # Procesar solo cuando se envía el formulario
        process_data(input1, input2)
```

### Seguridad y mejores prácticas

```python
import streamlit as st

# 1. Usar secrets para credenciales
# .streamlit/secrets.toml
# [database]
# username = "admin"
# password = "secret123"

db_username = st.secrets["database"]["username"]
db_password = st.secrets["database"]["password"]

# 2. Validar inputs del usuario
user_input = st.text_input("Ingresa un número:")
try:
    number = int(user_input)
    if number < 0 or number > 100:
        st.error("El número debe estar entre 0 y 100")
except ValueError:
    st.error("Por favor ingresa un número válido")

# 3. Sanitizar datos
import html
user_text = st.text_input("Comentario:")
safe_text = html.escape(user_text)

# 4. Limitar tamaño de archivos
uploaded_file = st.file_uploader("Sube un archivo", type=["csv"])
if uploaded_file is not None:
    if uploaded_file.size > 10 * 1024 * 1024:  # 10 MB
        st.error("El archivo es demasiado grande (máximo 10 MB)")
    else:
        # Procesar archivo
        pass

# 5. Rate limiting (ejemplo básico)
import time
if 'last_request' not in st.session_state:
    st.session_state.last_request = 0

if st.button("Hacer petición"):
    current_time = time.time()
    if current_time - st.session_state.last_request < 5:
        st.warning("Por favor espera 5 segundos entre peticiones")
    else:
        st.session_state.last_request = current_time
        # Hacer petición
        st.success("Petición realizada")
```


### Monitoreo y logging

```python
import streamlit as st
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Registrar eventos
def log_event(event_type, details):
    logging.info(f"{event_type}: {details}")

# Ejemplo de uso
if st.button("Procesar datos"):
    log_event("USER_ACTION", "Usuario inició procesamiento de datos")
    try:
        # Procesar datos
        result = process_data()
        log_event("SUCCESS", f"Datos procesados: {result}")
        st.success("Datos procesados correctamente")
    except Exception as e:
        log_event("ERROR", f"Error al procesar datos: {str(e)}")
        st.error("Error al procesar datos")

# Analytics básico
if 'page_views' not in st.session_state:
    st.session_state.page_views = 0
st.session_state.page_views += 1

if 'user_actions' not in st.session_state:
    st.session_state.user_actions = []

def track_action(action_name):
    st.session_state.user_actions.append({
        'action': action_name,
        'timestamp': datetime.now()
    })

# Mostrar estadísticas (solo para admin)
if st.sidebar.checkbox("Mostrar estadísticas"):
    st.sidebar.metric("Vistas de página", st.session_state.page_views)
    st.sidebar.write(f"Acciones: {len(st.session_state.user_actions)}")
```

### Testing

```python
# test_app.py
import pytest
from streamlit.testing.v1 import AppTest

def test_app_loads():
    at = AppTest.from_file("app.py")
    at.run()
    assert not at.exception

def test_button_click():
    at = AppTest.from_file("app.py")
    at.run()
    at.button[0].click()
    at.run()
    assert at.success[0].value == "¡Operación exitosa!"

def test_input_validation():
    at = AppTest.from_file("app.py")
    at.run()
    at.text_input[0].set_value("test@email.com")
    at.run()
    assert not at.error
```

---

## Recursos Adicionales

### Librerías útiles para Streamlit

```python
# Componentes adicionales
pip install streamlit-aggrid  # Tablas interactivas avanzadas
pip install streamlit-option-menu  # Menús personalizados
pip install streamlit-extras  # Componentes extra
pip install streamlit-lottie  # Animaciones Lottie
pip install streamlit-card  # Tarjetas personalizadas
pip install plotly  # Gráficos interactivos
pip install altair  # Visualizaciones declarativas
```

### Ejemplo con AG Grid

```python
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
import pandas as pd

df = pd.DataFrame({
    'Nombre': ['Ana', 'Juan', 'María'],
    'Edad': [25, 30, 28],
    'Ciudad': ['Madrid', 'Barcelona', 'Valencia']
})

gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_pagination(paginationAutoPageSize=True)
gb.configure_side_bar()
gb.configure_selection('multiple', use_checkbox=True)
gridOptions = gb.build()

AgGrid(
    df,
    gridOptions=gridOptions,
    enable_enterprise_modules=True,
    theme='streamlit'
)
```

### Ejemplo con Lottie

```python
import streamlit as st
from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
lottie_json = load_lottieurl(lottie_url)
st_lottie(lottie_json, height=300)
```

---

## Conclusión

Streamlit es una herramienta poderosa para crear aplicaciones de ciencia de datos de forma rápida y eficiente. Este manual cubre:

- Componentes básicos y avanzados
- Visualización de datos con múltiples librerías
- Conexión a bases de datos (SQL y NoSQL)
- Integración con APIs REST
- Creación de dashboards interactivos
- Machine learning interactivo
- Personalización de estilos y UX
- Despliegue en producción

### Próximos pasos

1. Practica con proyectos pequeños
2. Explora la [documentación oficial](https://docs.streamlit.io)
3. Únete a la [comunidad de Streamlit](https://discuss.streamlit.io)
4. Revisa ejemplos en [Streamlit Gallery](https://streamlit.io/gallery)
5. Contribuye con componentes personalizados

### Enlaces útiles

- Documentación oficial: https://docs.streamlit.io
- GitHub: https://github.com/streamlit/streamlit
- Foro de la comunidad: https://discuss.streamlit.io
- Galería de apps: https://streamlit.io/gallery
- Cheat sheet: https://docs.streamlit.io/library/cheatsheet

---

**¡Feliz desarrollo con Streamlit! 🎈**
