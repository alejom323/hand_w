import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Configuración de página Streamlit
st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='wide')

# Estilos personalizados: fondo negro, texto azul cielo
st.markdown("""
    <style>
        body, .stApp {
            background-color: black;
            color: #87CEEB;
        }

        html, body, [class*="css"] {
            color: #87CEEB;
        }

        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: #87CEEB !important;
        }

        section[data-testid="stSidebar"] {
            background-color: black !important;
        }

        section[data-testid="stSidebar"] * {
            color: #87CEEB !important;
        }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.title('Reconocimiento de Dígitos escritos a mano')
st.subheader("Dibuja el dígito en el panel y presiona 'Predecir'")

# Parámetros del canvas
drawing_mode = "freedraw"
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'  # Color del trazo: blanco
bg_color = '#87CEEB'      # Fondo del canvas: azul cielo

# Canvas para dibujar el dígito
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Color de relleno con opacidad
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

# Botón para predecir
if st.button('Predecir'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)  # ✅ Corrección del nombre de la función
        st.header('El Dígito es : ' + str(res))
    else:
        st.header('Por favor dibuja en el canvas el dígito.')

# Barra lateral
st.sidebar.title("Acerca de:")
st.sidebar.text("En esta aplicación se evalúa ")
st.sidebar.text("la capacidad de un RNA de reconocer") 
st.sidebar.text("dígitos escritos a mano.")
st.sidebar.text("Basado en desarrollo de Vinay Uniyal")
#st.sidebar.text("GitHub Repository")
#st.sidebar.write("[GitHub Repo Link](https://github.com/Vinay2022/Handwritten-Digit-Recognition)")

# Función de predicción
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32')
    img = img / 255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result
