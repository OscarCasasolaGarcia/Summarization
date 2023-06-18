import streamlit as st
from st_on_hover_tabs import on_hover_tabs

from datetime import date
import base64
from PIL import Image

from spaCy import *
from bartt import *
from t55 import *
from nltkk import *

# from subplots import get_candlestick_plot
img2 = Image.open('image/bg_main.png')
img_nltk= Image.open('image/img_nltk.png')
img_alg_nltk= Image.open('image/nltk.png')
img_spacy= Image.open('image/spacy.png')
img_bart= Image.open('image/bart.png')
img_t5= Image.open('image/T5.png')
# img_nltk= Image.open('logos/nltk.png')
st.set_page_config(page_title = 'Smart Summarizer', page_icon = img2, layout="wide")

st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)


def set_bg_hack(main_bg):
    
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
    )

with st.sidebar:
    # tabs = on_hover_tabs(tabName=['Intro', 'NLTK', 'SpaCy', 'BART', 'T5'], 
    #                      iconName=['home', '', 'book', 'book', 'book'], default_choice=0)
    tabs = on_hover_tabs(tabName=['Intro', 'NLTK', 'SpaCy', 'BART', 'T5'],
                            iconName=['home', 'summarize', 'summarize', 'summarize', 'summarize'],
                            styles = {'navtab': {'background-color':'#111',
                                                'color': '#818181',
                                                'font-size': '18px',
                                                'transition': '.3s',
                                                'white-space': 'nowrap',
                                                'text-transform': 'uppercase'},
                                    'tabOptionsStyle': {':hover :hover': {'color': 'white',
                                                                    'cursor': 'pointer'}},
                                    'iconStyle':{'position':'fixed',
                                                'left':'7.5px',
                                                'text-align': 'left'},
                                    'tabStyle' : {'list-style-type': 'none',
                                                    'margin-bottom': '30px',
                                                    'padding-left': '30px'}},
                            default_choice=0)

if tabs =='Intro':
    st.title("Smart Summarizer")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img2)
    with col2:
        st.header('Introducción')
        st.markdown('<div style="text-align: justify;">Smart Summarizer es una herramienta que permite generar resumenes automáticos al instante utilizando diversas técnicas de NLP. Esto se logra mediante cuatro modelos preentrenados distintos, como lo son: BART, T5, Pegasus y SpaCy. Esta implementación está hecha mediante una sintaxis similar a lo que ofrece ChatGPT, o mediante un enlace que promorcione el usuario para hacer este resumen.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('¿Por qué usar un resumidor de textos automático?')
        st.markdown('<div style="text-align: justify;"> Un resumidor de textos (text summarization) es una técnica o algoritmo que tiene como objetivo reducir la longitud de un texto manteniendo la información más relevante y significativa. Su propósito es crear un resumen conciso y coherente que capture los aspectos clave del texto original. El resumidor de texto puede ser utilizado para procesar documentos largos, artículos, noticias, informes, páginas web y otros tipos de contenido textual extenso.</div>', unsafe_allow_html=True)
    with col2:
        st.image(img_nltk)

    st.markdown('<div style="text-align: justify;">Proporciona una forma eficiente de extraer la información esencial de un texto y presentarla de manera más breve, lo que facilita su comprensión y permite a los lectores obtener rápidamente una idea general del contenido sin tener que leer el texto completo. El desarrollo de resumidores de texto ha sido impulsado por los avances en el procesamiento de lenguaje natural y el aprendizaje automático. Estos sistemas utilizan algoritmos y modelos de lenguaje pre-entrenados para realizar la resumización de texto de manera más eficiente y precisa. En resumen, la resumización de texto implica crear un resumen preciso y conciso de un documento de texto extenso. El objetivo principal de la resumización automática de texto es seleccionar la información más importante y presentarla de forma comprensible. Con el crecimiento de los datos textuales en línea, los métodos automáticos de resumización de texto son cada vez más útiles, ya que permiten obtener información relevante en menos tiempo.</div>', unsafe_allow_html=True)
    
    st.subheader('Modelos utilizados en Smart Summarizer')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(img_alg_nltk)
    with col2:
        st.image(img_spacy)
    with col3:
        st.image(img_bart)
    with col4:
        st.image(img_t5)

elif tabs == 'NLTK':
    main_nltk()

elif tabs == 'SpaCy':
    main_spacy()

elif tabs == 'BART':
    main_bart()

elif tabs == 'T5':
    main_t5()
    