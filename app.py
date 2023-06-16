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
# img_nltk= Image.open('logos/nltk.png')
st.set_page_config(page_title = 'Smart Summarizer', page_icon = img2, layout="wide")

st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)


# Function for Sumy Summarization
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

# Fetch Text From Url
@st.cache_data()
def get_text(raw_url):
	page = urlopen(raw_url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text

@st.cache_data()
def analyze_text(text):
	return nlp(text)



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
    tabs = on_hover_tabs(tabName=['Intro', 'NLTK', 'SpaCy', 'BART', 'T5', 'Pegasus'], 
                         iconName=['home', 'book', 'book', 'book', 'book', 'book'], default_choice=0)


if tabs =='Intro':
    col1, col2 = st.columns(2)
    with col1:
        st.image(img2)
    with col2:
        st.title("Smart Summarizer")
        st.header('Introducción')
        st.markdown('<div style="text-align: justify;">Smart Summarizer es una herramienta que permite generar resumenes automáticos al instante utilizando diversas técnicas de NLP. Esto se logra mediante cuatro modelos preentrenados distintos, como lo son: BART, T5, Pegasus y SpaCy. Esta implementación está hecha mediante una sintaxis similar a lo que ofrece ChatGPT, o mediante un enlace que promorcione el usuario para hacer este resumen.</div>', unsafe_allow_html=True)
    

    st.subheader('¿Por qué usar un resumidor de textos automático?')
    st.markdown('<div style="text-align: justify;"> Un resumidor de textos (text summarization) es una técnica o algoritmo que tiene como objetivo reducir la longitud de un texto manteniendo la información más relevante y significativa. Su propósito es crear un resumen conciso y coherente que capture los aspectos clave del texto original. El resumidor de texto puede ser utilizado para procesar documentos largos, artículos, noticias, informes, páginas web y otros tipos de contenido textual extenso. Proporciona una forma eficiente de extraer la información esencial de un texto y presentarla de manera más breve, lo que facilita su comprensión y permite a los lectores obtener rápidamente una idea general del contenido sin tener que leer el texto completo. El desarrollo de resumidores de texto ha sido impulsado por los avances en el procesamiento de lenguaje natural y el aprendizaje automático. Estos sistemas utilizan algoritmos y modelos de lenguaje pre-entrenados para realizar la resumización de texto de manera más eficiente y precisa. En resumen, la resumización de texto implica crear un resumen preciso y conciso de un documento de texto extenso. El objetivo principal de la resumización automática de texto es seleccionar la información más importante y presentarla de forma comprensible. Con el crecimiento de los datos textuales en línea, los métodos automáticos de resumización de texto son cada vez más útiles, ya que permiten obtener información relevante en menos tiempo.</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('¿Comó se realiza el análisis de sentimientos?')
        st.markdown('<div style="text-align: justify;">Se realiza mediante el uso de la libreria NLTK. El kit de herramientas de lenguaje natural (NLTK) es una biblioteca popular de Python para trabajar con datos de lenguaje humano. Proporciona una amplia gama de herramientas y recursos para tareas como tokenización, derivación, lematización, etiquetado de partes del discurso y análisis. Una de las características principales de NLTK es su extensa colección de corpus (grandes cuerpos de datos lingüísticos), que se pueden utilizar para entrenar y evaluar modelos de procesamiento de lenguaje natural. NLTK también incluye una gama de herramientas de preprocesamiento y visualización, así como interfaces para otras bibliotecas y herramientas como WordNet y Treebank. NLTK se usa ampliamente en investigación y educación, y también es una opción popular para proyectos de procesamiento de lenguaje natural en la industria. Está bien documentado y tiene una comunidad de usuarios grande y activa, lo que facilita encontrar ayuda y recursos en línea. En general, NLTK es una biblioteca poderosa y flexible que facilita el trabajo con datos de lenguaje humano en Python. Ya sea que sea un investigador, un estudiante o un desarrollador profesional, NLTK es un recurso valioso para cualquier proyecto de procesamiento de lenguaje natural.</div>', unsafe_allow_html=True)
    with col2:
        st.image(img_nltk)

elif tabs == 'NLTK':
    main_nltk()

elif tabs == 'SpaCy':
    main_spacy()

elif tabs == 'BART':
    main_bart()

elif tabs == 'T5':
    main_t5()

elif tabs == 'Pegasus':
     pass
    