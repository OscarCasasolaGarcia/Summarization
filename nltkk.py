from bs4 import BeautifulSoup #scraping content from website
import re # Regular expression
import requests # sending requests
import heapq # finding largest values

# sent_tokenize = valorizador de oraciones.
# word_tokenize = valorizador de palabras.
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords # stopwords = palabras reservadas.
import streamlit as st

def clean(text):
    text = re.sub(r"\[[0-9]*\]"," ",text)
    text = text.lower()
    text = re.sub(r'\s+'," ",text)
    text = re.sub(r","," ",text)
    return text

def get_key(val):
   for key, value in val.items():
      if val == value:
          return key

def main_nltk():
    st.subheader("Resumidor automático de textos utilizando NLTK")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    number = st.number_input("Ingresa el número de sentencias que quieres en el resumen...",min_value=1)
    option = st.selectbox('¿De qué fuente es tu texto?',('URL', 'Texto Plano'))
    if option == 'URL':
        url = st.text_input("Ingresa la URL del texto a resumir","https://en.wikipedia.org/wiki/Natural_language_processing")
        if st.button("Iniciar Resumen..."):
            res = requests.get(url,headers=headers)
            src_text = ""
            soup = BeautifulSoup(res.text,'html.parser')
            content = soup.findAll("p")
            for text in content:
                src_text +=text.text
            src_text = clean(src_text)

            ##Tokenixing
            sent_tokens = sent_tokenize(src_text)
            summary = re.sub(r"[^a-zA-z]"," ",src_text)
            word_tokens = word_tokenize(summary)
            
            ## Removing Stop words
            word_frequency = {}
            stopwordss =  set(stopwords.words("english"))
            
            ## Creating word Tokens and frequency
            for word in word_tokens:
                if word not in stopwordss:
                    if word not in word_frequency.keys():
                        word_frequency[word]=1
                    else:
                        word_frequency[word] +=1
            
            maximum_frequency = max(word_frequency.values())
            # st.info("La frecuencia máxima es: "+str(maximum_frequency))
            st.subheader("Frecuencia de palabras")
            
            for word in word_frequency.keys():
                word_frequency[word] = (word_frequency[word]/maximum_frequency)
            
            # st.text("Word Frequency: "+str(word_frequency))

            # En un dataframe, se crea una columna con las palabras y otra con la frecuencia, se ordena de mayor a menor, y se grafica con st.bar_chart
            # Crear el DataFrame
            import pandas as pd
            chart_data = pd.DataFrame(word_frequency.items(), columns=['Palabra', 'Frecuencia'])

            # Mostrar el DataFrame en Streamlit de mayor a menor
            st.data_editor(
                chart_data,
                column_config={
                    "Palabra": {
                        "editable": False,
                    },
                    "Frecuencia": st.column_config.ProgressColumn(
                        "Frecuencia",
                        # help="This is a help text",
                        format="",
                    ),
                },
                use_container_width=True,
                # hide_index=True,
            )

            # Graficar el DataFrame en Plotly, de mayor a menor y de rojo a azul
            import plotly.express as px
            fig = px.bar(chart_data.sort_values(by=['Frecuencia'], ascending=False), x="Palabra", y="Frecuencia", color='Frecuencia', color_continuous_scale='RdBu_r')
            st.plotly_chart(fig)
            
            ## Creating Sentence score
            sentences_score = {}
            for sentence in sent_tokens: 
                for word in word_tokenize(sentence):
                    if word in word_frequency.keys():
                        if (len(sentence.split(" "))) <30:
                            if sentence not in sentences_score.keys():
                                sentences_score[sentence] = word_frequency[word]
                            else:
                                sentences_score[sentence] += word_frequency[word]
            
            # st.text("Sentence Score: "+str(max(sentences_score.values())))
            st.subheader("Oraciones y su score")

            import numpy as np
            list_of_strings  = list(sentences_score.keys())
            # Convertimos el diccionario en un np.array
            sentence_oraciones_array = np.array(list_of_strings)

            # Convertimos el diccionario en un np.array
            sentence_scores_array = np.array(list(sentences_score.values()))
            
            # Se unen los dos arrays en un DataFrame
            df = pd.DataFrame({'Oraciones': sentence_oraciones_array, 'Score': sentence_scores_array})

            # Mostrar el DataFrame en Streamlit de mayor a menor
            st.data_editor(
                df,
                column_config={
                    "Oraciones": {
                        "editable": False,
                    },
                    "Score": st.column_config.ProgressColumn(
                        "Score",
                        # help="This is a help text",
                        format="",
                    ),
                },
                use_container_width=True,
                # hide_index=True,
            )
            
            # key = get_key(max(sentences_score.values()))
            # key =""
            for key, value in sentences_score.items():
                if sentences_score == value:
                    return key
            
            summary=heapq.nlargest(number,sentences_score,key=sentences_score.get)
            summary = ' '.join(summary)
            
            st.subheader("Texto original")
            st.info("Hay {} caracteres en total".format(len(src_text)))
            st.warning(src_text)
            st.subheader("Texto Resumido")
            st.info("Hay {} caracteres en total".format(len(summary)))
            st.success(summary)
            
            import pyperclip
            st.button("Copiar al portapapeles", pyperclip.copy(summary))

            # Mostramos una gráfica mostrando el porcentaje de palabras que se redujo
            import plotly.graph_objects as go
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = round((1-(len(summary)/len(src_text)))*100,4),
                title = {'text': "El texto se redujo en un {0}%".format(round((1-(len(summary)/len(src_text)))*100,2)) + " de su tamaño original"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {'axis': {'range': [None, 100]},
                        'steps' : [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "gray"},
                            {'range': [50, 75], 'color': "lightgray"},
                            {'range': [75, 100], 'color': "gray"}]}))
            st.plotly_chart(fig)


    if option == 'Texto Plano':
        src_text = st.text_area("Ingresa el texto a resumir aquí...","Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec euismod, nisl vitae aliquam ultricies, nunc nisl aliquet nunc, vitae")

        with st.expander("Iniciar Resumen..."):
            src_textt = str(src_text).replace('\n', ' ')
            st.subheader("Texto original")
            st.warning(src_textt)

            ##Tokenixing
            sent_tokens = sent_tokenize(src_textt)
            src_textt = re.sub(r"[^a-zA-z]"," ", src_textt)
            word_tokens = word_tokenize(src_textt)
            
            ## Removing Stop words
            word_frequency = {}
            stopwordss =  set(stopwords.words("english"))
            
            ## Creating word Tokens and frequency
            for word in word_tokens:
                if word not in stopwordss:
                    if word not in word_frequency.keys():
                        word_frequency[word]=1
                    else:
                        word_frequency[word] +=1
            
            maximum_frequency = max(word_frequency.values())
            st.subheader("Frecuencia de palabras")
            # st.info("La frecuencia máxima es: "+str(maximum_frequency))
            
            for word in word_frequency.keys():
                word_frequency[word] = (word_frequency[word]/maximum_frequency)
            
            # st.text("Word Frequency: "+str(word_frequency))

            # En un dataframe, se crea una columna con las palabras y otra con la frecuencia, se ordena de mayor a menor, y se grafica con st.bar_chart
            import pandas as pd
            chart_data = pd.DataFrame(word_frequency.items(), columns=['Palabra', 'Frecuencia'])

            # Mostrar el DataFrame en Streamlit de mayor a menor
            st.data_editor(
                chart_data,
                column_config={
                    "Palabra": {
                        "editable": False,
                    },
                    "Frecuencia": st.column_config.ProgressColumn(
                        "Frecuencia",
                        # help="This is a help text",
                        format="",
                    ),
                },
                use_container_width=True,
                # hide_index=True,
            )

            # Graficar el DataFrame en Plotly, de mayor a menor y de rojo a azul
            import plotly.express as px
            fig = px.bar(chart_data.sort_values(by=['Frecuencia'], ascending=False), x="Palabra", y="Frecuencia", color='Frecuencia', color_continuous_scale='RdBu_r')
            st.plotly_chart(fig)
            
            ## Creating Sentence score
            sentences_score = {}
            for sentence in sent_tokens: 
                for word in word_tokenize(sentence):
                    if word in word_frequency.keys():
                        if (len(sentence.split(" "))) <30:
                            if sentence not in sentences_score.keys():
                                sentences_score[sentence] = word_frequency[word]
                            else:
                                sentences_score[sentence] += word_frequency[word]
            
            # st.text("Sentence Score: "+str(max(sentences_score.values())))

            import numpy as np
            list_of_strings  = list(sentences_score.keys())
            # Convertimos el diccionario en un np.array
            sentence_oraciones_array = np.array(list_of_strings)

            # Convertimos el diccionario en un np.array
            sentence_scores_array = np.array(list(sentences_score.values()))
            
            # Se unen los dos arrays en un DataFrame
            df = pd.DataFrame({'Oraciones': sentence_oraciones_array, 'Score': sentence_scores_array})

            # Mostrar el DataFrame en Streamlit de mayor a menor
            st.subheader("Importancia de las oraciones")
            st.data_editor(
                df,
                column_config={
                    "Oraciones": {
                        "editable": False,
                    },
                    "Score": st.column_config.ProgressColumn(
                        "Score",
                        # help="This is a help text",
                        format="",
                    ),
                },
                use_container_width=True,
                # hide_index=True,
            )
            
            for key, value in sentences_score.items():
                if sentences_score == value:
                    return key
            
            summary=heapq.nlargest(number,sentences_score,key=sentences_score.get)
            summary = ' '.join(summary)


            st.subheader("Texto original")
            st.info("Hay {} caracteres en total".format(len(src_text)))
            st.warning(src_text)
            st.subheader("Texto Resumido")
            st.info("Hay {} caracteres en total".format(len(summary)))
            st.success(summary)
            
            import pyperclip
            st.button("Copiar al portapapeles", pyperclip.copy(summary))

            # Mostramos una gráfica mostrando el porcentaje de palabras que se redujo
            import plotly.graph_objects as go
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = round((1-(len(summary)/len(src_text)))*100,4),
                title = {'text': "El texto se redujo en un {0}%".format(round((1-(len(summary)/len(src_text)))*100,2)) + " de su tamaño original"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {'axis': {'range': [None, 100]},
                        'steps' : [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "gray"},
                            {'range': [50, 75], 'color': "lightgray"},
                            {'range': [75, 100], 'color': "gray"}]}))
            st.plotly_chart(fig)



            ############## EVALUATION METRICS
            import evaluate
            # Load the ROUGE evaluation metric
            rouge = evaluate.load('rouge')

            # Define the candidate predictions and reference sentences
            predictions = [summary]
            st.subheader("Evaluación de resultados")
            references1 = st.text_area("Ingresa el resumen hecho por ti","...")

            references = [references1]
            
            
            # Compute the ROUGE score
            st.subheader("Utilizando la métrica ROUGE")
            results = rouge.compute(predictions=predictions, references=references)
            ######## GRAFICAR LOS RESULTADOS
            results_array = np.array(list(results.items()))
            df = pd.DataFrame({'Metrica': results_array[:,0], 'Score': results_array[:,1]})
            st.data_editor(
                df,
                column_config={
                    "Metrica": {
                        "editable": False,
                    },
                    "Score": st.column_config.ProgressColumn(
                        "Score",
                        # help="This is a help text",
                        format="",
                    ),
                },
                use_container_width=True,
                # hide_index=True,
            )

            st.subheader("Utilizando la métrica BLEU")
            # Define the candidate predictions and reference sentences
            predictions = [summary]
            references = [[references1]]

            # Load the BLEU evaluation metric
            bleu = evaluate.load("bleu")

            # Compute the BLEU score
            results = bleu.compute(predictions=predictions, references=references)

            ######## Mostramos los resultados en una tabla
            df = pd.DataFrame({'Metrica': results.keys(), 'Score': results.values()})
            st.table(df)

                        