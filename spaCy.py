import streamlit as st
from st_on_hover_tabs import on_hover_tabs

from datetime import date
import base64
from PIL import Image

from gensim.summarization import summarize
# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
# Text Preprocessing Pkg
from spacy.lang.es.stop_words import STOP_WORDS
from string import punctuation
# Import Heapq
from heapq import nlargest
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

def main_spacy():
    # Text cleaning
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    from string import punctuation

    st.subheader("Resumidor automático de textos utilizando spaCy")
    raw_text = st.text_area("Enter Text Here","Almost two years ago, Tinder decided to move its platform to Kubernetes. Kubernetes afforded us an opportunity to drive Tinder Engineering toward containerization and low-touch operation through immutable deployment. Application build, deployment, and infrastructure would be defined as code. We were also looking to address challenges of scale and stability. When scaling became critical, we often suffered through several minutes of waiting for new EC2 instances to come online. The idea of containers scheduling and serving traffic within seconds as opposed to minutes was appealing to us. It wasn’t easy. During our migration in early 2019, we reached critical mass within our Kubernetes cluster and began encountering various challenges due to traffic volume, cluster size, and DNS. We solved interesting challenges to migrate 200 services and run a Kubernetes cluster at scale totaling 1,000 nodes, 15,000 pods, and 48,000 running containers. ")
    
    number = st.number_input("Ingresa el número de sentencias que quieres en el resumen...",min_value=1)

    with st.expander("Summarize"):
        stopwords = list(STOP_WORDS)
        # nlp = spacy.load("en_core_web_sm")
        doc = nlp(raw_text)

        # Word tokenization
        tokens = [token.text for token in doc]
        
        st.subheader("Tokens")
        
        # Convertimos tokens a un dataframe
        import pandas as pd
        pd.set_option('display.max_colwidth', 200)
        df = pd.DataFrame(tokens, columns=['Tokens'])
        # Ajustamos el tamaño del dataframe a la pantalla
        st.dataframe(df, use_container_width=True)

        punctuation = punctuation + "\n"
        # st.write(punctuation)
        word_frequencies = {}
        for word in doc:
            if word.text.lower() not in stopwords:
                if word.text.lower() not in punctuation:
                    if word.text not in word_frequencies.keys():
                        word_frequencies[word.text] = 1
                    else:
                        word_frequencies[word.text] += 1
        
        import numpy as np
        st.subheader("Frecuencia de palabras")
        chart_data = pd.DataFrame(word_frequencies.items(), columns=['Palabra', 'Frecuencia'])
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

        # Sentence tokenization
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word]/max_frequency
        
        # En un dataframe, se crea una columna con las palabras y otra con la frecuencia, se ordena de mayor a menor, y se grafica con st.bar_chart
        # Crear el DataFrame
        chart_data = pd.DataFrame(word_frequencies.items(), columns=['Palabra', 'Frecuencia'])

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
        
        sentence_tokens = [sent for sent in doc.sents]
        # st.info(sentence_tokens)


        # Word frequency table
        sentence_scores = {}
        for sent in sentence_tokens:
            for word in sent:
                if word.text.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]

        list_of_strings  = [i.text for i in sentence_scores.keys()]
        # Convertimos el diccionario en un np.array
        sentence_oraciones_array = np.array(list_of_strings)

        # Convertimos el diccionario en un np.array
        sentence_scores_array = np.array(list(sentence_scores.values()))
        
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


        ############# Summarization
        from heapq import nlargest
        # select_length = int(len(sentence_tokens)*0.3)

        summary = nlargest(number, sentence_scores, key = sentence_scores.get)

        final_summary = [word.text for word in summary]
        summary = " ".join(final_summary)

        st.subheader("Texto original")
        st.info("Hay {} caracteres en total".format(len(raw_text)))
        st.warning(raw_text)
        st.subheader("Texto Resumido")
        st.info("Hay {} caracteres en total".format(len(summary)))
        st.success(summary)
        import pyperclip
        st.button("Copiar al portapapeles", pyperclip.copy(summary))

        # Mostramos una gráfica mostrando el porcentaje de palabras que se redujo
        import plotly.graph_objects as go
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = round((1-(len(summary)/len(raw_text)))*100,4),
            title = {'text': "El texto se redujo en un {0}%".format(round((1-(len(summary)/len(raw_text)))*100,2)) + " de su tamaño original"},
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
