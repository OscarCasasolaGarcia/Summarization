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

    st.subheader("Summarize with NLP")
    raw_text = st.text_area("Enter Text Here","Type Here")

    if st.button("Summarize"):
        stopwords = list(STOP_WORDS)
        # nlp = spacy.load("en_core_web_sm")
        doc = nlp(raw_text)

        # Word tokenization
        tokens = [token.text for token in doc]
        st.info("Hay {} tokens".format(len(tokens)))
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
        # En un dataframe, se crea una columna con las palabras y otra con la frecuencia, se ordena de mayor a menor, y se grafica con st.bar_chart
        # Crear el DataFrame
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
        select_length = int(len(sentence_tokens)*0.3)

        summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)

        final_summary = [word.text for word in summary]
        summary = " ".join(final_summary)

        st.warning(raw_text)
        st.success(summary)