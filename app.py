import streamlit as st 


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
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

# Web Scraping Pkg
from bs4 import BeautifulSoup
from urllib.request import urlopen

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

def main():
	st.markdown(
	"""
		# Resumidor de texto (Text summarization)
	
	"""
	)

	st.markdown(
		"""
			## ¿Qué es?
		
			* Un resumidor de texto (text summarization) es una técnica o algoritmo que tiene como objetivo reducir la longitud de un texto manteniendo la información más relevante y significativa. Su propósito es crear un resumen conciso y coherente que capture los aspectos clave del texto original.

			* El resumidor de texto puede ser utilizado para procesar documentos largos, artículos, noticias, informes, páginas web y otros tipos de contenido textual extenso. Proporciona una forma eficiente de extraer la información esencial de un texto y presentarla de manera más breve, lo que facilita su comprensión y permite a los lectores obtener rápidamente una idea general del contenido sin tener que leer el texto completo.

			#### Existen dos enfoques principales para realizar el resumen de un texto:

			* Enfoque extractivo: Este enfoque consiste en identificar las oraciones más importantes o representativas del texto original y combinarlas para formar un resumen. Utiliza técnicas como el análisis de frecuencia de palabras, puntuación de oraciones y análisis de similitud semántica para seleccionar las oraciones más relevantes. El resumen generado es una extracción directa de partes del texto original. Este tiende a ser más simple y conserva el lenguaje y estilo del texto original, pero puede generar resúmenes menos coherentes. 

			* Enfoque abstractivo: A diferencia del enfoque extractivo, el enfoque abstractivo implica comprender el significado del texto original y generar un resumen utilizando técnicas de generación de lenguaje natural. Utiliza modelos de procesamiento de lenguaje natural (NLP) avanzados que pueden interpretar y comprender el contexto y la semántica del texto. El resumen generado es una síntesis nueva y original que no necesariamente se basa en oraciones o fragmentos específicos del texto original. Este puede generar resúmenes más coherentes y legibles, pero puede requerir un mayor nivel de procesamiento y comprensión del lenguaje.

		"""
	)

	st.markdown(
		"""
			El desarrollo de resumidores de texto ha sido impulsado por los avances en el procesamiento de lenguaje natural y el aprendizaje automático. Estos sistemas utilizan algoritmos y modelos de lenguaje pre-entrenados para realizar la resumización de texto de manera más eficiente y precisa.

			En resumen, la resumización de texto implica crear un resumen preciso y conciso de un documento de texto extenso. El objetivo principal de la resumización automática de texto es seleccionar la información más importante y presentarla de forma comprensible. Con el crecimiento de los datos textuales en línea, los métodos automáticos de resumización de texto son cada vez más útiles, ya que permiten obtener información relevante en menos tiempo.
		"""
	)

	st.markdown(
		"""
			#### Pasos para realizar la resumización de texto:
			
			* Limpieza de texto (Text cleaning):
				Antes de comenzar con el proceso de resumen, es importante realizar una limpieza del texto para eliminar cualquier ruido o información innecesaria. Esto puede implicar la eliminación de caracteres especiales, signos de puntuación, números, palabras vacías (stop words) y cualquier otro tipo de elementos que no contribuyan significativamente al significado del texto. La limpieza de texto puede llevarse a cabo utilizando técnicas como la eliminación de expresiones regulares, el uso de librerías de procesamiento de lenguaje natural (NLP) y la normalización de texto.
		"""
	)

	st.sidebar.subheader("Selecciona el módulo que deseas utilizar")

	activities = ["Summarize","NER Checker","NER For URL","Text Summarizer","Pegasus","BART","SpaCy","Test"]
	choice = st.sidebar.selectbox("Selecciona la opción de tu preferencia",activities)
	
	if choice == 'Summarize':
		st.subheader("Summarize with NLP")
		raw_text = st.text_area("Enter Text Here","Type Here")
		summarizer_type = st.selectbox("Summarizer Type",["Gensim","Sumy Lex Rank"])
		if st.button("Summarize"):
			if summarizer_type == "Gensim":
				summary_result = summarize(raw_text)

				st.write("Original Text\n")
				st.warning(raw_text)
				st.write("Total Length:",len(raw_text))

				st.write('\n\nSummarized Text\n')
				st.success(summary_result)
				st.write("Total Length:",len(summary_result))


			elif summarizer_type == "Sumy Lex Rank":
				summary_result = sumy_summarizer(raw_text)

				st.write("Original Text\n")
				st.warning(raw_text)
				st.write("Total Length:",len(raw_text))

				st.write('\n\nSummarized Text\n')
				st.success(summary_result)
				st.write("Total Length:",len(summary_result))

	if choice == 'NER Checker':
		st.subheader("Named Entity Recog with Spacy")
		raw_text = st.text_area("Enter Text Here","Type Here")
		if st.button("Analyze"):
			docx = analyze_text(raw_text)
			html = displacy.render(docx,style="ent")
			html = html.replace("\n\n","\n")
			st.markdown(html,unsafe_allow_html=True)

	if choice == 'NER For URL':
		st.subheader("Analysis on Text From URL")
		raw_url = st.text_input("Enter URL Here","Type here")
		text_preview_length = st.slider("Length to Preview",50,100)
		if st.button("Analyze"):
			if raw_url != "Type here":
				result = get_text(raw_url)
				len_of_full_text = len(result)
				len_of_short_text = round(len(result)/text_preview_length)
				st.success("Length of Full Text:{}".format(len_of_full_text))
				st.success("Length of Short Text:{}".format(len_of_short_text))
				st.info(result[:len_of_short_text])
				summarized_docx = sumy_summarizer(result)
				docx = analyze_text(summarized_docx)
				html = displacy.render(docx,style="ent")
				html = html.replace("\n\n","\n")
				st.markdown(html,unsafe_allow_html=True)
				# st.write(HTML_WRAPPER.format(html),unsafe_allow_html=True)
    
	if choice == 'Text Summarizer':
		import spacy
		from spacy.lang.en.stop_words import STOP_WORDS
		from string import punctuation
		from heapq import nlargest
		st.subheader("Summarize with NLP")
		raw_text = st.text_area("Enter Text Here","Type Here")
		nlp = spacy.load("en_core_web_sm")
		docx = nlp(raw_text)
		stopwords = list(STOP_WORDS)
		# Build Word Frequency
		# word.text is tokenization in spacy
		word_frequencies = {}  
		for word in docx:  
			if word.text not in stopwords:
				if word.text not in word_frequencies.keys():
					word_frequencies[word.text] = 1
				else:
					word_frequencies[word.text] += 1

		maximum_frequncy = max(word_frequencies.values())

		for word in word_frequencies.keys():  
			word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
		
		# Sentence Tokens
		sentence_list = [ sentence for sentence in docx.sents ]

		# Calculate Sentence Score and Ranking
		sentence_scores = {}  
		for sent in sentence_list:  
			for word in sent:
				if word.text.lower() in word_frequencies.keys():
					if len(sent.text.split(' ')) < 30:
						if sent not in sentence_scores.keys():
							sentence_scores[sent] = word_frequencies[word.text.lower()]
						else:
							sentence_scores[sent] += word_frequencies[word.text.lower()]

		# Find N Largest
		summary_sentences = nlargest(7, sentence_scores, key=sentence_scores.get)
		final_sentences = [ w.text for w in summary_sentences ]
		summary = ' '.join(final_sentences)
		

		st.write("Original Text\n")
		st.warning(raw_text)
		st.write("Total Length:",len(raw_text))

		st.write('\n\nSummarized Text\n')
		st.success(summary)
		st.write("Total Length:",len(summary))


	if choice == "SpaCy":
		# Text cleaning
		import spacy
		from spacy.lang.en.stop_words import STOP_WORDS
		from string import punctuation

		st.subheader("Summarize with NLP")
		raw_text = st.text_area("Enter Text Here","Type Here")

		if st.button("Summarize"):
			stopwords = list(STOP_WORDS)
			nlp = spacy.load("en_core_web_sm")
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

	
	if choice == "Test":
		# streamlit_app.py
		import spacy_streamlit

		models = ["en_core_web_sm", "en_core_web_md"]
		default_text = "Sundar Pichai is the CEO of Google."
		spacy_streamlit.visualize(models, default_text)

		
		
	
	if choice == 'Pegasus':
		# import gc
		# gc.collect()

		import torch
		from transformers import PegasusForConditionalGeneration, PegasusTokenizer

		# torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		st.subheader("Summarize with Pegasus")
		src_text = st.text_area("Enter Text Here","Type Here")

		model_name = 'google/pegasus-reddit_tifu'
		model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
		pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
		batch = pegasus_tokenizer.prepare_seq2seq_batch(src_text, truncation=True, padding='longest', return_tensors="pt").to(device)
		translated = model.generate(**batch)
		tgt_text = pegasus_tokenizer.batch_decode(translated, skip_special_tokens=True)

		st.write('\n\nSummarized Document\n')
		st.write(tgt_text[0])

	
	if choice == 'BART':
		import torch
		from transformers import BartTokenizer, BartForConditionalGeneration
		from transformers import T5Tokenizer, T5ForConditionalGeneration

		_num_beams = 4
		_no_repeat_ngram_size = 3
		_length_penalty = 1
		_min_length = 12
		_max_length = 128
		_early_stopping = True

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		st.subheader("Summarize with BART")
		src_text = st.text_area("Enter Text Here","Type Here")

		bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
		bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
		input_text = str(src_text)
		input_text = ' '.join(input_text.split())
		input_tokenized = bart_tokenizer.encode(input_text, return_tensors='pt').to(device)
		summary_ids = bart_model.generate(input_tokenized,
											num_beams=_num_beams,
											no_repeat_ngram_size=_no_repeat_ngram_size,
											length_penalty=_length_penalty,
											min_length=_min_length,
											max_length=_max_length,
											early_stopping=_early_stopping)

		output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
					summary_ids]

		st.write("Original Text\n")
		st.warning(src_text)
		st.write("Total Length:",len(src_text))

		st.write('\n\nSummarized Text\n')
		st.success(output[0])
		st.write("Total Length:",len(output[0]))


		

if __name__ == '__main__':
	main()