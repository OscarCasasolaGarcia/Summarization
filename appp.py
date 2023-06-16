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

nlp = spacy.load("en_core_web_sm")
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
		
			

			#### Existen dos enfoques principales para realizar el resumen de un texto:

			* Enfoque extractivo: Este enfoque consiste en identificar las oraciones más importantes o representativas del texto original y combinarlas para formar un resumen. Utiliza técnicas como el análisis de frecuencia de palabras, puntuación de oraciones y análisis de similitud semántica para seleccionar las oraciones más relevantes. El resumen generado es una extracción directa de partes del texto original. Este tiende a ser más simple y conserva el lenguaje y estilo del texto original, pero puede generar resúmenes menos coherentes. 

			* Enfoque abstractivo: A diferencia del enfoque extractivo, el enfoque abstractivo implica comprender el significado del texto original y generar un resumen utilizando técnicas de generación de lenguaje natural. Utiliza modelos de procesamiento de lenguaje natural (NLP) avanzados que pueden interpretar y comprender el contexto y la semántica del texto. El resumen generado es una síntesis nueva y original que no necesariamente se basa en oraciones o fragmentos específicos del texto original. Este puede generar resúmenes más coherentes y legibles, pero puede requerir un mayor nivel de procesamiento y comprensión del lenguaje.

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
		# nlp = spacy.load("en_core_web_sm")
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



if __name__ == '__main__':
	main()