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
	st.title("Summaryzer and Entity Checker")
	activities = ["Summarize","NER Checker","NER For URL","Text Summarizer"]
	choice = st.sidebar.selectbox("Select Activity", activities)
	
	if choice == 'Summarize':
		st.subheader("Summarize with NLP")
		raw_text = st.text_area("Enter Text Here","Type Here")
		summarizer_type = st.selectbox("Summarizer Type",["Gensim","Sumy Lex Rank"])
		if st.button("Summarize"):
			if summarizer_type == "Gensim":
				summary_result = summarize(raw_text)
			elif summarizer_type == "Sumy Lex Rank":
				summary_result = sumy_summarizer(raw_text)
			st.write(summary_result)

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
		st.subheader("Summarize with NLP")
		raw_text = st.text_area("Enter Text Here","Type Here")
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
		st.write("Original Document\n")
		st.write(raw_text)
		st.write("Total Length:",len(raw_text))
		st.write('\n\nSummarized Document\n')
		st.write(summary)
		st.write("Total Length:",len(summary))

if __name__ == '__main__':
	main()