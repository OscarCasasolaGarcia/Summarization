from bs4 import BeautifulSoup #scraping content from website
import re ##Regular expression
import requests ## sending requests
import heapq ## finding largest values
from nltk.tokenize import sent_tokenize,word_tokenize ## tokenizing
from nltk.corpus import stopwords ## removing stopwords
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
    st.subheader("Summarize with NLTK")
    # url = st.text_input("Enter URL Here","Type Here")
    src_text = st.text_area("Enter Text Here","Type Here")
    number = st.number_input("Enter the Number of Sentence you want in the summary",min_value=10)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    if st.button("Summarize"):
        # res = requests.get(url,headers=headers)
        # summary = ""
        # soup = BeautifulSoup(res.text,'html.parser')
        # content = soup.findAll("p")
        # for text in content:
        #     summary +=text.text
        
        # summary = clean(summary)
        summary = str(src_text).replace('\n', '')

        ##Tokenixing
        sent_tokens = sent_tokenize(summary)
        summary = re.sub(r"[^a-zA-z]"," ",summary)
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
        st.text("Maximum Frequency: "+str(maximum_frequency))
        
        for word in word_frequency.keys():
            word_frequency[word] = (word_frequency[word]/maximum_frequency)
        
        st.text("Word Frequency: "+str(word_frequency))
        
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
        
        st.text("Sentence Score: "+str(max(sentences_score.values())))
        
        # key = get_key(max(sentences_score.values()))
        # key =""
        for key, value in sentences_score.items():
            if sentences_score == value:
                return key
        
        st.text("Key: "+str(key) + "\n")
        st.text("Value: "+str(sentences_score[key]))
        summary=heapq.nlargest(7,sentences_score,key=sentences_score.get)
        summary = ' '.join(summary)
        st.success("Summary: "+str(summary))