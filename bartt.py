import torch
import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration

def main_bart():
    _num_beams = 4
    _no_repeat_ngram_size = 3
    _length_penalty = 1
    _min_length = 12
    _max_length = 128
    _early_stopping = True

    col1, col2, col3 = st.columns(3)
    _num_beams = col1.number_input("num_beams", value=_num_beams)
    _no_repeat_ngram_size = col2.number_input("no_repeat_ngram_size", value=_no_repeat_ngram_size)
    _length_penalty = col3.number_input("length_penalty", value=_length_penalty)

    col1, col2, col3 = st.columns(3)
    _min_length = col1.number_input("min_length", value=_min_length)
    _max_length = col2.number_input("max_length", value=_max_length)
    _early_stopping = col3.number_input("early_stopping", value=_early_stopping)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    st.subheader("Summarize with BART")
    src_text = st.text_area("Enter Text Here","Type Here")

    if st.button("Summarize"):
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