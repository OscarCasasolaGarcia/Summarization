import torch
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

def main_t5():
    _num_beams = 4
    _no_repeat_ngram_size = 3
    _length_penalty = 2
    _min_length = 30
    _max_length = 512
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

    st.subheader("Summarize with T5")
    src_text = st.text_area("Enter Text Here","Type Here")

    if st.button("Summarize"):
        t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        
        input_text = str(src_text).replace('\n', '')
        input_text = ' '.join(input_text.split())
        input_tokenized = t5_tokenizer.encode(input_text, return_tensors="pt").to(device)
        summary_task = torch.tensor([[21603, 10]]).to(device)
        input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1).to(device)
        summary_ids = t5_model.generate(input_tokenized,
                                        num_beams=_num_beams,
                                        no_repeat_ngram_size=_no_repeat_ngram_size,
                                        length_penalty=_length_penalty,
                                        min_length=_min_length,
                                        max_length=_max_length,
                                        early_stopping=_early_stopping)
        output = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                  summary_ids]

        st.write("Original Text\n")
        st.warning(src_text)
        st.write("Total Length:",len(src_text))

        st.write('\n\nSummarized Text\n')
        st.success(output[0])
        st.write("Total Length:",len(output[0]))