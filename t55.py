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


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    st.header("Resumidor Automático de Textos utilizando T5")
    src_text = st.text_area("Ingresa el texto a resumir","Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non risus. Suspendisse lectus tortor, dignissim sit amet, adipiscing nec, ultricies sed, dolor.")

    st.subheader("Calibración de los parámetros")
    col1, col2, col3 = st.columns(3)
    _num_beams = col1.number_input("num_beams", value=_num_beams)
    _no_repeat_ngram_size = col2.number_input("no_repeat_ngram_size", value=_no_repeat_ngram_size)
    _length_penalty = col3.number_input("length_penalty", value=_length_penalty)

    col1, col2, col3 = st.columns(3)
    _min_length = col1.number_input("min_length", value=_min_length)
    _max_length = col2.number_input("max_length", value=_max_length)
    _early_stopping = col3.number_input("early_stopping", value=_early_stopping)


    if st.button("Resumir Texto..."):
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

        st.subheader("Texto original")
        st.info("Hay {} caracteres en total".format(len(src_text)))
        st.warning(src_text)
        st.subheader("Texto Resumido")
        st.info("Hay {} caracteres en total".format(len(output[0])))
        st.success(output[0])
        import pyperclip
        st.button("Copiar al portapapeles", pyperclip.copy(output[0]))

        # Mostramos una gráfica mostrando el porcentaje de palabras que se redujo
        import plotly.graph_objects as go
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = round((1-(len(output[0])/len(src_text)))*100,4),
            title = {'text': "El texto se redujo en un {0}%".format(round((1-(len(output[0])/len(src_text)))*100,2)) + " de su tamaño original"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {'axis': {'range': [None, 100]},
                    'steps' : [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "gray"},
                        {'range': [50, 75], 'color': "lightgray"},
                        {'range': [75, 100], 'color': "gray"}]}))
        st.plotly_chart(fig)


        ############## EVALUATION METRICS
        # import evaluate
        # # Load the ROUGE evaluation metric
        # rouge = evaluate.load('rouge')

        # # Define the candidate predictions and reference sentences
        # predictions = [output[0]]
        # st.subheader("Evaluación de resultados")
        # references = ["Almost two years ago, Tinder decided to move its platform to Kubernetes. Kubernetes afforded us an opportunity to drive Tinder Engineering toward containerization and low-touch operation through immutable deployment. Application build, deployment, and infrastructure would be defined as code. We solved interesting challenges to migrate 200 services and run a Kubernetes cluster at scale totaling 1,000 nodes, 15,000 pods, and 48,000 running containers."]
        
        
        # # Compute the ROUGE score
        # st.subheader("Utilizando la métrica ROUGE")
        # results = rouge.compute(predictions=predictions, references=references)
        # ######## GRAFICAR LOS RESULTADOS

        # import pandas as pd
        # import numpy as np
        # results_array = np.array(list(results.items()))
        # df = pd.DataFrame({'Metrica': results_array[:,0], 'Score': results_array[:,1]})
        # st.data_editor(
        #     df,
        #     column_config={
        #         "Metrica": {
        #             "editable": False,
        #         },
        #         "Score": st.column_config.ProgressColumn(
        #             "Score",
        #             # help="This is a help text",
        #             format="",
        #         ),
        #     },
        #     use_container_width=True,
        #     # hide_index=True,
        # )

        # st.subheader("Utilizando la métrica BLEU")
        # # Define the candidate predictions and reference sentences
        # predictions = [output[0]]
        # references = [[references]]

        # # Load the BLEU evaluation metric
        # bleu = evaluate.load("bleu")

        # # Compute the BLEU score
        # results = bleu.compute(predictions=predictions, references=references)

        # ######## Mostramos los resultados en una tabla
        # df = pd.DataFrame({'Metrica': results.keys(), 'Score': results.values()})
        # st.table(df)