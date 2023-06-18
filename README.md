# Text Summarization using NLP (Natural Language Processing). SpaCy, BART, T5 and NLTK.

<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">Text Summarization</h3>

  <p align="center">
    Proyecto Final Análisis y Procesamiento Inteligente de Textos
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## Acerca del proyecto 

Este proyecto tiene el objetivo de implementar dos resumidores de textos automáticos utilizando spaCy y NLTK. Además, se utilizaron modelos pre-entrenados de BART y T5 para implementar dos resumidores automáticos más. Finalmente, se evaluaron los resumidores utilizando los puntajes BLEU y ROUGE.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Tecnologías y/o bibliotecas utilizadas

Este proyecto está construido con las siguientes tecnologías y/o bibliotecas:

* [Python](https://www.python.org/)
* [Streamlit](https://streamlit.io/)
* [Plotly](https://plotly.com/)
* [spaCy](https://spacy.io/)
* [NLTK](https://www.nltk.org/)
* [Hugging Face](https://huggingface.co/)
* [BART](https://huggingface.co/facebook/bart-large-cnn)
* [T5](https://huggingface.co/t5-base)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Introducción

A continuación, se muestran los pasos para ejecutar el proyecto.

### Pre-requisitos

Primero, se debe tener instalado Python 3.7 o superior. 

Segundo, se debe clona el repositorio en la carpeta deseada.

```sh
git clone https://github.com/OscarCasasolaGarcia/Summarization.git
```

Tercero, se debe crear un entorno virtual para instalar las dependencias del proyecto.

```sh
python -m venv .venv
```


### Instalación

1. Activar el entorno virtual
   ```sh
   source .venv/bin/activate
   ```

2. Una vez posicionados dentro de nuestro ambiente virtual, se procede con la instalación de todas las bibliotecas y dependencias mediante el siguiente comando:

    ```sh
    pip install -r requirements.txt
    ```

3. Finalmente, se ejecuta el proyecto mediante el siguiente comando:

    ```sh
    streamlit run app.py
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Agregar resumidor automático utilizando spaCy
- [x] Agregar resumidor automático utilizando NLTK
- [x] Agregar resumidor automático utilizando BART
- [x] Agregar resumidor automático utilizando T5
- [x] Agregar evaluación de resumidores utilizando puntajes BLEU y ROUGE
- [ ] Mejorar la interfaz gráfica
- [ ] Agregar soporte para múltiples idiomas
    - [x] Inglés
    - [ ] Español
    - [ ] Francés
- [ ] Agregar módulo Pegasus
- [ ] Agregar módulo BERT
- [ ] Agregar módulo GPT-3
- [ ] Migrar la aplicación a React.js, Flask y MySQL
- [ ] Agregar soporte para resumir documentos PDF
- [ ] Migrar la solución a GCP
- [ ] ...


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contribuciones

Las contribuciones son lo que hacen que la comunidad de código abierto sea un lugar tan increíble para aprender, inspirar y crear. Cualquier contribución que haga es **muy apreciada**.

Si tiene alguna sugerencia que hacer, por favor, primero discuta el cambio que desea realizar a través de un problema, correo electrónico o cualquier otro método con los propietarios de este repositorio antes de realizar un cambio.

1. Fork el proyecto
2. Cree su rama de características (`git checkout -b feature/AmazingFeature`)
3. Confirme sus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Empuje a la rama (`git push origin feature/AmazingFeature`)
5. Abra una solicitud de extracción

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->




<!-- CONTACT -->
## Contact

Oscar Casasola García

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->