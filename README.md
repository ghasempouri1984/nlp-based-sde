# nlp-based-sde
 this is nlp-based sde of letters based on streamlit.
 you need to run 'python -m spacy download en_core_web_sm' in terminal
 it is needed to install requirements.txt
 then you need to navigate to directory of app.py and in terminal run: streamlit run app.py

About the Project
This NLP App with Streamlit is an interactive web application that allows you to perform various natural language processing (NLP) tasks, such as tokenization, part of speech tagging, named entity recognition, and sentiment analysis. The app also generates word clouds, frequency plots, and part of speech histograms.

        ## Main Features

        - **Text Analysis:** Tokenize and analyze text, showing tokens, their shapes, parts of speech, tags, lemmas, and stopword status.
        - **Named Entity Recognition:** Extract and visualize entities such as names, dates, and organizations from the input text.
        - **Sentiment Analysis:** Determine the sentiment of the input text, classifying it as positive, negative, or neutral.
        - **Word Cloud Generation:** Create word clouds to visualize the most common words in the input text.
        - **Frequency Plots:** Generate bar plots to visualize the frequency of the most common tokens in the input text.
        - **Part of Speech Histograms:** Display histograms showing the distribution of different parts of speech in the input text.
        - **Interactive Maps:** Create interactive maps to visualize geographical entities extracted from the input text (optional).

        
        ### Why do we need a NLP-based scholarly digital edition of the letters of Vespasiano di Bisticci and how are we supposed to benefit from it?

        A Natural Language Processing (NLP)-based scholarly digital edition (SDE) of the letters of Vespasiano di Bisticci can provide several benefits:
        - **Enhanced Understanding:** NLP can help us understand the content of the letters in a more nuanced way. It can reveal patterns, themes, and sentiments that might not be immediately apparent from a casual reading. This can provide new insights into Vespasiano di Bisticci's thoughts, feelings, and experiences, as well as the broader historical and cultural context in which he lived [1,2].
        - **Efficient Analysis:** Analyzing historical texts manually can be a time-consuming process, especially when dealing with a large number of documents. NLP can automate much of this process, making it possible to analyze the letters more efficiently [3].
        - **Interactive Exploration:** The interactive features of the SDE, such as word clouds, frequency plots, and part-of-speech histograms, can provide a more engaging and immersive experience for users. They can explore the text in a dynamic way, generating visualizations based on their interests [4].
        - **Facilitating Research:** The NLP-based SDE can be a valuable resource for scholars studying Vespasiano di Bisticci, the Renaissance period, or historical linguistics. It can facilitate research by providing a rich, interactive, and easily accessible dataset for analysis [5].

        
        ### What do we need to create a NLP-based SDE and what kind of edition is best suited to our case?
        
        Creating a Natural Language Processing (NLP)-based scholarly digital edition (SDE) of the letters of Vespasiano di Bisticci involves several steps and considerations:
        - **Accurate Transcription and Translation:** The letters must be accurately transcribed and, if necessary, translated. This is a critical step as the quality of the NLP analysis depends on the quality of the input text [6]
        - **Selection of NLP Tools:** The choice of NLP tools depends on the specific tasks we want to perform (e.g., tokenization, part-of-speech tagging, named entity recognition, sentiment analysis). Python has several libraries for NLP, such as NLTK, SpaCy, and TextBlob, which can be used for these tasks [7].
        - **Development of the Web Application:** Streamlit is a great choice for developing interactive web applications for machine learning and data science. It allows you to create interactive features like word clouds, frequency plots, and part-of-speech histograms [8].
        - **Inclusion of Scholarly Input:** The NLP analysis should be complemented with scholarly input to provide context and interpretation. This could involve annotating the text with additional information or providing essays that discuss the findings of the NLP analysis [2].


        ### Which technologies best achieve the expected results?
        The app is built using the following Python libraries:

        - **Streamlit:** A framework for building web applications.
        - **spaCy:** A library for advanced NLP tasks.
        - **neattext:** A text cleaning library.
        - **TextBlob:** A library for sentiment analysis.
        - **nltk:** A library for working with human language data (text).
        - **wordcloud:** A library for generating word clouds.
        - **matplotlib, seaborn:** Libraries for creating static, animated, and interactive visualizations.
        - **pandas:** A library for data manipulation and analysis.
        - **folium:** A library for creating interactive maps.

        You can find the complete source code for this app on [GitHub](https://github.com/ghasempouri1984/VespasianodaBisticci). Feel free to contribute or report any issues you find.

        **References:**
        1. Language technology for digital humanities (https://link.springer.com/article/10.1007/s10579-019-09482-4)
        2. Digital Humanities and Natural Language Processing (http://digitalhumanities.org:8081/dhq/vol/14/2/000454/000454.html)
        3. Computational text analysis within the Humanities: How to combine working practices from the contributing fields? (https://link.springer.com/article/10.1007/s10579-019-09459-3)
        4. Natural Language Processing in the Humanities: A Case Study in Automated Metadata Enhancement (https://journal.code4lib.org/articles/14834)
        5. LL(O)D and NLP perspectives on semantic change for humanities research (https://content.iospress.com/articles/semantic-web/sw222848)
        6. Reflexivity in Issues of Scale and Representation in a Digital Humanities Project (https://arxiv.org/abs/2109.14184)
        7. Teaching Computational Methods to Humanities Students (https://www.semanticscholar.org/paper/Teaching-Computational-Methods-to-Humanities-%C3%96hman/068248315c50889b4ab56bbdd1f88f76785cc1c5)
        8. How to Build an NLP Machine Learning App-End to End (https://medium.com/geekculture/how-to-build-an-nlp-machine-learning-app-end-to-end-76404ea9f6e8)
