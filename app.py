import streamlit as st
import streamlit.components.v1 as stc

# additional pkgs
# eda pkgs
import pandas as pd

# nlp pkgs
import spacy

nlp = spacy.load('en_core_web_sm')
from spacy import displacy
from bs4 import BeautifulSoup

# text cleaning pkgs
import neattext as nt
import neattext.functions as nfx

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from textblob import TextBlob

from collections import Counter
import seaborn as sns

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')

import base64

import folium
from streamlit_folium import folium_static

import datefinder

import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

import re



# funcs
def text_analyzer(my_text):
    docx = nlp(my_text)
    allData = [(token.text, token.shape_,token.pos_,token.tag_,token.lemma_,token.is_alpha,token.is_stop) for token in docx]
    df = pd.DataFrame(allData, columns=['Token', 'shape', 'PoS', 'Tag', 'Lemma', 'IsAlpha', 'Is_Stopword'])
    return df

def get_entities(my_text):
    docx = nlp(my_text)
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    return entities

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding:1rem">{}</div>"""
# @st.cache
def render_entitis(rawtext):
    docx = nlp(rawtext)
    html = displacy.render(docx, style="ent")
    html = html.replace("\n\n", "\n")
    result = HTML_WRAPPER.format(html)
    return result

def plot_wordcloud(text):
    #wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
    #plt.imshow(wordcloud, interpolation='bilinear')
    #plt.axis('off')
    #plt.show()
    wordcloud = WordCloud(stopwords=set(stopwords.words('english')), max_words=100, background_color='white').generate(text)
    #wordcloud = WordCloud(stopwords=nfx.ENGLISH_STOP_WORDS, max_font_size=50, max_words=100, background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def extract_date(texts):
    matches = datefinder.find_dates(texts)
    date = None
    for match in matches:
        date = match
        break
    return date

def extract_date2(texts):

    date = [entity.text for entity in texts.ents if entity.label_ == "DATE"]
    return date

def analyze_sentiment(letters):
    sentiment_data = []

    for letter in letters:
        doc = nlp(letter)
        date = extract_date2(doc)
        sentiment_polarity = TextBlob(letter).sentiment.polarity

        if sentiment_polarity > 0:
            sentiment = "Positive"
        elif sentiment_polarity < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        sentiment_data.append((letter, date, sentiment))

    return sentiment_data

def analyze_sentiment1(documents):
    sentiment_data = []
    doc = nlp(documents)
    date = extract_date2(doc)

    blob = TextBlob(documents)
    sentiment_polarity = blob.sentiment.polarity
    #sentiment_subjectivity = blob.sentiment.subjectivity

    if sentiment_polarity > 0:
        sentiment = "Positive"
    elif sentiment_polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    #return sentiment, sentiment_polarity, sentiment_subjectivity
    #sentiment = TextBlob(documents).sentiment.polarity
    sentiment_data.append((documents, date, sentiment))
    
    return sentiment_data

def analyze_temporal_spatial(letters):
    temporal_spatial_data = []
    lat = [43.7492, 43.7695, 43.7696, 43.7695, 40.8518, 43.7750, 41.9028, 40.8369, 44.4938, 41.0594]
    long = [11.7894, 11.1951, 11.2558, 11.1951, 14.2681, 11.2444, 12.4964, 14.1222, 11.3426, 14.2056]
    i=0
    for letter in letters:
        doc = nlp(letter)
        date = extract_date2(doc)
        
        #lat, lon = extract_geolocation(doc)

        temporal_spatial_data.append((letter, date, lat[i], long[i]))
        i=i+1
    return temporal_spatial_data

def plot_pos_histogram(text):
    docx = nlp(text)
    pos_counts = Counter([token.pos_ for token in docx])
    pos_df = pd.DataFrame(pos_counts.items(), columns=["POS", "Count"])
    plt.figure(figsize=(10, 5))
    sns.barplot(x="POS", y="Count", data=pos_df, palette="viridis")
    plt.title("Part of Speech Histogram")
    plt.show()

def plot_word_freq(df, num_of_most_common):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Word', y='Frequency', data=df[:num_of_most_common])
    plt.xticks(rotation=45)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    st.pyplot()

def create_map():
    lat = [43.7492, 43.7695, 43.7696, 43.7695, 40.8518, 43.7750, 41.9028, 40.8369, 44.4938, 41.0594]
    long = [11.7894, 11.1951, 11.2558, 11.1951, 14.2681, 11.2444, 12.4964, 14.1222, 11.3426, 14.2056]
    ratings = [3, 2, 5, 1, 4, 3, 5, 2, 1, 4]
    letters_received = [7, 12, 5, 9, 15, 4, 11, 6, 14, 8]

    df_cord = pd.DataFrame(columns=["Lat", "Long", "Ratings", "Letters Received"])
    df_cord["Lat"] = lat
    df_cord["Long"] = long
    df_cord["Ratings"] = ratings
    df_cord["Letters Received"] = letters_received

    m = folium.Map(df_cord[['Lat', 'Long']].mean().values.tolist(), zoom_start=10)

    cold_colors = ['#96d5e5', '#5cc5e5', '#5c99e5', '#5c5ce5', '#965ce5']
    warm_colors = ['#e5965c', '#e5c25c', '#e5e55c', '#c2e55c', '#5ce59b']

    for lat, lon, r, l in zip(df_cord['Lat'], df_cord['Long'], df_cord['Ratings'], df_cord['Letters Received']):
        if r <= 3:
            color = cold_colors[r - 1]
        else:
            color = warm_colors[r - 4]
        folium.CircleMarker([lat, lon], radius=l*1.5, color=color, fill=True, fill_color=color, fill_opacity=0.5).add_child(
            folium.Popup(f"Rating: {r}<br>Letters Received: {l}")
        ).add_to(m)

    sw = df_cord[['Lat', 'Long']].min().values.tolist()
    ne = df_cord[['Lat', 'Long']].max().values.tolist()

    m.fit_bounds([sw, ne])
    return m

def word_stats(text):
    docx = nlp(text)
    num_tokens = len(docx)
    num_sentences = len(list(docx.sents))
    num_words = len([token for token in docx if token.is_alpha])
    num_stopwords = len([token for token in docx if token.is_stop])
    return num_tokens, num_sentences, num_words, num_stopwords

def preprocess(text):
    # Tokenize the text
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in words if word.isalpha() and word not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return words

def create_dictionary_and_corpus(texts):
    processed_texts = [preprocess(text) for text in texts]
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    return dictionary, corpus

def get_lda_topics(texts, num_topics=5, passes=20):
    dictionary, corpus = create_dictionary_and_corpus(texts)
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)
    return lda_model.print_topics()

def remove_initial_numbers(text):
    # The regex pattern to match any sequence of {number} at the beginning of a string
    pattern = r"^\{\d+\}"
    
    # Split the text into paragraphs
    paragraphs = text.split("\n")

    # Remove the {number} sequence at the beginning of each paragraph
    cleaned_paragraphs = [re.sub(pattern, '', paragraph).lstrip() for paragraph in paragraphs]

    # Join the cleaned paragraphs back together
    cleaned_text = "\n".join(cleaned_paragraphs)

    # Remove bracketed numbers [number]
    cleaned_text = re.sub(r"\[\d+\]", "", cleaned_text)
    
    return cleaned_text

import xml.etree.ElementTree as ET

def generate_tei_xml_v2(doc):
    soup = BeautifulSoup(features='xml')

    tei = soup.new_tag('TEI')
    soup.append(tei)

    text = soup.new_tag('text')
    tei.append(text)

    body = soup.new_tag('body')
    text.append(body)

    div = soup.new_tag('div')
    body.append(div)

    p = None

    for sent in doc.sents:
        if not p:
            p = soup.new_tag('p')
        for token in sent:
            if token.ent_type_ == 'PERSON':
                persName = soup.new_tag('persName')
                persName.string = token.text
                p.append(persName)
            elif token.ent_type_ == 'DATE':
                date = soup.new_tag('date')
                date.string = token.text
                p.append(date)
            elif token.ent_type_ == 'GPE':
                placeName = soup.new_tag('placeName')
                placeName.string = token.text
                p.append(placeName)
            else:
                p.append(' ')
                p.append(token.text)
        if len(p.contents) > 0:  # check if 'p' has any children
            div.append(p)
        p = None

    tei_xml = str(soup)
    return tei_xml

def main():
    st.title("NLP-Based Scholarly Digital Edition of Letters")
    menu = ["Run The Tool", "About The Project", "Documentation"]

    choice = st.sidebar.selectbox("Menu", menu)
    if 'letters' not in st.session_state:
            st.session_state['letters'] = []

    if choice == "Run The Tool":
        st.subheader("Analyze Text:")
         # Add file uploader and text area widgets
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        raw_text = st.text_area("Enter Text Here")
        
        # Handle uploaded file content
        if uploaded_file is not None:
            raw_text = uploaded_file.read().decode("utf-8")
        #raw_text = st.text_area("Enter Text Here")
        num_of_most_common = st.sidebar.number_input("Most Common Tokens", 5, 15)
        if st.button("Analyze"):
            if raw_text.strip() == "":
                st.warning("Text box is empty. Please enter text.")
            else:

                st.session_state.letters.append(raw_text)  # Add the current letter to the session state
                letters = st.session_state.letters
                with st.expander('Original Text'):
                    st.write(raw_text)
                
                with st.expander('Text Analysis'):
                    token_result_df = text_analyzer(raw_text)
                    st.dataframe(token_result_df)
                    #st.write(raw_text)
                
                with st.expander('Entities'):
                    #entity_result = get_entities(raw_text)
                    #  st.write(entity_result)

                    entity_result = render_entitis(raw_text)
                    stc.html(entity_result, height=1000, scrolling=True)

                with st.expander('Generate TEI XML'):
                    #first try
                    #tei_xml = generate_tei_xml(raw_text)
                    #st.text(tei_xml)
                    
                    #second try
                    #doc = nlp(raw_text)
                    #tei_xml = generate_tei_xml_v2(doc)
                    #st.text(tei_xml)
                    # Preprocess the raw text
                    cleaned_text = remove_initial_numbers(raw_text)
                    doc = nlp(cleaned_text)
                    tei_xml = generate_tei_xml_v2(doc)
                    
                    # Convert BeautifulSoup object to a string
                    tei_xml_str = str(tei_xml)

                    # Create a download button for the TEI XML
                    st.download_button(
                        label="Download TEI XML",
                        data=tei_xml_str,
                        file_name="tei.xml",
                        mime="application/xml",
                    )


                # Layouts
                col1,col2 = st.columns(2)

                with col1:
                    with st.expander('Folium Map'):
                        map_fig = create_map()
                        folium_static(map_fig)

                    with st.expander("Word Stats"):
                        num_tokens, num_sentences, num_words, num_stopwords = word_stats(raw_text)
                        st.markdown(f"**Number of Tokens:** {num_tokens}")
                        st.markdown(f"**Number of Sentences:** {num_sentences}")
                        st.markdown(f"**Number of Words:** {num_words}")
                        st.markdown(f"**Number of Stopwords:** {num_stopwords}")
                    
                    with st.expander("Top Keywords"):
                        pass
                    
                    with st.expander("Sentiment"):
                        sentiment_data = analyze_sentiment(letters)
                        temporal_spatial_data = analyze_temporal_spatial(letters)
                        sentiment_df = pd.DataFrame(sentiment_data, columns=["Document", "Date", "Sentiment"])
                        temporal_spatial_df = pd.DataFrame(temporal_spatial_data, columns=["Document", "Date", "Latitude", "Longitude"])

                        st.write(sentiment_df)
                        st.write(temporal_spatial_df)

                                           
                
                with col2:
                    with st.expander("Plot Word Freq"):
                        cleaned_text = nfx.remove_stopwords(raw_text)
                        token_df = text_analyzer(cleaned_text)
                        word_frequency = token_df['Token'].value_counts().reset_index().rename(columns={'index': 'Word', 'Token': 'Frequency'})
                        plot_word_freq(word_frequency, num_of_most_common)

                    with st.expander("Plot Part of Speech"):
                        st.pyplot(plot_pos_histogram(raw_text))

                    with st.expander("Plot WordCloud"):
                        cleaned_text = nfx.remove_stopwords(raw_text)
                        st.pyplot(plot_wordcloud(cleaned_text))

                with st.expander("Topic Modeling"):
                    
                    #num_topics = st.slider("Number of Topics", 2, 10, 5)
                    #num_passes = st.slider("Number of Passes", 10, 50, 20)
                    num_topics = 3
                    num_passes = 3
                    lda_topics = get_lda_topics([raw_text], num_topics=num_topics, passes=num_passes)
                    for idx, topic in lda_topics:
                        st.markdown(f"**Topic {idx + 1}:** {topic}")
                                                           
                with st.expander("Download Text Analysis Results "):
                    token_result_df = text_analyzer(raw_text)
                    csv_data = token_result_df.to_csv(index=False)
                    b64 = base64.b64encode(csv_data.encode()).decode()
                    download_filename = "text_analysis_results.csv"
                    download_link = f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">Download CSV File</a>'
                    st.markdown(download_link, unsafe_allow_html=True)
            
    
    #elif choice == "NLP(files)":
    #    st.subheader("NLP Task")
    elif choice == "About The Project":
        st.subheader("About The Project")
        st.markdown("""
        This NLP App with Streamlit is an interactive web application that allows you to perform various natural language processing (NLP) tasks, such as tokenization, part of speech tagging, named entity recognition, and sentiment analysis. The app also generates word clouds, frequency plots, and part of speech histograms.

        ## Main Features

        - **Text Analysis:** Tokenize and analyze text, showing tokens, their shapes, parts of speech, tags, lemmas, and stopword status.
        - **Named Entity Recognition:** Extract and visualize entities such as names, dates, and organizations from the input text.
        - **TEI XML Generation:** Takes a document as a sequence of tokens as input and generates a downloadable Text Encoding Initiative (TEI) compliant XML document.
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

        """)

        
    else:
        st.subheader("Documentation")
        st.markdown("""
        
        ##### 1. The 'app.py' is the main Streamlit app that allows users to analyze and explore text data. Here's an explanation of the functions and methods used in the code:
        - **text_analyzer(my_text):** This function takes a text as input and performs analysis on it using spaCy. It extracts various attributes for each token in the text, such as token shape, part-of-speech tag, lemma, and whether it is alphabetic or a stopword. The function returns the analysis results as a pandas DataFrame.
        - **get_entities(my_text):** This function takes a text as input and uses spaCy to extract named entities from the text. It returns a list of tuples, where each tuple contains the text of the entity and its label.
        - **render_entitis(rawtext):** This function takes raw text as input and uses spaCy's displacy module to visualize the named entities in the text. It returns an HTML string containing the rendered visualization.
        - **generate_tei_xml_v2(doc):** Takes a 'spaCy' document as input and generates a Text Encoding Initiative (TEI) compliant XML document.
        - **plot_wordcloud(text):** This function generates a word cloud visualization from the given text using the WordCloud module from the wordcloud library. It removes stopwords and generates a word cloud based on the word frequencies in the text.
        - **extract_date(texts) and extract_date2(texts):** These functions extract dates from the given text using the datefinder library. The first function returns a single date, while the second function returns a list of dates.
        - **analyze_sentiment(letters) and analyze_sentiment1(documents):** These functions analyze the sentiment of each letter or document using the TextBlob library. They calculate the sentiment polarity (positive, negative, or neutral) and return a list of tuples containing the letter/document, date, and sentiment.
        - **analyze_temporal_spatial(letters):** This function associates each letter with a specific location (latitude and longitude) and returns a list of tuples containing the letter, date, latitude, and longitude. The location data is hardcoded in the lat and long lists.
        - **plot_pos_histogram(text):** This function generates a histogram plot showing the distribution of part-of-speech tags in the given text. It uses the Counter class from the collections module and seaborn for visualization.
        - **plot_word_freq(df, num_of_most_common):** This function generates a bar plot showing the most common words and their frequencies in the given DataFrame. It uses the barplot function from the seaborn library.
        - **create_map():** This function creates an interactive map using the folium library. It adds circle markers on the map based on latitude, longitude, ratings, and the number of letters received.
        - **word_stats(text):** This function calculates the number of tokens, sentences, words, and stopwords in the given text using spaCy.
        - **preprocess(text):** This function preprocesses the text by tokenizing it, removing stopwords, and lemmatizing the words. It returns a list of preprocessed words.
        - **create_dictionary_and_corpus(texts):** This function preprocesses a list of texts, creates a dictionary, and converts the texts into bag-of-words representation (corpus) using the gensim library.
        - **get_lda_topics(texts, num_topics, passes):** This function performs Latent Dirichlet Allocation (LDA) topic modeling on the given texts. It creates a dictionary and corpus using create_dictionary_and_corpus(), and then trains an LDA model using the ldamodel

        ##### 2. The 'rdflib_vespasiano.ipynb' notebook performs a sequence of data extraction, transformation, and loading (ETL) operations on the RDF dataset and it consists of three parts as follows.
        **The goal of the first part of the script is to load and parse the RDF dataset, extract relevant information from it, and store the extracted data for further analysis. This involves RDF data manipulation, data validation, and the creation of a CSV file and a pandas DataFrame to store the processed data.
        The steps can be summarized as follows:**
        1. **Importing Libraries:** Essential libraries, such as rdflib, are imported for RDF manipulation.
        2. **Loading RDF data:** RDF data is loaded from a file stored in Google Drive and a graph is created for manipulation.
        3. **Exploring the Data:** Initial data exploration is conducted to understand the dataset's structure and content.
        4. **Data Extraction:** Information about the content of various letters is extracted and stored in a list. This includes identifying the content of a specific letter, creating a list of letter URIs, and creating a list of letter contents.
        5. **Data Validation:** A check is performed to ensure that the number of letter URIs and their contents match.
        6. **DataFrame Creation and Saving:** A DataFrame is created to store the extracted letter contents and their URIs, and this DataFrame is then saved to a CSV file.
        7. **Additional Data Extraction:** Additional information about another specific letter is extracted and stored as a pandas Series.

        **The second part of the script focuses on extracting temporal information, i.e., dates from the dataset. It uses regular expressions to find and sort dates, attaches this date information to the previously extracted letter content, and saves this combined data for future use.**
        The steps can be summarized as follows:
        1. **Listing Letters:** A list of letters (excluding a problematic one) is created.
        2. **Date Collection:** For each letter, the date of writing is identified from the RDF dataset and stored in a DataFrame.
        3. **Sorting Dates:** Regular expressions are used to sort the dates based on their formatting. Any discrepancies in the dates are addressed.
        4. **Date Update:** The sorted dates replace the original dates in the DataFrame.
        5. **Data Consolidation:** The date information is added to the table of letters and contents, combining all relevant information in one place.
        6. **Data Export:** The consolidated data, consisting of letters, their content, and the dates, is saved to a new CSV file named 'letters-with-dates.csv'.

        **The third part focuess on enriching the DataFrame extended_fianl with additional information drawn from the RDF graph. This part is focused on refining the uri_from column and adding a few new columns to the DataFrame.**
        Here is a summary of the objectives of this part:
        1. **Refinement of Place Information:** The uri_from column, which stores the place information, is further refined. It was initially filled based on the uri_maintext column, but in this part, another approach is also used to update the uri_from values. This second approach uses the 'p-sender-letter' URIs instead of the main text URIs to find and update the corresponding place information.
        2. **Addition of New Columns:** The DataFrame is enriched with additional columns that represent new information:
        - uri_expr: The URI of the expressions associated with each letter.
        - sender-att: The URI of the sender attributions associated with each expression.
        - p-sender-letter: The URI of the entity generated by each sender attribution.
        3. **Data Export:** The final DataFrame, now with enriched information, is exported to a CSV file named 'extended-data.csv'.

        """)
        


if __name__ == '__main__':
    main()
