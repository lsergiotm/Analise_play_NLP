import streamlit as st
from unidecode import unidecode
from bertopic import BERTopic
from umap import UMAP
import pandas as pd
import spacy
from google_play_scraper import Sort, reviews, app
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import nltk
from io import BytesIO
import PyPDF2
import docx2txt
from bs4 import BeautifulSoup
import requests

# Baixar pacotes necessários do NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Funções auxiliares
def extract_text_from_pdf(file):
    buffer = BytesIO(file.read())
    reader = PyPDF2.PdfReader(buffer)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    text = docx2txt.process(file_path)
    return text

def get_text_from_web(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

def remove_stopwords(text):
    stop_words = set(nltk.corpus.stopwords.words('portuguese'))
    words = text.lower().split()
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return filtered_words

def generate_bar_chart(word_freq):
    words, frequencies = zip(*word_freq)
    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Palavras')
    plt.ylabel('Frequência')
    plt.title('Top 20 Palavras Mais Frequentes')
    st.pyplot(plt)

# Função principal
def main():
    st.title('Análise de Avaliações de Aplicativos')
    
    id_app = st.text_input('Insira o ID do aplicativo Google Play', 'com.twitter.android')
    
    if st.button('Analisar'):
        result, continuation_token = reviews(
            id_app,
            lang='pt',
            country='br',
            sort=Sort.MOST_RELEVANT,
            count=1000,
        )

        df = pd.DataFrame(result)
        df = df[df['score'].isin([1, 2])]
        sample_size = min(500, len(df))
        df = df.sample(sample_size, replace=False)
        df['content'] = df['content'].apply(lambda x: str(x).encode('utf-8', errors='replace').decode())

        # Carregar modelo de linguagem
        nlp = spacy.load('pt_core_news_sm')
        df['clear_content'] = df['content'].apply(lambda x: [token.lemma_ for token in nlp(x.lower()) if (token.is_alpha and not token.is_stop)])
        df['clear'] = df['clear_content'].apply(lambda arr: ' '.join(arr))
        df['clear'] = df['clear'].apply(unidecode)

        model = BERTopic(language="portuguese", nr_topics='auto')
        docs = df['clear'].values
        docs_clean = [doc for doc in docs if len(doc) > 40]

        if len(docs_clean) > 0:
            try:
                topics, probs = model.fit_transform(docs_clean)
                freq = model.get_topic_info()
                st.write(freq)

                data = {'Doc': docs_clean, 'Topic': topics, 'Prob': probs}
                df_topics = pd.DataFrame(data)

                st.write(df_topics.sample(30))

                # Distribuição dos tópicos
                plt.figure(figsize=(10, 6))
                sns.countplot(y='Topic', data=df_topics, order=df_topics['Topic'].value_counts().index)
                plt.title('Distribuição dos Tópicos')
                plt.xlabel('Número de Documentos')
                plt.ylabel('Tópico')
                st.pyplot(plt)

                # Nuvem de palavras
                topic_num = 5
                topic_docs = df_topics[df_topics['Topic'] == topic_num]['Doc']
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(topic_docs))
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Nuvem de Palavras para o Tópico {topic_num}')
                st.pyplot(plt)

                # Análise de sentimento
                df['sentiment'] = df['content'].apply(lambda x: TextBlob(x).sentiment.polarity)
                df_topics['sentiment'] = df_topics['Doc'].apply(lambda x: TextBlob(x).sentiment.polarity)
                
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='Topic', y='sentiment', data=df_topics)
                plt.title('Sentimento por Tópico')
                plt.xlabel('Tópico')
                plt.ylabel('Sentimento')
                st.pyplot(plt)

            except IndexError:
                st.write("Não foi possível gerar tópicos suficientes.")
        else:
            st.write("Não há documentos suficientes para a modelagem de tópicos.")
        
        # Análise temporal das avaliações
        df['date'] = pd.to_datetime(df['at'])
        df.set_index('date', inplace=True)
        df.resample('M').size().plot()
        plt.title('Número de Avaliações por Mês')
        plt.xlabel('Mês')
        plt.ylabel('Número de Avaliações')
        st.pyplot(plt)

if __name__ == "__main__":
    main()
