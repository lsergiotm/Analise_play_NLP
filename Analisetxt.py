import streamlit as st
from google_play_scraper import app
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Função para obter dados de um aplicativo específico do Google Play
def get_app_data(app_id):
    result = app(app_id)
    return result

# Função para visualizar avaliações do aplicativo
def plot_app_reviews(reviews):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='score', data=reviews)
    plt.title('Distribuição das Avaliações')
    plt.xlabel('Pontuação')
    plt.ylabel('Contagem')
    st.pyplot(plt)

# Interface Streamlit
st.title('Análise de Aplicativos do Google Play')
app_id = st.text_input('Insira o ID do aplicativo do Google Play:', 'com.example.app')

if st.button('Obter Dados do Aplicativo'):
    if app_id:
        app_data = get_app_data(app_id)
        reviews = pd.DataFrame(app_data['comments'])

        st.subheader('Dados do Aplicativo')
        st.write(app_data)

        st.subheader('Distribuição das Avaliações')
        plot_app_reviews(reviews)
    else:
        st.error('Por favor, insira um ID de aplicativo válido.')
