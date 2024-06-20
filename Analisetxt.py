import sys
from collections import Counter
from Teste_trab_final import get_text_from_web, remove_stopwords, generate_bar_chart

def main():
    if len(sys.argv) != 2:
        print("Usage: python Teste_trab_final.py <play_store_app_link>")
        return

    app_link = sys.argv[1]
    print(f"Fetching data from: {app_link}")

    # Extraindo o texto da página da Play Store
    text = get_text_from_web(app_link)

    if not text:
        print("No text found at the provided link.")
        return

    # Removendo as palavras de parada
    filtered_words = remove_stopwords(text)

    # Contando a frequência das palavras
    word_freq = Counter(filtered_words).most_common(20)

    # Gerando o gráfico de frequência das palavras
    generate_bar_chart(word_freq)

if __name__ == "__main__":
    main()
