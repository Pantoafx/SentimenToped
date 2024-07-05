import streamlit as st
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load model from .pkl file
with open('msvmstrem.pkl', 'rb') as model_file:
    modelsvc_loaded = pickle.load(model_file)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W+', ' ', text)  # Remove symbols and non-alphanumeric characters
    text = text.lower()  # Convert text to lowercase
    stemmer = StemmerFactory().create_stemmer()  # Initialize stemmer
    text = stemmer.stem(text)  # Stemming
    return text

# Determine color based on sentiment
def get_sentiment_color(sentimen):
    if sentimen == 'positif':
        return 'green'
    elif sentimen == 'negatif':
        return 'red'
    else:
        return 'black'  # Default if sentiment is unknown

# Main page function
def main():
    uhm = 'logo.png'
    uhm1 = 'logo.png'
    sidebar_logo = uhm 
    main_body_logo = uhm1
    st.logo(sidebar_logo, icon_image=main_body_logo)
    st.sidebar.markdown('https://handayani.ac.id')

    st.markdown("<h1 style='text-align: center; color: #ff6347;'>Analisis Sentimen Ulasan Tokopedia</h1>", unsafe_allow_html=True)

    # Input text from user
    user_input = st.text_area('Paste Ulasan Tokopedia disini...')

    # Button for sentiment analysis
    if st.button('Analisis Sentimen'):
        if user_input:
            # Preprocess the text
            text_clean = preprocess_text(user_input)

            # Transform text with the model and predict sentiment
            text_vector = modelsvc_loaded['vectorizer'].transform([text_clean])
            prediction = modelsvc_loaded['classifier'].predict(text_vector)

            # Determine sentiment label
            sentiment_label = 'positif' if prediction[0] == 'positif' else 'negatif'

            # Determine emoji based on sentiment
            emoji = '😃' if sentiment_label == 'positif' else '😟'

            # Determine color based on sentiment
            color = get_sentiment_color(sentiment_label)

            # Display sentiment result with color, larger text, and emoji
            st.markdown(f"<p style='font-size: 32px; color: {color};'>Sentimen : {sentiment_label} {emoji}</p>", unsafe_allow_html=True)
        else:
            st.warning('Masukkan teks untuk menganalisis.')

if __name__ == '__main__':
    main()
