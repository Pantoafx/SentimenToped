import streamlit as st
import pickle
import re
import langid
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from PIL import Image

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

# Function to detect language
def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

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
    userText = st.text_input('Halo', placeholder='Paste Ulasan Disini..')
    
    # Button for sentiment analysis
    if st.button('Analysis'):
        if userText:
            # Detect language
            lang = detect_language(userText)
            
            if lang == 'id':
                # Preprocess the text
                text_clean = preprocess_text(userText)

                # Transform text with the model and predict sentiment
                text_vector = modelsvc_loaded['vectorizer'].transform([text_clean])
                prediction = modelsvc_loaded['classifier'].predict(text_vector)

                # Determine sentiment label
                sentiment_label = 'positif' if prediction[0] == 'positif' else 'negatif'

                # Load corresponding image
                if sentiment_label == 'positif':
                    image = Image.open('./images/positive.PNG')
                else:
                    image = Image.open('./images/negative.PNG')

                # Determine color based on sentiment
                color = get_sentiment_color(sentiment_label)

                # Display sentiment result with color and larger text
                st.markdown(f"<p style='font-size: 32px; color: {color};'>Sentimen : {sentiment_label}</p>", unsafe_allow_html=True)
                
                # Display image
                st.image(image, caption=sentiment_label)

                # Display language and dummy rating (since we don't calculate actual rating here)
                col1, col2 = st.columns(2)
                col1.metric("Perkiraan Rating", "N/A", None)
                col2.metric("Bahasa", "Indonesia", None)
            else:
                st.warning('Mohon masukkan teks dalam Bahasa Indonesia.')
        else:
            st.warning('Masukkan teks untuk menganalisis.')

if __name__ == '__main__':
    main()
