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

# Function to predict rating
def predict_rating(proba):
    # Convert probability to a rating scale of 1 to 5
    rating = round(proba * 5, 2)
    return rating

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
                prediction_proba = modelsvc_loaded['classifier'].predict_proba(text_vector)
                
                # Get the probability of the positive class
                proba_positif = prediction_proba[0][1]
                
                # Predict rating
                rating = predict_rating(proba_positif)

                # Determine sentiment label
                sentiment_label = 'positif' if proba_positif >= 0.5 else 'negatif'

                # Load corresponding image
                if sentiment_label == 'positif':
                    image = Image.open('./images/positive.png')
                else:
                    image = Image.open('./images/negative.png')

                # Determine color based on sentiment
                color = 'green' if sentiment_label == 'positif' else 'red'

                # Display results in three columns with spacing
                st.components.v1.html("""
                                <h3 style="color: #0284c7; font-family: Source Sans Pro, sans-serif; font-size: 28px; margin-bottom: 10px; margin-top: 50px;">Hasil Analisa</h3>
                                """, height=100)
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                col1.image(image, caption=sentiment_label)
                col2.metric("Perkiraan Rating", rating, None)
                col3.metric("Bahasa", "Indonesia", None)
            else:
                st.warning('Mohon masukkan teks dalam Bahasa Indonesia.')
        else:
            st.warning('Masukkan teks untuk menganalisis.')

if __name__ == '__main__':
    main()
