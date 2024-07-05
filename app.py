import streamlit as st
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Muat model dan vectorizer dari file .pkl
with open('model_svm.pkl', 'rb') as model_file:
    modelsvc_loaded = pickle.load(model_file)

# Fungsi pra-pemrosesan
def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = re.sub(r'\W+', ' ', text)  # Hapus simbol dan karakter non-alfanumerik
    text = text.lower()  # Ubah teks menjadi lowercase
    stemmer = StemmerFactory().create_stemmer()  # Inisialisasi stemmer
    text = stemmer.stem(text)  # Stemming
    return text

# Streamlit app
st.title('Sentimen Analisis')

text_input = st.text_area("Masukkan teks untuk analisis sentimen:")

if st.button('Analisis'):
    if text_input:
        text_clean = preprocess_text(text_input)
        text_vector = modelsvc_loaded.named_steps['vectorizer'].transform([text_clean])
        prediction = modelsvc_loaded.named_steps['classifier'].predict(text_vector)
        sentiment_label = 'positif' if prediction[0] == 'positif' else 'negatif'
        st.write(f'Sentimen: {sentiment_label}')
    else:
        st.write('Teks tidak boleh kosong')
