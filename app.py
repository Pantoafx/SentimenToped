import streamlit as st
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Muat model dari file .pkl
with open('msvmstrem.pkl', 'rb') as model_file:
    modelsvc_loaded = pickle.load(model_file)

# Fungsi pra-pemrosesan
def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = re.sub(r'\W+', ' ', text)  # Hapus simbol dan karakter non-alfanumerik
    text = text.lower()  # Ubah teks menjadi lowercase
    stemmer = StemmerFactory().create_stemmer()  # Inisialisasi stemmer
    text = stemmer.stem(text)  # Stemming
    return text

# Halaman Utama
def main():
    uhm = 'logo.png'
    uhm1 = 'logo.png'
    sidebar_logo = uhm 
    main_body_logo = uhm1
    st.logo(sidebar_logo, icon_image=main_body_logo)
    st.sidebar.markdown('https://handayani.ac.id')

    st.markdown("<h1 style='text-align: center; color: #ff6347;'>Analisis Sentimen Ulasan Tokopedia</h1>", unsafe_allow_html=True)

    # Input teks dari pengguna
    user_input = st.text_area('Paste Ulasan Tokopedia disini...')

    # Tombol untuk analisis sentimen
    if st.button('Analisis Sentimen'):
        if user_input:
            # Pra-pemrosesan teks
            text_clean = preprocess_text(user_input)

            # Transformasi teks dengan model dan prediksi sentimen
            text_vector = modelsvc_loaded['vectorizer'].transform([text_clean])
            prediction = modelsvc_loaded['classifier'].predict(text_vector)

            # Tentukan label sentimen
            sentiment_label = 'positif' if prediction[0] == 'positif' else 'negatif'

            # Menambahkan emoji berdasarkan sentimen
            emoji = ':smiley:' if sentiment_label == 'positif' else ':persevere:'

            st.write(f'Sentimen : {sentiment_label} {emoji}')
        else:
            st.warning('Masukkan teks untuk menganalisis.')

if __name__ == '__main__':
    main()
