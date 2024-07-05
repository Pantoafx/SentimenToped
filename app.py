import streamlit as st
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError

# Muat model dari file .pkl
with open('svm_model.pkl', 'rb') as model_file:
    modelsvc_loaded = pickle.load(model_file)

# Muat vectorizer dari file .pkl
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer_loaded = pickle.load(vectorizer_file)

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
    st.title('Aplikasi Analisis Sentimen')

    st.write('Masukkan teks untuk menganalisis sentimen:')

    # Input teks dari pengguna
    user_input = st.text_area('Teks')

    # Tombol untuk analisis sentimen
    if st.button('Analisis Sentimen'):
        if user_input:
            # Pra-pemrosesan teks
            text_clean = preprocess_text(user_input)

            try:
                # Transformasi teks dengan vectorizer
                text_vector = vectorizer_loaded.transform([text_clean])

                # Prediksi sentimen
                prediction = modelsvc_loaded.predict(text_vector)

                # Tentukan label sentimen
                sentiment_label = 'positif' if prediction[0] == 'positif' else 'negatif'

                st.write(f'Sentimen teks adalah: {sentiment_label}')
            except NotFittedError as e:
                st.error(f'Model TF-IDF belum difit dengan benar. Pastikan model telah dilatih sebelum digunakan.')
        else:
            st.warning('Masukkan teks untuk menganalisis.')

if __name__ == '__main__':
    main()
