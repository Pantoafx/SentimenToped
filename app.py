import streamlit as st
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.exceptions import NotFittedError

# Muat model dari file .pkl
with open('svm_model.pkl', 'rb') as model_file:
    modelsvc_loaded = pickle.load(model_file)

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
                # Transformasi teks dengan vectorizer dan prediksi sentimen
                text_vector = modelsvc_loaded.named_steps['vectorizer'].transform([text_clean])
                prediction = modelsvc_loaded.named_steps['classifier'].predict(text_vector)

                # Tentukan label sentimen
                sentiment_label = 'positif' if prediction[0] == 'positif' else 'negatif'

                st.write(f'Sentimen teks adalah: {sentiment_label}')
            except NotFittedError as e:
                st.error(f'Model TF-IDF belum difit dengan benar. Pastikan model telah dilatih sebelum digunakan.')
        else:
            st.warning('Masukkan teks untuk menganalisis.')

# Fungsi pra-pemrosesan
def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = re.sub(r'\W+', ' ', text)  # Hapus simbol dan karakter non-alfanumerik
    text = text.lower()  # Ubah teks menjadi lowercase
    stemmer = StemmerFactory().create_stemmer()  # Inisialisasi stemmer
    text = stemmer.stem(text)  # Stemming
    return text

if __name__ == '__main__':
    main()
