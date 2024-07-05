from flask import Flask, request, jsonify, render_template
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

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

# Halaman Utama
@app.route('/')
def home():
    return render_template('index.html')  # Render template HTML

# Endpoint untuk menganalisis sentimen
@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()  # Ambil data dari request
    text = data.get('text')  # Ambil teks dari data

    if not text:
        return jsonify({'error': 'Teks tidak boleh kosong'}), 400

    # Pra-pemrosesan teks
    text_clean = preprocess_text(text)

    # Transformasi teks dengan vectorizer dan prediksi sentimen
    text_vector = modelsvc_loaded.named_steps['vectorizer'].transform([text_clean])
    prediction = modelsvc_loaded.named_steps['classifier'].predict(text_vector)

    # Tentukan label sentimen
    sentiment_label = 'positif' if prediction[0] == 'positif' else 'negatif'

    return jsonify({'sentiment': sentiment_label})

if __name__ == '__main__':
    app.run(debug=True)
