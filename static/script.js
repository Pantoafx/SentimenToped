function analyzeSentiment() {
    const text = document.getElementById('text').value;
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        const sentimentLabel = document.getElementById('sentimentLabel');
        resultDiv.innerText = ''; // Clear previous result
        if (data.sentiment === 'positif') {
            sentimentLabel.innerHTML = '😊 Sentimen: Positif';
            sentimentLabel.style.color = '#3ad15d'; // green color for positive sentiment
        } else if (data.sentiment === 'negatif') {
            sentimentLabel.innerHTML = '😡 Sentimen: Negatif';
            sentimentLabel.style.color = '#dc3545'; // red color for negative sentiment
        } else {
            sentimentLabel.innerText = 'Sentimen: ' + data.sentiment;
            sentimentLabel.style.color = '#ffc107'; // yellow color for other sentiments
        }
        sentimentLabel.style.display = 'flex';
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function clearText() {
    document.getElementById('text').value = '';
    document.getElementById('result').innerText = '';
    document.getElementById('sentimentLabel').innerText = '';
    document.getElementById('sentimentLabel').style.display = 'none';
}
