from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

# Load model architecture
with open('sentiment_emotion_model_architecture.json', 'r') as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
    
# Load model weights
with open('sentiment_emotion_model_weights.pkl', 'rb') as f:
    weights = pickle.load(f)

model.set_weights(weights)

# Load tokenizer and emotion labels
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('emotion_labels.pkl', 'rb') as f:
    emotion_labels = pickle.load(f)

# Determine input length for padding
input_length = 100  # Adjust this to match your training maxlen

# Preprocess text function (without using NLTK)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()  # Split by whitespace
    return ' '.join(tokens)  # Join tokens back to a single string

# Prediction function
def predict_emotion(sentence):
    processed_sentence = preprocess_text(sentence)
    sequence = tokenizer.texts_to_sequences([processed_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=input_length)
    prediction = model.predict(padded_sequence)
    emotion_index = np.argmax(prediction)  # Get the index of the highest probability
    emotion = emotion_labels[emotion_index]
    return emotion

@app.route('/')
def home():
    return render_template('samo.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        input_text = data.get('text', '')
        
        if not input_text:
            return jsonify({"error": "No text provided"}), 400

        predicted_emotion = predict_emotion(input_text)
        
        return jsonify({
            "input_text": input_text,
            "predicted_emotion": predicted_emotion
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/input', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        
        # Call the prediction function
        predicted_emotion = predict_emotion(user_input)
        
        return render_template('textinput.html', user_input=user_input, predicted_emotion=predicted_emotion)
    
    return render_template('textinput.html')

if __name__ == '__main__':
    app.run(debug=True)
