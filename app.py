from flask import Flask, request, jsonify, render_template
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

# Path to English to Italian translation model and tokenizer
translate_eng_to_it_model_path = r"NLPProjectModels/translation_model_en_to_it"
translate_eng_to_it_tokenizer_path = r"NLPProjectModels/translation_tokenizer_en_to_it"

# Load translation model & tokenizer
translate_eng_to_it_model = MarianMTModel.from_pretrained(translate_eng_to_it_model_path)
translate_eng_to_it_tokenizer = MarianTokenizer.from_pretrained(translate_eng_to_it_tokenizer_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text')
    target_lang = data.get('target_lang')  # should be 'it'

    if not text or target_lang != 'it':
        return jsonify({"error": "Missing input text or invalid target_lang"}), 400

    # Translation: English to Italian
    inputs = translate_eng_to_it_tokenizer(text, return_tensors="pt")
    translated_tokens = translate_eng_to_it_model.generate(**inputs)
    translated_text = translate_eng_to_it_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return jsonify({
        "translated_text": translated_text,
        "sentiment_class": "N/A",
        "confidence": "N/A",
        "ner": ["Not available"],
        "pos": ["Not available"]
    })

if __name__ == '__main__':
    app.run(debug=True)
