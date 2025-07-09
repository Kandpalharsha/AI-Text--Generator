from flask import Flask, request, render_template, jsonify
from transformers import pipeline, set_seed

app = Flask(__name__)

# Load GPT-2 model
generator = pipeline('text-generation', model='gpt2')
set_seed(42)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json['prompt']
    result = generator(prompt, max_length=100, num_return_sequences=1)
    return jsonify({'generated_text': result[0]['generated_text']})

if __name__ == '__main__':
    app.run(debug=True)
