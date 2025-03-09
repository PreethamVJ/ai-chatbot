from flask import Flask, request, jsonify
from flask_cors import CORS  # Enable CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"  # You can use "small" or "large" if you want
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model and tokenizer loaded successfully!")

@app.route('/chat', methods=['POST'])
def chat():
    # Get the user's message from the request
    user_message = request.json.get('message')
    print(f"Received message: {user_message}")

    # Encode the user's input and generate a response
    inputs = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    bot_response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    print(f"Generated response: {bot_response}")

    # Return the bot's response as JSON
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask server in debug mode