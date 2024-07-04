import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to generate response
def generate_response(prompt, model, tokenizer, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit app
def main():
    st.title("GPT-2 Chatbot")
    user_input = st.text_input("You: ", "")

    if user_input:
        response = generate_response(user_input, model, tokenizer)
        st.text_area("Bot:", value=response, height=200)

if __name__ == "__main__":
    main()
