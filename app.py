import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer

# Initialize your text generation model and tokenizer
model = BartForConditionalGeneration.from_pretrained("/content/drive/MyDrive/hinglish/final_model")
tokenizer = BartTokenizer.from_pretrained("/content/drive/MyDrive/hinglish/final_model")

# Define the Streamlit app
st.title("Text Generation App")
user_input = st.text_area("Enter a text prompt", "")
generate_button = st.button("Generate Text")

if generate_button and user_input:
    input_ids = tokenizer.encode(user_input, return_tensors="pt", max_length=1024, truncation=True)
    output = model.generate(input_ids, max_length=150, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    st.subheader("Generated Text:")
    st.write(generated_text)

st.write("")

# Optionally, add more features and descriptions to your app
