import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Language pair mapping
model_map = {
    "English to Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "English to French": "Helsinki-NLP/opus-mt-en-fr",
    "English to German": "Helsinki-NLP/opus-mt-en-de",
    "English to Spanish": "Helsinki-NLP/opus-mt-en-es"
}

@st.cache_resource
def load_model(name):
    tokenizer = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name)
    return tokenizer, model

st.title("🌍 Real-Time Language Translator using NMT")
lang_option = st.sidebar.selectbox("Select Language Pair", list(model_map.keys()))
model_name = model_map[lang_option]
tokenizer, model = load_model(model_name)

text = st.text_area("Enter text to translate:")

if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        tokens = tokenizer([text], return_tensors="pt", padding=True)
        translation = model.generate(**tokens)
        translated = tokenizer.decode(translation[0], skip_special_tokens=True)
        st.success("Translated Text:")
        st.write(translated)
