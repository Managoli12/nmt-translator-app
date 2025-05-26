import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Language pair mapping
language_pairs = {
    ("English", "Hindi"): "Helsinki-NLP/opus-mt-en-hi",
    ("English", "French"): "Helsinki-NLP/opus-mt-en-fr",
    ("English", "German"): "Helsinki-NLP/opus-mt-en-de",
    ("English", "Spanish"): "Helsinki-NLP/opus-mt-en-es"
}

@st.cache_resource
def load_model(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Streamlit UI
st.title("Real-Time Language Translation")

# Dropdowns
source_lang = st.selectbox("Source Language", sorted(set(x[0] for x in language_pairs)))
target_lang = st.selectbox("Target Language", sorted(set(x[1] for x in language_pairs)))

# Check if pair exists
model_name = language_pairs.get((source_lang, target_lang))
if model_name:
    tokenizer, model = load_model(model_name)

    # Text input
    text = st.text_input("Enter text:")

    if st.button("Translate"):
        if text.strip() == "":
            st.warning("Please enter some text.")
        else:
            tokens = tokenizer([text], return_tensors="pt", padding=True)
            translated = model.generate(**tokens)
            result = tokenizer.decode(translated[0], skip_special_tokens=True)
            st.markdown("**Translated Text:**")
            st.write(result)
else:
    st.error("Translation for this language pair is not available.")
