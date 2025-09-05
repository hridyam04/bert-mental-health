import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = "../models/bert_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

st.title("🧠 Mental Health Sentiment & Emotion Detector")
user_input = st.text_area("Enter your text:")

if st.button("Analyze"):
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).tolist()[0]
    st.write("Emotion probabilities:", probs)
