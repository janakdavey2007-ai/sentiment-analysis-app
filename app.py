import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.title("Twitter Sentiment Analyzer")

df = pd.read_csv("train.csv")

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return text

df["clean_tweet"] = df["tweet"].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_tweet"])
y = df["label"]

model = LogisticRegression()
model.fit(X,y)

tweet = st.text_input("Enter a tweet")

if st.button("Predict Sentiment"):

    vec = vectorizer.transform([tweet])
    pred = model.predict(vec)

    if pred[0] == 1:
        st.success("Positive Sentiment 😊")
    else:
        st.error("Negative Sentiment 😠")