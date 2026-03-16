import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Page config
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🤖", layout="wide")

# Animated background
st.markdown("""
<style>
.stApp {
background: linear-gradient(120deg,#1f4037,#99f2c8);
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("AI Dashboard")
page = st.sidebar.radio("Navigation", ["Sentiment Analyzer", "Live Dashboard", "Dataset Explorer", "About"])

# Load dataset
df = pd.read_csv("train.csv")

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vectorized, y)

# -------- SENTIMENT ANALYZER --------

if page == "Sentiment Analyzer":

    st.title("🤖 AI Sentiment Analyzer")

    text = st.text_area("Enter a Tweet")

    if st.button("Analyze Sentiment"):

        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        confidence = max(probability) * 100

        if prediction == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😡 Negative Sentiment")

        st.subheader("AI Confidence Meter")

        st.progress(int(confidence))
        st.write(f"Confidence Level: {confidence:.2f}%")

# -------- LIVE DASHBOARD --------

elif page == "Live Dashboard":

    st.title("📊 Live Tweet Sentiment Dashboard")

    sentiment_counts = df["label"].value_counts()

    chart_data = pd.DataFrame({
        "Sentiment": ["Negative", "Positive"],
        "Count": sentiment_counts
    })

    fig = px.bar(chart_data, x="Sentiment", y="Count", title="Sentiment Distribution")

    st.plotly_chart(fig)

    st.subheader("Dataset Preview")

    st.dataframe(df.sample(10))

# -------- DATASET EXPLORER --------

elif page == "Dataset Explorer":

    st.title("Dataset Explorer")

    st.write("First 10 rows of dataset")

    st.dataframe(df.head(10))

# -------- ABOUT --------

else:

    st.title("About This Project")

    st.write("""
This project demonstrates **AI-based Sentiment Analysis** using:

• TF-IDF Vectorization  
• Logistic Regression  
• Real-time predictions  
• Interactive Dashboard  

Built using Python and Streamlit.
""")
