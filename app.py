import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🤖", layout="wide")

# Background style
st.markdown("""
<style>
.stApp {
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("AI Dashboard")

page = st.sidebar.radio(
"Navigation",
["Sentiment Analyzer","Live Dashboard","Dataset Explorer","About"]
)

# Load dataset
df = pd.read_csv("train_reviews_dataset.csv")

# Remove empty rows
df = df.dropna()

# Detect text column automatically
text_column = None
for col in df.columns:
    if df[col].dtype == "object":
        text_column = col
        break

# Detect label column
label_column = [c for c in df.columns if c != text_column][0]

# Prepare data
X = df[text_column].astype(str)
y = df[label_column]

# Train model
vectorizer = TfidfVectorizer(stop_words="english")
X_vector = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vector, y)

# Sentiment Analyzer
if page == "Sentiment Analyzer":

    st.title("AI Tweet Sentiment Analyzer")

    user_text = st.text_area("Enter tweet")

    if st.button("Analyze Sentiment"):

        vec = vectorizer.transform([user_text])
        pred = model.predict(vec)[0]
        confidence = max(model.predict_proba(vec)[0]) * 100

        if pred == 1:
            st.success("Positive Sentiment 😊")
        else:
            st.error("Negative Sentiment 😡")

        st.progress(int(confidence))
        st.write("Confidence:", round(confidence,2), "%")

# Dashboard
elif page == "Live Dashboard":

    st.title("Sentiment Dashboard")

    counts = df[label_column].value_counts()

    chart_data = pd.DataFrame({
        "Sentiment": counts.index.astype(str),
        "Count": counts.values
    })

    fig = px.bar(chart_data,x="Sentiment",y="Count",color="Sentiment")
    st.plotly_chart(fig)

# Dataset Explorer
elif page == "Dataset Explorer":

    st.title("Dataset Preview")
    st.dataframe(df.head(20))

# About
else:

    st.title("About Project")

    st.write("""
This AI web app performs sentiment analysis on tweets using machine learning.
Built using Python, Streamlit, and Scikit-Learn.
""")
