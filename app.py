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

@keyframes gradientMove {
0% {background-position:0% 50%;}
50% {background-position:100% 50%;}
100% {background-position:0% 50%;}
}

.stApp {
background: linear-gradient(270deg,#0f2027,#203a43,#2c5364);
background-size: 600% 600%;
animation: gradientMove 15s ease infinite;
color: white;
}

[data-testid="stSidebar"] {
background: linear-gradient(180deg,#000428,#004e92);
}

</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🤖 AI Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Sentiment Analyzer", "Live Dashboard", "Dataset Explorer", "About"]
)

# Load dataset
df = pd.read_csv("train.csv")

# Auto detect dataset columns
text_column = df.columns[0]
label_column = df.columns[1]

X = df.iloc[:,0]
y = df.iloc[:,1]

# Train model
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vectorized, y)

# ---------------- SENTIMENT ANALYZER ----------------

if page == "Sentiment Analyzer":

    st.title("🤖 AI Tweet Sentiment Analyzer")

    st.write("Enter a tweet and the AI will detect its sentiment.")

    user_input = st.text_area("Enter Tweet")

    if st.button("Analyze Sentiment"):

        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        confidence = max(probability) * 100

        if prediction == 1:
            st.success("😊 Positive Sentiment Detected")
        else:
            st.error("😡 Negative Sentiment Detected")

        st.subheader("AI Confidence Meter")

        st.progress(int(confidence))

        st.write(f"Confidence: {confidence:.2f}%")

# ---------------- LIVE DASHBOARD ----------------

elif page == "Live Dashboard":

    st.title("📊 Live Tweet Sentiment Dashboard")

    sentiment_counts = df[label_column].value_counts()

    chart_data = pd.DataFrame({
        "Sentiment": sentiment_counts.index.astype(str),
        "Count": sentiment_counts.values
    })

    fig = px.bar(
        chart_data,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        title="Sentiment Distribution"
    )

    st.plotly_chart(fig)

    st.subheader("Random Tweets from Dataset")

    st.dataframe(df.sample(10))

# ---------------- DATASET EXPLORER ----------------

elif page == "Dataset Explorer":

    st.title("📂 Dataset Explorer")

    st.write("Preview of dataset")

    st.dataframe(df.head(20))

# ---------------- ABOUT ----------------

else:

    st.title("About This Project")

    st.write("""
This AI application performs **Sentiment Analysis on Tweets**.

### Technologies Used

• Python  
• Streamlit  
• TF-IDF Vectorization  
• Logistic Regression  
• Interactive Dashboard

This project demonstrates how Machine Learning models can analyze social media sentiment in real-time.
""")
