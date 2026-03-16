import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🤖", layout="wide")

# Animated background
st.markdown("""
<style>

@keyframes gradient {
0% {background-position:0% 50%;}
50% {background-position:100% 50%;}
100% {background-position:0% 50%;}
}

.stApp {
background: linear-gradient(270deg,#0f2027,#203a43,#2c5364);
background-size: 600% 600%;
animation: gradient 15s ease infinite;
color: white;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.title("🤖 AI Dashboard")

page = st.sidebar.radio(
"Navigation",
["Sentiment Analyzer","Live Dashboard","Dataset Explorer","About"]
)

# Load dataset
df = pd.read_csv("train.csv")

# Automatically detect columns
text_column = df.columns[0]
label_column = df.columns[1]

X = df[text_column]
y = df[label_column]

# Train model
vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vector,y)

# SENTIMENT ANALYZER
if page == "Sentiment Analyzer":

    st.title("🤖 AI Tweet Sentiment Analyzer")

    text = st.text_area("Enter Tweet")

    if st.button("Analyze"):

        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]
        confidence = max(model.predict_proba(vector)[0])*100

        if prediction == 1:
            st.success("Positive Sentiment 😊")
        else:
            st.error("Negative Sentiment 😡")

        st.progress(int(confidence))
        st.write("Confidence:",round(confidence,2),"%")

# DASHBOARD
elif page == "Live Dashboard":

    st.title("📊 Sentiment Dashboard")

    sentiment_counts = df[label_column].value_counts()

    chart_data = pd.DataFrame({
        "Sentiment": sentiment_counts.index.astype(str),
        "Count": sentiment_counts.values
    })

    fig = px.bar(chart_data,x="Sentiment",y="Count",color="Sentiment")

    st.plotly_chart(fig)

# DATASET
elif page == "Dataset Explorer":

    st.title("Dataset Preview")
    st.dataframe(df.head(20))

# ABOUT
else:

    st.title("About Project")

    st.write("""
This project uses Machine Learning to detect sentiment from tweets.

Technologies used:
- Python
- Streamlit
- TF-IDF
- Logistic Regression
""")
