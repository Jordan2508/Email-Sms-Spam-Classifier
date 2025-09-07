import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer3.pkl','rb'))
model = pickle.load(open('model3.pkl','rb'))

# --- CSS Styling with classy design ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
    }

    /* Make all text bold and black */
    .stApp, .stMarkdown, .stTextInput, .stTextArea, .stButton, .stHeader, .stSubheader, .stTitle {
        color: black !important;
        font-weight: bold !important;
    }

    /* Force widget labels to black */
    label, .stTextArea label, .stTextInput label {
        color: black !important;
        font-weight: bold !important;
        font-size: 18px !important;
    }

    /* Input box styling */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.85);
        font-weight: bold;
        color: black;
        border-radius: 10px;
    }

    /* Button styling */
    .stButton button {
        background: rgba(255, 255, 255, 0.95);
        color: black;
        font-weight: bold;
        border-radius: 12px;
        border: 2px solid black;
        font-size: 18px;
        transition: 0.3s;
    }
    .stButton button:hover {
        background: yellow !important;
        color: black !important;
        border: 2px solid black;
    }

    /* Result box styling */
    .result-box {
        background: rgba(255, 255, 255, 0.85);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app title
st.title("✉️📱 Email / SMS Spam Classifier")

# Input area
input_sms = st.text_area("✍️ Write a message:")

# Predict button
if st.button("🔍 Predict"):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display result
    if result == 1:
        st.markdown('<div class="result-box">🚨 This message looks like <span style="color:red;">SPAM</span>!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box">✅ This message is <span style="color:green;">NOT SPAM</span>.</div>', unsafe_allow_html=True)




