import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

# Load dataset
df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None
)

# Add column names
df.columns = ["sentiment","id","date","query","user","text"]

df = df[["sentiment","text"]]

# Convert labels
df["sentiment"] = df["sentiment"].replace(0,"negative")
df["sentiment"] = df["sentiment"].replace(4,"positive")

# Reduce dataset size for faster training
df = df.sample(10000, random_state=42)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = text.lower()
    text = re.sub("[^a-zA-Z]"," ",text)
    words = text.split()

    words = [stemmer.stem(w) for w in words if w not in stop_words]

    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]

# Train test split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

# Prediction
y_pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))

# Test custom input
while True:

    text = input("Enter sentence (or exit): ")

    if text == "exit":
        break

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    pred = model.predict(vector)

    print("Sentiment:",pred[0])