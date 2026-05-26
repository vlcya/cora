# cora
# ex 1
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://quotes.toscrape.com/"

response = requests.get(url)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

quotes_data = []

quotes = soup.find_all("div", class_="quote")

for quote in quotes:
    text = quote.find("span", class_="text").get_text(strip=True)
    author = quote.find("small", class_="author").get_text(strip=True)

    tags = [
        tag.get_text(strip=True)
        for tag in quote.find_all("a", class_="tag")
    ]

    quotes_data.append({
        "quote": text,
        "author": author,
        "tags": tags
    })

df = pd.DataFrame(quotes_data)

print(df.head(10))


# ex 2

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Clasele folosite
categories = ['comp.graphics', 'rec.autos']

# Încărcăm doar aceste două categorii
data = fetch_20newsgroups(
    subset='all',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

X = data.data
y = data.target

# Împărțim datele în train și test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Vectorizare text
vectorizer = CountVectorizer(stop_words='english')

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Model 1: Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_vectorized, y_train)

y_pred_logistic = logistic_model.predict(X_test_vectorized)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)

print("Logistic Regression accuracy:", accuracy_logistic)

# Model 2: Multinomial Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized, y_train)

y_pred_nb = nb_model.predict(X_test_vectorized)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

print("Multinomial Naive Bayes accuracy:", accuracy_nb)



ex 3


ex 4
