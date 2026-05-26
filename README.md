# Exercițiul 1

Exercițiul 1: Extrageți citatul, autorul și tag-urile de pe prima pagină
a site-ului https://quotes.toscrape.com. Normalizați tag-urile ca listă de string-uri,
apoi stocați rezultatele într-un DataFrame pandas și afișați primele 10 rânduri.

```python
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
```

---

# Exercițiul 2

Exercițiul 2: Creați un model de clasificare text folosind un subset din 20 Newsgroups,
cu două clase: 'comp.graphics' și 'rec.autos'. Vectorizați textul, antrenați modelul și
calculați acuratețea pe setul de test.

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

categories = ['comp.graphics', 'rec.autos']

data = fetch_20newsgroups(
    subset='all',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = CountVectorizer(stop_words='english')

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Clase:", data.target_names)
```

Variantă alternativă cu Naive Bayes:

```python
nb_model = MultinomialNB()

nb_model.fit(X_train_vectorized, y_train)

y_pred_nb = nb_model.predict(X_test_vectorized)

accuracy_nb = accuracy_score(y_test, y_pred_nb)

print("Multinomial Naive Bayes accuracy:", accuracy_nb)
```

---

# Exercițiul 3

Exercițiul 3: Folosiți setul de date Diabetes din Scikit-learn pentru a
antrena un model de regresie Ridge. Folosiți GridSearchCV pentru a căuta
parametrul optim alpha și afișați eroarea MSE pe setul de test.

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])

param_grid = {
    "ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error"
)

grid_search.fit(X_train, y_train)

print("Cel mai bun alpha:", grid_search.best_params_["ridge__alpha"])

y_pred = grid_search.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("MSE pe setul de test:", mse)
```

Afișarea rezultatelor pentru toate valorile lui `alpha`:

```python
results = pd.DataFrame(grid_search.cv_results_)

print(results[[
    "param_ridge__alpha",
    "mean_test_score",
    "rank_test_score"
]])
```

---

# Exercițiul 4

Exercițiul 4: Implementați o rețea neuronală simplă folosind PyTorch pentru a clasifica
datele din setul Digits. Antrenați modelul pentru 100 de epoci și afișați acuratețea pe
setul de test.

```python
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

digits = load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)

model = SimpleNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()

    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoca [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")

model.eval()

with torch.no_grad():
    test_outputs = model(X_test_tensor)

    predicted_classes = torch.argmax(test_outputs, dim=1)

    accuracy = (predicted_classes == y_test_tensor).float().mean()

print("Acuratețea pe setul de test:", accuracy.item())
```
