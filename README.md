# cora
ex 1
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


ex 2



ex 3


ex 4
