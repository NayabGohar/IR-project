import pandas as pd

df = pd.read_csv(
    r"C:\Users\Lenovo\Documents\Information Retrieval\ir_system\data\news-articles.csv",
    encoding="latin1"
)

print(df.head())
