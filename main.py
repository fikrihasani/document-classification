import model
import pandas as pd

content_type = "title"
df = pd.read_csv(
    "data/data-"+content_type+".csv")

X = df["Judul"].str.lower()
y = df["Topik"]

model.build(X,y)
