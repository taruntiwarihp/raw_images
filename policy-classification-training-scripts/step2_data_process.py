import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

df = pd.read_csv("full_final_data.csv")
# This page intentionally left blank

print(df.shape)

df.drop_duplicates(subset=['Sentence'], keep='first', inplace=True)

df['Sentence'].replace('', np.nan, inplace=True)

df.dropna(subset=['Sentence'], inplace=True)

df = df[df.astype(str)['Sentence'] != "This page intentionally left blank"]
df["lemm_sent"] = df["Sentence"].apply(lambda text: lemmatize_words(text))
df["lemm_sent"] = df["lemm_sent"].str.lower()
df["lemm_multi"] = df["Sentence"].apply(lambda text: lemmatize_words(text))
df["lemm_multi"] = df["lemm_multi"].str.lower()
df.reset_index(drop=True, inplace=True)
print(df.shape)

#  ['Labels', 'Img_pth', 'Texts', 'Sentence','Multi_column', 'lemm_sent', 'lemm_multi']


df.to_csv("data/final_filtered_data.csv", index=False)