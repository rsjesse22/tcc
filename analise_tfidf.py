import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

#TF-IDF para identificar termos mais relevantes em todos os títulos analisados.

df = pd.read_csv(r'G:\Meu Drive\Tcc\dados\noticias_desastres.csv')
print(df.columns)

df.drop('Unnamed: 0', axis=1, inplace=True)

textos = (
    df['Título'].astype(str) + ' ' +
    df['Resumo'].astype(str)
)

nltk.download('stopwords')
stopwords_pt = stopwords.words('portuguese')
print(stopwords_pt[:20])

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words= stopwords_pt,
    max_df=0.95,
    min_df=2
)
tfidf_matrix = vectorizer.fit_transform(textos)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)
ranking = tfidf_df.sum(axis=0).sort_values(ascending=False)

print(ranking.head(20))
