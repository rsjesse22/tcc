import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt


df = pd.read_csv(r'G:\Meu Drive\Tcc\dados\noticias_desastres.csv')

textos = (
    df['Título'].astype(str) + ' ' +
    df['Resumo'].astype(str)
)

sentimento = pipeline(
    "sentiment-analysis",
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")


def analisar_sentimento(texto):

    resultado = sentimento(texto[:512])[0]
    label = resultado['label'].lower()
    score = resultado['score']
    if 0.50 <= score <= 0.60:
        label = 'neutral'

    return pd.Series([
        label,
        score
    ])

df[['sentimento', 'confianca']] = textos.apply(
    analisar_sentimento
)

print(df[['Título', 'sentimento']].head())

df['sentimento'].value_counts().plot(kind='bar')

plt.title('Sentimentos das notícias')

plt.show()
df.to_csv(r'G:\Meu Drive\Tcc\dados\noticias_com_sentimento_final.csv')


