import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt


df = pd.read_csv(r'G:\Meu Drive\Tcc\dados\noticias_desastres_jp.csv')

textos = (
    df['Título'].astype(str) + ' ' +
    df['Resumo'].astype(str)
)

sentimento = pipeline(
    "sentiment-analysis",
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")


def analisar_sentimento(texto):

    resultado = sentimento(texto[:512])[0]

    return pd.Series([
        resultado['label'],
        resultado['score']
    ])

df[['sentimento', 'confianca']] = textos.apply(
    analisar_sentimento
)

print(df[['Título', 'sentimento']].head())

df['sentimento'].value_counts().plot(kind='bar')

plt.title('Sentimentos das notícias')

plt.show()
df.to_csv(r'G:\Meu Drive\Tcc\dados\noticias_com_sentimento_final.csv')


