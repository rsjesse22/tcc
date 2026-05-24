import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# df = pd.read_csv(r'G:\Meu Drive\Tcc\dados\noticias_desastres.csv')
# print(len(df))

# df_rotulo, df_restante = train_test_split(
#     df,
#     test_size=0.8,
#     random_state=42
# )


# print(len(df_rotulo))
# print(len(df_restante))


# df_rotulo.to_csv(r'G:\Meu Drive\Tcc\dados\dados_para_rotular.csv',index=False, encoding="utf-8-sig")
# df_restante.to_csv(r'G:\Meu Drive\Tcc\dados\dados_restantes.csv',index=False,encoding='utf-8-sig')

df_rotulado = pd.read_csv(r'G:\Meu Drive\Tcc\dados\dados_rotulados.csv', sep=";", encoding='latin1')
df_restante = pd.read_csv(r'G:\Meu Drive\Tcc\dados\dados_restantes.csv')

X_train = (df_rotulado['Título'].astype(str) + ' ' + df_rotulado['Resumo'].astype(str))
y_train = df_rotulado['sentimento']

X_pred = (df_restante['Título'].astype(str) + ' ' +df_restante['Resumo'].astype(str))

vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_pred_tfidf = vectorizer.transform(X_pred)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train_tfidf, y_train)

predicoes = modelo.predict(X_pred_tfidf)
probabilidades = modelo.predict_proba(X_pred_tfidf)

df_restante['sentimento'] = predicoes
df_restante['confianca'] = probabilidades.max(axis=1)

df_restante.to_csv(r'G:\Meu Drive\Tcc\dados\noticias_com_sentimento_final_reg.csv',index=False,encoding='utf-8-sig')

