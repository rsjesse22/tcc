import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv(r'G:\Meu Drive\Tcc\dados\noticias_desastres_jp.csv')
print(len(df))

df_rotulo, df_restante = train_test_split(
    df,
    test_size=0.8,
    random_state=42
)

print(len(df_rotulo))
print(len(df_restante))


df_rotulo.to_csv(r'G:\Meu Drive\Tcc\dados\dados_para_rotular.csv')