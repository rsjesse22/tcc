import pandas as pd
from matplotlib import pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)
df = pd.read_csv(r"G:\Meu Drive\Tcc\dados\noticias_desastres_jp_puro.csv")
df2 = pd.read_excel(r"G:\Meu Drive\Tcc\dados\clima_G1_Folha_LIMPO.xlsx")



# Mudanças dados JP



df.insert(0, "Portal", "Joven Pan")
df.rename(columns={'titulo': 'Título'}, inplace = True)
# print(df.head())
# print(df.columns)


aleatorio = df['descricao']
df.drop('descricao', axis = 1, inplace = True)
df.insert(2,'Resumo', aleatorio)
# print(df.columns)
# print(df.head())

aleatorio_dois = df['data']
df.drop('data', axis = 1, inplace = True)
df.insert(3,"Data", aleatorio_dois)
df.rename(columns={'link': 'URL'}, inplace=True)
# print(df.columns)
# print(df.head())


df2['Data'] = pd.to_datetime(df2["Data"], format="%a, %d %b %Y %H:%M:%S GMT")
df2['Data'].dt.strftime('%Y-%m-%d %H:%M:%S')

df['Data'] = pd.to_datetime(
    df['Data'],
    format='%d/%m/%Y %Hh%M%S'
)
print(df["Data"])

print(df2["Data"])

df_final = pd.concat([df, df2])


print(df_final["Data"].dtypes)


noticias_por_ano = (
    df_final.groupby(df_final["Data"].dt.year)
      .size()
      .reset_index(name="quantidade_noticias")
)

noticias_por_ano.plot(
    x="Data",
    y="quantidade_noticias",
    kind="bar"
)

plt.xlabel("Ano")
plt.ylabel("Quantidade de Notícias")
plt.title("Notícias por Ano")

plt.show()

# print(df_final.head)
df_final.to_csv(r"G:\Meu Drive\Tcc\dados\noticias_desastres.csv", index=False, encoding="utf-8-sig")