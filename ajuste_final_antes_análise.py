import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv(r"G:\Meu Drive\Tcc\dados\noticias_desastres_jp.csv")
df2 = pd.read_excel(r"G:\Meu Drive\Tcc\dados\clima_G1_Folha_LIMPO.xlsx")



#Mudanças dados JP


# df.drop('Unnamed: 0', axis = 1, inplace = True)
# df.insert(0, "Portal", "Joven Pan")
# df.rename(columns={'titulo': 'Título'}, inplace = True)
# df.drop('site', axis = 1, inplace = True)


# aleatorio = df['descricao']
# df.drop('descricao', axis = 1, inplace = True)
# df.insert(2,'Resumo', aleatorio)


# aleatorio_dois = df['data']
# df.drop('data', axis = 1, inplace = True)
# df.insert(3,"Data", aleatorio_dois)
# df.rename(columns={'link': 'URL'}, inplace=True)



# df2['Data'] = pd.to_datetime(df2["Data"], format="%a, %d %b %Y %H:%M:%S GMT")
# df2['Data'].dt.strftime('%Y-%m-%d %H:%M:%S')


# print(df["Data"])

# print(df2["Data"])

# df_final = pd.concat([df, df2])


# print(df_final["Data"].dtypes)

# print(df2["Data"])

# df_final["Data"] = pd.to_datetime(df_final["Data"])

# noticias_por_ano = (
#     df_final.groupby(df_final["Data"].dt.year)
#       .size()
#       .reset_index(name="quantidade_noticias")
# )

# noticias_por_ano.plot(
#     x="Data",
#     y="quantidade_noticias",
#     kind="bar"
# )

# plt.xlabel("Ano")
# plt.ylabel("Quantidade de Notícias")
# plt.title("Notícias por Ano")

# plt.show()

df.drop('Unnamed: 0', inplace=True, axis=1)
print(df.head)
df.to_csv(r"G:\Meu Drive\Tcc\dados\noticias_desastres_jp.csv")