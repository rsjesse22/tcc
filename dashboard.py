import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
#import warnings
#warnings.filterwarnings("ignore")

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard de Sentimentos",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta de cores ───────────────────────────────────────────────────────────
CORES = {
    "negative": "#E24B4A",
    "positive": "#639922",
    "neutral":  "#888780",
    "NEG":      "#E24B4A",
    "POS":      "#639922",
    "NEU":      "#888780",
}
MODELO_NOMES = {
    "csv1": "robertuito (títulos)",
    "csv2": "distilbert (título + resumo)",
    "csv3": "Regressão Logística + TF-IDF",
}

# ── CSS customizado ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 500; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; color: #888; }
    .section-title {
        font-size: 0.72rem; letter-spacing: 0.1em; color: #aaa;
        text-transform: uppercase; margin-bottom: 0.5rem; margin-top: 1.5rem;
    }
    div[data-testid="stHorizontalBlock"] { align-items: stretch; }
</style>
""", unsafe_allow_html=True)


# ── Funções de carregamento ───────────────────────────────────────────────────
@st.cache_data
def carregar_csv1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # extrai sentimento e confiança do dicionário na coluna analise_sentimento
    def extrair(val):
        try:
            d = ast.literal_eval(str(val))
            return d.get("sentimento", None), float(d.get("confianca", 0))
        except Exception:
            return None, None
    df[["sentimento", "confianca"]] = df["analise_sentimento"].apply(
        lambda v: pd.Series(extrair(v))
    )
    # normaliza labels NEG/POS/NEU → negative/positive/neutral
    mapa = {"NEG": "negative", "POS": "positive", "NEU": "neutral"}
    df["sentimento"] = df["sentimento"].map(mapa).fillna(df["sentimento"])
    df["modelo"] = MODELO_NOMES["csv1"]
    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"], errors="coerce")
    return df

@st.cache_data
def carregar_csv2_csv3(path: str, modelo_key: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # detecta colunas de sentimento e confiança
    col_sent = next((c for c in df.columns if c.lower() in ["sentimento", "sentiment"]), None)
    col_conf = next((c for c in df.columns if c.lower() in ["confianca", "confiança", "confidence"]), None)
    if col_sent:
        df["sentimento"] = df[col_sent].str.lower().str.strip()
    if col_conf:
        df["confianca"] = pd.to_numeric(df[col_conf], errors="coerce")
    df["modelo"] = MODELO_NOMES[modelo_key]
    col_data = next((c for c in df.columns if c.lower() == "data"), None)
    if col_data:
        df["data"] = pd.to_datetime(df[col_data], errors="coerce")
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📁 Carregar arquivos CSV")
    st.caption("Faça upload dos três arquivos gerados pelo seu pipeline NLP.")

    upload1 = st.file_uploader("CSV 1 — robertuito",        type="csv", key="u1")
    upload2 = st.file_uploader("CSV 2 — distilbert",         type="csv", key="u2")
    upload3 = st.file_uploader("CSV 3 — Regressão Logística", type="csv", key="u3")

    st.divider()
    st.markdown("## 🎛️ Filtros")
    filtro_sentimento = st.multiselect(
        "Sentimento",
        options=["negative", "positive", "neutral"],
        default=["negative", "positive", "neutral"],
    )
    filtro_confianca = st.slider(
        "Confiança mínima", min_value=0.0, max_value=1.0, value=0.0, step=0.05
    )

    st.divider()
    st.markdown("## ℹ️ Sobre")
    st.caption(
        "Dashboard desenvolvido para análise comparativa de sentimentos "
        "em notícias sobre desastres naturais — TCC."
    )


# ── Estado: sem uploads ───────────────────────────────────────────────────────
if not upload1 and not upload2 and not upload3:
    st.title("📰 Dashboard de Análise de Sentimentos")
    st.info(
        "👈 Faça upload dos seus arquivos CSV na barra lateral para começar. "
        "Você pode carregar um, dois ou os três arquivos ao mesmo tempo.",
        icon="📂",
    )
    st.markdown("""
    **Estrutura esperada de cada arquivo:**

    | Arquivo | Modelo | Colunas principais |
    |---------|--------|--------------------|
    | CSV 1 | robertuito-sentiment-analysis | `titulo_original`, `analise_sentimento` (dict) |
    | CSV 2 | distilbert-multilingual | `Título`, `Resumo`, `sentimento`, `confianca` |
    | CSV 3 | Regressão Logística + TF-IDF | `Título`, `Resumo`, `sentimento`, `confianca` |
    """)
    st.stop()


# ── Carregamento dos dados ────────────────────────────────────────────────────
dfs = []
if upload1:
    dfs.append(carregar_csv1(upload1))
if upload2:
    dfs.append(carregar_csv2_csv3(upload2, "csv2"))
if upload3:
    dfs.append(carregar_csv2_csv3(upload3, "csv3"))

df_todos = pd.concat(dfs, ignore_index=True)

# aplica filtros
df_filtrado = df_todos[
    df_todos["sentimento"].isin(filtro_sentimento) &
    (df_todos["confianca"] >= filtro_confianca)
].copy()

modelos_carregados = df_filtrado["modelo"].unique().tolist()


# ── Cabeçalho ─────────────────────────────────────────────────────────────────
st.title("📰 Dashboard de Análise de Sentimentos")
st.caption(f"{len(modelos_carregados)} modelo(s) carregado(s) · {len(df_filtrado):,} notícias após filtros")
st.divider()


# ── Métricas globais ──────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Visão geral</p>', unsafe_allow_html=True)

contagem = df_filtrado["sentimento"].value_counts()
total    = len(df_filtrado)
conf_med = df_filtrado["confianca"].mean()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total de notícias", f"{total:,}")
col2.metric("Negativas",  f"{contagem.get('negative', 0):,}",
            f"{contagem.get('negative',0)/total*100:.1f}%" if total else "—")
col3.metric("Positivas",  f"{contagem.get('positive', 0):,}",
            f"{contagem.get('positive',0)/total*100:.1f}%" if total else "—")
col4.metric("Neutras",    f"{contagem.get('neutral', 0):,}",
            f"{contagem.get('neutral',0)/total*100:.1f}%" if total else "—")
col5.metric("Confiança média", f"{conf_med:.3f}" if not pd.isna(conf_med) else "—")

st.divider()


# ── Aba de navegação ──────────────────────────────────────────────────────────
aba1, aba2, aba3, aba4 = st.tabs([
    "📊 Distribuição",
    "📈 Confiança",
    "🔀 Comparação entre modelos",
    "🗞️ Explorar notícias",
])


# ═══════════════════════════════════════════════════════════════════════════════
# ABA 1 — Distribuição
# ═══════════════════════════════════════════════════════════════════════════════
with aba1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Distribuição geral de sentimentos")
        fig_pie = px.pie(
            df_filtrado,
            names="sentimento",
            color="sentimento",
            color_discrete_map=CORES,
            hole=0.45,
        )
        fig_pie.update_traces(textinfo="percent+label", pull=[0.03]*3)
        fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("#### Volume por sentimento")
        df_bar = df_filtrado["sentimento"].value_counts().reset_index()
        df_bar.columns = ["sentimento", "count"]
        fig_bar = px.bar(
            df_bar, x="sentimento", y="count",
            color="sentimento", color_discrete_map=CORES,
            text="count",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            showlegend=False,
            xaxis_title="", yaxis_title="Notícias",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Série temporal — só se houver coluna data
    if "data" in df_filtrado.columns and df_filtrado["data"].notna().any():
        st.markdown("#### Evolução temporal dos sentimentos")
        df_tempo = (
            df_filtrado.dropna(subset=["data"])
            .assign(mes=lambda d: d["data"].dt.to_period("M").astype(str))
            .groupby(["mes", "sentimento"])
            .size()
            .reset_index(name="count")
        )
        fig_line = px.line(
            df_tempo, x="mes", y="count",
            color="sentimento", color_discrete_map=CORES,
            markers=True,
        )
        fig_line.update_layout(
            xaxis_title="Mês", yaxis_title="Notícias",
            legend_title="Sentimento",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # Por portal — se existir coluna
    col_portal = next(
        (c for c in df_filtrado.columns if c.lower() in ["portal", "site", "fonte"]), None
    )
    if col_portal:
        st.markdown("#### Sentimentos por portal")
        df_portal = (
            df_filtrado.groupby([col_portal, "sentimento"])
            .size().reset_index(name="count")
        )
        fig_portal = px.bar(
            df_portal, x=col_portal, y="count",
            color="sentimento", color_discrete_map=CORES,
            barmode="group",
        )
        fig_portal.update_layout(
            xaxis_title="", yaxis_title="Notícias",
            legend_title="Sentimento",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_portal, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ABA 2 — Confiança
# ═══════════════════════════════════════════════════════════════════════════════
with aba2:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Distribuição de confiança por sentimento")
        fig_box = px.box(
            df_filtrado.dropna(subset=["confianca"]),
            x="sentimento", y="confianca",
            color="sentimento", color_discrete_map=CORES,
            points="outliers",
        )
        fig_box.update_layout(
            showlegend=False,
            xaxis_title="", yaxis_title="Confiança",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with c2:
        st.markdown("#### Histograma de confiança")
        fig_hist = px.histogram(
            df_filtrado.dropna(subset=["confianca"]),
            x="confianca", color="sentimento",
            color_discrete_map=CORES,
            nbins=20, barmode="overlay", opacity=0.75,
        )
        fig_hist.update_layout(
            xaxis_title="Score de confiança", yaxis_title="Frequência",
            legend_title="Sentimento",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### Confiança média por sentimento e modelo")
    df_conf_modelo = (
        df_filtrado.dropna(subset=["confianca"])
        .groupby(["modelo", "sentimento"])["confianca"]
        .mean().reset_index()
    )
    df_conf_modelo["confianca"] = df_conf_modelo["confianca"].round(4)
    fig_conf = px.bar(
        df_conf_modelo, x="modelo", y="confianca",
        color="sentimento", color_discrete_map=CORES,
        barmode="group", text="confianca",
    )
    fig_conf.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_conf.update_layout(
        xaxis_title="", yaxis_title="Confiança média",
        legend_title="Sentimento", yaxis_range=[0, 1.05],
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    # Dispersão confiança × sentimento (violin)
    st.markdown("#### Violin plot — densidade de confiança")
    fig_vio = px.violin(
        df_filtrado.dropna(subset=["confianca"]),
        x="sentimento", y="confianca",
        color="sentimento", color_discrete_map=CORES,
        box=True, points=False,
    )
    fig_vio.update_layout(
        showlegend=False,
        xaxis_title="", yaxis_title="Confiança",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_vio, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ABA 3 — Comparação entre modelos
# ═══════════════════════════════════════════════════════════════════════════════
with aba3:
    if len(modelos_carregados) < 2:
        st.info("Carregue pelo menos 2 arquivos CSV para habilitar a comparação entre modelos.")
    else:
        st.markdown("#### Distribuição de sentimentos por modelo")
        df_mod = (
            df_filtrado.groupby(["modelo", "sentimento"])
            .size().reset_index(name="count")
        )
        df_mod_total = df_filtrado.groupby("modelo").size().reset_index(name="total")
        df_mod = df_mod.merge(df_mod_total, on="modelo")
        df_mod["pct"] = (df_mod["count"] / df_mod["total"] * 100).round(1)

        fig_mod = px.bar(
            df_mod, x="modelo", y="pct",
            color="sentimento", color_discrete_map=CORES,
            barmode="stack", text="pct",
        )
        fig_mod.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
        fig_mod.update_layout(
            xaxis_title="", yaxis_title="Percentual (%)",
            legend_title="Sentimento",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_mod, use_container_width=True)

        st.markdown("#### Radar — perfil comparativo dos modelos")
        categorias = ["negative", "positive", "neutral"]
        fig_radar = go.Figure()
        for modelo in modelos_carregados:
            sub = df_filtrado[df_filtrado["modelo"] == modelo]
            total_mod = len(sub)
            valores = [
                sub[sub["sentimento"] == s].shape[0] / total_mod * 100
                if total_mod else 0
                for s in categorias
            ]
            valores_fechados = valores + [valores[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=valores_fechados,
                theta=categorias + [categorias[0]],
                fill="toself",
                name=modelo,
                opacity=0.6,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            legend_title="Modelo",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("#### Tabela comparativa de métricas por modelo")
        rows = []
        for modelo in modelos_carregados:
            sub = df_filtrado[df_filtrado["modelo"] == modelo]
            tot = len(sub)
            for sent in ["negative", "positive", "neutral"]:
                n = sub[sub["sentimento"] == sent].shape[0]
                conf_m = sub[sub["sentimento"] == sent]["confianca"].mean()
                rows.append({
                    "Modelo": modelo,
                    "Sentimento": sent,
                    "Qtd": n,
                    "%": f"{n/tot*100:.1f}%" if tot else "—",
                    "Confiança média": f"{conf_m:.4f}" if not pd.isna(conf_m) else "—",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ABA 4 — Explorar notícias
# ═══════════════════════════════════════════════════════════════════════════════
with aba4:
    st.markdown("#### Explorar notícias individualmente")

    col_busca, col_sent_f, col_mod_f = st.columns([3, 2, 2])
    with col_busca:
        busca = st.text_input("🔍 Buscar por palavra no título", "")
    with col_sent_f:
        sent_f = st.selectbox("Sentimento", ["Todos"] + ["negative", "positive", "neutral"])
    with col_mod_f:
        mod_f  = st.selectbox("Modelo", ["Todos"] + modelos_carregados)

    df_expl = df_filtrado.copy()

    col_titulo = next(
        (c for c in df_expl.columns if c.lower() in ["título", "titulo", "titulo_original"]), None
    )

    if busca and col_titulo:
        df_expl = df_expl[df_expl[col_titulo].str.contains(busca, case=False, na=False)]
    if sent_f != "Todos":
        df_expl = df_expl[df_expl["sentimento"] == sent_f]
    if mod_f != "Todos":
        df_expl = df_expl[df_expl["modelo"] == mod_f]

    st.caption(f"{len(df_expl):,} notícias encontradas")

    colunas_exibir = ["sentimento", "confianca", "modelo"]
    if col_titulo:
        colunas_exibir = [col_titulo] + colunas_exibir
    if "data" in df_expl.columns:
        colunas_exibir = colunas_exibir + ["data"]
    col_portal2 = next(
        (c for c in df_expl.columns if c.lower() in ["portal", "site", "fonte"]), None
    )
    if col_portal2 and col_portal2 not in colunas_exibir:
        colunas_exibir = [col_portal2] + colunas_exibir

    colunas_exibir = [c for c in colunas_exibir if c in df_expl.columns]

    st.dataframe(
        df_expl[colunas_exibir].sort_values("confianca", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
