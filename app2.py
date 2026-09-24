import streamlit as st
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

# Configuração da página web
st.set_page_config(page_title="Motor de Auditoria IA", page_icon="🏥", layout="wide")

st.title("🏥 Motor de Auditoria Clínica e Atuarial com IA")
st.markdown(
    "Sistema automatizado para previsão de custos médicos e justificativa de sinistralidade extrema (Outliers) através de Machine Learning e SHAP.")


# =====================================================================
# 1. CACHE: CARREGAMENTO E TREINO
# =====================================================================
@st.cache_resource
def carregar_e_treinar_modelo():
    arquivo = "insurance.csv" if os.path.exists("insurance.csv") else "medical-charges.csv"
    df = pd.read_csv(arquivo, sep=None, engine="python", encoding="latin1")

    df = df.rename(columns={
        "age": "idade", "sex": "genero", "children": "filhos",
        "smoker": "fumante", "region": "região", "charges": "custo", "imc": "bmi"
    })

    if df["custo"].dtype == 'object':
        df["custo"] = df["custo"].astype(str).str.replace(".", "", regex=False).str.replace(",", ".",
                                                                                            regex=False).astype(float)
    if df["bmi"].dtype == 'object':
        df["bmi"] = df["bmi"].astype(str).str.replace(",", ".", regex=False).astype(float)

    df["genero"] = df["genero"].replace({"male": "masculino", "female": "feminino"})
    df["fumante"] = df["fumante"].replace({"yes": "sim", "no": "nao"})
    df["região"] = df["região"].replace({
        "southwest": "sudoeste", "southeast": "sudeste",
        "northwest": "noroeste", "northeast": "nordeste"
    })

    df["perfil_altissimo_risco"] = ((df["bmi"] >= 30) & (df["fumante"] == "sim")).astype(int)
    df_encoded = pd.get_dummies(df, columns=["genero", "fumante", "região"], drop_first=True)

    X = df_encoded.drop(columns=["custo"])
    y = df_encoded["custo"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    predicoes = modelo.predict(X_test)
    rmse = mean_squared_error(y_test, predicoes) ** 0.5
    r2 = r2_score(y_test, predicoes)
    mae = mean_absolute_error(y_test, predicoes)

    return modelo, X_test, y_test, r2, mae, rmse


modelo, X_test, y_test, r2, mae, rmse = carregar_e_treinar_modelo()

# =====================================================================
# 2. SEÇÃO VISUAL: DESEMPENHO DO MODELO
# =====================================================================
st.header("📊 Desempenho Global do Algoritmo")
col1, col2, col3 = st.columns(3)

# Acurácia ajustada para %
col1.metric("Acurácia da Base (R²)", f"{(r2 * 100):.2f}%", "Confiança Alta")
col2.metric("Margem de Erro Média (MAE)", f"${mae:,.2f}")
col3.metric("Desvio Padrão (RMSE)", f"${rmse:,.2f}")

st.divider()

# =====================================================================
# 3. SEÇÃO VISUAL: AUDITORIA DE OUTLIER (SHAP)
# =====================================================================
st.header("🔎 Auditoria do Maior Outlier (Conta Extrema)")

indice_outlier = y_test.argmax()
paciente_x = X_test.iloc[indice_outlier:indice_outlier + 1]
custo_real = y_test.iloc[indice_outlier]
custo_previsto = modelo.predict(paciente_x)[0]

explainer = shap.TreeExplainer(modelo)
shap_values = explainer.shap_values(paciente_x)

try:
    valor_base = float(explainer.expected_value[0])
except TypeError:
    valor_base = float(explainer.expected_value)

colA, colB, colC = st.columns(3)
colA.metric("Custo Faturado (Real)", f"${custo_real:,.2f}")
colB.metric("Custo Estimado (IA)", f"${custo_previsto:,.2f}")
colC.metric("Custo Médio da Operadora", f"${valor_base:,.2f}")

dicionario_fatores = {
    "perfil_altissimo_risco": "Perfil Altíssimo Risco (Obesidade + Fumo)",
    "fumante_sim": "Tabagismo",
    "idade": "Idade do Paciente",
    "bmi": "Índice de Massa Corporal (IMC)",
    "genero_masculino": "Gênero (Masculino)",
    "região_noroeste": "Residência (Noroeste)",
    "região_sudoeste": "Residência (Sudoeste)",
    "região_sudeste": "Residência (Sudeste)",
    "filhos": "Dependentes na Apólice"
}


def formatar_valor(fator, valor):
    if fator in ["perfil_altissimo_risco", "fumante_sim", "genero_masculino", "região_noroeste", "região_sudoeste",
                 "região_sudeste"]:
        return "Sim" if valor in [1, 1.0, True, "True"] else "Não"
    if fator == "bmi": return f"{float(valor):.1f}"
    if fator in ["idade", "filhos"]: return str(int(valor))
    return str(valor)


dados_tabela = []
agravantes, atenuantes = [], []

pesos = pd.DataFrame({
    'Fator Técnico': paciente_x.columns,
    'Valor do Fator': paciente_x.iloc[0].values,
    'Impacto ($)': shap_values[0]
}).sort_values(by='Impacto ($)', ascending=False)

for index, row in pesos.iterrows():
    fator_tecnico = row['Fator Técnico']
    valor_tecnico = row['Valor do Fator']
    impacto = row['Impacto ($)']

    if abs(impacto) < 50: continue

    fator_legivel = dicionario_fatores.get(fator_tecnico, fator_tecnico)
    valor_legivel = formatar_valor(fator_tecnico, valor_tecnico)

    sinal = "🔺 Agravante" if impacto > 0 else "🔽 Atenuante"
    dados_tabela.append([sinal, fator_legivel, valor_legivel, f"${abs(impacto):,.2f}"])

    if impacto > 0:
        agravantes.append((fator_legivel, impacto, fator_tecnico))
    else:
        atenuantes.append((fator_legivel, abs(impacto), fator_tecnico))

df_tabela = pd.DataFrame(dados_tabela,
                         columns=["Efeito", "Variável Analisada", "Registro do Paciente", "Peso Financeiro"])
st.subheader("Extrato Atuarial do Paciente")
st.dataframe(df_tabela, width="stretch")

# =====================================================================
# 4. PARECER ANALÍTICO E RECOMENDAÇÃO (Motor Lógico)
# =====================================================================
st.subheader("📝 Parecer Gerado Automaticamente")

agravantes = sorted(agravantes, key=lambda x: x[1], reverse=True)
atenuantes = sorted(atenuantes, key=lambda x: x[1], reverse=True)

maior_agravante = agravantes[0] if agravantes else ("Nenhum", 0, "nenhum")
maior_atenuante = atenuantes[0] if atenuantes else ("Nenhum", 0, "nenhum")

# Texto reescrito para evitar que o Markdown aglutine as palavras
texto_parecer = (
    f"O paciente analisado representa um evento de custo extremo, com risco atuarial estimado em **$ {custo_previsto:,.2f}**. "
    f"O principal fator de inflação que justificou o rompimento da curva de custos foi o registro de **{maior_agravante[0]}**, "
    f"correspondendo isoladamente a um acréscimo de **$ {maior_agravante[1]:,.2f}** no resultado final."
)

if atenuantes:
    texto_parecer += f" Em contrapartida, a variável **{maior_atenuante[0]}** atuou como mitigadora, reduzindo a exposição financeira em **$ {abs(maior_atenuante[1]):,.2f}**."

st.info(texto_parecer)

if "perfil_altissimo_risco" in maior_agravante[2] or "fumante" in maior_agravante[2] or "bmi" in maior_agravante[2]:
    st.error(
        "🚨 **RECOMENDAÇÃO DE AUDITORIA:** O ofensor primário da sinistralidade é comportamental/clínico. Falhas na medicina preventiva permitiram a escalada deste custo. Exige intervenção ativa da gestão de crônicos.")
else:
    st.warning(
        "⚠️ **RECOMENDAÇÃO DE AUDITORIA:** O custo extremo é tracionado por questões demográficas sem margem para intervenção médica preventiva direta. Monitorar evolução.")

# =====================================================================
# 5. METODOLOGIA
# =====================================================================
with st.expander("⚙️ METODOLOGIA: Como o motor de IA calcula estes valores?"):
    st.write("""
    * **Motor Preditivo (Random Forest):** Analisa cruzamentos de dados demográficos simulando centenas de cenários simultâneos em formato de árvore de decisão para chegar a uma estimativa base de custo livre de vieses únicos.
    * **Motor de Explicabilidade (SHAP):** Aplica conceitos de Teoria dos Jogos para destrinchar a predição. Ele isola exatamente quantos dólares cada variável (jogador) contribuiu para encarecer ou baratear a conta médica, quebrando a 'caixa preta' do algoritmo.
    * **Motor de Linguagem (NLG):** Interpreta os vetores financeiros negativos e positivos do SHAP. Se o maior peso causador da sinistralidade for passível de tratamento (como IMC alto ou Tabagismo), ele gera automaticamente um alerta acionável para a área de Prevenção e Promoção à Saúde (Promoprev).
    """)