# 🏥 Motor de Auditoria Clínica e Atuarial com IA

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://motor-auditoria-ia.streamlit.app/)

Uma aplicação de Inteligência Artificial focada em saúde corporativa, desenvolvida para prever custos médicos e auditar contas extremas (outliers) utilizando Machine Learning e Explainable AI (SHAP).

## 🎯 O Problema de Negócio
Operadoras de saúde e seguradoras perdem milhares de horas tentando auditar e entender os gatilhos de sinistralidade de contas médicas de alto custo. Modelos tradicionais entregam apenas um valor numérico frio, sem justificativa clínica para a área de negócios.

## 💡 A Solução
Este motor atuarial não apenas prevê o custo do paciente, mas gera um **parecer textual automático**, explicando dólar a dólar o que inflacionou a conta e recomendando ações de medicina preventiva. 

O sistema opera em uma arquitetura de três camadas:
1. **Motor Preditivo:** Algoritmo `RandomForestRegressor` treinado para identificar padrões de custo na base, imune a ruídos isolados.
2. **Motor de Explicabilidade:** Integração com `SHAP` (Teoria dos Jogos) para abrir a "caixa preta" da IA e isolar o peso financeiro exato de cada variável.
3. **Motor de Linguagem (NLG) e Frontend:** Regras lógicas interpretam a matriz matemática e geram um alerta acionável em um dashboard interativo.

## 🚀 Acesso à Aplicação
A aplicação está 100% hospedada na nuvem e pode ser acessada de qualquer navegador. 
👉 **[Clique aqui para acessar o Motor de Auditoria ao vivo](https://motor-auditoria-ia.streamlit.app/)**

## 🛠️ Como Executar Localmente (Para Desenvolvedores)
Caso queira rodar o código na sua própria máquina:

1. Clone o repositório:
```bash
git clone [https://github.com/victormagal9778-cloud/motor-auditoria-ia.git](https://github.com/victormagal9778-cloud/motor-auditoria-ia.git)
