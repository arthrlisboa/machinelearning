"""
=============================================================================
APLICAÇÃO STREAMLIT - ANÁLISE DE CRÉDITO PARA EMPRÉSTIMOS
=============================================================================

Esta aplicação usa um modelo de Machine Learning treinado para prever
se um empréstimo será aprovado ou não com base nas características do cliente.

Para rodar:
streamlit run loan_app.py

Arquivos necessários na mesma pasta:
- best_loan_model.pkl
- scaler.pkl
- encoders.pkl
- target_encoder.pkl
- model_info.pkl

=============================================================================
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image

# Configuração da página
st.set_page_config(
    page_title="Análise de Crédito",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 1. FUNÇÕES DE CARREGAMENTO
# =============================================================================

@st.cache_resource
def carregar_modelo_e_preprocessadores():
    """Carrega modelo e todos os preprocessadores necessários"""
    try:
        with open('best_loan_model.pkl', 'rb') as f:
            modelo = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        with open('target_encoder.pkl', 'rb') as f:
            target_encoder = pickle.load(f)
        
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        return modelo, scaler, encoders, target_encoder, model_info
    
    except FileNotFoundError as e:
        st.error(f"❌ Erro: Arquivo não encontrado - {e}")
        st.info("📁 Certifique-se de que todos os arquivos .pkl estão na mesma pasta")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivos: {str(e)}")
        st.stop()

# Carregar modelo e preprocessadores
modelo, scaler, encoders, target_encoder, model_info = carregar_modelo_e_preprocessadores()

# =============================================================================
# 2. FUNÇÃO DE PREPROCESSAMENTO
# =============================================================================

def preprocessar_input(dados_usuario):
    """
    Preprocessa os dados do usuário no mesmo formato do treinamento
    
    Args:
        dados_usuario: dict com os dados do formulário
    
    Returns:
        numpy array pronto para predição
    """
    # Criar DataFrame com as features na ordem correta
    df = pd.DataFrame([dados_usuario])
    
    # Encontrar nomes reais das colunas nos encoders (case-insensitive)
    for col_original, encoder in encoders.items():
        col_lower = col_original.lower()
        for col_df in df.columns:
            if col_df.lower() == col_lower:
                # Aplicar encoding
                df[col_df] = encoder.transform([str(df[col_df].values[0])])
                break
    
    # Reordenar colunas para corresponder ao treinamento
    df = df[model_info['feature_names']]
    
    # Aplicar scaling nas colunas numéricas
    numeric_cols = model_info['numeric_columns']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df.values

# =============================================================================
# 3. INTERFACE STREAMLIT
# =============================================================================

# Cabeçalho
st.title("🏦 Sistema de Análise de Crédito para Empréstimos")
st.markdown("""
Bem-vindo ao sistema automatizado de análise de crédito! Preencha o formulário abaixo 
para verificar se seu empréstimo será aprovado.
""")

st.divider()

# Sidebar com informações
with st.sidebar:
    st.header("ℹ️ Sobre o Sistema")
    st.markdown(f"""
    **Modelo:** {model_info['best_model_name']}  
    **Status:** ✅ Ativo
    
    ---
    
    ### 📊 Informações Técnicas
    
    **Features utilizadas:**  
    {len(model_info['feature_names'])} variáveis
    
    **Classes:**
    - ✅ Aprovado
    - ❌ Negado
    
    ---
    
    ### 🔒 Privacidade
    
    Seus dados são processados localmente e não são armazenados.
    """)

# =============================================================================
# 4. FORMULÁRIO DE ENTRADA
# =============================================================================

st.header("📋 Dados do Solicitante")

# Layout em colunas
col1, col2 = st.columns(2)

with col1:
    st.subheader("Informações Pessoais")
    
    # Gênero (Gender)
    gender = st.radio(
        "Gênero:",
        options=["Male", "Female"],
        horizontal=True,
        help="Selecione seu gênero"
    )
    
    # Estado Civil (Married)
    married = st.radio(
        "Estado Civil:",
        options=["Yes", "No"],
        format_func=lambda x: "Casado(a)" if x == "Yes" else "Solteiro(a)",
        horizontal=True,
        help="Você é casado(a)?"
    )
    
    # Dependentes (Dependents)
    dependents = st.selectbox(
        "Número de Dependentes:",
        options=[0, 1, 2, 3],
        help="Quantas pessoas dependem de você financeiramente?"
    )
    
    # Educação (Education)
    education = st.radio(
        "Nível de Educação:",
        options=["Graduate", "Not Graduate"],
        format_func=lambda x: "Graduado" if x == "Graduate" else "Não Graduado",
        help="Você possui ensino superior completo?"
    )

with col2:
    st.subheader("Informações Financeiras")
    
    # Autônomo (Self_Employed)
    self_employed = st.radio(
        "Trabalha como Autônomo:",
        options=["Yes", "No"],
        format_func=lambda x: "Sim" if x == "Yes" else "Não",
        horizontal=True,
        help="Você trabalha por conta própria?"
    )
    
    # Renda do Solicitante (ApplicantIncome)
    applicant_income = st.number_input(
        "Renda Mensal (R$):",
        min_value=0,
        max_value=1000000,
        value=5000,
        step=100,
        help="Informe sua renda mensal em reais"
    )
    
    # Valor do Empréstimo (LoanAmount)
    loan_amount = st.number_input(
        "Valor do Empréstimo Solicitado (R$):",
        min_value=0,
        max_value=10000000,
        value=150000,
        step=1000,
        help="Quanto você deseja solicitar de empréstimo?"
    )

st.divider()

# =============================================================================
# 5. BOTÃO DE ANÁLISE E RESULTADOS
# =============================================================================

# Centralizar o botão
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    botao_analisar = st.button(
        "🔍 Analisar Crédito",
        type="primary",
        use_container_width=True
    )

if botao_analisar:
    
    # Preparar dados do usuário
    dados_usuario = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'LoanAmount': loan_amount
    }
    
    # Converter nomes para lowercase para matching (se necessário)
    dados_usuario_processado = {}
    for key, value in dados_usuario.items():
        # Encontrar a chave correta no model_info
        for feat_name in model_info['feature_names']:
            if feat_name.lower() == key.lower():
                dados_usuario_processado[feat_name] = value
                break
        else:
            # Se não encontrar, usar o nome original
            dados_usuario_processado[key] = value
    
    try:
        # Preprocessar dados
        X_input = preprocessar_input(dados_usuario_processado)
        
        # Fazer predição
        predicao = modelo.predict(X_input)[0]
        
        # Obter probabilidades (se o modelo suportar)
        tem_proba = hasattr(modelo, 'predict_proba')
        if tem_proba:
            probabilidades = modelo.predict_proba(X_input)[0]
            prob_classe_predita = probabilidades[predicao]
        
        # Decodificar predição
        status_texto = target_encoder.inverse_transform([predicao])[0]
        
        st.divider()
        
        # =============================================================================
        # 6. EXIBIÇÃO DOS RESULTADOS
        # =============================================================================
        
        st.header("📊 Resultado da Análise")
        
        # Resultado principal
        if status_texto in ['Y', 'Yes', '1', 'Approved', 'approved']:
            st.success("### ✅ EMPRÉSTIMO APROVADO!")
            st.balloons()
            
            resultado_col1, resultado_col2 = st.columns(2)
            
            with resultado_col1:
                st.markdown("""
                **Parabéns!** Sua solicitação de empréstimo foi aprovada pelo nosso sistema.
                
                **Próximos passos:**
                1. Nossa equipe entrará em contato em até 48h
                2. Documentação necessária será solicitada
                3. Análise final e liberação do crédito
                """)
            
            with resultado_col2:
                if tem_proba:
                    st.metric(
                        "Confiança da Aprovação",
                        f"{prob_classe_predita*100:.1f}%",
                        help="Probabilidade calculada pelo modelo"
                    )
                    
                    # Barra de progresso
                    st.progress(prob_classe_predita)
        
        else:
            st.error("### ❌ EMPRÉSTIMO NEGADO")
            
            resultado_col1, resultado_col2 = st.columns(2)
            
            with resultado_col1:
                st.markdown("""
                Infelizmente, sua solicitação não foi aprovada no momento.
                
                **Possíveis motivos:**
                - Renda insuficiente para o valor solicitado
                - Histórico de crédito
                - Perfil de risco alto
                
                **Recomendações:**
                - Solicite um valor menor
                - Melhore seu score de crédito
                - Tente novamente em 6 meses
                """)
            
            with resultado_col2:
                if tem_proba:
                    st.metric(
                        "Confiança da Negação",
                        f"{prob_classe_predita*100:.1f}%",
                        help="Probabilidade calculada pelo modelo"
                    )
                    
                    # Barra de progresso
                    st.progress(prob_classe_predita)
        
        # Detalhes técnicos (expandível)
        with st.expander("🔍 Ver Detalhes Técnicos"):
            st.markdown("### Dados Processados")
            
            col_det1, col_det2 = st.columns(2)
            
            with col_det1:
                st.markdown("**Entrada do Usuário:**")
                for key, value in dados_usuario.items():
                    st.text(f"{key}: {value}")
            
            with col_det2:
                if tem_proba:
                    st.markdown("**Probabilidades por Classe:**")
                    for idx, classe in enumerate(target_encoder.classes_):
                        st.text(f"{classe}: {probabilidades[idx]*100:.2f}%")
                
                st.markdown(f"**Modelo Utilizado:** {model_info['best_model_name']}")
                st.markdown(f"**Predição Bruta:** {predicao}")
        
        # Análise de risco
        st.divider()
        st.subheader("📈 Análise de Perfil")
        
        col_analise1, col_analise2, col_analise3 = st.columns(3)
        
        with col_analise1:
            # Proporção empréstimo/renda
            proporcao = (loan_amount / applicant_income) if applicant_income > 0 else 0
            st.metric(
                "Proporção Empréstimo/Renda",
                f"{proporcao:.1f}x",
                help="Quantas vezes sua renda representa o empréstimo"
            )
        
        with col_analise2:
            # Categoria de renda
            if applicant_income < 3000:
                cat_renda = "Baixa"
            elif applicant_income < 8000:
                cat_renda = "Média"
            else:
                cat_renda = "Alta"
            
            st.metric(
                "Categoria de Renda",
                cat_renda
            )
        
        with col_analise3:
            # Score fictício baseado na probabilidade
            if tem_proba:
                score = int(prob_classe_predita * 1000)
                st.metric(
                    "Score de Crédito",
                    score,
                    help="Score estimado (0-1000)"
                )
    
    except Exception as e:
        st.error(f"❌ Erro ao processar dados: {str(e)}")
        st.info("Por favor, verifique se todos os campos foram preenchidos corretamente.")

# =============================================================================
# 7. RODAPÉ
# =============================================================================

st.divider()

st.markdown("""
---
### 

**Dados utilizados na análise:**
- Informações pessoais (gênero, estado civil, dependentes, educação)
- Situação profissional (autônomo ou não)
- Dados financeiros (renda mensal, valor solicitado)

---
🏦 Sistema de Análise de Crédito | 🤖 Powered by Machine Learning
""")