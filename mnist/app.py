"""
=============================================================================
APLICAÇÃO STREAMLIT - RECONHECEDOR DE DÍGITOS (0-9)
=============================================================================

Esta aplicação permite:
1. Desenhar um dígito (0-9) na tela
2. Fazer upload de uma imagem
3. Usar imagens de exemplo do MNIST
4. Ver a predição do modelo Random Forest otimizado

Requisitos:
- streamlit
- streamlit-drawable-canvas
- pickle
- numpy
- PIL
- scikit-learn

Para rodar:
streamlit run app.py
=============================================================================
"""

import streamlit as st
import pickle
import numpy as np
from PIL import Image, ImageOps
import io

# Configuração da página
st.set_page_config(
    page_title="Reconhecedor de Dígitos MNIST",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 1. CARREGAMENTO DO MODELO
# =============================================================================

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo Random Forest otimizado"""
    try:
        with open('best_random_forest_mnist.pkl', 'rb') as f:
            modelo = pickle.load(f)
        return modelo
    except FileNotFoundError:
        st.error("❌ Erro: Arquivo 'best_random_forest_mnist.pkl' não encontrado!")
        st.info("📁 Certifique-se de que o arquivo está na mesma pasta do app.py")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {str(e)}")
        st.stop()

# Carregando o modelo
modelo = carregar_modelo()

# =============================================================================
# 2. FUNÇÕES DE PROCESSAMENTO DE IMAGEM
# =============================================================================

def preprocessar_imagem(imagem):
    """
    Preprocessa a imagem para o formato MNIST (28x28, escala de cinza, normalizada)
    
    Args:
        imagem: PIL Image
    
    Returns:
        numpy array (1, 784) pronto para predição
    """
    # Converter para escala de cinza
    imagem_gray = ImageOps.grayscale(imagem)
    
    # Redimensionar para 28x28
    imagem_resized = imagem_gray.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Converter para array numpy
    img_array = np.array(imagem_resized)
    
    # Inverter cores se necessário (MNIST tem fundo preto e dígito branco)
    # Se o fundo for branco, inverte
    if img_array.mean() > 127:
        img_array = 255 - img_array
    
    # Normalizar (0-255 → 0-1)
    img_array = img_array / 255.0
    
    # Achatar para vetor (784,) e adicionar dimensão (1, 784)
    img_flatten = img_array.flatten().reshape(1, -1)
    
    return img_flatten, imagem_resized

def fazer_predicao(imagem_processada):
    """
    Faz a predição usando o modelo
    
    Args:
        imagem_processada: numpy array (1, 784)
    
    Returns:
        tuple: (predição, probabilidades)
    """
    predicao = modelo.predict(imagem_processada)[0]
    
    # Tentar obter probabilidades (nem todos os modelos têm)
    try:
        probabilidades = modelo.predict_proba(imagem_processada)[0]
    except:
        probabilidades = None
    
    return int(predicao), probabilidades

# =============================================================================
# 3. INTERFACE STREAMLIT
# =============================================================================

# Título e descrição
st.title("🔢 Reconhecedor de Dígitos MNIST")
st.markdown("""
Esta aplicação usa um **Random Forest otimizado** treinado no dataset MNIST 
para reconhecer dígitos escritos à mão (0-9).

**Acurácia do modelo:** 98.60% 🏆
""")

# Divisor
st.divider()

# Sidebar com informações
with st.sidebar:
    st.header("ℹ️ Informações do Modelo")
    st.markdown(f"""
    **Modelo:** Random Forest  
    **Status:** ✅ Carregado  
    **Features:** 784 (28x28 pixels)  
    **Classes:** 10 (dígitos 0-9)
    
    ---
    
    ### 📊 Hiperparâmetros Otimizados
    """)
    
    # Mostrar alguns parâmetros do modelo
    params = modelo.get_params()
    st.markdown(f"""
    - **n_estimators:** {params.get('n_estimators', 'N/A')}
    - **max_depth:** {params.get('max_depth', 'N/A')}
    - **min_samples_split:** {params.get('min_samples_split', 'N/A')}
    - **min_samples_leaf:** {params.get('min_samples_leaf', 'N/A')}
    """)
    
    st.divider()
    
    st.markdown("""
    ### 🎨 Dicas para Desenhar
    - Use fundo **branco** e traço **preto**
    - Desenhe no **centro** do canvas
    - Faça o dígito **grande**
    - Evite traços muito finos
    """)

# =============================================================================
# 4. MÉTODOS DE ENTRADA
# =============================================================================

# Tabs para diferentes métodos de entrada
tab1, tab2, tab3 = st.tabs(["✏️ Desenhar", "📤 Upload", "🎲 Exemplos MNIST"])

# ---------- TAB 1: DESENHAR ----------
with tab1:
    st.subheader("Desenhe um dígito (0-9)")
    
    # Tentar importar streamlit-drawable-canvas
    try:
        from streamlit_drawable_canvas import st_canvas
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Canvas de Desenho:**")
            
            # Canvas para desenhar
            canvas_result = st_canvas(
                fill_color="white",
                stroke_width=20,
                stroke_color="black",
                background_color="white",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
            
            if st.button("🗑️ Limpar Canvas", use_container_width=True):
                st.rerun()
        
        with col2:
            st.markdown("**Resultado da Predição:**")
            
            if canvas_result.image_data is not None:
                # Verificar se há algo desenhado
                if np.any(canvas_result.image_data[:, :, :3] < 255):
                    # Converter para PIL Image
                    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                    img = img.convert('RGB')
                    
                    # Preprocessar
                    img_processada, img_28x28 = preprocessar_imagem(img)
                    
                    # Fazer predição
                    predicao, probabilidades = fazer_predicao(img_processada)
                    
                    # Mostrar imagem processada (28x28)
                    st.markdown("**Imagem Processada (28x28):**")
                    st.image(img_28x28, width=140)
                    
                    # Resultado
                    st.markdown("---")
                    st.markdown(f"### Predição: **{predicao}**")
                    
                    # Mostrar probabilidades se disponível
                    if probabilidades is not None:
                        st.markdown("**Confiança por Classe:**")
                        prob_dict = {i: float(prob) for i, prob in enumerate(probabilidades)}
                        prob_sorted = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                        
                        for digito, prob in prob_sorted[:3]:
                            st.progress(prob, text=f"Dígito {digito}: {prob:.2%}")
                else:
                    st.info("👆 Desenhe algo no canvas à esquerda!")
    
    except ImportError:
        st.error("❌ Biblioteca 'streamlit-drawable-canvas' não instalada!")
        st.code("pip install streamlit-drawable-canvas", language="bash")
        st.info("💡 Use a aba 'Upload' ou 'Exemplos MNIST' enquanto isso.")

# ---------- TAB 2: UPLOAD ----------
with tab2:
    st.subheader("Faça upload de uma imagem")
    
    uploaded_file = st.file_uploader(
        "Escolha uma imagem (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        help="Envie uma imagem de um dígito escrito à mão"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Imagem Original:**")
            imagem_original = Image.open(uploaded_file)
            st.image(imagem_original, width=280)
        
        with col2:
            st.markdown("**Resultado da Predição:**")
            
            # Preprocessar
            img_processada, img_28x28 = preprocessar_imagem(imagem_original)
            
            # Mostrar imagem processada
            st.markdown("**Imagem Processada (28x28):**")
            st.image(img_28x28, width=140)
            
            # Fazer predição
            predicao, probabilidades = fazer_predicao(img_processada)
            
            # Resultado
            st.markdown("---")
            st.markdown(f"### Predição: **{predicao}**")
            
            # Mostrar probabilidades
            if probabilidades is not None:
                st.markdown("**Confiança por Classe:**")
                prob_dict = {i: float(prob) for i, prob in enumerate(probabilidades)}
                prob_sorted = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                
                for digito, prob in prob_sorted[:5]:
                    st.progress(prob, text=f"Dígito {digito}: {prob:.2%}")

# ---------- TAB 3: EXEMPLOS MNIST ----------
with tab3:
    st.subheader("Teste com exemplos reais do MNIST")
    
    try:
        from sklearn.datasets import fetch_openml
        
        @st.cache_data
        def carregar_exemplos_mnist():
            """Carrega alguns exemplos do MNIST"""
            mnist = fetch_openml('mnist_784', version=1, parser='auto')
            X = np.array(mnist.data, dtype=np.float32)
            y = np.array(mnist.target, dtype=np.int64)
            
            # Pegar 10 exemplos aleatórios
            np.random.seed(42)
            indices = np.random.choice(len(X), 20, replace=False)
            
            return X[indices], y[indices]
        
        X_exemplos, y_exemplos = carregar_exemplos_mnist()
        
        st.markdown("Selecione um exemplo abaixo:")
        
        # Criar grid de exemplos
        cols = st.columns(5)
        
        exemplo_selecionado = None
        
        for idx in range(20):
            col_idx = idx % 5
            
            with cols[col_idx]:
                # Reshape para 28x28
                img_array = X_exemplos[idx].reshape(28, 28)
                img_pil = Image.fromarray((img_array).astype('uint8'))
                
                # Botão com a imagem
                if st.button(f"Exemplo {idx+1}", key=f"exemplo_{idx}", use_container_width=True):
                    exemplo_selecionado = idx
                
                st.image(img_pil, width=100)
                st.caption(f"Label: {y_exemplos[idx]}")
        
        # Mostrar predição do exemplo selecionado
        if exemplo_selecionado is not None:
            st.divider()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"**Exemplo {exemplo_selecionado + 1} Selecionado:**")
                img_array = X_exemplos[exemplo_selecionado].reshape(28, 28)
                img_pil = Image.fromarray((img_array).astype('uint8'))
                st.image(img_pil, width=200)
                st.markdown(f"**Label Real:** {y_exemplos[exemplo_selecionado]}")
            
            with col2:
                st.markdown("**Resultado da Predição:**")
                
                # Normalizar (já vem 0-255, precisa 0-1)
                img_processada = X_exemplos[exemplo_selecionado].reshape(1, -1) / 255.0
                
                # Fazer predição
                predicao, probabilidades = fazer_predicao(img_processada)
                
                # Resultado
                if predicao == y_exemplos[exemplo_selecionado]:
                    st.success(f"### ✅ Predição: **{predicao}** (CORRETO!)")
                else:
                    st.error(f"### ❌ Predição: **{predicao}** (Esperado: {y_exemplos[exemplo_selecionado]})")
                
                # Mostrar probabilidades
                if probabilidades is not None:
                    st.markdown("**Confiança por Classe:**")
                    prob_dict = {i: float(prob) for i, prob in enumerate(probabilidades)}
                    prob_sorted = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                    
                    for digito, prob in prob_sorted[:5]:
                        if digito == y_exemplos[exemplo_selecionado]:
                            st.progress(prob, text=f"Dígito {digito}: {prob:.2%} ⭐")
                        else:
                            st.progress(prob, text=f"Dígito {digito}: {prob:.2%}")
    
    except Exception as e:
        st.error(f"Erro ao carregar exemplos: {str(e)}")
        st.info("Use as outras abas para testar o modelo!")

# =============================================================================
# 5. RODAPÉ
# =============================================================================

st.divider()

st.markdown("""
---
### 📝 Informações Técnicas

**Como funciona:**
1. A imagem é redimensionada para 28x28 pixels (formato MNIST)
2. Convertida para escala de cinza
3. Normalizada (valores 0-1)
4. Achatada em um vetor de 784 features
5. Passada para o modelo Random Forest

**Modelo:** Random Forest otimizado via RandomizedSearchCV  
**Acurácia:** 98.60% no conjunto de teste  
**Dataset:** MNIST (70,000 imagens de dígitos manuscritos)

---
🚀 Desenvolvido com Streamlit | 🤖 Powered by scikit-learn
""")