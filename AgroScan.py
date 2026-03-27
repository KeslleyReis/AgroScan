#python -m streamlit run AgroScan.py

import streamlit as st
import cv2
import numpy as np
import time
import base64 
from io import BytesIO

# 🎨 Paleta profissional
cores = {
    "Fértil": (34,139,34),
    "Seco": (210,180,140),
    "Rochoso": (80,80,80),
    "Úmido": (0,100,200),
    "Arenoso": (194,178,128)
}

# Função para converter imagem np para base64
def img_to_base64(img):
    # Converter BGR para RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, img_encoded = cv2.imencode('.png', img_rgb)
    img_base64 = base64.b64encode(img_encoded).decode()
    return img_base64

# Função para criar o slider interativo com mouse
def interactive_image_comparison(img_original, img_mapa, initial=50):
    img_b64_original = img_to_base64(img_original)
    img_b64_mapa = img_to_base64(img_mapa)

    h, w = img_original.shape[:2]
    max_display = min(w, 900)  # manter tamanho máximo para consistência visual
    aspect_percentage = (h / w) * 100

    html_code = """
    <div style="position: relative; width: 100%; max-width: {max_display}px; margin: auto;">
        <div style="position: absolute; top: 8px; left: 8px; z-index: 12; color: white; font-weight: bold; font-size: 14px; text-shadow: 0 0 6px rgba(0, 0, 0, 0.8);">Comparação: <span id="percentage-label">{initial}%</span></div>
        <div id="comparison-container" style="position: relative; width: 100%; padding-bottom: {aspect_percentage}%; background: #000; cursor: col-resize; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.25);">
            <img id="img-original" src="data:image/png;base64,{img_b64_original}" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover;"/>
            <div id="img-mapa-container" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: hidden; clip-path: inset(0 50% 0 0);">
                <img id="img-mapa" src="data:image/png;base64,{img_b64_mapa}" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover;"/>
            </div>
            <div id="divider-line" style="position: absolute; top: 0; left: 50%; width: 3px; height: 100%; background: rgba(0,255,255,0.9); box-shadow: 0 0 15px rgba(0,255,255,0.8); z-index: 10;"></div>
        </div>
    </div>

    <script>
        const container = document.getElementById('comparison-container');
        const mapaContainer = document.getElementById('img-mapa-container');
        const dividerLine = document.getElementById('divider-line');
        const percentageLabel = document.getElementById('percentage-label');

        function applyPercentage(p) {{
            const clamped = Math.max(0, Math.min(100, p));
            mapaContainer.style.clipPath = 'inset(0 ' + (100 - clamped) + '% 0 0)';
            dividerLine.style.left = clamped + '%';
            percentageLabel.innerText = Math.round(clamped) + '%';
        }}

        function updatePosition(e) {{
            const rect = container.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const percentage = (x / rect.width) * 100;
            applyPercentage(percentage);
        }}

        container.addEventListener('mousemove', updatePosition);
        container.addEventListener('touchmove', (e) => {{
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {{
                clientX: touch.clientX,
                clientY: touch.clientY
            }});
            updatePosition(mouseEvent);
        }});

        applyPercentage({initial});
    </script>
    """.format(
        max_display=max_display,
        aspect_percentage=aspect_percentage,
        img_b64_original=img_b64_original,
        img_b64_mapa=img_b64_mapa,
        initial=initial
    )

    st.components.v1.html(html_code, height=int((max_display * h) / w) + 20)


st.title("🌱 AgroScan Drone - Análise e recomendação de Terreno")

uploaded_file = st.file_uploader("Envie a foto do terreno", type=["jpg","png"])

if uploaded_file:

    # Carregar imagem
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Imagem analisada", use_container_width=True)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 🌱 NDVI FAKE (índice de vegetação mais realista)
    b, g, r = cv2.split(img.astype("float"))

    ndvi = (g - r) / (g + r + 1e-5)

    ndvi_normalizado = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX)
    ndvi_uint8 = ndvi_normalizado.astype(np.uint8)

    # 🎨 mapa profissional
    mapa_ndvi = cv2.applyColorMap(ndvi_uint8, cv2.COLORMAP_TURBO)

    # Máscaras
    verde = cv2.inRange(hsv, (35,40,40), (85,255,255))
    amarelo = cv2.inRange(hsv, (20,100,100), (35,255,255))
    cinza = cv2.inRange(hsv, (0,0,50), (180,50,200))
    azul = cv2.inRange(hsv, (90,50,50), (130,255,255))

    total_pixels = img.shape[0] * img.shape[1]

    p_verde = np.sum(verde > 0) / total_pixels
    p_amarelo = np.sum(amarelo > 0) / total_pixels
    p_cinza = np.sum(cinza > 0) / total_pixels
    p_azul = np.sum(azul > 0) / total_pixels

    # Resultado geral
    st.subheader("📊 Resultado geral")

    if p_verde > 0.4:
        resultado = "Terreno fértil"
    elif p_amarelo > 0.3:
        resultado = "Terreno seco"
    elif p_cinza > 0.3:
        resultado = "Terreno rochoso"
    elif p_azul > 0.2:
        resultado = "Terreno úmido"
    else:
        resultado = "Terreno arenoso"

    st.success(resultado)
    st.metric("Saúde da vegetação", f"{round(np.mean(ndvi)*100,1)}%")

    st.write(f"Fértil: {round(p_verde*100,2)}%")
    st.write(f"Seco: {round(p_amarelo*100,2)}%")
    st.write(f"Rochoso: {round(p_cinza*100,2)}%")
    st.write(f"Úmido: {round(p_azul*100,2)}%")

    # 🗺️ Mapa geral
    mapa = mapa_ndvi.copy()

    mapa[verde > 0] = cores["Fértil"]
    mapa[amarelo > 0] = cores["Seco"]
    mapa[cinza > 0] = cores["Rochoso"]
    mapa[azul > 0] = cores["Úmido"]

    mapa = cv2.GaussianBlur(mapa, (11,11), 0)

    st.subheader(" Comparação Interativa: Passe o mouse para comparar")
    
    if 'compare_pct' not in st.session_state:
        st.session_state.compare_pct = 50

    cols = st.columns(3)
    if cols[0].button('Reset 50%'):
        st.session_state.compare_pct = 50
    if cols[1].button('Original'):
        st.session_state.compare_pct = 0
    if cols[2].button('Mapa completo'):
        st.session_state.compare_pct = 100

    # Garantir que a imagem do mapa tem a mesma altura e largura
    h, w = img.shape[:2]
    mapa_resized = cv2.resize(mapa, (w, h))
    
    # Usar o componente interativo com mouse
    interactive_image_comparison(img, mapa_resized, initial=st.session_state.compare_pct)

    # GRID 4

    h, w, _ = img.shape

    st.subheader("Análise por regiões")

    num_grids = st.radio("Selecione o número de grids:", [4, 8, 16, 32], index=0)

    # Definir rows e cols baseado no num_grids
    if num_grids == 4:
        rows, cols = 2, 2
    elif num_grids == 8:
        rows, cols = 2, 4
    elif num_grids == 16:
        rows, cols = 4, 4
    elif num_grids == 32:
        rows, cols = 4, 8

    # Criar regiões e coordenadas
    regioes = {}
    coords = {}
    for i in range(rows):
        for j in range(cols):
            name = f"Grid {i*cols + j + 1}"
            y_start = i * (h // rows)
            y_end = (i + 1) * (h // rows)
            x_start = j * (w // cols)
            x_end = (j + 1) * (w // cols)
            regioes[name] = hsv[y_start:y_end, x_start:x_end]
            coords[name] = (x_start, x_end, y_start, y_end)

    def analisar_regiao(hsv_area, img_area):

        b, g, r = cv2.split(img_area.astype("float"))
        ndvi = (g - r) / (g + r + 1e-5)

        ndvi_mean = np.mean(ndvi)

        gray = cv2.cvtColor(img_area, cv2.COLOR_BGR2GRAY)
        textura = cv2.Laplacian(gray, cv2.CV_64F)
        textura_mean = np.mean(np.abs(textura))

        umidade = np.mean(b > g)

        if ndvi_mean > 0.3:
            tipo = "Fértil"
        elif textura_mean > 40:
            tipo = "Rochoso"
        elif umidade > 0.4:
            tipo = "Úmido"
        elif ndvi_mean < 0.1:
            tipo = "Seco"
        else:
            tipo = "Arenoso"

        return tipo, p_verde, p_amarelo, p_cinza, p_azul


        # 🤖 RECOMENDAÇÃO AUTOMÁTICA
    def recomendacao(tipo):

        if tipo == "Fértil":
            return {
                "plantio": "Alta produtividade. Ideal para culturas como milho, soja e feijão.",
                "irrigacao": "Baixa necessidade. Monitoramento semanal é suficiente.",
                "acao": "Manter nutrientes e evitar erosão do solo."
            }

        elif tipo == "Seco":
            return {
                "plantio": "Indicado para culturas resistentes como sorgo ou mandioca.",
                "irrigacao": "Irrigação frequente recomendada (2x ao dia em períodos críticos).",
                "acao": "Aplicar cobertura morta para retenção de umidade."
            }

        elif tipo == "Úmido":
            return {
                "plantio": "Bom para arroz ou culturas adaptadas à alta umidade.",
                "irrigacao": "Evitar irrigação excessiva.",
                "acao": "Melhorar drenagem do solo para evitar fungos."
            }

        elif tipo == "Rochoso":
            return {
                "plantio": "Baixa viabilidade agrícola.",
                "irrigacao": "Não recomendado.",
                "acao": "Considerar correção do solo ou uso alternativo."
            }

        else:  # Arenoso
            return {
                "plantio": "Culturas leves como cenoura, batata-doce e amendoim.",
                "irrigacao": "Irrigação constante, pois perde água rápido.",
                "acao": "Adicionar matéria orgânica para melhorar retenção."
            }


    iniciar = st.button("Iniciar análise")

    placeholder_img = st.empty()
    placeholder_text = st.empty()
    
    if iniciar:
        for nome, regiao in regioes.items():
            st.session_state.trail = []
        resultados = {}

        # função de texto digitando
        def escrever_texto(texto):
            texto_animado = ""
            for char in texto:
                texto_animado += char
                placeholder_text.markdown(texto_animado)
                time.sleep(0.02)

        for nome, regiao in regioes.items():

            x_start, x_end, y_start, y_end = coords[nome]

            img_area = img[y_start:y_end, x_start:x_end]

            tipo, _, _, _, _ = analisar_regiao(regiao, img_area)

            resultados[nome] = tipo

            # ANIMAÇÃO DO DRONE (CORRIGIDA)

            passos_x = 20
            passos_y = 6

            y_positions = np.linspace(y_start, y_end, passos_y, endpoint=True)

            for idx, py in enumerate(y_positions):

                if idx % 2 == 0:
                    x_positions = np.linspace(x_start, x_end, passos_x)
                else:
                    x_positions = np.linspace(x_end, x_start, passos_x)

                for px in x_positions:

                    frame = img.copy()

                    # grade geral
                    for j in range(1, cols):
                        x_line = j * (w // cols)
                        cv2.line(frame, (x_line, 0), (x_line, h), (255,255,255), 2)
                    for i in range(1, rows):
                        y_line = i * (h // rows)
                        cv2.line(frame, (0, y_line), (w, y_line), (255,255,255), 2)

                    # destaque da região
                    cv2.rectangle(frame, (x_start,y_start), (x_end,y_end), (255,100,0), 2)

                    # suavização (movimento fluido)
                    if 'drone_x' not in st.session_state:
                        st.session_state.drone_x = px
                        st.session_state.drone_y = py

                    alpha = 0.15  # menor = mais suave

                    st.session_state.drone_x = (1 - alpha)*st.session_state.drone_x + alpha*px
                    st.session_state.drone_y = (1 - alpha)*st.session_state.drone_y + alpha*py

                    cx = int(st.session_state.drone_x)
                    cy = int(st.session_state.drone_y)
                    # 🚁 Drone (novo visual)
                    cv2.circle(frame, (cx, cy), 14, (255, 100, 0), -1)   # azul escuro (BGR)
                    cv2.circle(frame, (cx, cy), 6, (255, 255, 0), -1)    # núcleo ciano

                    # hélices
                    cv2.line(frame, (cx-16, cy), (cx+16, cy), (200,200,200), 2)
                    cv2.line(frame, (cx, cy-16), (cx, cy+16), (200,200,200), 2)

                    if 'trail' not in st.session_state:
                        st.session_state.trail = []

                    #st.session_state.trail.append((cx, cy))

                    # limitar tamanho do rastro
                    if len(st.session_state.trail) > 7:
                        st.session_state.trail.pop(0)

                    # desenhar rastro
                    for i in range(1, len(st.session_state.trail), 2):
                        cv2.line(frame,
                                st.session_state.trail[i-1],
                                st.session_state.trail[i],
                                (0,255,255),2)
                        
                    # 🧱 criar máscara do grid atual
                    mask = np.zeros_like(frame)

                    cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), (255,255,255), -1)

                    # 📡 radar com máscara (não vaza do grid)
                    tempo = time.time()
                    
                    radar_layer = np.zeros_like(frame)

                    for i in range(3):
                        raio = int((time.time()*80) % 40)
                        cv2.circle(radar_layer, (cx, cy), raio, (0,255,0), 2)

                    # aplicar máscara
                    radar_filtrado = cv2.bitwise_and(radar_layer, mask)

                    # juntar com frame
                    frame = cv2.add(frame, radar_filtrado)

                    # linha de scan
                    scan_layer = np.zeros_like(frame)
                    cv2.line(scan_layer, (x_start, cy), (x_end, cy), (255,100,0), 2)

                    scan_filtrado = cv2.bitwise_and(scan_layer, mask)
                    frame = cv2.add(frame, scan_filtrado)

                    if int(px) % 2 == 0:  # só atualiza 1 a cada 3 frames
                        placeholder_img.image(frame, use_container_width=True)

                    # velocidade REAL
                    time.sleep(0.01)

            # TEXTO DENTRO DO LOOP
            escrever_texto(f"{nome}: {tipo}")
            time.sleep(0.3)

        # LIMPAR TEXTO TEMPORÁRIO
        placeholder_text.empty()

        # imagem final (sem destaque azul)
        frame_final = img.copy()
        # grade geral
        for j in range(1, cols):
            x_line = j * (w // cols)
            cv2.line(frame_final, (x_line, 0), (x_line, h), (255,255,255), 2)
        for i in range(1, rows):
            y_line = i * (h // rows)
            cv2.line(frame_final, (0, y_line), (w, y_line), (255,255,255), 2)

        placeholder_img.image(frame_final, use_container_width=True)

        #Resumofinal
        st.subheader("Legenda do mapa")

        for tipo, cor in cores.items():
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:10px;'>"
                f"<div style='width:20px; height:20px; background-color:rgb{cor}; border-radius:4px;'></div>"
                f"<span>{tipo}</span>"
                f"</div>",
                unsafe_allow_html=True
            )        

        st.subheader("Recomendações Inteligentes")

        for nome, tipo in resultados.items():

            rec = recomendacao(tipo)

            st.markdown(f"""
            ### {nome} - {tipo}

            🌱 **Plantio:** {rec['plantio']}  
            💧 **Irrigação:** {rec['irrigacao']}  
            ⚙️ **Ação recomendada:** {rec['acao']}
            """)
