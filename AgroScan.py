#python -m streamlit run AgroScan.py

import streamlit as st
import cv2
import numpy as np
import time

st.title("🌱 AgroScan Drone - Análise de Terreno")

uploaded_file = st.file_uploader("Envie a foto do terreno", type=["jpg","png"])

if uploaded_file:

    # Carregar imagem
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Imagem analisada", use_container_width=True)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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

    st.write(f"Fértil: {round(p_verde*100,2)}%")
    st.write(f"Seco: {round(p_amarelo*100,2)}%")
    st.write(f"Rochoso: {round(p_cinza*100,2)}%")
    st.write(f"Úmido: {round(p_azul*100,2)}%")

    # 🗺️ Mapa geral
    mapa = np.zeros_like(img)

    mapa[verde > 0] = [0,255,0]
    mapa[amarelo > 0] = [255,255,0]
    mapa[cinza > 0] = [128,128,128]
    mapa[azul > 0] = [0,0,255]

    st.subheader("🗺️ Mapa geral do terreno")
    st.image(mapa, use_container_width=True)

    # GRID 4

    h, w, _ = img.shape

    regioes = {
        "Grid 1": hsv[0:h//2, 0:w//2],
        "Grid 2": hsv[0:h//2, w//2:w],
        "Grid 3": hsv[h//2:h, 0:w//2],
        "Grid 4": hsv[h//2:h, w//2:w]
    }

    def analisar_regiao(hsv_area):
        verde = cv2.inRange(hsv_area, (35,40,40), (85,255,255))
        amarelo = cv2.inRange(hsv_area, (20,100,100), (35,255,255))
        cinza = cv2.inRange(hsv_area, (0,0,50), (180,50,200))
        azul = cv2.inRange(hsv_area, (90,50,50), (130,255,255))

        total = hsv_area.shape[0] * hsv_area.shape[1]

        p_verde = np.sum(verde > 0) / total
        p_amarelo = np.sum(amarelo > 0) / total
        p_cinza = np.sum(cinza > 0) / total
        p_azul = np.sum(azul > 0) / total

        if p_verde > 0.4:
            tipo = "Fértil"
        elif p_amarelo > 0.3:
            tipo = "Seco"
        elif p_cinza > 0.3:
            tipo = "Rochoso"
        elif p_azul > 0.2:
            tipo = "Úmido"
        else:
            tipo = "Arenoso"

        return tipo, p_verde, p_amarelo, p_cinza, p_azul

    st.subheader("Análise por regiões")

    iniciar = st.button("Iniciar análise")

    placeholder_img = st.empty()
    placeholder_text = st.empty()

    if iniciar:

        resultados = {}

        # função de texto digitando
        def escrever_texto(texto):
            texto_animado = ""
            for char in texto:
                texto_animado += char
                placeholder_text.markdown(texto_animado)
                time.sleep(0.02)

        for nome, regiao in regioes.items():

            tipo, v, a, c, az = analisar_regiao(regiao)
            resultados[nome] = tipo

            # coordenadas do grid
            if nome == "Grid 1":
                x_start, x_end = 0, w//2
                y_start, y_end = 0, h//2
            elif nome == "Grid 2":
                x_start, x_end = w//2, w
                y_start, y_end = 0, h//2
            elif nome == "Grid 3":
                x_start, x_end = 0, w//2
                y_start, y_end = h//2, h
            else:
                x_start, x_end = w//2, w
                y_start, y_end = h//2, h

            # ANIMAÇÃO DO DRONE (CORRIGIDA)

            passos_x = 25
            passos_y = 6

            y_positions = np.linspace(y_start, y_end, passos_y)

            for idx, py in enumerate(y_positions):

                if idx % 2 == 0:
                    x_positions = np.linspace(x_start, x_end, passos_x)
                else:
                    x_positions = np.linspace(x_end, x_start, passos_x)

                for px in x_positions:

                    frame = img.copy()

                    # grade "+"
                    cv2.line(frame, (w//2, 0), (w//2, h), (255,255,255), 2)
                    cv2.line(frame, (0, h//2), (w, h//2), (255,255,255), 2)

                    # destaque da região
                    cv2.rectangle(frame, (x_start,y_start), (x_end,y_end), (255,0,0), 2)

                    # Drone (melhorado)
                    cx, cy = int(px), int(py)

                    cv2.circle(frame, (cx, cy), 12, (0,0,255), -1)
                    cv2.line(frame, (cx-14, cy), (cx+14, cy), (255,255,255), 2)
                    cv2.line(frame, (cx, cy-14), (cx, cy+14), (255,255,255), 2)

                    # radar
                    raio = int((time.time()*150) % 25)
                    cv2.circle(frame, (cx, cy), raio, (0,255,255), 1)

                    # linha de scan
                    cv2.line(frame, (x_start, cy), (x_end, cy), (0,255,255), 2)

                    # OVERLAY (AQUI É O LUGAR CERTO)
                    mapa_suave = cv2.GaussianBlur(mapa, (21,21), 0)
                    frame = cv2.addWeighted(frame, 0.8, mapa_suave, 0.2, 0)

                    placeholder_img.image(frame, use_container_width=True)

                    # velocidade REAL
                    time.sleep(0.02)

            # TEXTO DENTRO DO LOOP
            escrever_texto(f"{nome}: {tipo}")
            time.sleep(0.3)

        # LIMPAR TEXTO TEMPORÁRIO
        placeholder_text.empty()

        # imagem final (sem destaque azul)
        frame_final = img.copy()
        cv2.line(frame_final, (w//2, 0), (w//2, h), (255,255,255), 2)
        cv2.line(frame_final, (0, h//2), (w, h//2), (255,255,255), 2)

        placeholder_img.image(frame_final, use_container_width=True)

        
        # 📊 RESULTADO FINAL (FADE NORMAL)
        

        st.subheader("Resumo das regiões")

        for nome, tipo in resultados.items():
            st.write(f"{nome}: {tipo}")