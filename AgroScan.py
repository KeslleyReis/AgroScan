# python -m streamlit run AgroScan.py
# Dependências: pip install streamlit opencv-python-headless numpy pandas requests anthropic reportlab

import streamlit as st
import cv2
import numpy as np
import base64
import json
import io
import os
import time
import sqlite3
import hashlib
from datetime import datetime

import pandas as pd
import requests

# ═══════════════════════════════════════════════════════════════
# CONSTANTES GLOBAIS
# ═══════════════════════════════════════════════════════════════

MAX_W = 720
MAX_H = 480
APP_VERSION = "4.0"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
DB_PATH = os.path.join(DATA_DIR, "agroscan.db")

CORES_BGR = {
    "Fertil":  (34,  139, 34),
    "Umido":   (139, 90,  0),
    "Seco":    (30,  144, 200),
    "Arenoso": (80,  150, 210),
    "Rochoso": (80,  80,  80),
}

LABEL_PT = {
    "Fertil":  "Fértil",
    "Umido":   "Úmido",
    "Seco":    "Seco",
    "Arenoso": "Arenoso",
    "Rochoso": "Rochoso",
}

_VERDE   = dict(lo=(35, 40,  40),  hi=(85,  255, 255))
_AMARELO = dict(lo=(20, 100, 100), hi=(35,  255, 255))
_AZUL    = dict(lo=(90, 50,  50),  hi=(130, 255, 255))
_CINZA   = dict(lo=(0,  0,   50),  hi=(180, 50,  200))


# ═══════════════════════════════════════════════════════════════
# CSS GLOBAL — animações de digitação e fade-in
# ═══════════════════════════════════════════════════════════════

def inject_global_css():
    st.markdown("""
<style>
/* ── Animação de digitação (textos do topo) ──────────────────── */
@keyframes typing {
  from { width: 0; }
  to   { width: 100%; }
}
@keyframes blink {
  50% { border-color: transparent; }
}
.typewriter {
  display: inline-block;
  overflow: hidden;
  white-space: nowrap;
  border-right: 2px solid #22a622;
  animation: typing 1.8s steps(40, end) forwards,
             blink   .75s step-end 3;
  max-width: 100%;
}

/* ── Fade-in esquerda→direita curtíssimo (textos pós-drone) ─── */
@keyframes fadeSlideIn {
  from {
    opacity: 0;
    clip-path: inset(0 100% 0 0);
  }
  to {
    opacity: 1;
    clip-path: inset(0 0% 0 0);
  }
}
.fade-slide {
  animation: fadeSlideIn 0.45s ease-out forwards;
  opacity: 0;
}
/* escalonamento por índice (até 20 itens) */
.fade-slide:nth-child(1)  { animation-delay: 0.00s; }
.fade-slide:nth-child(2)  { animation-delay: 0.07s; }
.fade-slide:nth-child(3)  { animation-delay: 0.14s; }
.fade-slide:nth-child(4)  { animation-delay: 0.21s; }
.fade-slide:nth-child(5)  { animation-delay: 0.28s; }
.fade-slide:nth-child(6)  { animation-delay: 0.35s; }
.fade-slide:nth-child(7)  { animation-delay: 0.42s; }
.fade-slide:nth-child(8)  { animation-delay: 0.49s; }
.fade-slide:nth-child(9)  { animation-delay: 0.56s; }
.fade-slide:nth-child(10) { animation-delay: 0.63s; }
.fade-slide:nth-child(11) { animation-delay: 0.70s; }
.fade-slide:nth-child(12) { animation-delay: 0.77s; }
.fade-slide:nth-child(13) { animation-delay: 0.84s; }
.fade-slide:nth-child(14) { animation-delay: 0.91s; }
.fade-slide:nth-child(15) { animation-delay: 0.98s; }
.fade-slide:nth-child(16) { animation-delay: 1.05s; }
.fade-slide:nth-child(17) { animation-delay: 1.12s; }
.fade-slide:nth-child(18) { animation-delay: 1.19s; }
.fade-slide:nth-child(19) { animation-delay: 1.26s; }
.fade-slide:nth-child(20) { animation-delay: 1.33s; }
@media (max-width: 768px) {
  .block-container {
    padding: 1rem 0.75rem 4rem;
  }
  h1 {
    font-size: 1.35rem !important;
  }
  h3 {
    font-size: 1rem !important;
  }
  div[data-baseweb="tab-list"] {
    flex-wrap: wrap;
    gap: 4px;
  }
  div[data-testid="stHorizontalBlock"] {
    gap: 0.5rem;
  }
  .stButton > button,
  .stDownloadButton > button {
    width: 100%;
  }
}
</style>
""", unsafe_allow_html=True)


def typewriter(texto: str, tag="h3", extra_style=""):
    """Renderiza texto com efeito de digitação."""
    st.markdown(
        f"<{tag} style='margin:0 0 8px;{extra_style}'>"
        f"<span class='typewriter'>{texto}</span></{tag}>",
        unsafe_allow_html=True,
    )


def fade_slide_wrap(html_items: list[str]) -> str:
    """Envolve cada item HTML em .fade-slide para animação sequencial."""
    itens = "".join(f"<div class='fade-slide'>{item}</div>" for item in html_items)
    return f"<div>{itens}</div>"


# ═══════════════════════════════════════════════════════════════
# UTILITÁRIOS DE IMAGEM
# ═══════════════════════════════════════════════════════════════

def redimensionar(img: np.ndarray, max_w=MAX_W, max_h=MAX_H) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return img
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def img_to_base64(img: np.ndarray, quality=85) -> str:
    """Converte BGR → JPEG base64 (mantém BGR para JPEG nativo)."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def img_rgb_to_base64(img_rgb: np.ndarray, quality=85) -> str:
    """Converte RGB → JPEG base64 (para imagens já em RGB)."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def numpy_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode(".png", img_rgb)
    return bytes(buf)


def ensure_data_dirs():
    for path in (DATA_DIR, IMAGES_DIR):
        os.makedirs(path, exist_ok=True)


def get_db_connection():
    ensure_data_dirs()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            nome TEXT NOT NULL,
            cidade TEXT,
            latitude REAL,
            longitude REAL,
            area_ha REAL DEFAULT 0,
            observacoes TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            property_id INTEGER NOT NULL,
            image_source TEXT,
            image_path TEXT,
            map_path TEXT,
            resultado_geral TEXT,
            ndvi_global REAL,
            score_medio REAL,
            total_grids INTEGER,
            weather_json TEXT,
            risks_json TEXT,
            metricas_json TEXT,
            analises_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(property_id) REFERENCES properties(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            property_id INTEGER,
            label TEXT NOT NULL,
            ndvi REAL NOT NULL,
            textura REAL NOT NULL,
            umidade REAL NOT NULL,
            saturacao REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(property_id) REFERENCES properties(id)
        )
        """
    )
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def criar_usuario(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    if len(username) < 3 or len(password) < 4:
        return False, "Usuário deve ter 3+ caracteres e senha 4+ caracteres."

    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, hash_password(password), datetime.now().isoformat()),
        )
        conn.commit()
        return True, "Conta criada com sucesso."
    except sqlite3.IntegrityError:
        return False, "Este usuário já existe."
    finally:
        conn.close()


def autenticar_usuario(username: str, password: str):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT id, username, created_at FROM users WHERE username = ? AND password_hash = ?",
        (username.strip().lower(), hash_password(password)),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def salvar_propriedade(user_id: int, nome: str, cidade: str,
                       latitude: float, longitude: float,
                       area_ha: float, observacoes: str) -> tuple[bool, str]:
    nome = nome.strip()
    if not nome:
        return False, "Informe o nome da propriedade."

    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO properties (user_id, nome, cidade, latitude, longitude, area_ha, observacoes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            nome,
            cidade.strip(),
            float(latitude),
            float(longitude),
            float(area_ha),
            observacoes.strip(),
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return True, "Propriedade salva com sucesso."


def listar_propriedades(user_id: int) -> list[dict]:
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT * FROM properties WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def contar_amostras_modelo(user_id: int) -> int:
    conn = get_db_connection()
    total = conn.execute(
        "SELECT COUNT(*) AS total FROM model_samples WHERE user_id = ?",
        (user_id,),
    ).fetchone()["total"]
    conn.close()
    return int(total or 0)


@st.cache_data(ttl=1800, show_spinner=False)
def obter_previsao_tempo(latitude: float, longitude: float) -> dict:
    try:
        resposta = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
                "forecast_days": 3,
            },
            timeout=15,
        )
        resposta.raise_for_status()
        dados = resposta.json()
        atual = dados.get("current", {})
        diario = dados.get("daily", {})

        previsao = []
        datas = diario.get("time", [])
        maxs = diario.get("temperature_2m_max", [])
        mins = diario.get("temperature_2m_min", [])
        chuvas = diario.get("precipitation_sum", [])
        for data, tmax, tmin, chuva in zip(datas, maxs, mins, chuvas):
            previsao.append({
                "data": data,
                "temp_max": tmax,
                "temp_min": tmin,
                "chuva_mm": chuva,
            })

        return {
            "temperatura": atual.get("temperature_2m"),
            "umidade": atual.get("relative_humidity_2m"),
            "precipitacao": atual.get("precipitation"),
            "vento": atual.get("wind_speed_10m"),
            "previsao": previsao,
        }
    except Exception as e:
        return {"erro": str(e), "previsao": []}


def prever_tipo_modelo_local(ndvi: float, textura: float,
                             umidade_frac: float, sat_mean: float):
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT label, COUNT(*) AS total,
               AVG(ndvi) AS ndvi,
               AVG(textura) AS textura,
               AVG(umidade) AS umidade,
               AVG(saturacao) AS saturacao
        FROM model_samples
        GROUP BY label
        HAVING COUNT(*) >= 3
        """
    ).fetchall()
    conn.close()

    if not rows:
        return None, 0.0, 0

    amostra = np.array([ndvi, textura / 80.0, umidade_frac, sat_mean], dtype=float)
    melhor_label = None
    melhor_dist = None
    total_amostras = 0

    for row in rows:
        total_amostras += int(row["total"])
        centro = np.array([
            float(row["ndvi"]),
            float(row["textura"]) / 80.0,
            float(row["umidade"]),
            float(row["saturacao"]),
        ], dtype=float)
        dist = float(np.linalg.norm(amostra - centro))
        if melhor_dist is None or dist < melhor_dist:
            melhor_dist = dist
            melhor_label = row["label"]

    confianca = 0.0 if melhor_dist is None else max(0.0, min(0.95, 1.0 - melhor_dist / 1.5))
    return melhor_label, round(confianca, 2), total_amostras


def registrar_amostras_modelo(user_id: int, property_id: int, analises: dict):
    conn = get_db_connection()
    agora = datetime.now().isoformat()
    for dados in analises.values():
        conn.execute(
            """
            INSERT INTO model_samples (user_id, property_id, label, ndvi, textura, umidade, saturacao, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                property_id,
                dados["tipo"],
                float(dados["ndvi"]),
                float(dados["textura"]),
                float(dados["umidade_frac"]),
                float(dados.get("sat_mean", 0.0)),
                agora,
            ),
        )
    conn.commit()
    conn.close()


def detectar_riscos(analises: dict, metricas: dict, clima: dict | None = None) -> list[dict]:
    riscos = []
    score_medio = float(metricas.get("score_medio", 0))
    ndvi_global = float(metricas.get("ndvi_global", 0))
    p_amarelo = float(metricas.get("p_amarelo", 0))
    p_azul = float(metricas.get("p_azul", 0))
    p_cinza = float(metricas.get("p_cinza", 0))
    textura_media = float(np.mean([d["textura"] for d in analises.values()])) if analises else 0.0

    def add_risk(titulo: str, nivel: str, descricao: str, acao: str):
        riscos.append({
            "titulo": titulo,
            "nivel": nivel,
            "descricao": descricao,
            "acao": acao,
        })

    if ndvi_global < 0.12 or p_amarelo > 0.28:
        add_risk(
            "Risco de seca",
            "Alto" if ndvi_global < 0.05 else "Médio",
            "A vegetação está com baixo vigor e parte do terreno apresenta estresse hídrico.",
            "Priorizar gotejamento, cobertura morta e monitorar umidade do solo nos próximos 7 dias.",
        )

    if p_azul > 0.24 or (clima and clima.get("previsao") and any((dia.get("chuva_mm") or 0) > 20 for dia in clima.get("previsao", []))):
        add_risk(
            "Risco de Alagamento",
            "Médio",
            "Há sinais de umidade elevada e possibilidade de acúmulo de água em parte da área.",
            "Abrir drenagem lateral, nivelar pontos críticos e revisar compactação superficial.",
        )

    if p_cinza > 0.20 or textura_media > 42:
        add_risk(
            "Risco de erosão / solo exposto",
            "Médio",
            "O terreno tem áreas com textura elevada, rochosidade ou pouca cobertura vegetal.",
            "Aplicar plantio em curva de nível e cobertura vegetal para estabilizar o solo.",
        )

    if score_medio < 45:
        add_risk(
            "Baixa aptidão agrícola",
            "Alto",
            "O score médio da área indica necessidade de correção estrutural antes de culturas mais exigentes.",
            "Fazer análise química do solo, corrigir pH e aumentar matéria orgânica.",
        )

    if clima and not clima.get("erro") and (clima.get("umidade") or 0) > 80 and 0.10 < ndvi_global < 0.32:
        add_risk(
            "Risco de pragas e fungos",
            "Médio",
            "Umidade do ar elevada favorece pressão fitossanitária em culturas tropicais.",
            "Inspecionar folhas semanalmente e programar manejo preventivo.",
        )

    return riscos


def gerar_resumo_executivo(resultado_geral: str, metricas: dict, riscos: list[dict]) -> list[str]:
    resumo = [
        f"Resultado geral: {resultado_geral}.",
        f"NDVI médio estimado em {round(metricas.get('ndvi_global', 0) * 100, 1)}% e score médio de {round(metricas.get('score_medio', 0), 1)}/100.",
    ]
    if riscos:
        prioridades = ", ".join(f"{r['titulo']} ({r['nivel']})" for r in riscos[:3])
        resumo.append(f"Prioridades imediatas: {prioridades}.")
    else:
        resumo.append("Nenhum risco crítico foi identificado na triagem automática.")
    return resumo


def salvar_analise_historico(user_id: int, property_id: int, origem_imagem: str,
                             img_orig: np.ndarray, img_mapa: np.ndarray,
                             resultado_geral: str, metricas: dict, analises: dict,
                             clima: dict, riscos: list[dict]) -> int:
    ensure_data_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = os.path.join(IMAGES_DIR, f"img_{timestamp}.png")
    map_path = os.path.join(IMAGES_DIR, f"mapa_{timestamp}.png")

    with open(image_path, "wb") as f:
        f.write(numpy_to_png_bytes(img_orig))
    with open(map_path, "wb") as f:
        f.write(numpy_to_png_bytes(img_mapa))

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO analyses (
            user_id, property_id, image_source, image_path, map_path,
            resultado_geral, ndvi_global, score_medio, total_grids,
            weather_json, risks_json, metricas_json, analises_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            property_id,
            origem_imagem,
            image_path,
            map_path,
            resultado_geral,
            float(metricas.get("ndvi_global", 0.0)),
            float(metricas.get("score_medio", 0.0)),
            int(metricas.get("total_grids", len(analises))),
            json.dumps(clima or {}, ensure_ascii=False),
            json.dumps(riscos or [], ensure_ascii=False),
            json.dumps(metricas or {}, ensure_ascii=False),
            json.dumps(analises or {}, ensure_ascii=False),
            datetime.now().isoformat(),
        ),
    )
    analysis_id = cur.lastrowid
    conn.commit()
    conn.close()
    return analysis_id


def listar_analises_salvas(user_id: int, property_id: int | None = None) -> list[dict]:
    conn = get_db_connection()
    query = (
        "SELECT a.*, p.nome AS propriedade_nome, p.cidade AS propriedade_cidade "
        "FROM analyses a "
        "LEFT JOIN properties p ON p.id = a.property_id "
        "WHERE a.user_id = ?"
    )
    params: list = [user_id]
    if property_id is not None:
        query += " AND a.property_id = ?"
        params.append(property_id)
    query += " ORDER BY a.created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def obter_analise_salva(analysis_id: int):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT a.*, p.nome AS propriedade_nome, p.cidade AS propriedade_cidade, p.latitude, p.longitude "
        "FROM analyses a LEFT JOIN properties p ON p.id = a.property_id WHERE a.id = ?",
        (analysis_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None

    item = dict(row)
    item["weather"] = json.loads(item.get("weather_json") or "{}")
    item["risks"] = json.loads(item.get("risks_json") or "[]")
    item["metricas"] = json.loads(item.get("metricas_json") or "{}")
    item["analises"] = json.loads(item.get("analises_json") or "{}")
    return item


def render_auth_screen():
    st.info("Faça login para acessar o painel completo, histórico e gestão por fazenda.")
    tab_login, tab_registro = st.tabs(["Entrar", "Criar conta"])

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Usuário")
            password = st.text_input("Senha", type="password")
            entrou = st.form_submit_button("Entrar", use_container_width=True)
        if entrou:
            user = autenticar_usuario(username, password)
            if user:
                st.session_state["current_user"] = user
                st.success("Login realizado com sucesso.")
                st.rerun()
            else:
                st.error("Usuário ou senha inválidos.")

    with tab_registro:
        with st.form("register_form", clear_on_submit=True):
            novo_user = st.text_input("Novo usuário")
            nova_senha = st.text_input("Nova senha", type="password")
            criar = st.form_submit_button("Criar conta", use_container_width=True)
        if criar:
            ok, msg = criar_usuario(novo_user, nova_senha)
            if ok:
                st.success(msg)
            else:
                st.warning(msg)


# ═══════════════════════════════════════════════════════════════
# MAPA DE TERRENO  — TURBO colormap (mais informativo)
# ═══════════════════════════════════════════════════════════════

def gerar_mapa_terreno(img: np.ndarray) -> np.ndarray:
    """
    NDVI aproximado com colormap TURBO:
      Azul escuro  = sem vegetação / solo exposto
      Ciano/Verde  = vegetação moderada
      Amarelo/Verm = vegetação densa / stress hídrico
    Retorna em BGR (mesmo espaço da imagem original).
    """
    b_f, g_f, r_f = cv2.split(img.astype("float32"))
    ndvi = (g_f - r_f) / (g_f + r_f + 1e-5)
    ndvi_norm = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mapa = cv2.applyColorMap(ndvi_norm, cv2.COLORMAP_TURBO)   # ← TURBO substituindo VIRIDIS
    return mapa                                                 # BGR


# ═══════════════════════════════════════════════════════════════
# ANÁLISE DETALHADA POR REGIÃO
# ═══════════════════════════════════════════════════════════════

def extrair_features_regiao(img_area: np.ndarray):
    b, g, r = cv2.split(img_area.astype("float32"))
    eps = 1e-5
    ndvi = float(np.mean((g - r) / (g + r + eps)))
    gray = cv2.cvtColor(img_area, cv2.COLOR_BGR2GRAY)
    textura = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_64F))))
    umidade_frac = float(np.mean(b > g + 10))
    hsv_area = cv2.cvtColor(img_area, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv_area[:, :, 1])) / 255.0
    return ndvi, textura, umidade_frac, sat_mean


def analisar_regiao(img_area: np.ndarray) -> dict:
    ndvi, textura, umidade_frac, sat_mean = extrair_features_regiao(img_area)

    if ndvi > 0.30:
        tipo_heuristico = "Fertil"
    elif umidade_frac > 0.35:
        tipo_heuristico = "Umido"
    elif textura > 45:
        tipo_heuristico = "Rochoso"
    elif ndvi < 0.08 and sat_mean < 0.25:
        tipo_heuristico = "Seco"
    else:
        tipo_heuristico = "Arenoso"

    tipo_modelo, confianca_modelo, total_amostras = prever_tipo_modelo_local(
        ndvi, textura, umidade_frac, sat_mean
    )

    tipo = tipo_heuristico
    origem_modelo = "Heurístico"
    if tipo_modelo and total_amostras >= 6 and confianca_modelo >= 0.60:
        tipo = tipo_modelo
        origem_modelo = f"Híbrido ({total_amostras} amostras reais)"

    s_ndvi = max(0.0, min(1.0, (ndvi + 0.2) / 0.7)) * 40
    s_umidade = (1.0 - abs(umidade_frac - 0.25) / 0.25) * 25
    s_textura = max(0.0, 1.0 - textura / 80.0) * 20
    s_sat = sat_mean * 15
    score = int(min(100, max(0, s_ndvi + s_umidade + s_textura + s_sat)))

    culturas_map = {
        "Fertil": ["Soja", "Milho", "Feijao", "Cana-de-acucar"],
        "Umido": ["Arroz irrigado", "Banana", "Coco", "Taro"],
        "Seco": ["Sorgo", "Mandioca", "Palma forrageira", "Sisal"],
        "Arenoso": ["Amendoim", "Batata-doce", "Cenoura", "Melancia"],
        "Rochoso": ["Sem uso agricola direto"],
    }
    irrig_map = {
        "Fertil": 4.0,
        "Umido": 1.5,
        "Seco": 8.0,
        "Arenoso": 6.0,
        "Rochoso": 0.0,
    }

    return {
        "tipo": tipo,
        "tipo_heuristico": tipo_heuristico,
        "tipo_modelo": tipo_modelo or "Indisponível",
        "origem_modelo": origem_modelo,
        "confianca_modelo": confianca_modelo,
        "ndvi": round(ndvi, 3),
        "textura": round(textura, 1),
        "umidade_frac": round(umidade_frac, 3),
        "sat_mean": round(sat_mean, 3),
        "score": score,
        "culturas": culturas_map.get(tipo, ["Indefinido"]),
        "irrigacao_mm": irrig_map.get(tipo, 5.0),
    }


# ═══════════════════════════════════════════════════════════════
# RECOMENDAÇÕES TEXTUAIS (sem IA)
# ═══════════════════════════════════════════════════════════════

def recomendacao_local(dados: dict) -> dict:
    tipo  = dados["tipo"]
    irr   = dados["irrigacao_mm"]
    score = dados["score"]
    ndvi  = dados["ndvi"]
    tex   = dados["textura"]

    rec = {
        "Fertil": {
            "opcao_1": {
                "perfil":    "Producao convencional",
                "culturas":  "Soja, Milho ou Feijao",
                "plantio":   f"Solo em boas condicoes (Score {score}/100). Plantio direto recomendado para manter estrutura.",
                "irrigacao": f"~{irr} mm/dia via aspersao convencional. Monitoramento quinzenal de umidade.",
                "acao":      "Analise de solo a cada 2 anos. Rotacao com gramíneas para fixar nitrogenio.",
            },
            "opcao_2": {
                "perfil":    "Agricultura organica",
                "culturas":  "Hortalicas, Frutas ou Leguminosas",
                "plantio":   f"NDVI {ndvi} indica boa biomassa. Ideal para transicao organica com menor insumo.",
                "irrigacao": f"~{irr-1:.1f} mm/dia via gotejamento subsuperficial. Economia de 30% de agua.",
                "acao":      "Adubacao verde com crotalaria. Compostagem no proprio terreno.",
            },
            "opcao_3": {
                "perfil":    "Alta produtividade",
                "culturas":  "Cana-de-acucar ou Eucalipto",
                "plantio":   f"Terreno com Score {score}/100 suporta culturas de ciclo longo e alta demanda.",
                "irrigacao": f"~{irr+1:.1f} mm/dia com pivo central ou gotejamento de longa duracao.",
                "acao":      "Calagem preventiva. Monitoramento mensal de pragas e nutricao foliar.",
            },
        },
        "Umido": {
            "opcao_1": {
                "perfil":    "Aproveitamento da umidade",
                "culturas":  "Arroz irrigado ou Taro",
                "plantio":   f"Umidade elevada favorece culturas semiaquaticas. Score {score}/100.",
                "irrigacao": f"~{irr} mm/dia. Manter lamina d'agua controlada de 5-10 cm.",
                "acao":      "Nivelar o terreno para uniformizar a lamina. Evitar areas de estagnacao.",
            },
            "opcao_2": {
                "perfil":    "Fruticultura adaptada",
                "culturas":  "Banana, Coco ou Acai",
                "plantio":   f"NDVI {ndvi} indica condicoes favoraveis para fruticultura tropical umida.",
                "irrigacao": f"~{irr-0.5:.1f} mm/dia. Drenagem lateral essencial para evitar podridao de raiz.",
                "acao":      "Instalar valetas de drenagem. Cobertura de palha entre plantas.",
            },
            "opcao_3": {
                "perfil":    "Reflorestamento produtivo",
                "culturas":  "Seringueira ou Palmito",
                "plantio":   f"Solo umido com Score {score}/100 e ideal para especies de alto valor umido.",
                "irrigacao": f"Minimo — ~{max(0.5, irr-1):.1f} mm/dia apenas em secas prolongadas.",
                "acao":      "Adubacao fosfatada no plantio. Controle de invasoras nos 2 primeiros anos.",
            },
        },
        "Seco": {
            "opcao_1": {
                "perfil":    "Culturas resistentes a seca",
                "culturas":  "Sorgo ou Mandioca",
                "plantio":   f"NDVI baixo ({ndvi}) confirma stress hidrico. Culturas CAM e C4 sao ideais.",
                "irrigacao": f"~{irr} mm/dia gotejamento subsuperficial. Irrigar no periodo da manha.",
                "acao":      "Mulching com palha seca para reduzir evaporacao. Cova profunda no plantio.",
            },
            "opcao_2": {
                "perfil":    "Pecuaria extensiva",
                "culturas":  "Palma forrageira ou Capim buffel",
                "plantio":   f"Score {score}/100. Forrageiras resistentes a seca sao mais viaveis que graos.",
                "irrigacao": f"~{irr-2:.1f} mm/dia. Cisternas de captacao de agua da chuva como suporte.",
                "acao":      "Terraceamento para captacao de chuva. Calcario para corrigir pH acido.",
            },
            "opcao_3": {
                "perfil":    "Recuperacao gradual",
                "culturas":  "Sisal ou Agave",
                "plantio":   f"Terreno seco com Score {score}/100 — plantas xerofitas preparam o solo para futuras culturas.",
                "irrigacao": f"~{max(1.0, irr-3):.1f} mm/dia apenas no estabelecimento (primeiros 60 dias).",
                "acao":      "Introducao de materia organica por compostagem. Plantio em curvas de nivel.",
            },
        },
        "Arenoso": {
            "opcao_1": {
                "perfil":    "Hortalicas de raiz",
                "culturas":  "Cenoura, Batata-doce ou Amendoim",
                "plantio":   f"Solo arenoso (textura {tex}) facilita crescimento de raizes. Score {score}/100.",
                "irrigacao": f"~{irr} mm/dia fracionado em 3x ao dia. Solo arenoso nao reten agua.",
                "acao":      "Incorporar 20% de argila e composto organico antes do plantio.",
            },
            "opcao_2": {
                "perfil":    "Fruticultura irrigada",
                "culturas":  "Melancia, Melao ou Morango",
                "plantio":   f"NDVI {ndvi} em solo arenoso indica boa drenagem para fruticultura de mesa.",
                "irrigacao": f"~{irr-1:.1f} mm/dia via gotejamento com fertirrigacao.",
                "acao":      "Adubacao solavel parcelada semanalmente. Cobertura plastica (mulching).",
            },
            "opcao_3": {
                "perfil":    "Melhoria estrutural do solo",
                "culturas":  "Feijao-de-porco ou Mucuna (adubacao verde)",
                "plantio":   f"Score {score}/100. Ciclo de adubacao verde por 1-2 safras antes de culturas comerciais.",
                "irrigacao": f"~{irr-2:.1f} mm/dia. Priorizar agua no periodo critico de florescimento.",
                "acao":      "Incorporar biomassa no solo apos florescimento. Aguardar 30 dias para novo plantio.",
            },
        },
        "Rochoso": {
            "opcao_1": {
                "perfil":    "Uso minimo viavel",
                "culturas":  "Cactos ornamentais ou Suculentas",
                "plantio":   f"Solo rochoso (textura {tex}, Score {score}/100). Uso agricola direto inviavel.",
                "irrigacao": "Minimo — apenas para estabelecimento inicial.",
                "acao":      "Dinamite ou subsolagem profunda para quebrar camada rochosa superficial.",
            },
            "opcao_2": {
                "perfil":    "Reflorestamento nativo",
                "culturas":  "Especies nativas da regiao (aroeira, umbu)",
                "plantio":   f"NDVI {ndvi} em area rochosa — vegetacao nativa e a opcao mais sustentavel.",
                "irrigacao": "Zero apos estabelecimento (90 dias). Especies adaptadas a litossolos.",
                "acao":      "Coveamento manual de 40x40x40 cm com substrato enriquecido.",
            },
            "opcao_3": {
                "perfil":    "Aproveitamento alternativo",
                "culturas":  "Geracao fotovoltaica ou turismo rural",
                "plantio":   f"Score {score}/100 indica baixo potencial agricola. Uso alternativo e mais rentavel.",
                "irrigacao": "Nao aplicavel.",
                "acao":      "Estudo topografico para implantacao de paineis solares ou trilhas ecoturisticas.",
            },
        },
    }

    return rec.get(tipo, rec["Arenoso"])


# ═══════════════════════════════════════════════════════════════
# ANÁLISE COM IA (Claude)
# ═══════════════════════════════════════════════════════════════

def analisar_com_ia(img: np.ndarray, analises: dict, api_key: str) -> str:
    try:
        import anthropic

        linhas = []
        for nome, d in analises.items():
            linhas.append(
                f"- {nome}: {d['tipo']} | NDVI={d['ndvi']} | "
                f"Score={d['score']}/100 | Irrigacao={d['irrigacao_mm']}mm/dia"
            )
        resumo = "\n".join(linhas)

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1800,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_to_base64(img, quality=75),
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Voce e um agronomo especialista em agricultura tropical brasileira.\n"
                            "Analise esta imagem de terreno capturada por drone.\n\n"
                            "A analise automatica classificou os grids assim:\n"
                            f"{resumo}\n\n"
                            "Forneca:\n"
                            "1. Avaliacao geral do terreno (2-3 frases).\n"
                            "2. Para cada tipo de solo: culturas rentaveis, manejo de irrigacao,"
                            " correcao de solo.\n"
                            "3. Riscos identificados.\n"
                            "4. Prioridades de intervencao.\n\n"
                            "Responda em portugues, de forma direta e pratica."
                        ),
                    },
                ],
            }],
        )
        return msg.content[0].text
    except Exception as e:
        return f"**Erro na analise IA:** {e}"


# ═══════════════════════════════════════════════════════════════
# SLIDER COMPARATIVO — tonalidade corrigida (BGR→RGB uniforme)
# ═══════════════════════════════════════════════════════════════

def slider_comparacao(img_orig_bgr: np.ndarray, img_mapa_bgr: np.ndarray, initial=50):
    """
    Ambas as imagens são convertidas para RGB antes de codificar em JPEG,
    garantindo que a tonalidade exibida seja idêntica à imagem original.
    """
    h, w = img_orig_bgr.shape[:2]

    # ── Correção de cor: BGR → RGB para exibição correta no browser ──
    orig_rgb = cv2.cvtColor(img_orig_bgr, cv2.COLOR_BGR2RGB)
    mapa_rgb = cv2.cvtColor(img_mapa_bgr, cv2.COLOR_BGR2RGB)

    b64o = img_rgb_to_base64(orig_rgb)
    b64m = img_rgb_to_base64(mapa_rgb)

    asp = round((h / w) * 100, 2)
    inv = 100 - initial

    html = f"""
<div style="position:relative;width:100%;max-width:{w}px;margin:0 auto;">
  <div style="position:absolute;top:8px;left:10px;z-index:12;color:#fff;
              font-size:13px;font-weight:600;text-shadow:0 1px 4px rgba(0,0,0,.9);">
    <span id="lbl">{initial}%</span>
  </div>
  <div id="cmp" style="position:relative;width:100%;padding-bottom:{asp}%;
                        cursor:col-resize;border-radius:8px;overflow:hidden;">
    <img src="data:image/jpeg;base64,{b64o}"
         style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;"/>
    <div id="clip" style="position:absolute;inset:0;overflow:hidden;
                           clip-path:inset(0 {inv}% 0 0);">
      <img src="data:image/jpeg;base64,{b64m}"
           style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;"/>
    </div>
    <div id="line" style="position:absolute;top:0;left:{initial}%;width:2px;height:100%;
                            background:#0ff;box-shadow:0 0 8px #0ff;z-index:10;"/>
  </div>
</div>
<script>
(function(){{
  const cmp=document.getElementById('cmp'),
        clip=document.getElementById('clip'),
        line=document.getElementById('line'),
        lbl=document.getElementById('lbl');
  function set(p){{
    p=Math.max(0,Math.min(100,p));
    clip.style.clipPath='inset(0 '+(100-p)+'% 0 0)';
    line.style.left=p+'%'; lbl.textContent=Math.round(p)+'%';
  }}
  function ev(e){{ const r=cmp.getBoundingClientRect(); set((e.clientX-r.left)/r.width*100); }}
  cmp.addEventListener('mousemove',ev);
  cmp.addEventListener('touchmove',e=>ev(e.touches[0]),{{passive:true}});
  set({initial});
}})();
</script>"""
    st.components.v1.html(html, height=h + 10)


# ═══════════════════════════════════════════════════════════════
# ANIMAÇÃO DO DRONE
# ═══════════════════════════════════════════════════════════════

def drone_animation_component(img: np.ndarray, analises: dict,
                               coords: dict, rows: int, cols: int) -> str:
    h, w    = img.shape[:2]
    # Drone usa BGR corrigido para RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_b64 = img_rgb_to_base64(img_rgb)

    grids = []
    for nome, dados in analises.items():
        xs, xe, ys, ye = coords[nome]
        grids.append({
            "nome":  nome,
            "tipo":  dados["tipo"],
            "score": dados["score"],
            "xs": int(xs), "xe": int(xe),
            "ys": int(ys), "ye": int(ye),
        })

    grids_json = json.dumps(grids, ensure_ascii=False)

    return f"""
<div style="max-width:{w}px;margin:0 auto;">
<canvas id="dc" style="width:100%;height:auto;border-radius:8px;display:block;"></canvas>
<div id="dl" style="font-family:monospace;font-size:12px;color:#aaa;
                     min-height:20px;margin-top:5px;"></div>
</div>
<script>
(function(){{
  const B64={json.dumps(img_b64)},GRIDS={grids_json};
  const ROWS={rows},COLS={cols},IW={w},IH={h};
  const cv=document.getElementById('dc'),ctx=cv.getContext('2d');
  const dl=document.getElementById('dl');
  cv.width=IW; cv.height=IH;
  const bg=new Image();
  bg.onload=()=>run();
  bg.src='data:image/jpeg;base64,'+B64;

  function grid(hl){{
    ctx.strokeStyle='rgba(255,255,255,0.55)';ctx.lineWidth=1.5;
    for(let j=1;j<COLS;j++){{
      const x=j*Math.floor(IW/COLS);
      ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,IH);ctx.stroke();
    }}
    for(let i=1;i<ROWS;i++){{
      const y=i*Math.floor(IH/ROWS);
      ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(IW,y);ctx.stroke();
    }}
    if(hl){{
      ctx.strokeStyle='rgba(255,210,0,0.95)';ctx.lineWidth=2.5;
      ctx.strokeRect(hl.xs,hl.ys,hl.xe-hl.xs,hl.ye-hl.ys);
    }}
  }}

  function drone(cx,cy,t){{
    const r=(t*60)%42;
    ctx.save();
    ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);
    ctx.strokeStyle=`rgba(0,255,100,${{Math.max(0,.65-r/55)}})`;
    ctx.lineWidth=1.2;ctx.stroke();
    ctx.restore();
    ctx.beginPath();ctx.arc(cx,cy,13,0,Math.PI*2);
    ctx.fillStyle='rgba(255,120,0,.95)';ctx.fill();
    ctx.beginPath();ctx.arc(cx,cy,5,0,Math.PI*2);
    ctx.fillStyle='rgba(255,240,0,.95)';ctx.fill();
    ctx.strokeStyle='rgba(200,200,200,.85)';ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(cx-15,cy);ctx.lineTo(cx+15,cy);ctx.stroke();
    ctx.beginPath();ctx.moveTo(cx,cy-15);ctx.lineTo(cx,cy+15);ctx.stroke();
  }}

  const sleep=ms=>new Promise(r=>setTimeout(r,ms));

  async function scan(g){{
    const{{xs,xe,ys,ye,nome,tipo,score}}=g;
    let dx=xs,dy=ys;const a=0.18;let t=0;
    for(let yi=0;yi<5;yi++){{
      const py=ys+(ye-ys)*yi/4;
      for(let xi=0;xi<18;xi++){{
        const f=xi/17;
        const px=yi%2===0?xs+(xe-xs)*f:xe-(xe-xs)*f;
        dx=(1-a)*dx+a*px;dy=(1-a)*dy+a*py;
        ctx.drawImage(bg,0,0,IW,IH);
        grid(g);drone(dx,dy,t);t+=0.016;
        await sleep(13);
      }}
    }}
    dl.textContent=nome+': '+tipo+' | Score: '+score+'/100';
  }}

  async function run(){{
    for(const g of GRIDS){{await scan(g);await sleep(260);}}
    ctx.drawImage(bg,0,0,IW,IH);grid(null);
    dl.textContent='Concluido - '+GRIDS.length+' grids analisados';
  }}
}})();
</script>"""


# ═══════════════════════════════════════════════════════════════
# EXPORTAR PDF  — migrado para reportlab (sem fpdf2)
# ═══════════════════════════════════════════════════════════════

def gerar_pdf(img_orig: np.ndarray, img_mapa: np.ndarray,
              analises: dict, rec_texto, meta: dict | None = None) -> io.BytesIO | None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            Image as RLImage, HRFlowable,
        )
        from reportlab.lib.enums import TA_CENTER
        from reportlab.graphics.shapes import Drawing
        from reportlab.graphics.charts.barcharts import VerticalBarChart

        meta = meta or {}
        propriedade = meta.get("propriedade", {})
        clima = meta.get("clima", {})
        riscos = meta.get("riscos", [])
        resumo_executivo = meta.get("resumo_executivo", [])
        fonte = meta.get("origem_imagem", "N/D")
        usuario = meta.get("usuario", "N/D")

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=15 * mm, rightMargin=15 * mm,
            topMargin=15 * mm, bottomMargin=15 * mm,
        )

        styles = getSampleStyleSheet()
        VERDE = colors.HexColor("#22a622")
        VERDE_ESCURO = colors.HexColor("#145214")
        CINZA = colors.HexColor("#646464")
        VERMELHO = colors.HexColor("#b03030")

        titulo_style = ParagraphStyle(
            "Titulo", parent=styles["Title"],
            textColor=VERDE, fontSize=19, alignment=TA_CENTER, spaceAfter=4,
        )
        subtitulo_style = ParagraphStyle(
            "Subtitulo", parent=styles["Normal"],
            textColor=CINZA, fontSize=9, alignment=TA_CENTER, spaceAfter=8,
        )
        secao_style = ParagraphStyle(
            "Secao", parent=styles["Heading2"],
            textColor=VERDE, fontSize=12, spaceBefore=8, spaceAfter=4,
        )
        corpo_style = ParagraphStyle(
            "Corpo", parent=styles["Normal"],
            fontSize=8, leading=11, spaceAfter=3,
        )
        opcao_titulo_style = ParagraphStyle(
            "OpcaoTitulo", parent=styles["Normal"],
            fontSize=9, leading=12, textColor=VERDE_ESCURO,
            fontName="Helvetica-Bold",
        )
        opcao_body_style = ParagraphStyle(
            "OpcaoBody", parent=styles["Normal"],
            fontSize=8, leading=11, leftIndent=8,
        )

        def safe(txt) -> str:
            return (str(txt)
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))

        story = []

        story.append(Paragraph("AgroScan Drone", titulo_style))
        story.append(Paragraph("Relatório Executivo de Diagnóstico Agrícola", subtitulo_style))
        story.append(Paragraph(
            f"Gerado em {time.strftime('%d/%m/%Y às %H:%M')} • Usuário: {safe(usuario)}",
            subtitulo_style,
        ))
        story.append(HRFlowable(width="100%", thickness=1, color=VERDE, spaceAfter=8))

        info_table = Table([
            ["Propriedade", safe(propriedade.get("nome", "Não informada")), "Cidade/Região", safe(propriedade.get("cidade", "N/D"))],
            ["Área (ha)", safe(propriedade.get("area_ha", "0")), "Origem da imagem", safe(fonte)],
            ["Latitude", safe(propriedade.get("latitude", "N/D")), "Longitude", safe(propriedade.get("longitude", "N/D"))],
        ], colWidths=[28 * mm, 58 * mm, 28 * mm, 58 * mm])
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fbf8")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7e7d7")),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 4 * mm))

        story.append(Paragraph("Resumo Executivo", secao_style))
        for linha in resumo_executivo or ["Resumo não disponível."]:
            story.append(Paragraph(f"• {safe(linha)}", corpo_style))

        if clima and not clima.get("erro"):
            story.append(Paragraph("Clima e Janela Operacional", secao_style))
            story.append(Paragraph(
                f"Temperatura atual: {safe(clima.get('temperatura', 'N/D'))} °C • "
                f"Umidade: {safe(clima.get('umidade', 'N/D'))}% • "
                f"Vento: {safe(clima.get('vento', 'N/D'))} km/h • "
                f"Precipitação: {safe(clima.get('precipitacao', 'N/D'))} mm",
                corpo_style,
            ))
            for dia in clima.get("previsao", [])[:3]:
                story.append(Paragraph(
                    f"{safe(dia['data'])}: mín {safe(dia['temp_min'])} °C / máx {safe(dia['temp_max'])} °C • chuva {safe(dia['chuva_mm'])} mm",
                    corpo_style,
                ))

        if riscos:
            story.append(Paragraph("Riscos Prioritários", secao_style))
            for risco in riscos:
                cor = VERMELHO if risco["nivel"] == "Alto" else colors.HexColor("#d4a017")
                destaque = Paragraph(
                    f"<b>{safe(risco['titulo'])}</b> ({safe(risco['nivel'])}) — {safe(risco['descricao'])}<br/><b>Ação:</b> {safe(risco['acao'])}",
                    ParagraphStyle("Risco", parent=corpo_style, textColor=cor),
                )
                story.append(destaque)

        story.append(Paragraph("Imagens do Terreno", secao_style))
        ob = io.BytesIO(numpy_to_png_bytes(img_orig))
        mb = io.BytesIO(numpy_to_png_bytes(img_mapa))
        img_w = 85 * mm
        img_h = 55 * mm

        img_table = Table(
            [[RLImage(ob, width=img_w, height=img_h), RLImage(mb, width=img_w, height=img_h)]],
            colWidths=[img_w + 5 * mm, img_w + 5 * mm],
        )
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 4 * mm))

        story.append(Paragraph("Resultado por Grid", secao_style))
        header = ["Grid", "Tipo", "Score", "Irrigação mm/d", "Culturas recomendadas"]
        data = [header]
        for nome, d in analises.items():
            cult = ", ".join(d["culturas"][:2])
            data.append([nome, d["tipo"], f"{d['score']}/100", f"{d['irrigacao_mm']} mm", cult])

        tabela = Table(data, colWidths=[35 * mm, 28 * mm, 20 * mm, 28 * mm, 59 * mm], repeatRows=1)
        tabela.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), VERDE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f8f0"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ]))
        story.append(tabela)
        story.append(Spacer(1, 4 * mm))

        chart_labels = list(analises.keys())[:12]
        chart_values = [analises[nome]["score"] for nome in chart_labels]
        if chart_values:
            story.append(Paragraph("Gráfico de Score por Grid", secao_style))
            drawing = Drawing(450, 180)
            chart = VerticalBarChart()
            chart.x = 35
            chart.y = 30
            chart.height = 110
            chart.width = 360
            chart.data = [chart_values]
            chart.categoryAxis.categoryNames = chart_labels
            chart.categoryAxis.labels.angle = 25
            chart.categoryAxis.labels.boxAnchor = "ne"
            chart.valueAxis.valueMin = 0
            chart.valueAxis.valueMax = 100
            chart.valueAxis.valueStep = 20
            chart.bars[0].fillColor = VERDE
            drawing.add(chart)
            story.append(drawing)

        story.append(Paragraph("Recomendações por Grid", secao_style))
        if isinstance(rec_texto, str):
            for linha in rec_texto.split("\n"):
                if linha.strip():
                    story.append(Paragraph(safe(linha), corpo_style))
        else:
            for nome, d in analises.items():
                rec = recomendacao_local(d)
                cabecalho_grid = (
                    f"{safe(nome)} — {safe(d['tipo'])} | Score {d['score']}/100 | "
                    f"Irrigação {d['irrigacao_mm']} mm/dia"
                )
                story.append(Paragraph(cabecalho_grid, secao_style))

                for chave_op in ("opcao_1", "opcao_2", "opcao_3"):
                    op = rec[chave_op]
                    num = chave_op.split("_")[1]
                    story.append(Paragraph(f"Opção {num} — {safe(op['perfil'])}", opcao_titulo_style))
                    for rotulo, chave in [
                        ("Culturas", "culturas"),
                        ("Plantio", "plantio"),
                        ("Irrigação", "irrigacao"),
                        ("Ação", "acao"),
                    ]:
                        story.append(Paragraph(f"<b>{safe(rotulo)}:</b> {safe(op[chave])}", opcao_body_style))
                    story.append(Spacer(1, 2 * mm))

                story.append(Paragraph(
                    f"<font color='#888888' size='7'>NDVI: {d['ndvi']} | Textura: {d['textura']} | Umidade: {d['umidade_frac']}</font>",
                    corpo_style,
                ))
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd"), spaceAfter=4))

        doc.build(story)
        buf.seek(0)
        return buf

    except Exception as e:
        st.error(f"Erro ao gerar PDF: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# ═══════════════════════════════════════════════════════════════
# INTERFACE STREAMLIT
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title="AgroScan Drone", page_icon="🌱", layout="wide")
inject_global_css()
init_db()

# Título com efeito de digitação
typewriter("🌱 AgroScan Drone — Plataforma de Identificação e Recomendação de terreno", tag="h1",
           extra_style="font-size:1.8rem;")

if "current_user" not in st.session_state:
    render_auth_screen()
    st.stop()

current_user = st.session_state["current_user"]
amostras_modelo = contar_amostras_modelo(current_user["id"])
propriedades = listar_propriedades(current_user["id"])
propriedade_ativa = None

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configurações")
    st.success(f"Usuário conectado: {current_user['username']}")
    if st.button("Sair", use_container_width=True):
        for chave in [
            "current_user", "analises", "coords", "rows", "cols", "img_orig",
            "img_mapa", "riscos", "metricas_globais", "clima_atual",
            "resumo_executivo", "selected_property_id", "propriedade_ativa",
        ]:
            st.session_state.pop(chave, None)
        st.rerun()

    st.subheader("Análise com IA")
    usar_ia = st.toggle("Ativar análise com Claude IA", value=False)
    api_key = ""
    if usar_ia:
        st.caption(
            "Chave no formato sk-ant-...  \n"
            "Obtenha em console.anthropic.com  \n"
            "Cada análise consome ~$0.01."
        )
        api_key = st.text_input("Chave API Anthropic", type="password")

    st.divider()
    st.subheader("Fazenda / Propriedade")
    with st.form("property_form", clear_on_submit=True):
        nome_prop = st.text_input("Nome da propriedade")
        cidade_prop = st.text_input("Cidade / Região")
        lat_prop = st.number_input("Latitude", value=-15.7801, format="%.6f")
        lon_prop = st.number_input("Longitude", value=-47.9292, format="%.6f")
        area_prop = st.number_input("Área (ha)", min_value=0.0, value=10.0, step=1.0)
        obs_prop = st.text_area("Observações")
        salvar_prop = st.form_submit_button("Salvar propriedade", use_container_width=True)
    if salvar_prop:
        ok, msg = salvar_propriedade(
            current_user["id"], nome_prop, cidade_prop, lat_prop, lon_prop, area_prop, obs_prop
        )
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.warning(msg)

    propriedades = listar_propriedades(current_user["id"])
    if propriedades:
        labels_props = [f"{p['nome']} — {p['cidade'] or 'Sem região'}" for p in propriedades]
        default_index = 0
        selected_prop_id = st.session_state.get("selected_property_id")
        if selected_prop_id:
            for idx, prop in enumerate(propriedades):
                if prop["id"] == selected_prop_id:
                    default_index = idx
                    break
        escolha_prop = st.selectbox("Propriedade ativa", labels_props, index=default_index)
        propriedade_ativa = propriedades[labels_props.index(escolha_prop)]
        st.session_state["selected_property_id"] = propriedade_ativa["id"]
        st.caption(
            f"Lat: {propriedade_ativa['latitude']}, Lon: {propriedade_ativa['longitude']} • "
            f"Área: {propriedade_ativa['area_ha']} ha"
        )
    else:
        st.info("Cadastre uma propriedade para habilitar histórico, mapa e previsões.")

    st.divider()
    st.caption(f"Modelo local: {amostras_modelo} amostras reais acumuladas")
    st.caption(f"AgroScan v{APP_VERSION} — upload, câmera, histórico e campo")

# ── Painel da fazenda ─────────────────────────────────────────────
typewriter("🧭 Painel da Propriedade", tag="h3")
clima_atual = {}
historico_prop = []
if propriedade_ativa:
    historico_prop = listar_analises_salvas(current_user["id"], property_id=propriedade_ativa["id"])
    if propriedade_ativa.get("latitude") is not None and propriedade_ativa.get("longitude") is not None:
        clima_atual = obter_previsao_tempo(float(propriedade_ativa["latitude"]), float(propriedade_ativa["longitude"]))

    ultimo_score = round(float(historico_prop[0]["score_medio"]), 1) if historico_prop else 0.0
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("🏡 Propriedade", propriedade_ativa["nome"])
    p2.metric("📚 Análises salvas", len(historico_prop))
    p3.metric("🧠 Amostras reais", amostras_modelo)
    p4.metric("📈 Último score", f"{ultimo_score}/100" if historico_prop else "N/D")

    d1, d2 = st.columns([1.2, 1])
    with d1:
        st.markdown(
            f"**Localização:** {propriedade_ativa['cidade'] or 'Não informada'}  \n"
            f"**Área:** {propriedade_ativa['area_ha']} ha  \n"
            f"**Observações:** {propriedade_ativa['observacoes'] or 'Sem observações'}"
        )
        if propriedade_ativa.get("latitude") is not None and propriedade_ativa.get("longitude") is not None:
            mapa_local = pd.DataFrame({
                "lat": [float(propriedade_ativa["latitude"])],
                "lon": [float(propriedade_ativa["longitude"])],
            })
            st.map(mapa_local)

    with d2:
        st.markdown("**Previsão climática integrada (Open-Meteo)**")
        if clima_atual and not clima_atual.get("erro"):
            c1, c2 = st.columns(2)
            c1.metric("🌡️ Temp.", f"{clima_atual.get('temperatura', 'N/D')} °C")
            c2.metric("💧 Umidade", f"{clima_atual.get('umidade', 'N/D')}%")
            c3, c4 = st.columns(2)
            c3.metric("🌧️ Chuva", f"{clima_atual.get('precipitacao', 'N/D')} mm")
            c4.metric("💨 Vento", f"{clima_atual.get('vento', 'N/D')} km/h")
            if clima_atual.get("previsao"):
                clima_df = pd.DataFrame([
                    {
                        "Data": dia["data"],
                        "Mín (°C)": dia["temp_min"],
                        "Máx (°C)": dia["temp_max"],
                        "Chuva (mm)": dia["chuva_mm"],
                    }
                    for dia in clima_atual["previsao"]
                ])
                st.dataframe(clima_df, use_container_width=True, hide_index=True)
        else:
            st.info("Previsão indisponível no momento para esta localização.")
else:
    st.info("Cadastre uma propriedade para ativar mapa geográfico, clima, histórico e painel por fazenda.")

# ── Modelo local treinável ───────────────────────────────────────
typewriter("🧠 Modelo Local com Imagens Reais", tag="h3")
if amostras_modelo >= 6:
    st.success(
        f"O classificador híbrido já está ativo com {amostras_modelo} amostras reais salvas no banco local."
    )
else:
    st.info(
        f"O sistema já acumula {amostras_modelo} amostras. Conforme novas análises forem salvas, o modelo local ganhará precisão."
    )

# ── Entrada da imagem ─────────────────────────────────────────────
typewriter("📷 Entrada da Imagem", tag="h3")
aba_upload, aba_camera = st.tabs(["Upload de arquivo", "Usar câmera"])

with aba_upload:
    arquivo_upload = st.file_uploader(
        "Envie a foto do terreno",
        type=["jpg", "jpeg", "png"],
    )

with aba_camera:
    foto_camera = st.camera_input("Tire uma foto do terreno")
    st.caption("Permita o acesso à câmera no navegador para capturar a imagem.")

imagem_entrada = None
origem_imagem = ""
if foto_camera is not None:
    imagem_entrada = foto_camera
    origem_imagem = "Câmera"
elif arquivo_upload is not None:
    imagem_entrada = arquivo_upload
    origem_imagem = "Upload"
else:
    st.info("Envie uma imagem ou tire uma foto pela câmera para iniciar a análise.")

if imagem_entrada:
    img_bytes = imagem_entrada.getvalue()
    raw = np.frombuffer(img_bytes, dtype=np.uint8)
    img_full = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    if img_full is None:
        st.error("Não foi possível ler a imagem enviada.")
    else:
        img = redimensionar(img_full, MAX_W, MAX_H)
        H, W = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        DISPLAY_W = min(W, 380)
        img_rgb_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_b64_upload = img_rgb_to_base64(img_rgb_display)
        st.markdown(
            f"<div style='display:flex;flex-direction:column;align-items:center;margin:0 0 12px'>"
            f"<img src='data:image/jpeg;base64,{img_b64_upload}' style='width:{DISPLAY_W}px;border-radius:8px;'/>"
            f"<span style='font-size:12px;color:#888;margin-top:4px'>Imagem carregada via {origem_imagem} ({W} x {H} px)</span></div>",
            unsafe_allow_html=True,
        )

        b_f, g_f, r_f = cv2.split(img.astype("float32"))
        ndvi_global = float(np.mean((g_f - r_f) / (g_f + r_f + 1e-5)))
        total = H * W
        p_verde = np.sum(cv2.inRange(hsv, _VERDE["lo"], _VERDE["hi"]) > 0) / total
        p_amarelo = np.sum(cv2.inRange(hsv, _AMARELO["lo"], _AMARELO["hi"]) > 0) / total
        p_cinza = np.sum(cv2.inRange(hsv, _CINZA["lo"], _CINZA["hi"]) > 0) / total
        p_azul = np.sum(cv2.inRange(hsv, _AZUL["lo"], _AZUL["hi"]) > 0) / total

        if p_verde > 0.4:
            resultado_geral = "Terreno predominantemente fértil"
        elif p_amarelo > 0.3:
            resultado_geral = "Terreno predominantemente seco"
        elif p_cinza > 0.3:
            resultado_geral = "Terreno predominantemente rochoso"
        elif p_azul > 0.2:
            resultado_geral = "Terreno predominantemente úmido"
        else:
            resultado_geral = "Terreno predominantemente arenoso"

        metricas_globais = {
            "ndvi_global": ndvi_global,
            "p_verde": float(p_verde),
            "p_amarelo": float(p_amarelo),
            "p_cinza": float(p_cinza),
            "p_azul": float(p_azul),
        }

        typewriter("📊 Resultado Geral", tag="h3")
        st.success(resultado_geral)
        st.metric("NDVI médio (saúde da vegetação)", f"{round(ndvi_global * 100, 1)}%")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌿 Fértil", f"{round(p_verde * 100, 1)}%")
        c2.metric("☀️ Seco", f"{round(p_amarelo * 100, 1)}%")
        c3.metric("🪨 Rochoso", f"{round(p_cinza * 100, 1)}%")
        c4.metric("💧 Úmido", f"{round(p_azul * 100, 1)}%")

        mapa = gerar_mapa_terreno(img)

        typewriter("🗺️ Mapa de Identificação NDVI", tag="h3")
        if "cmp_pct" not in st.session_state:
            st.session_state.cmp_pct = 50

        bc1, bc2, bc3 = st.columns([1, 1, 1])
        with bc1:
            if st.button("Reset 50%", use_container_width=True):
                st.session_state.cmp_pct = 50
        with bc2:
            if st.button("Original", use_container_width=True):
                st.session_state.cmp_pct = 0
        with bc3:
            if st.button("Mapa completo", use_container_width=True):
                st.session_state.cmp_pct = 100

        slider_comparacao(img, mapa, st.session_state.cmp_pct)

        legenda_html = (
            "<div style='display:flex;flex-direction:column;gap:6px;margin:8px 0 16px'>"
            "<div style='display:flex;align-items:center;gap:8px'>"
            "<div style='width:120px;height:14px;border-radius:3px;background:linear-gradient(to right,#30123b,#4887f7,#1de4b1,#a2fc3c,#fba318,#7a0403);'></div>"
            "<span style='font-size:12px'>Escala NDVI: Baixo → Alto</span></div>"
            "<div style='display:flex;gap:16px;flex-wrap:wrap;font-size:12px'>"
            "<span>🟣 Sem vegetação/Rochoso</span>"
            "<span>🔵 Solo exposto/Seco</span>"
            "<span>🟢 Vegetação moderada</span>"
            "<span>🟡 Vegetação densa</span>"
            "<span>🔴 Stress hídrico</span>"
            "</div></div>"
        )
        st.markdown(legenda_html, unsafe_allow_html=True)

        typewriter("🔲 Análise por Regiões", tag="h3")
        num_grids = st.radio("Número de grids:", [4, 8, 16, 32], horizontal=True)
        grid_map = {4: (2, 2), 8: (2, 4), 16: (4, 4), 32: (4, 8)}
        rows, cols = grid_map[num_grids]

        coords: dict = {}
        for i in range(rows):
            for j in range(cols):
                nome = f"Grid {i * cols + j + 1}"
                ys_, ye_ = i * (H // rows), (i + 1) * (H // rows)
                xs_, xe_ = j * (W // cols), (j + 1) * (W // cols)
                coords[nome] = (xs_, xe_, ys_, ye_)

        if st.button("Iniciar análise de grids", type="primary", use_container_width=True):
            analises: dict = {}
            for nome, (xs_, xe_, ys_, ye_) in coords.items():
                analises[nome] = analisar_regiao(img[ys_:ye_, xs_:xe_])

            score_medio = round(sum(d["score"] for d in analises.values()) / len(analises), 1)
            metricas_globais.update({
                "score_medio": score_medio,
                "total_grids": len(analises),
            })
            riscos = detectar_riscos(analises, metricas_globais, clima_atual)
            resumo_executivo = gerar_resumo_executivo(resultado_geral, metricas_globais, riscos)

            analysis_id = None
            if propriedade_ativa:
                analysis_id = salvar_analise_historico(
                    current_user["id"],
                    propriedade_ativa["id"],
                    origem_imagem,
                    img,
                    mapa,
                    resultado_geral,
                    metricas_globais,
                    analises,
                    clima_atual,
                    riscos,
                )
                registrar_amostras_modelo(current_user["id"], propriedade_ativa["id"], analises)

            st.session_state.update({
                "analises": analises,
                "coords": coords,
                "rows": rows,
                "cols": cols,
                "img_orig": img,
                "img_mapa": mapa,
                "riscos": riscos,
                "metricas_globais": metricas_globais,
                "clima_atual": clima_atual,
                "origem_imagem": origem_imagem,
                "analysis_id": analysis_id,
                "resumo_executivo": resumo_executivo,
                "propriedade_ativa": propriedade_ativa,
                "resultado_geral": resultado_geral,
            })

            if analysis_id:
                st.success(f"Análise #{analysis_id} salva no histórico da propriedade.")
            else:
                st.info("Análise gerada. Cadastre uma propriedade para salvar histórico e comparação.")

# ── Exibir resultados salvos na sessão ───────────────────────────
if "analises" in st.session_state:
    analises = st.session_state["analises"]
    _coords = st.session_state["coords"]
    _rows = st.session_state["rows"]
    _cols = st.session_state["cols"]
    _img = st.session_state["img_orig"]
    riscos = st.session_state.get("riscos", [])
    metricas_salvas = st.session_state.get("metricas_globais", {})

    typewriter("🚁 Drone em operação", tag="h3")
    html_anim = drone_animation_component(_img, analises, _coords, _rows, _cols)
    iframe_h = int(_img.shape[0]) + 40
    st.components.v1.html(html_anim, height=iframe_h)

    st.markdown("<h3 class='fade-slide'>🧾 Resumo Executivo</h3>", unsafe_allow_html=True)
    resumo_cards = [
        f"<div style='padding:8px 12px;border-left:3px solid #22a622;background:#162716;border-radius:6px;margin:6px 0'>{linha}</div>"
        for linha in st.session_state.get("resumo_executivo", [])
    ]
    if resumo_cards:
        st.markdown(fade_slide_wrap(resumo_cards), unsafe_allow_html=True)

    st.markdown("<h3 class='fade-slide'>📈 Score Agrícola por Grid</h3>", unsafe_allow_html=True)
    score_medio = round(sum(d["score"] for d in analises.values()) / len(analises), 1)
    st.metric("Score médio do terreno", f"{score_medio}/100")

    barras = []
    for nome, d in list(analises.items())[:16]:
        cor = "#22a622" if d["score"] >= 60 else "#d4a017" if d["score"] >= 35 else "#b03030"
        barras.append(
            f"<div style='display:flex;align-items:center;gap:10px;margin:3px 0'>"
            f"<span style='min-width:70px;font-size:13px'>{nome}</span>"
            f"<div style='flex:1;background:#333;border-radius:4px;height:14px'>"
            f"<div style='width:{d['score']}%;background:{cor};border-radius:4px;height:14px'></div></div>"
            f"<span style='font-size:12px;min-width:140px'>{d['tipo']} {d['score']}/100</span>"
            f"</div>"
        )
    st.markdown(fade_slide_wrap(barras), unsafe_allow_html=True)

    st.markdown("<h3 class='fade-slide'>⚠️ Riscos Locais</h3>", unsafe_allow_html=True)
    if riscos:
        cards_risco = []
        for risco in riscos:
            borda = "#b03030" if risco["nivel"] == "Alto" else "#d4a017"
            cards_risco.append(
                f"<div style='border-left:4px solid {borda};padding:8px 12px;margin:8px 0;background:#231919;border-radius:0 6px 6px 0'>"
                f"<strong>{risco['titulo']} — {risco['nivel']}</strong><br/>{risco['descricao']}<br/>"
                f"<strong>Ação:</strong> {risco['acao']}</div>"
            )
        st.markdown(fade_slide_wrap(cards_risco), unsafe_allow_html=True)
    else:
        st.success("Nenhum risco crítico foi identificado na análise automática atual.")

    st.markdown("<h3 class='fade-slide'>🌾 Recomendações por Grid</h3>", unsafe_allow_html=True)
    if usar_ia and api_key:
        with st.spinner("Consultando Claude IA..."):
            texto_ia = analisar_com_ia(_img, analises, api_key)
        st.session_state["rec_ia"] = texto_ia
        st.markdown(f"<div class='fade-slide'>{texto_ia}</div>", unsafe_allow_html=True)
    else:
        for nome, d in analises.items():
            rec = recomendacao_local(d)
            with st.expander(
                f"{nome} — {d['tipo']} | Score: {d['score']}/100 | Irrigação: {d['irrigacao_mm']} mm/dia | {d.get('origem_modelo', 'Heurístico')}",
                expanded=False,
            ):
                opcoes_html = []
                for chave_op in ("opcao_1", "opcao_2", "opcao_3"):
                    op = rec[chave_op]
                    num = chave_op.split("_")[1]
                    opcoes_html.append(
                        f"<div style='border-left:3px solid #22a622;padding:6px 12px;margin:8px 0;border-radius:0 6px 6px 0;background:#1a2e1a'>"
                        f"<strong>Opção {num} — {op['perfil']}</strong><br/>"
                        f"🌱 <strong>Culturas:</strong> {op['culturas']}<br/>"
                        f"🌿 <strong>Plantio:</strong> {op['plantio']}<br/>"
                        f"💧 <strong>Irrigação:</strong> {op['irrigacao']}<br/>"
                        f"⚙️ <strong>Ação:</strong> {op['acao']}"
                        f"</div>"
                    )
                st.markdown(fade_slide_wrap(opcoes_html), unsafe_allow_html=True)
                st.markdown(
                    f"<small style='color:#666'>NDVI: {d['ndvi']} | Textura: {d['textura']} | Umidade: {d['umidade_frac']} | Modelo: {d.get('origem_modelo', 'Heurístico')}</small>",
                    unsafe_allow_html=True,
                )
        st.session_state.pop("rec_ia", None)

    st.divider()
    st.markdown("<h3 class='fade-slide'>📄 Exportar Relatório</h3>", unsafe_allow_html=True)
    if st.button("Gerar PDF", use_container_width=True):
        with st.spinner("Gerando relatório executivo com reportlab..."):
            rec_cont = st.session_state.get("rec_ia") or analises
            pdf_buf = gerar_pdf(
                st.session_state["img_orig"],
                st.session_state["img_mapa"],
                analises,
                rec_cont,
                meta={
                    "usuario": current_user["username"],
                    "origem_imagem": st.session_state.get("origem_imagem", "N/D"),
                    "propriedade": st.session_state.get("propriedade_ativa") or propriedade_ativa or {},
                    "clima": st.session_state.get("clima_atual", {}),
                    "riscos": st.session_state.get("riscos", []),
                    "resumo_executivo": st.session_state.get("resumo_executivo", []),
                },
            )
        if pdf_buf:
            st.download_button(
                "⬇️ Baixar relatório PDF",
                data=pdf_buf,
                file_name=f"agroscan_{time.strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

# ── Histórico e comparação temporal ──────────────────────────────
st.divider()
typewriter("🗂️ Histórico e Comparação Temporal", tag="h3")
if propriedade_ativa:
    historico_prop = listar_analises_salvas(current_user["id"], property_id=propriedade_ativa["id"])
    if historico_prop:
        hist_df = pd.DataFrame([
            {
                "ID": item["id"],
                "Data": item["created_at"].replace("T", " ")[:16],
                "Fonte": item["image_source"],
                "Resultado": item["resultado_geral"],
                "NDVI": round(float(item["ndvi_global"]), 3),
                "Score": round(float(item["score_medio"]), 1),
            }
            for item in historico_prop
        ])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        if len(historico_prop) >= 2:
            labels_hist = [
                f"#{item['id']} • {item['created_at'].replace('T', ' ')[:16]} • Score {float(item['score_medio']):.1f}/100"
                for item in historico_prop
            ]
            h1, h2 = st.columns(2)
            with h1:
                escolha_a = st.selectbox("Análise base", labels_hist, index=min(1, len(labels_hist) - 1))
            with h2:
                escolha_b = st.selectbox("Análise comparada", labels_hist, index=0)

            registro_a = historico_prop[labels_hist.index(escolha_a)]
            registro_b = historico_prop[labels_hist.index(escolha_b)]

            if registro_a["id"] != registro_b["id"]:
                analise_a = obter_analise_salva(registro_a["id"])
                analise_b = obter_analise_salva(registro_b["id"])
                if analise_a and analise_b:
                    delta_ndvi = float(analise_b["ndvi_global"]) - float(analise_a["ndvi_global"])
                    delta_score = float(analise_b["score_medio"]) - float(analise_a["score_medio"])
                    x1, x2, x3 = st.columns(3)
                    x1.metric("Δ NDVI", f"{delta_ndvi:+.3f}")
                    x2.metric("Δ Score", f"{delta_score:+.1f}")
                    x3.metric("Grids analisados", int(analise_b.get("total_grids") or 0))

                    ic1, ic2 = st.columns(2)
                    with ic1:
                        st.image(analise_a["image_path"], caption=f"Base #{analise_a['id']}", use_container_width=True)
                    with ic2:
                        st.image(analise_b["image_path"], caption=f"Comparada #{analise_b['id']}", use_container_width=True)

                    riscos_a = ", ".join(r["titulo"] for r in analise_a.get("risks", [])) or "Sem riscos críticos"
                    riscos_b = ", ".join(r["titulo"] for r in analise_b.get("risks", [])) or "Sem riscos críticos"
                    st.markdown(f"**Riscos da análise base:** {riscos_a}")
                    st.markdown(f"**Riscos da análise comparada:** {riscos_b}")
        else:
            st.info("Salve pelo menos 2 análises desta propriedade para comparar evolução no tempo.")
    else:
        st.info("Ainda não há análises salvas para esta propriedade.")
else:
    st.info("Cadastre uma propriedade para ativar o histórico, o painel por fazenda e a comparação temporal.")
