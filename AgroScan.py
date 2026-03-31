# python -m streamlit run AgroScan.py
# Dependências: pip install streamlit opencv-python-headless numpy pandas requests anthropic reportlab

import streamlit as st
import cv2
import numpy as np
import base64
import json
import io
import os
import re
import sys
import time
import sqlite3
import hashlib
import importlib
import subprocess
import unicodedata
from datetime import datetime

import pandas as pd
import pydeck as pdk
import requests

# ═══════════════════════════════════════════════════════════════
# CONSTANTES GLOBAIS
# ═══════════════════════════════════════════════════════════════

MAX_W = 720
MAX_H = 480
APP_VERSION = "5.1"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
DB_PATH = os.path.join(DATA_DIR, "agroscan.db")
IBGE_MUNICIPIOS_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
IBGE_INDICADORES_URL = "https://apisidra.ibge.gov.br/values/t/4714/n6/{municipio_id}/v/93,6318,614/p/2022?formato=json"
RESULT_SESSION_KEYS = [
    "analises", "coords", "rows", "cols", "img_orig", "img_mapa",
    "riscos", "metricas_globais", "clima_atual", "resumo_executivo",
    "origem_imagem", "analysis_id", "propriedade_ativa",
    "resultado_geral", "rec_ia", "localized_risks", "risk_overlay",
    "current_image_source", "current_image_loaded_at", "new_image_reset_notice",
    "alertas_inteligentes", "plano_acao", "risk_geo_points", "risk_level_filter",
]

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
    theme_mode = st.session_state.get("theme_mode", "Escuro")
    palette = {
        "Escuro": {
            "bg": "radial-gradient(circle at top left, rgba(34,166,34,0.12), transparent 0 24%), linear-gradient(180deg, #07110d 0%, #0a1712 100%)",
            "text": "#eef9ef",
            "sub": "#b6d3bc",
            "panel": "linear-gradient(180deg, rgba(17,30,24,0.96) 0%, rgba(12,22,18,0.96) 100%)",
            "panel_soft": "rgba(15, 28, 22, 0.90)",
            "sidebar": "linear-gradient(180deg, rgba(9,20,16,0.98) 0%, rgba(16,31,24,0.98) 100%)",
            "border": "rgba(79, 160, 97, 0.22)",
            "shadow": "0 8px 18px rgba(0,0,0,.20)",
            "hover": "rgba(24, 44, 33, 0.98)",
            "chip": "rgba(34,166,34,0.12)",
        },
        "Claro": {
            "bg": "radial-gradient(circle at top left, rgba(34,166,34,0.10), transparent 0 24%), linear-gradient(180deg, #f7fbf8 0%, #edf5ef 100%)",
            "text": "#132317",
            "sub": "#47614e",
            "panel": "linear-gradient(180deg, rgba(255,255,255,0.99) 0%, rgba(244,248,245,0.98) 100%)",
            "panel_soft": "rgba(248, 251, 248, 0.98)",
            "sidebar": "linear-gradient(180deg, rgba(248,250,248,0.99) 0%, rgba(238,244,239,0.99) 100%)",
            "border": "rgba(57, 122, 72, 0.22)",
            "shadow": "0 8px 18px rgba(32, 55, 40, 0.10)",
            "hover": "rgba(233, 245, 236, 0.98)",
            "chip": "rgba(34,166,34,0.10)",
        },
    }
    p = palette["Claro" if theme_mode == "Claro" else "Escuro"]
    st.markdown(
        f"""
<style>
:root {{
  --ag-bg: {p['bg']};
  --ag-text: {p['text']};
  --ag-sub: {p['sub']};
  --ag-panel: {p['panel']};
  --ag-panel-soft: {p['panel_soft']};
  --ag-sidebar: {p['sidebar']};
  --ag-border: {p['border']};
  --ag-shadow: {p['shadow']};
  --ag-hover: {p['hover']};
  --ag-chip: {p['chip']};
  --ag-success: #22a622;
  --ag-info: #1f8ef1;
  --ag-warn: #ffb020;
  --ag-danger: #ff5c5c;
}}

html, body, [data-testid='stAppViewContainer'], [data-testid='stHeader'] {{
  color: var(--ag-text) !important;
  background: var(--ag-bg) !important;
}}
.stApp {{
  background: var(--ag-bg) !important;
  color: var(--ag-text) !important;
}}
.block-container {{
  padding-top: 1.1rem;
}}
section[data-testid='stSidebar'] {{
  background: var(--ag-sidebar) !important;
  border-right: 1px solid var(--ag-border) !important;
}}
h1, h2, h3, h4, h5, h6,
p, li, label, small {{
  color: var(--ag-text);
}}
.agroscan-subtitle, .agroscan-muted, [data-testid='stCaptionContainer'] {{
  color: var(--ag-sub) !important;
}}
div[data-testid='stMetric'],
div[data-baseweb='tab'],
div[data-testid='stExpander'],
div[data-testid='stForm'] {{
  background: var(--ag-panel) !important;
  border: 1px solid var(--ag-border) !important;
  box-shadow: var(--ag-shadow) !important;
  color: var(--ag-text) !important;
  border-radius: 14px;
}}
div[data-testid='stMetric'] {{
  padding: 10px 12px;
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}}
div[data-testid='stMetric']:hover {{
  transform: translateY(-2px);
  border-color: rgba(34,166,34,0.32) !important;
}}
div[data-baseweb='tab-list'] {{
  gap: 8px;
}}
div[data-baseweb='tab'] {{
  border-radius: 999px;
  padding: 7px 14px;
  transition: transform .18s ease, background-color .18s ease, border-color .18s ease;
}}
div[data-baseweb='tab']:hover {{
  transform: translateY(-1px);
  background: var(--ag-hover) !important;
}}
div[data-baseweb='tab'][aria-selected='true'] {{
  background: linear-gradient(90deg, rgba(34,166,34,0.24), rgba(60,120,255,0.16)) !important;
  color: var(--ag-text) !important;
  border-color: rgba(34,166,34,0.28) !important;
}}
.stButton > button,
.stDownloadButton > button {{
  border-radius: 12px;
  border: 1px solid var(--ag-border) !important;
  background: var(--ag-panel) !important;
  color: var(--ag-text) !important;
  transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
}}
.stButton > button:hover,
.stDownloadButton > button:hover {{
  transform: translateY(-1px);
  box-shadow: 0 10px 22px rgba(34,166,34,0.14);
  filter: brightness(1.02);
}}
.stButton > button:focus,
.stDownloadButton > button:focus {{
  border-color: rgba(34,166,34,0.38) !important;
  box-shadow: 0 0 0 0.16rem rgba(34,166,34,0.14) !important;
}}
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
div[data-baseweb='select'] > div,
div[data-baseweb='base-input'] input {{
  background: var(--ag-panel-soft) !important;
  color: var(--ag-text) !important;
  border: 1px solid var(--ag-border) !important;
}}
.agroscan-tour {{
  background: var(--ag-panel);
  border: 1px solid var(--ag-border);
  border-radius: 16px;
  padding: 14px 16px;
  margin: 8px 0 14px;
  box-shadow: var(--ag-shadow);
}}
.agroscan-tour-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
  margin-top: 10px;
}}
.agroscan-tour-step,
.agroscan-shortcuts a,
.agroscan-notice,
.agroscan-card {{
  background: var(--ag-panel);
  color: var(--ag-text) !important;
  border: 1px solid var(--ag-border);
  border-radius: 12px;
  box-shadow: var(--ag-shadow);
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}}
.agroscan-tour-step {{
  padding: 10px 12px;
}}
.agroscan-tour-step:hover,
.agroscan-shortcuts a:hover,
.agroscan-notice:hover,
.agroscan-card:hover {{
  transform: translateY(-2px);
  border-color: rgba(34,166,34,0.34);
}}
.agroscan-shortcuts {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
  margin: 6px 0 14px;
}}
.agroscan-shortcuts a {{
  text-decoration: none !important;
  padding: 10px 12px;
  display: block;
}}
.agroscan-shortcuts span {{
  display: block;
  font-size: 12px;
  color: var(--ag-sub) !important;
  margin-top: 4px;
}}
.agroscan-notice {{
  border-left: 5px solid var(--ag-info);
  padding: 10px 12px;
  margin: 8px 0;
}}
.agroscan-notice.high {{ border-left-color: var(--ag-danger); }}
.agroscan-notice.medium {{ border-left-color: var(--ag-warn); }}
.agroscan-notice.low {{ border-left-color: var(--ag-success); }}
.agroscan-card {{
  padding: 10px 12px;
  margin: 8px 0;
}}
.agroscan-card.accent-success {{ border-left: 4px solid var(--ag-success); }}
.agroscan-card.accent-info {{ border-left: 4px solid var(--ag-info); }}
.agroscan-card.accent-warn {{ border-left: 4px solid var(--ag-warn); }}
.agroscan-card.accent-danger {{ border-left: 4px solid var(--ag-danger); }}
.agroscan-chip {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--ag-chip);
  color: var(--ag-text) !important;
  font-size: 12px;
  border: 1px solid var(--ag-border);
}}
.agroscan-stepper {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px;
  margin: 8px 0 14px;
}}
.agroscan-step {{
  background: var(--ag-panel);
  border: 1px solid var(--ag-border);
  border-radius: 14px;
  padding: 10px 12px;
}}
.agroscan-step.done {{
  border-color: rgba(59, 201, 103, 0.35);
}}
.agroscan-step-title {{
  font-weight: 700;
  color: var(--ag-text);
}}
.agroscan-step-status {{
  font-size: 12px;
  color: var(--ag-sub);
}}
.agroscan-subtitle {{
  margin: -2px 0 10px;
  font-size: 0.95rem;
}}
.agroscan-score-row {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 6px 0;
}}
.agroscan-score-label {{
  min-width: 72px;
  font-size: 13px;
}}
.agroscan-score-track {{
  flex: 1;
  background: rgba(127, 145, 132, 0.22);
  border-radius: 999px;
  height: 14px;
  overflow: hidden;
}}
.agroscan-loading-card {{
  background: var(--ag-panel);
  border: 1px solid var(--ag-border);
  border-radius: 14px;
  padding: 12px 14px;
  margin: 8px 0 12px;
  box-shadow: var(--ag-shadow);
}}
.agroscan-loading-track {{
  width: 100%;
  height: 8px;
  border-radius: 999px;
  margin-top: 10px;
  overflow: hidden;
  background: rgba(127, 145, 132, 0.18);
}}
.agroscan-loading-fill {{
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #22a622 0%, #1f8ef1 100%);
}}
@keyframes agPulse {{
  0%, 100% {{ opacity: .35; transform: scale(.95); }}
  50% {{ opacity: 1; transform: scale(1.08); }}
}}
.agroscan-loading-dot {{
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #22a622;
  animation: agPulse 1.1s infinite;
}}
.agroscan-floating-legend {{
  position: sticky;
  top: 72px;
  z-index: 9;
  background: var(--ag-panel);
  color: var(--ag-text);
  border: 1px solid var(--ag-border);
  border-radius: 14px;
  padding: 10px 12px;
  margin: 10px 0 14px;
  box-shadow: var(--ag-shadow);
}}
.agroscan-legend-row {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 6px;
  font-size: 12px;
  color: var(--ag-text);
}}
.agroscan-legend-dot {{
  width: 10px;
  height: 10px;
  border-radius: 50%;
  display: inline-block;
}}
.agroscan-legend-dot.high {{ background: #ff5c5c; }}
.agroscan-legend-dot.medium {{ background: #ffb020; }}
.agroscan-legend-dot.low {{ background: #39d98a; }}
.agroscan-legend-dot.ndvi {{ background: linear-gradient(90deg, #4887f7 0%, #1de4b1 50%, #fba318 100%); width: 16px; border-radius: 999px; }}
.agroscan-executive-banner {{
  background: var(--ag-panel);
  border: 1px solid var(--ag-border);
  border-left: 5px solid var(--ag-info);
  border-radius: 14px;
  padding: 12px 14px;
  margin: 8px 0 14px;
  box-shadow: var(--ag-shadow);
}}

/* ── Animação de digitação (textos do topo) ──────────────────── */
@keyframes typing {{
  from {{ width: 0; }}
  to   {{ width: 100%; }}
}}
@keyframes blink {{
  50% {{ border-color: transparent; }}
}}
.typewriter {{
  display: inline-block;
  overflow: hidden;
  white-space: nowrap;
  border-right: 2px solid #22a622;
  animation: typing 1.8s steps(40, end) forwards, blink .75s step-end 3;
  max-width: 100%;
}}

/* ── Fade-in esquerda→direita curtíssimo (textos pós-drone) ─── */
@keyframes fadeSlideIn {{
  from {{ opacity: 0; clip-path: inset(0 100% 0 0); }}
  to {{ opacity: 1; clip-path: inset(0 0% 0 0); }}
}}
.fade-slide {{
  animation: fadeSlideIn 0.45s ease-out forwards;
  opacity: 0;
}}
.fade-slide:nth-child(1)  {{ animation-delay: 0.00s; }}
.fade-slide:nth-child(2)  {{ animation-delay: 0.07s; }}
.fade-slide:nth-child(3)  {{ animation-delay: 0.14s; }}
.fade-slide:nth-child(4)  {{ animation-delay: 0.21s; }}
.fade-slide:nth-child(5)  {{ animation-delay: 0.28s; }}
.fade-slide:nth-child(6)  {{ animation-delay: 0.35s; }}
.fade-slide:nth-child(7)  {{ animation-delay: 0.42s; }}
.fade-slide:nth-child(8)  {{ animation-delay: 0.49s; }}
.fade-slide:nth-child(9)  {{ animation-delay: 0.56s; }}
.fade-slide:nth-child(10) {{ animation-delay: 0.63s; }}
.fade-slide:nth-child(11) {{ animation-delay: 0.70s; }}
.fade-slide:nth-child(12) {{ animation-delay: 0.77s; }}
.fade-slide:nth-child(13) {{ animation-delay: 0.84s; }}
.fade-slide:nth-child(14) {{ animation-delay: 0.91s; }}
.fade-slide:nth-child(15) {{ animation-delay: 0.98s; }}
.fade-slide:nth-child(16) {{ animation-delay: 1.05s; }}
.fade-slide:nth-child(17) {{ animation-delay: 1.12s; }}
.fade-slide:nth-child(18) {{ animation-delay: 1.19s; }}
.fade-slide:nth-child(19) {{ animation-delay: 1.26s; }}
.fade-slide:nth-child(20) {{ animation-delay: 1.33s; }}

@media (max-width: 768px) {{
  .block-container {{ padding: 1rem 0.75rem 4rem; }}
  h1 {{ font-size: 1.35rem !important; }}
  h3 {{ font-size: 1rem !important; }}
  div[data-baseweb='tab-list'] {{ flex-wrap: wrap; gap: 4px; }}
  div[data-testid='stHorizontalBlock'] {{ gap: 0.5rem; }}
  .stButton > button, .stDownloadButton > button {{ width: 100%; }}
  .agroscan-tour, .agroscan-notice, .agroscan-floating-legend {{ padding: 10px 12px; position: static; }}
  .agroscan-shortcuts {{ grid-template-columns: 1fr 1fr; }}
}}
</style>
        """,
        unsafe_allow_html=True,
    )


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


def render_workflow_stepper(has_property: bool, has_image: bool, has_analysis: bool):
    etapas = [
        ("1", "Propriedade (opcional)", has_property),
        ("2", "Clima + IBGE", has_property),
        ("3", "Imagem", has_image),
        ("4", "Análise de grids", has_analysis),
        ("5", "Plano / PDF", has_analysis),
    ]
    blocos = []
    for numero, titulo, concluido in etapas:
        if "(opcional)" in titulo and not concluido:
            status = "➖ Opcional"
        else:
            status = "✅ Concluído" if concluido else "⏳ Aguardando"
        classe = "agroscan-step done" if concluido else "agroscan-step"
        blocos.append(
            f"<div class='{classe}'>"
            f"<div class='agroscan-step-title'>{numero}. {titulo}</div>"
            f"<div class='agroscan-step-status'>{status}</div>"
            f"</div>"
        )
    st.markdown(f"<div class='agroscan-stepper'>{''.join(blocos)}</div>", unsafe_allow_html=True)


def render_guided_tour():
    if not st.session_state.get("show_guided_tour", False):
        return

    st.markdown(
        """
        <div class='agroscan-tour'>
            <h4 style='margin:0 0 6px'>👋 Tour guiado do AgroScan</h4>
            <div style='font-size:0.95rem'>Na primeira abertura, siga este fluxo rápido para chegar no diagnóstico com menos cliques.</div>
            <div class='agroscan-tour-grid'>
                <div class='agroscan-tour-step'><strong>1. Escolha o modo</strong><br/><small>Use propriedade cadastrada ou análise avulsa na hora.</small></div>
                <div class='agroscan-tour-step'><strong>2. Envie a imagem</strong><br/><small>Faça upload ou capture pela câmera do dispositivo.</small></div>
                <div class='agroscan-tour-step'><strong>3. Rode os grids</strong><br/><small>Inicie a leitura e veja alertas, mapa e plano.</small></div>
                <div class='agroscan-tour-step'><strong>4. Compare e exporte</strong><br/><small>Use o slider premium e gere o PDF final.</small></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("✅ Começar agora", key="tour_start_btn", use_container_width=True):
            st.session_state["show_guided_tour"] = False
            st.session_state["tour_seen"] = True
            st.rerun()
    with c2:
        if st.button("🙈 Fechar tour", key="tour_close_btn", use_container_width=True):
            st.session_state["show_guided_tour"] = False
            st.session_state["tour_seen"] = True
            st.rerun()
    with c3:
        st.caption("Você pode reabrir o tour pela sidebar sempre que quiser revisar o fluxo.")


def render_mobile_shortcuts(has_image: bool, has_analysis: bool):
    status_imagem = "Imagem pronta" if has_image else "Enviar imagem"
    status_resultado = "Resultados ativos" if has_analysis else "Aguardando leitura"
    st.markdown(
        f"""
        <div class='agroscan-shortcuts'>
            <a href='#secao-propriedade'><strong>🏡 Propriedade</strong><span>Escolher fazenda ou modo avulso</span></a>
            <a href='#secao-upload'><strong>📷 Captura</strong><span>{status_imagem}</span></a>
            <a href='#secao-resultados'><strong>📊 Resultados</strong><span>{status_resultado}</span></a>
            <a href='#secao-historico'><strong>🗂️ Histórico</strong><span>Comparar antes x depois</span></a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_priority_notice(title: str, message: str, level: str = "info"):
    css_class = {
        "high": "high",
        "medium": "medium",
        "low": "low",
        "info": "",
    }.get(level, "")
    icon = {
        "high": "🔴",
        "medium": "🟠",
        "low": "🟢",
        "info": "🔵",
    }.get(level, "🔵")
    st.markdown(
        f"<div class='agroscan-notice {css_class}'><strong>{icon} {title}</strong><br/>{message}</div>",
        unsafe_allow_html=True,
    )


def render_analysis_progress(stage_box, progress_bar, percent: int, title: str, detail: str):
    percent = max(0, min(100, int(percent)))
    progress_bar.progress(percent)
    stage_box.markdown(
        f"""
        <div class='agroscan-loading-card'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start;gap:12px;'>
                <div>
                    <strong>{title}</strong><br/>
                    <span class='agroscan-muted'>{detail}</span>
                </div>
                <div class='agroscan-loading-dot'></div>
            </div>
            <div class='agroscan-loading-track'>
                <div class='agroscan-loading-fill' style='width:{percent}%'></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_floating_grid_legend():
    st.markdown(
        """
        <div class='agroscan-floating-legend'>
            <strong>🧭 Legenda fixa do mapa e dos grids</strong>
            <div class='agroscan-legend-row'><span class='agroscan-legend-dot ndvi'></span> Escala NDVI / saúde da vegetação</div>
            <div class='agroscan-legend-row'><span class='agroscan-legend-dot high'></span> Risco alto priorizado</div>
            <div class='agroscan-legend-row'><span class='agroscan-legend-dot medium'></span> Risco médio em monitoramento</div>
            <div class='agroscan-legend-row'><span class='agroscan-legend-dot low'></span> Risco baixo / preventivo</div>
            <div class='agroscan-legend-row'><span class='agroscan-chip'>Hover</span> Passe o mouse e clique para receber feedback visual.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_interactive_geo_map(
    points_df: pd.DataFrame,
    center_lat: float | None = None,
    center_lon: float | None = None,
    zoom: float = 12.2,
):
    if points_df is None or points_df.empty or not {"lat", "lon"}.issubset(points_df.columns):
        st.info("Sem coordenadas suficientes para exibir o mapa interativo.")
        return

    df = points_df.copy()
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)

    if "radius" not in df.columns:
        df["radius"] = 110
    if "color" not in df.columns:
        df["color"] = [[34, 166, 34, 190] for _ in range(len(df))]
    if "label" not in df.columns:
        df["label"] = ["Ponto monitorado" for _ in range(len(df))]
    if "info" not in df.columns:
        df["info"] = ["Sem detalhamento adicional" for _ in range(len(df))]

    lat_ref = float(center_lat) if center_lat is not None else float(df["lat"].mean())
    lon_ref = float(center_lon) if center_lon is not None else float(df["lon"].mean())
    pitch = 28 if len(df) > 1 else 6
    theme_mode = st.session_state.get("theme_mode", "Escuro")
    map_style = "dark" if theme_mode == "Escuro" else "light"
    tooltip_bg = "rgba(10, 22, 18, 0.92)" if theme_mode == "Escuro" else "rgba(255, 255, 255, 0.96)"
    tooltip_color = "#eef9ef" if theme_mode == "Escuro" else "#142117"

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="color",
        get_line_color=[255, 255, 255, 210],
        line_width_min_pixels=1,
        filled=True,
        stroked=True,
        pickable=True,
        auto_highlight=True,
    )

    deck = pdk.Deck(
        map_provider="carto",
        map_style=map_style,
        initial_view_state=pdk.ViewState(
            latitude=lat_ref,
            longitude=lon_ref,
            zoom=zoom,
            pitch=pitch,
        ),
        layers=[layer],
        tooltip={
            "html": "<b>{label}</b><br/>{info}",
            "style": {
                "backgroundColor": tooltip_bg,
                "color": tooltip_color,
                "fontSize": "12px",
                "border": "1px solid rgba(85,170,105,.20)",
            },
        },
    )
    st.pydeck_chart(deck, use_container_width=True)


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


def carregar_imagem_bgr(path: str | None) -> np.ndarray | None:
    if not path or not os.path.exists(path):
        return None
    try:
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except Exception:
        return None


def ensure_data_dirs():
    for path in (DATA_DIR, IMAGES_DIR):
        os.makedirs(path, exist_ok=True)


def get_db_connection():
    ensure_data_dirs()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_resource(show_spinner=False)
def ensure_optional_dependency(module_name: str, pip_name: str | None = None) -> tuple[bool, str]:
    pip_target = pip_name or module_name
    try:
        importlib.import_module(module_name)
        return True, f"{pip_target} disponível no ambiente atual."
    except ModuleNotFoundError:
        install_cmd = [sys.executable, "-m", "pip", "install", pip_target]
        try:
            proc = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except Exception as exc:
            return False, f"Falha ao preparar {pip_target}: {exc}"

        if proc.returncode != 0:
            detalhe = (proc.stderr or proc.stdout or "").strip()
            return False, (
                f"Não foi possível instalar {pip_target} automaticamente. "
                f"Comando sugerido: {' '.join(install_cmd)}. Detalhes: {detalhe}"
            )

        try:
            importlib.invalidate_caches()
            importlib.import_module(module_name)
            return True, f"{pip_target} instalado automaticamente neste ambiente."
        except Exception as exc:
            return False, f"{pip_target} foi instalado, mas a importação ainda falhou: {exc}"


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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            property_id INTEGER,
            analysis_id INTEGER,
            grid_name TEXT NOT NULL,
            original_label TEXT,
            confirmed_label TEXT NOT NULL,
            risk_status TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(property_id) REFERENCES properties(id),
            FOREIGN KEY(analysis_id) REFERENCES analyses(id)
        )
        """
    )
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(analyses)").fetchall()}
    if "coords_json" not in existing_cols:
        cur.execute("ALTER TABLE analyses ADD COLUMN coords_json TEXT")
    if "localized_risks_json" not in existing_cols:
        cur.execute("ALTER TABLE analyses ADD COLUMN localized_risks_json TEXT")

    conn.commit()
    conn.close()


def reset_analysis_state():
    for chave in RESULT_SESSION_KEYS:
        st.session_state.pop(chave, None)


def build_image_signature(image_bytes: bytes, origem_imagem: str) -> str:
    return hashlib.sha256(image_bytes + origem_imagem.encode("utf-8")).hexdigest()


def sync_image_state(image_bytes: bytes, origem_imagem: str) -> bool:
    nonce = int(st.session_state.get("image_input_nonce", 0))
    nova_assinatura = f"{build_image_signature(image_bytes, origem_imagem)}::{nonce}"
    assinatura_atual = st.session_state.get("active_image_signature")
    mudou = nova_assinatura != assinatura_atual

    if mudou:
        havia_resultado = "analises" in st.session_state
        reset_analysis_state()
        st.session_state["active_image_signature"] = nova_assinatura
        st.session_state["cmp_pct"] = 50
        st.session_state["new_image_reset_notice"] = havia_resultado
        st.session_state["current_image_source"] = origem_imagem
        st.session_state["current_image_loaded_at"] = datetime.now().isoformat()

    return mudou


def parse_numeric_value(value):
    if value is None:
        return None

    texto = str(value).strip().replace("\xa0", "")
    if not texto or texto in {"...", "-", "X"}:
        return None

    if "," in texto and "." in texto:
        if texto.rfind(",") > texto.rfind("."):
            texto = texto.replace(".", "").replace(",", ".")
        else:
            texto = texto.replace(",", "")
    elif "," in texto:
        texto = texto.replace(".", "").replace(",", ".")

    try:
        return float(texto)
    except ValueError:
        return None


def formatar_numero_br(valor: float | None, casas: int = 2) -> str:
    if valor is None:
        return "N/D"
    numero = f"{valor:,.{casas}f}"
    return numero.replace(",", "_").replace(".", ",").replace("_", ".")


def handle_new_image_input(widget_key: str, origem_imagem: str):
    arquivo = st.session_state.get(widget_key)
    if arquivo is None or not hasattr(arquivo, "getvalue"):
        return

    image_bytes = arquivo.getvalue()
    if not image_bytes:
        return

    novo_nonce = int(st.session_state.get("image_input_nonce", 0)) + 1
    st.session_state["image_input_nonce"] = novo_nonce
    nova_assinatura = f"{build_image_signature(image_bytes, origem_imagem)}::{novo_nonce}"
    assinatura_atual = st.session_state.get("active_image_signature")
    havia_resultado = "analises" in st.session_state

    if havia_resultado or nova_assinatura != assinatura_atual:
        reset_analysis_state()
        st.session_state["cmp_pct"] = 50
        st.session_state["new_image_reset_notice"] = havia_resultado

    st.session_state["active_image_signature"] = nova_assinatura
    st.session_state["current_image_source"] = origem_imagem
    st.session_state["current_image_loaded_at"] = datetime.now().isoformat()
    st.session_state["last_image_widget"] = widget_key


def slugify_municipio(nome: str) -> str:
    nome_limpo = re.split(r"\s*[\/|]\s*", (nome or "").strip())[0]
    nome_limpo = re.split(r"\s+[–—-]\s+", nome_limpo)[0]
    nome_ascii = unicodedata.normalize("NFKD", nome_limpo).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "-", nome_ascii.lower()).strip("-")


@st.cache_data(ttl=86400, show_spinner=False)
def obter_indicadores_municipio_ibge(codigo_municipio: int | None) -> dict:
    if not codigo_municipio:
        return {}

    try:
        resposta = requests.get(
            IBGE_INDICADORES_URL.format(municipio_id=int(codigo_municipio)),
            timeout=15,
        )
        resposta.raise_for_status()
        dados = resposta.json()
        if not isinstance(dados, list) or len(dados) <= 1:
            return {}

        indicadores = {
            "ano_base": "2022",
            "fonte_indicadores": "IBGE SIDRA / Censo 2022",
        }

        for item in dados[1:]:
            nome_variavel = unicodedata.normalize("NFKD", str(item.get("D2N", "")))
            nome_variavel = nome_variavel.encode("ascii", "ignore").decode("ascii").lower()
            valor = parse_numeric_value(item.get("V"))
            if valor is None:
                continue

            if "populacao residente" in nome_variavel:
                indicadores["populacao_residente"] = int(round(valor))
            elif "area da unidade territorial" in nome_variavel:
                indicadores["area_km2"] = round(valor, 3)
            elif "densidade demografica" in nome_variavel:
                indicadores["densidade_hab_km2"] = round(valor, 2)

        return indicadores
    except Exception as e:
        return {"erro_indicadores": str(e)}


@st.cache_data(ttl=86400, show_spinner=False)
def obter_dados_municipio_ibge(cidade: str) -> dict:
    slug = slugify_municipio(cidade)
    if not slug:
        return {}

    try:
        resposta = requests.get(f"{IBGE_MUNICIPIOS_URL}/{slug}", timeout=10)
        resposta.raise_for_status()
        dados = resposta.json()
        item = dados[0] if isinstance(dados, list) and dados else dados
        if not isinstance(item, dict):
            return {}

        regiao_imediata = item.get("regiao-imediata", {}) or {}
        regiao_intermediaria = regiao_imediata.get("regiao-intermediaria", {}) or {}
        microrregiao = item.get("microrregiao", {}) or {}
        mesorregiao = microrregiao.get("mesorregiao", {}) or {}
        uf = mesorregiao.get("UF", {}) or regiao_intermediaria.get("UF", {}) or {}
        regiao = uf.get("regiao", {}) or {}
        codigo_municipio = item.get("id")
        indicadores = obter_indicadores_municipio_ibge(codigo_municipio)

        return {
            "municipio": item.get("nome"),
            "id": codigo_municipio,
            "uf": uf.get("sigla"),
            "uf_nome": uf.get("nome"),
            "regiao": regiao.get("nome"),
            "microrregiao": microrregiao.get("nome"),
            "mesorregiao": mesorregiao.get("nome"),
            "regiao_imediata": regiao_imediata.get("nome"),
            "regiao_intermediaria": regiao_intermediaria.get("nome"),
            "fonte": "IBGE Localidades",
            **indicadores,
        }
    except Exception as e:
        return {"erro": str(e)}


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
                "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,precipitation",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
                "forecast_days": 7,
            },
            timeout=15,
        )
        resposta.raise_for_status()
        dados = resposta.json()
        atual = dados.get("current", {})
        diario = dados.get("daily", {})
        horario = dados.get("hourly", {})

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

        previsao_24h = []
        horas = horario.get("time", [])[:24]
        temps = horario.get("temperature_2m", [])[:24]
        umidades = horario.get("relative_humidity_2m", [])[:24]
        probs = horario.get("precipitation_probability", [])[:24]
        chuvas_h = horario.get("precipitation", [])[:24]
        for data, temp, umid, prob, chuva in zip(horas, temps, umidades, probs, chuvas_h):
            previsao_24h.append({
                "data": data,
                "temperatura": temp,
                "umidade": umid,
                "prob_chuva": prob,
                "chuva_mm": chuva,
            })

        chuva_24h = round(sum(float(item.get("chuva_mm") or 0) for item in previsao_24h), 1)
        pico_temp_24h = round(max((float(item.get("temperatura") or 0) for item in previsao_24h), default=float(atual.get("temperature_2m") or 0)), 1)
        umidade_media_24h = round(
            sum(float(item.get("umidade") or 0) for item in previsao_24h) / max(len(previsao_24h), 1), 1
        ) if previsao_24h else atual.get("relative_humidity_2m")
        chuva_3d = round(sum(float(item.get("chuva_mm") or 0) for item in previsao[:3]), 1)
        chuva_7d = round(sum(float(item.get("chuva_mm") or 0) for item in previsao[:7]), 1)

        melhor_janela = None
        if previsao:
            melhor_janela = min(
                previsao[:7],
                key=lambda dia: (float(dia.get("chuva_mm") or 0), float(dia.get("temp_max") or 99)),
            )

        return {
            "temperatura": atual.get("temperature_2m"),
            "umidade": atual.get("relative_humidity_2m"),
            "precipitacao": atual.get("precipitation"),
            "vento": atual.get("wind_speed_10m"),
            "previsao": previsao,
            "previsao_24h": previsao_24h,
            "janela_24h": {
                "chuva_total_mm": chuva_24h,
                "pico_temp": pico_temp_24h,
                "umidade_media": umidade_media_24h,
            },
            "janela_3d": {
                "chuva_total_mm": chuva_3d,
                "dias_monitorados": len(previsao[:3]),
            },
            "janela_7d": {
                "chuva_total_mm": chuva_7d,
                "dias_monitorados": len(previsao[:7]),
            },
            "melhor_janela_operacao": melhor_janela,
        }
    except Exception as e:
        return {"erro": str(e), "previsao": [], "previsao_24h": []}


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


def registrar_feedback_usuario(user_id: int, property_id: int | None, analysis_id: int | None,
                               grid_name: str, dados_grid: dict, rotulo_final: str,
                               risco_confirmado: str, observacao: str = "") -> tuple[bool, str]:
    if not dados_grid or not rotulo_final:
        return False, "Feedback inválido para o grid informado."

    conn = get_db_connection()
    agora = datetime.now().isoformat()
    try:
        conn.execute(
            """
            INSERT INTO feedback_events (
                user_id, property_id, analysis_id, grid_name,
                original_label, confirmed_label, risk_status, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                property_id,
                analysis_id,
                grid_name,
                dados_grid.get("tipo"),
                rotulo_final,
                risco_confirmado,
                observacao.strip(),
                agora,
            ),
        )
        conn.execute(
            """
            INSERT INTO model_samples (user_id, property_id, label, ndvi, textura, umidade, saturacao, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                property_id,
                rotulo_final,
                float(dados_grid.get("ndvi", 0.0)),
                float(dados_grid.get("textura", 0.0)),
                float(dados_grid.get("umidade_frac", 0.0)),
                float(dados_grid.get("sat_mean", 0.0)),
                agora,
            ),
        )
        conn.commit()
        return True, f"Feedback salvo para {grid_name}. O modelo local aprendeu com essa correção."
    except Exception as e:
        return False, f"Não foi possível registrar o feedback: {e}"
    finally:
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


def faixa_posicional(idx: int, total: int, eixo: str) -> str:
    labels = ["Norte", "Centro-norte", "Centro", "Centro-sul", "Sul"]
    if eixo == "horizontal":
        labels = ["Oeste", "Centro-oeste", "Centro", "Centro-leste", "Leste"]

    if total <= 1:
        return labels[len(labels) // 2]

    pos = idx / max(total - 1, 1)
    return labels[min(len(labels) - 1, round(pos * (len(labels) - 1)))]


def descrever_localizacao_grid(grid_idx: int, rows: int, cols: int, area_total_ha: float = 0.0) -> dict:
    row_idx = grid_idx // cols
    col_idx = grid_idx % cols
    vertical = faixa_posicional(row_idx, rows, "vertical")
    horizontal = faixa_posicional(col_idx, cols, "horizontal")
    area_pct = round(100.0 / max(rows * cols, 1), 2)
    area_ha = round(area_total_ha * area_pct / 100.0, 2) if area_total_ha else None

    if vertical == "Centro" and horizontal == "Centro":
        local = "Centro do terreno"
    elif horizontal == "Centro":
        local = f"Faixa {vertical}"
    elif vertical == "Centro":
        local = f"Faixa {horizontal}"
    else:
        local = f"Setor {vertical} / {horizontal}"

    return {
        "local": local,
        "vertical": vertical,
        "horizontal": horizontal,
        "area_pct": area_pct,
        "area_ha": area_ha,
    }


def identificar_riscos_localizados(analises: dict, coords: dict, clima: dict | None,
                                   rows: int, cols: int, area_total_ha: float = 0.0) -> list[dict]:
    pontos = []
    clima_ok = bool(clima and not clima.get("erro"))
    chuva_prevista = float(sum((dia.get("chuva_mm") or 0) for dia in (clima or {}).get("previsao", [])[:3])) if clima_ok else 0.0
    temp_atual = float((clima or {}).get("temperatura") or 0) if clima_ok else 0.0
    umidade_ar = float((clima or {}).get("umidade") or 0) if clima_ok else 0.0
    prioridade = {"Baixo": 1, "Médio": 2, "Alto": 3}

    def adicionar(nome_grid: str, idx: int, titulo: str, nivel: str,
                  descricao: str, acao: str, gatilho_clima: str, score_ref: int):
        xs, xe, ys, ye = coords[nome_grid]
        posicao = descrever_localizacao_grid(idx, rows, cols, area_total_ha)
        pontos.append({
            "grid": nome_grid,
            "titulo": titulo,
            "nivel": nivel,
            "localizacao": posicao["local"],
            "area_pct": posicao["area_pct"],
            "area_ha": posicao["area_ha"],
            "descricao": descricao,
            "acao": acao,
            "gatilho_clima": gatilho_clima,
            "score_referencia": int(score_ref),
            "bbox": {"xs": int(xs), "xe": int(xe), "ys": int(ys), "ye": int(ye)},
        })

    for idx, (nome, dados) in enumerate(analises.items()):
        ndvi = float(dados.get("ndvi", 0))
        umidade_frac = float(dados.get("umidade_frac", 0))
        textura = float(dados.get("textura", 0))
        score = int(dados.get("score", 0))
        tipo = dados.get("tipo", "")

        if tipo == "Seco" or ndvi < 0.10 or (temp_atual >= 32 and chuva_prevista < 6 and umidade_frac < 0.18):
            nivel = "Alto" if ndvi < 0.04 or score < 35 else "Médio"
            adicionar(
                nome,
                idx,
                "Seca localizada",
                nivel,
                "Vegetação com baixo vigor e umidade insuficiente nesta faixa do terreno.",
                "Priorizar irrigação localizada e cobertura morta nas próximas 48h.",
                f"Janela climática: {chuva_prevista:.1f} mm previstos e temperatura atual de {temp_atual:.1f} °C.",
                score,
            )

        if tipo == "Umido" or umidade_frac > 0.38 or (chuva_prevista > 20 and umidade_frac > 0.20):
            nivel = "Alto" if umidade_frac > 0.48 and chuva_prevista > 25 else "Médio"
            adicionar(
                nome,
                idx,
                "Encharcamento provável",
                nivel,
                "A área mostra retenção de água acima do ideal e pode perder trafegabilidade.",
                "Revisar drenagem lateral e evitar máquinas pesadas até a estabilização.",
                f"Previsão de chuva acumulada em 3 dias: {chuva_prevista:.1f} mm.",
                score,
            )

        if tipo == "Rochoso" or textura > 42 or (tipo == "Arenoso" and chuva_prevista > 18):
            nivel = "Alto" if tipo == "Rochoso" or textura > 52 else "Médio"
            adicionar(
                nome,
                idx,
                "Erosão / solo exposto",
                nivel,
                "Cobertura vegetal limitada com textura elevada, favorecendo perda superficial do solo.",
                "Usar cobertura vegetal, curva de nível e correção estrutural antes do plantio principal.",
                f"Risco ampliado com chuva prevista de {chuva_prevista:.1f} mm sobre solo mais exposto.",
                score,
            )

        if umidade_ar > 80 and 0.10 < ndvi < 0.32:
            adicionar(
                nome,
                idx,
                "Pragas e fungos",
                "Médio",
                "Microclima úmido com vigor intermediário favorece pressão fitossanitária local.",
                "Intensificar inspeção foliar e manejo preventivo nesta faixa do terreno.",
                f"Umidade relativa do ar em {umidade_ar:.0f}% no momento.",
                score,
            )

    pontos.sort(key=lambda item: (prioridade.get(item["nivel"], 0), 100 - item["score_referencia"]), reverse=True)
    return pontos[:16]


def gerar_mapa_riscos_localizados(img: np.ndarray, riscos_localizados: list[dict]) -> np.ndarray:
    mapa = img.copy()
    if not riscos_localizados:
        return mapa

    severidade = {"Baixo": 1, "Médio": 2, "Alto": 3}
    riscos_dominantes: dict[str, dict] = {}
    for risco in riscos_localizados:
        atual = riscos_dominantes.get(risco["grid"])
        if atual is None or severidade.get(risco["nivel"], 0) > severidade.get(atual["nivel"], 0):
            riscos_dominantes[risco["grid"]] = risco

    cores = {
        "Seca localizada": (0, 140, 255),
        "Encharcamento provável": (255, 120, 0),
        "Erosão / solo exposto": (120, 120, 120),
        "Pragas e fungos": (180, 70, 200),
    }

    overlay = img.copy()
    for risco in riscos_dominantes.values():
        bbox = risco["bbox"]
        xs, xe, ys, ye = bbox["xs"], bbox["xe"], bbox["ys"], bbox["ye"]
        cor = cores.get(risco["titulo"], (0, 255, 255))
        cv2.rectangle(overlay, (xs, ys), (xe, ye), cor, -1)

    mapa = cv2.addWeighted(overlay, 0.22, img, 0.78, 0)
    for risco in riscos_dominantes.values():
        bbox = risco["bbox"]
        xs, xe, ys, ye = bbox["xs"], bbox["xe"], bbox["ys"], bbox["ye"]
        cor = cores.get(risco["titulo"], (0, 255, 255))
        cv2.rectangle(mapa, (xs, ys), (xe, ye), cor, 2)
        legenda = f"{risco['grid']} • {risco['nivel'][0]}"
        topo = max(18, ys)
        cv2.rectangle(mapa, (xs, topo - 18), (min(xe, xs + 120), topo), (15, 15, 15), -1)
        cv2.putText(mapa, legenda, (xs + 4, topo - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return mapa


def gerar_alertas_inteligentes(metricas: dict, riscos: list[dict], clima: dict | None,
                               riscos_localizados: list[dict]) -> list[dict]:
    alertas = []
    prioridade = {"Baixo": 1, "Médio": 2, "Alto": 3}
    previsao = (clima or {}).get("previsao", [])
    chuva_3d = round(sum(float(dia.get("chuva_mm") or 0) for dia in previsao[:3]), 1)
    chuva_7d = round(sum(float(dia.get("chuva_mm") or 0) for dia in previsao[:7]), 1)
    temp_atual = float((clima or {}).get("temperatura") or 0)
    umidade_ar = float((clima or {}).get("umidade") or 0)
    vento = float((clima or {}).get("vento") or 0)
    ndvi_global = float(metricas.get("ndvi_global", 0))

    def adicionar(nivel: str, titulo: str, janela: str, descricao: str, acao: str):
        alertas.append({
            "nivel": nivel,
            "titulo": titulo,
            "janela": janela,
            "descricao": descricao,
            "acao": acao,
        })

    if any(r.get("titulo") == "Seca localizada" and r.get("nivel") == "Alto" for r in riscos_localizados) and chuva_3d < 8:
        adicionar(
            "Alto",
            "Irrigação prioritária",
            "Próximas 24h",
            "Há setores com estresse hídrico elevado e baixa chuva prevista no curto prazo.",
            "Abrir irrigação localizada imediatamente nos grids críticos e revisar cobertura do solo.",
        )

    if any(r.get("titulo") == "Encharcamento provável" for r in riscos_localizados) and chuva_3d >= 20:
        adicionar(
            "Alto" if chuva_3d >= 35 else "Médio",
            "Drenagem preventiva",
            "Próximos 3 dias",
            f"Há previsão acumulada de {chuva_3d} mm com faixas já úmidas no terreno.",
            "Desobstruir saídas d'água, evitar maquinário pesado e priorizar inspeção nos pontos baixos.",
        )

    if umidade_ar >= 80 and any("Pragas e fungos" in r.get("titulo", "") for r in riscos_localizados + riscos):
        adicionar(
            "Médio",
            "Janela fitossanitária crítica",
            "48h",
            "A combinação de umidade alta e vigor intermediário aumenta o risco de pragas e fungos.",
            "Planejar inspeção foliar e manejo preventivo nos setores sinalizados.",
        )

    if vento >= 18:
        adicionar(
            "Médio",
            "Operação de campo com restrição",
            "Hoje",
            f"Vento em {vento:.1f} km/h pode prejudicar pulverização e aplicação foliar.",
            "Priorizar janelas de menor vento e evitar aplicação em horários de rajada.",
        )

    if ndvi_global >= 0.22 and chuva_7d <= 12 and not alertas:
        adicionar(
            "Baixo",
            "Monitoramento programado",
            "Próximos 7 dias",
            "O cenário geral está estável, porém convém manter acompanhamento preventivo.",
            "Revisar os grids com menor score e comparar com a próxima captura.",
        )

    alertas.sort(key=lambda item: (prioridade.get(item["nivel"], 0), item["janela"]), reverse=True)
    return alertas[:6]


def montar_plano_acao_prioritario(analises: dict, riscos_localizados: list[dict],
                                  clima: dict | None, propriedade: dict | None = None) -> list[dict]:
    plano = []
    vistos = set()
    chuva_3d = round(sum(float(dia.get("chuva_mm") or 0) for dia in (clima or {}).get("previsao", [])[:3]), 1)

    for item in riscos_localizados:
        grid = item.get("grid")
        if grid in vistos:
            continue
        vistos.add(grid)
        nivel = item.get("nivel", "Médio")
        prazo = "24h" if nivel == "Alto" else "3 dias" if nivel == "Médio" else "7 dias"
        impacto = (
            "Redução imediata do risco operacional e preservação do potencial produtivo."
            if nivel == "Alto"
            else "Estabilização da área e prevenção de perdas progressivas."
        )
        plano.append({
            "prioridade": len(plano) + 1,
            "objetivo": item.get("titulo", "Área de atenção"),
            "grid": grid,
            "setor": item.get("localizacao", "Setor não identificado"),
            "acao": item.get("acao", "Monitorar em campo."),
            "prazo": prazo,
            "gatilho": item.get("gatilho_clima", "Leitura local do terreno"),
            "impacto": impacto,
        })
        if len(plano) >= 6:
            break

    if not plano and analises:
        ordenados = sorted(analises.items(), key=lambda kv: kv[1].get("score", 0))[:3]
        for nome, dados in ordenados:
            plano.append({
                "prioridade": len(plano) + 1,
                "objetivo": f"Melhorar score agrícola de {nome}",
                "grid": nome,
                "setor": "Setor monitorado",
                "acao": "Executar correção leve do solo, revisar irrigação e repetir captura comparativa.",
                "prazo": "7 dias",
                "gatilho": f"Score atual {dados.get('score', 0)}/100 com chuva prevista de {chuva_3d} mm.",
                "impacto": "Ganho gradual de estabilidade e melhor tomada de decisão para o próximo ciclo.",
            })

    return plano


def estimar_coordenadas_talhoes_risco(riscos_localizados: list[dict], propriedade: dict | None) -> list[dict]:
    if not riscos_localizados or not propriedade:
        return []
    if propriedade.get("latitude") is None or propriedade.get("longitude") is None:
        return []

    lat0 = float(propriedade["latitude"])
    lon0 = float(propriedade["longitude"])
    area_ha = float(propriedade.get("area_ha") or 10.0)
    lado_m = max(40.0, float(np.sqrt(max(area_ha, 0.1) * 10000.0)))
    max_x = max((int((item.get("bbox") or {}).get("xe", 0)) for item in riscos_localizados), default=1)
    max_y = max((int((item.get("bbox") or {}).get("ye", 0)) for item in riscos_localizados), default=1)
    metros_por_grau_lat = 111_320.0
    metros_por_grau_lon = max(1.0, 111_320.0 * float(np.cos(np.radians(lat0))))

    pontos = []
    vistos = set()
    for item in riscos_localizados:
        grid = item.get("grid", "Grid")
        if grid in vistos:
            continue
        vistos.add(grid)
        bbox = item.get("bbox") or {}
        cx = ((float(bbox.get("xs", 0)) + float(bbox.get("xe", 0))) / 2.0) / max(max_x, 1)
        cy = ((float(bbox.get("ys", 0)) + float(bbox.get("ye", 0))) / 2.0) / max(max_y, 1)
        desloc_east = (cx - 0.5) * lado_m
        desloc_north = (0.5 - cy) * lado_m
        lat = lat0 + desloc_north / metros_por_grau_lat
        lon = lon0 + desloc_east / metros_por_grau_lon
        pontos.append({
            "lat": lat,
            "lon": lon,
            "grid": grid,
            "nivel": item.get("nivel", "Médio"),
            "risco": item.get("titulo", "Área de atenção"),
            "localizacao": item.get("localizacao", "Setor monitorado"),
        })

    return pontos


def salvar_analise_historico(user_id: int, property_id: int, origem_imagem: str,
                             img_orig: np.ndarray, img_mapa: np.ndarray,
                             resultado_geral: str, metricas: dict, analises: dict,
                             clima: dict, riscos: list[dict],
                             coords: dict | None = None,
                             riscos_localizados: list[dict] | None = None) -> int:
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
            weather_json, risks_json, metricas_json, analises_json,
            coords_json, localized_risks_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            json.dumps(coords or {}, ensure_ascii=False),
            json.dumps(riscos_localizados or [], ensure_ascii=False),
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
    item["coords"] = json.loads(item.get("coords_json") or "{}")
    item["localized_risks"] = json.loads(item.get("localized_risks_json") or "[]")
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

def slider_comparacao(
    img_orig_bgr: np.ndarray,
    img_mapa_bgr: np.ndarray,
    initial=50,
    left_label: str = "Antes",
    right_label: str = "Depois",
    component_key: str = "cmp_default",
):
    """Comparador premium com ids únicos, labels e slider auxiliar."""
    img_a = redimensionar(img_orig_bgr, MAX_W, MAX_H)
    img_b = redimensionar(img_mapa_bgr, MAX_W, MAX_H)
    h, w = img_a.shape[:2]
    img_b = cv2.resize(img_b, (w, h))

    orig_rgb = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    mapa_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

    b64o = img_rgb_to_base64(orig_rgb)
    b64m = img_rgb_to_base64(mapa_rgb)

    asp = round((h / w) * 100, 2)
    inv = 100 - initial
    uid = re.sub(r"[^a-zA-Z0-9_]+", "_", str(component_key))

    html = f"""
<div style="width:100%;max-width:{w}px;margin:0 auto;">
  <div style="display:flex;justify-content:space-between;gap:8px;margin:0 0 8px;flex-wrap:wrap;">
    <span style="padding:4px 10px;border-radius:999px;background:rgba(34,166,34,.14);font-size:12px;"><strong>{left_label}</strong></span>
    <span style="padding:4px 10px;border-radius:999px;background:rgba(31,142,241,.14);font-size:12px;"><strong>{right_label}</strong></span>
  </div>
  <div style="position:relative;width:100%;">
    <div style="position:absolute;top:8px;left:10px;z-index:12;color:#fff;font-size:13px;font-weight:600;text-shadow:0 1px 4px rgba(0,0,0,.9);">
      <span id="lbl-{uid}">{initial}%</span>
    </div>
    <div id="cmp-{uid}" style="position:relative;width:100%;padding-bottom:{asp}%;cursor:col-resize;border-radius:12px;overflow:hidden;box-shadow:0 8px 18px rgba(0,0,0,.18);">
      <img src="data:image/jpeg;base64,{b64o}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;"/>
      <div id="clip-{uid}" style="position:absolute;inset:0;overflow:hidden;clip-path:inset(0 {inv}% 0 0);">
        <img src="data:image/jpeg;base64,{b64m}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;"/>
      </div>
      <div id="line-{uid}" style="position:absolute;top:0;left:{initial}%;width:3px;height:100%;background:#00d9ff;box-shadow:0 0 10px #00d9ff;z-index:10;"></div>
    </div>
  </div>
  <div style="margin-top:10px;display:flex;align-items:center;gap:10px;">
    
    <input id="range-{uid}" type="range" min="0" max="100" value="{initial}" style="width:100%;accent-color:#22a622;" />
    
  </div>

</div>
<script>
(function(){{
  const cmp = document.getElementById('cmp-{uid}');
  const clip = document.getElementById('clip-{uid}');
  const line = document.getElementById('line-{uid}');
  const lbl = document.getElementById('lbl-{uid}');
  const range = document.getElementById('range-{uid}');

  function setValue(p) {{
    p = Math.max(0, Math.min(100, Number(p)));
    clip.style.clipPath = 'inset(0 ' + (100 - p) + '% 0 0)';
    line.style.left = p + '%';
    lbl.textContent = Math.round(p) + '%';
    if (range && document.activeElement !== range) range.value = p;
  }}

  function updateFromEvent(e) {{
    const evt = e.touches ? e.touches[0] : e;
    const rect = cmp.getBoundingClientRect();
    setValue(((evt.clientX - rect.left) / rect.width) * 100);
  }}

  cmp.addEventListener('mousemove', updateFromEvent);
  cmp.addEventListener('click', updateFromEvent);
  cmp.addEventListener('touchmove', updateFromEvent, {{ passive: true }});
  range.addEventListener('input', (e) => setValue(e.target.value));
  setValue({initial});
}})();
</script>
"""
    st.components.v1.html(html, height=h + 98)


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


def interactive_risk_map_component(img: np.ndarray, riscos_localizados: list[dict],
                                   rows: int, cols: int) -> str:
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_b64 = img_rgb_to_base64(img_rgb)
    prioridade = {"Baixo": 1, "Médio": 2, "Alto": 3}

    agrupados: dict[str, dict] = {}
    for risco in riscos_localizados:
        grid = risco.get("grid", "Grid")
        item = agrupados.setdefault(
            grid,
            {
                "grid": grid,
                "localizacao": risco.get("localizacao", "Faixa não identificada"),
                "area_pct": risco.get("area_pct"),
                "area_ha": risco.get("area_ha"),
                "bbox": risco.get("bbox", {}),
                "nivel": risco.get("nivel", "Médio"),
                "titulo": risco.get("titulo", "Área de atenção"),
                "gatilho_clima": risco.get("gatilho_clima", ""),
                "acao": risco.get("acao", ""),
                "detalhes": [],
            },
        )
        item["detalhes"].append({
            "titulo": risco.get("titulo", "Risco"),
            "nivel": risco.get("nivel", "Médio"),
            "descricao": risco.get("descricao", ""),
            "acao": risco.get("acao", ""),
            "gatilho_clima": risco.get("gatilho_clima", ""),
        })

        if prioridade.get(risco.get("nivel", "Baixo"), 0) >= prioridade.get(item.get("nivel", "Baixo"), 0):
            item["nivel"] = risco.get("nivel", item["nivel"])
            item["titulo"] = risco.get("titulo", item["titulo"])
            item["gatilho_clima"] = risco.get("gatilho_clima", item["gatilho_clima"])
            item["acao"] = risco.get("acao", item["acao"])

    pinos = []
    for indice, item in enumerate(agrupados.values(), start=1):
        bbox = item.get("bbox") or {}
        xs = int(bbox.get("xs", 0))
        xe = int(bbox.get("xe", xs))
        ys = int(bbox.get("ys", 0))
        ye = int(bbox.get("ye", ys))
        item["cx"] = int((xs + xe) / 2)
        item["cy"] = int((ys + ye) / 2)
        item["pin"] = indice
        pinos.append(item)

    component_id = f"riskmap_{hashlib.md5((img_b64 + str(len(pinos))).encode('utf-8')).hexdigest()[:10]}"
    pinos_json = json.dumps(pinos, ensure_ascii=False)

    return f"""
<div style="max-width:{w}px;margin:0 auto;">
  <div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;margin:0 0 8px;">
    <span style="font-size:12px;color:#a8c8a8">Clique em um ponto para ver o risco e a ação recomendada.</span>
    <span style="font-size:12px;color:#a8c8a8">{len(pinos)} área(s) sinalizadas na própria malha de grids.</span>
  </div>
  <div style="position:relative;">
    <div style="position:absolute;top:12px;right:12px;z-index:12;padding:8px 10px;border-radius:10px;background:rgba(8,16,24,.86);color:#ecffec;font-size:11px;border:1px solid rgba(255,255,255,.10);box-shadow:0 8px 18px rgba(0,0,0,.22);pointer-events:none;">
      <div style="font-weight:700;margin-bottom:4px;">🧭 Legenda fixa</div>
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;"><span style="width:9px;height:9px;border-radius:50%;background:#ff5c5c;display:inline-block;"></span> Risco alto</div>
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;"><span style="width:9px;height:9px;border-radius:50%;background:#ffb020;display:inline-block;"></span> Risco médio</div>
      <div style="display:flex;align-items:center;gap:6px;"><span style="width:9px;height:9px;border-radius:50%;background:#39d98a;display:inline-block;"></span> Risco baixo</div>
    </div>
    <canvas id="{component_id}" style="width:100%;height:auto;border-radius:10px;display:block;border:1px solid rgba(255,255,255,.08);background:#0b1010;"></canvas>
  </div>
  <div style="display:flex;gap:14px;flex-wrap:wrap;font-size:12px;margin:8px 0 10px;">
    <span>🔴 Alto</span><span>🟠 Médio</span><span>🟢 Baixo</span><span>✨ Hover realça o grid</span>
  </div>
  <div id="{component_id}_detail" style="padding:10px 12px;background:#142119;border-left:3px solid #22a622;border-radius:8px;font-size:13px;color:#ecffec;box-shadow:0 8px 18px rgba(0,0,0,.18);">
    {'Nenhuma área crítica foi marcada nesta leitura.' if not pinos else 'Selecione um ponto no mapa para abrir o detalhamento interativo.'}
  </div>
</div>
<script>
(function(){{
  const IW={w}, IH={h}, ROWS={rows}, COLS={cols};
  const PINS={pinos_json};
  const canvas=document.getElementById('{component_id}');
  const detail=document.getElementById('{component_id}_detail');
  const ctx=canvas.getContext('2d');
  canvas.width=IW; canvas.height=IH;
  const bg=new Image();
  const colors={{'Alto':'#ff5c5c','Médio':'#ffb020','Baixo':'#39d98a'}};
  let hoverIndex=-1, selectedIndex=PINS.length ? 0 : -1;

  function colorFor(level) {{
    return colors[level] || '#37b5ff';
  }}

  function formatArea(pin) {{
    if (pin.area_ha !== null && pin.area_ha !== undefined) return `${{pin.area_ha}} ha`;
    if (pin.area_pct !== null && pin.area_pct !== undefined) return `${{pin.area_pct}}% da área`;
    return 'N/D';
  }}

  function renderDetail(idx) {{
    if (idx < 0 || !PINS[idx]) {{
      detail.innerHTML = 'Selecione um ponto no mapa para abrir o detalhamento interativo.';
      return;
    }}
    const pin = PINS[idx];
    const lista = (pin.detalhes || []).map(item =>
      `<li><strong>${{item.titulo}}</strong> (${{item.nivel}}) — ${{item.gatilho_clima || item.descricao || 'Monitorar'}}<br/><span style="color:#b9dcb9">Ação:</span> ${{item.acao || 'Monitoramento em campo.'}}</li>`
    ).join('');
    detail.innerHTML = `
      <strong>${{pin.grid}}</strong> — ${{pin.localizacao}}<br/>
      <span style="color:#b9dcb9">Área estimada:</span> ${{formatArea(pin)}} ·
      <span style="color:#b9dcb9">Risco dominante:</span> ${{pin.titulo}} (${{pin.nivel}})<br/>
      <span style="color:#b9dcb9">Gatilho climático:</span> ${{pin.gatilho_clima || 'Leitura local do terreno'}}
      <ul style="margin:8px 0 0 18px;padding:0">${{lista}}</ul>
    `;
  }}

  function draw() {{
    ctx.clearRect(0,0,IW,IH);
    ctx.drawImage(bg,0,0,IW,IH);
    ctx.strokeStyle='rgba(255,255,255,0.45)';
    ctx.lineWidth=1.2;
    for(let j=1;j<COLS;j++) {{
      const x=j*Math.floor(IW/COLS);
      ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,IH); ctx.stroke();
    }}
    for(let i=1;i<ROWS;i++) {{
      const y=i*Math.floor(IH/ROWS);
      ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(IW,y); ctx.stroke();
    }}

    PINS.forEach((pin, idx) => {{
      const active = idx === hoverIndex || idx === selectedIndex;
      const color = colorFor(pin.nivel);
      const bbox = pin.bbox || {{}};
      if (active) {{
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(bbox.xs, bbox.ys, bbox.xe - bbox.xs, bbox.ye - bbox.ys);
        ctx.fillStyle = color + '22';
        ctx.fillRect(bbox.xs, bbox.ys, bbox.xe - bbox.xs, bbox.ye - bbox.ys);
        ctx.restore();
      }}
      ctx.beginPath();
      ctx.arc(pin.cx, pin.cy, active ? 11 : 9, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#081018';
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(pin.pin), pin.cx, pin.cy + 0.5);
    }});
  }}

  function findPin(evt) {{
    const rect = canvas.getBoundingClientRect();
    const sx = IW / rect.width;
    const sy = IH / rect.height;
    const x = (evt.clientX - rect.left) * sx;
    const y = (evt.clientY - rect.top) * sy;
    let nearest = -1;
    let minDist = 18;
    PINS.forEach((pin, idx) => {{
      const dist = Math.hypot(pin.cx - x, pin.cy - y);
      if (dist < minDist) {{
        minDist = dist;
        nearest = idx;
      }}
    }});
    return nearest;
  }}

  canvas.addEventListener('mousemove', (evt) => {{
    hoverIndex = findPin(evt);
    draw();
  }});
  canvas.addEventListener('mouseleave', () => {{
    hoverIndex = -1;
    draw();
  }});
  canvas.addEventListener('click', (evt) => {{
    selectedIndex = findPin(evt);
    renderDetail(selectedIndex);
    draw();
  }});

  bg.onload = () => {{
    draw();
    renderDetail(selectedIndex);
  }};
  bg.src = 'data:image/jpeg;base64,' + {json.dumps(img_b64)};
}})();
</script>"""


# ═══════════════════════════════════════════════════════════════
# EXPORTAR PDF  — migrado para reportlab (sem fpdf2)
# ═══════════════════════════════════════════════════════════════

def gerar_pdf(img_orig: np.ndarray, img_mapa: np.ndarray,
              analises: dict, rec_texto, meta: dict | None = None) -> io.BytesIO | None:
    try:
        dep_ok, dep_msg = ensure_optional_dependency("reportlab", "reportlab")
        if not dep_ok:
            raise RuntimeError(
                "A biblioteca responsável pelo PDF não está disponível neste ambiente Python. "
                f"{dep_msg} | Python em uso: {sys.executable}"
            )

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
        ibge = meta.get("ibge", {})
        clima = meta.get("clima", {})
        riscos = meta.get("riscos", [])
        riscos_localizados = meta.get("localized_risks", [])
        resumo_executivo = meta.get("resumo_executivo", [])
        alertas_inteligentes = meta.get("alertas_inteligentes", [])
        plano_acao = meta.get("plano_acao", [])
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
            ["Código IBGE", safe(ibge.get("id", "N/D")), "Extensão territorial", safe(f"{formatar_numero_br(ibge.get('area_km2'), 2)} km²" if ibge.get("area_km2") is not None else "N/D")],
            ["Densidade pop.", safe(f"{formatar_numero_br(ibge.get('densidade_hab_km2'), 2)} hab/km²" if ibge.get("densidade_hab_km2") is not None else "N/D"), "Microrregião", safe(ibge.get("microrregiao", "N/D"))],
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

        if alertas_inteligentes:
            story.append(Paragraph("Alertas Inteligentes", secao_style))
            for alerta in alertas_inteligentes[:6]:
                cor = VERMELHO if alerta["nivel"] == "Alto" else colors.HexColor("#d4a017") if alerta["nivel"] == "Médio" else VERDE_ESCURO
                story.append(Paragraph(
                    f"<b>{safe(alerta['titulo'])}</b> ({safe(alerta['nivel'])}) — {safe(alerta['janela'])}<br/>"
                    f"{safe(alerta['descricao'])}<br/><b>Ação:</b> {safe(alerta['acao'])}",
                    ParagraphStyle("Alerta", parent=corpo_style, textColor=cor),
                ))

        if riscos_localizados:
            story.append(Paragraph("Áreas de Risco no Terreno", secao_style))
            for ponto in riscos_localizados[:8]:
                area_txt = f"{ponto['area_ha']} ha" if ponto.get("area_ha") is not None else f"{ponto.get('area_pct', 0)}% da área"
                story.append(Paragraph(
                    f"<b>{safe(ponto['grid'])}</b> — {safe(ponto['localizacao'])} ({safe(area_txt)})<br/>"
                    f"<b>{safe(ponto['titulo'])}</b> ({safe(ponto['nivel'])}) — {safe(ponto['gatilho_clima'])}",
                    corpo_style,
                ))

        if plano_acao:
            story.append(Paragraph("Plano de Ação Prioritário", secao_style))
            for etapa in plano_acao[:6]:
                story.append(Paragraph(
                    f"<b>Prioridade {safe(etapa['prioridade'])}</b> — {safe(etapa['objetivo'])}<br/>"
                    f"<b>Local:</b> {safe(etapa['grid'])} / {safe(etapa['setor'])}<br/>"
                    f"<b>Prazo:</b> {safe(etapa['prazo'])} • <b>Ação:</b> {safe(etapa['acao'])}<br/>"
                    f"<b>Impacto esperado:</b> {safe(etapa['impacto'])}",
                    corpo_style,
                ))

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
        install_hint = f"{sys.executable} -m pip install reportlab"
        st.caption("Se o relatório não abrir, confirme se o Streamlit foi iniciado com o mesmo Python do projeto.")
        st.code(f"Python em uso: {sys.executable}\nComando sugerido: {install_hint}")
        import traceback
        st.code(traceback.format_exc())
        return None
        return None


# ═══════════════════════════════════════════════════════════════
# INTERFACE STREAMLIT
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title="AgroScan Drone", page_icon="🌱", layout="wide")
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Escuro"
if "show_guided_tour" not in st.session_state:
    st.session_state["show_guided_tour"] = True
if "tour_seen" not in st.session_state:
    st.session_state["tour_seen"] = False
if "executive_mode" not in st.session_state:
    st.session_state["executive_mode"] = False

inject_global_css()
init_db()

# Título com efeito de digitação
typewriter("🌱 AgroScan Drone — Plataforma de Identificação e Recomendação de terreno", tag="h1",
           extra_style="font-size:1.8rem;")
st.markdown(
    "<div class='agroscan-subtitle'>Diagnóstico agrícola com clima, território IBGE, mapa interativo e plano de ação em uma experiência visual mais fluida.</div>",
    unsafe_allow_html=True,
)

if "current_user" not in st.session_state:
    render_auth_screen()
    st.stop()

current_user = st.session_state["current_user"]
amostras_modelo = contar_amostras_modelo(current_user["id"])
propriedades = listar_propriedades(current_user["id"])
propriedade_ativa = None
executive_mode = st.session_state.get("executive_mode", False)

if executive_mode:
    st.markdown(
        """
        <div class='agroscan-executive-banner'>
            <strong>🎯 Modo executivo ativado</strong><br/>
            Layout mais limpo para apresentação, com foco nos indicadores, mapa e plano de ação.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    render_guided_tour()
    render_mobile_shortcuts(
        bool(st.session_state.get("active_image_signature")),
        "analises" in st.session_state,
    )

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configurações")
    st.success(f"Usuário conectado: {current_user['username']}")
    theme_choice = st.radio(
        "Tema visual",
        ["Escuro", "Claro"],
        index=0 if st.session_state.get("theme_mode", "Escuro") == "Escuro" else 1,
        horizontal=True,
    )
    if theme_choice != st.session_state.get("theme_mode"):
        st.session_state["theme_mode"] = theme_choice
        st.rerun()

    executive_choice = st.toggle(
        "Modo executivo para apresentação",
        value=st.session_state.get("executive_mode", False),
        help="Simplifica a visualização e deixa o painel mais limpo para reuniões, projeção e demonstrações.",
    )
    if executive_choice != st.session_state.get("executive_mode", False):
        st.session_state["executive_mode"] = executive_choice
        st.rerun()

    if st.button("🧭 Ver tour guiado", use_container_width=True):
        st.session_state["show_guided_tour"] = True
        st.rerun()

    if st.button("Sair", use_container_width=True):
        reset_analysis_state()
        for chave in [
            "current_user", "selected_property_id", "active_image_signature", "cmp_pct",
            "last_image_widget", "upload_image_input", "camera_image_input", "image_input_nonce",
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
        opcao_nenhuma = "Nenhuma (sem histórico)"
        labels_props = [opcao_nenhuma] + [f"{p['nome']} — {p['cidade'] or 'Sem região'}" for p in propriedades]
        default_index = 0
        selected_prop_id = st.session_state.get("selected_property_id")
        if selected_prop_id:
            for idx, prop in enumerate(propriedades, start=1):
                if prop["id"] == selected_prop_id:
                    default_index = idx
                    break
        escolha_prop = st.selectbox(
            "Propriedade ativa",
            labels_props,
            index=default_index,
            help="Selecione 'Nenhuma' para uma análise rápida sem salvar histórico nem comparar antes/depois.",
        )
        if escolha_prop == opcao_nenhuma:
            propriedade_ativa = None
            st.session_state["selected_property_id"] = None
            st.caption("Modo avulso ativo: a análise acontece na hora e não gera histórico/comparação automática.")
        else:
            propriedade_ativa = propriedades[labels_props.index(escolha_prop) - 1]
            st.session_state["selected_property_id"] = propriedade_ativa["id"]
            st.caption(
                f"Lat: {propriedade_ativa['latitude']}, Lon: {propriedade_ativa['longitude']} • "
                f"Área: {propriedade_ativa['area_ha']} ha"
            )
    else:
        st.info("Nenhuma propriedade cadastrada ainda. Você pode seguir com análise avulsa normalmente, sem histórico nem comparação temporal.")

    st.divider()
    st.caption(f"Modelo local: {amostras_modelo} amostras reais acumuladas")
    st.caption(f"AgroScan v{APP_VERSION} — upload, câmera, histórico e campo")

# ── Painel da fazenda ─────────────────────────────────────────────
st.markdown("<div id='secao-propriedade'></div>", unsafe_allow_html=True)
render_workflow_stepper(
    propriedade_ativa is not None,
    bool(st.session_state.get("active_image_signature")),
    "analises" in st.session_state,
)
typewriter("🧭 Painel da Propriedade", tag="h3")
clima_atual = {}
historico_prop = []
ibge_info = {}
if propriedade_ativa:
    historico_prop = listar_analises_salvas(current_user["id"], property_id=propriedade_ativa["id"])
    if propriedade_ativa.get("latitude") is not None and propriedade_ativa.get("longitude") is not None:
        clima_atual = obter_previsao_tempo(float(propriedade_ativa["latitude"]), float(propriedade_ativa["longitude"]))
    if propriedade_ativa.get("cidade"):
        ibge_info = obter_dados_municipio_ibge(propriedade_ativa["cidade"])

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
                "label": [propriedade_ativa["nome"]],
                "info": [
                    f"{propriedade_ativa['cidade'] or 'Localização cadastrada'}<br/>Área: {propriedade_ativa['area_ha']} ha"
                ],
                "radius": [180],
                "color": [[34, 166, 34, 190]],
            })
            st.caption("Mapa interativo da propriedade — arraste, dê zoom e passe o mouse para ver os detalhes.")
            render_interactive_geo_map(
                mapa_local,
                center_lat=float(propriedade_ativa["latitude"]),
                center_lon=float(propriedade_ativa["longitude"]),
                zoom=12.6,
            )

    with d2:
        st.markdown("**Previsão climática integrada (Open-Meteo)**")
        if clima_atual and not clima_atual.get("erro"):
            janela24 = clima_atual.get("janela_24h", {})
            janela3 = clima_atual.get("janela_3d", {})
            janela7 = clima_atual.get("janela_7d", {})
            melhor_janela = clima_atual.get("melhor_janela_operacao") or {}

            c1, c2, c3 = st.columns(3)
            c1.metric("🌡️ Temp. atual", f"{clima_atual.get('temperatura', 'N/D')} °C")
            c2.metric("💧 Umidade", f"{clima_atual.get('umidade', 'N/D')}%")
            c3.metric("💨 Vento", f"{clima_atual.get('vento', 'N/D')} km/h")
            c4, c5, c6 = st.columns(3)
            c4.metric("🌧️ Chuva 24h", f"{janela24.get('chuva_total_mm', 'N/D')} mm")
            c5.metric("📆 Chuva 3 dias", f"{janela3.get('chuva_total_mm', 'N/D')} mm")
            c6.metric(
                "✅ Melhor janela",
                melhor_janela.get("data", "N/D"),
                f"{melhor_janela.get('chuva_mm', 'N/D')} mm",
            )

            st.caption(f"Precipitação acumulada em 7 dias: {janela7.get('chuva_total_mm', 'N/D')} mm")
            clima_tab_24h, clima_tab_7d = st.tabs(["Próximas 24h", "Próximos 7 dias"])
            with clima_tab_24h:
                horas_df = pd.DataFrame([
                    {
                        "Hora": item["data"][11:16] if "T" in item["data"] else item["data"],
                        "Temperatura (°C)": item["temperatura"],
                        "Umidade (%)": item["umidade"],
                        "Prob. chuva (%)": item["prob_chuva"],
                    }
                    for item in clima_atual.get("previsao_24h", [])
                ])
                if not horas_df.empty:
                    st.line_chart(horas_df.set_index("Hora"), use_container_width=True)

            with clima_tab_7d:
                clima_df = pd.DataFrame([
                    {
                        "Data": dia["data"],
                        "Mín (°C)": dia["temp_min"],
                        "Máx (°C)": dia["temp_max"],
                        "Chuva (mm)": dia["chuva_mm"],
                    }
                    for dia in clima_atual.get("previsao", [])
                ])
                if not clima_df.empty:
                    st.dataframe(clima_df, use_container_width=True, hide_index=True)
                    st.bar_chart(clima_df.set_index("Data")[["Chuva (mm)"]], use_container_width=True)
        else:
            st.info("Previsão indisponível no momento para esta localização.")

        st.markdown("**Referência territorial IBGE**")
        if ibge_info and not ibge_info.get("erro"):
            st.markdown(
                f"**Município oficial:** {ibge_info.get('municipio', 'N/D')} / {ibge_info.get('uf', 'N/D')}  \n"
                f"**Código IBGE:** {ibge_info.get('id', 'N/D')}  \n"
                f"**Microrregião:** {ibge_info.get('microrregiao', 'N/D')}  \n"
                f"**Mesorregião:** {ibge_info.get('mesorregiao', 'N/D')}"
            )
            i1, i2 = st.columns(2)
            i1.metric(
                "🗺️ Extensão territorial",
                f"{formatar_numero_br(ibge_info.get('area_km2'), 2)} km²" if ibge_info.get("area_km2") is not None else "N/D",
            )
            i2.metric(
                "👥 Densidade populacional",
                f"{formatar_numero_br(ibge_info.get('densidade_hab_km2'), 2)} hab/km²" if ibge_info.get("densidade_hab_km2") is not None else "N/D",
            )
            if ibge_info.get("populacao_residente") is not None:
                st.caption(
                    f"População residente: {formatar_numero_br(float(ibge_info['populacao_residente']), 0)} habitantes • "
                    f"Fonte: {ibge_info.get('fonte')} + {ibge_info.get('fonte_indicadores', 'IBGE')}"
                )
            elif ibge_info.get("erro_indicadores"):
                st.caption("Município validado no IBGE, mas os indicadores oficiais estão indisponíveis no momento.")
        elif propriedade_ativa.get("cidade"):
            st.caption("Não foi possível validar automaticamente o município no IBGE com o texto cadastrado.")
else:
    render_priority_notice(
        "Modo avulso ativo",
        "Você pode analisar a imagem normalmente agora. Se quiser clima, histórico e comparação temporal, selecione uma propriedade cadastrada.",
        level="low",
    )

# ── Modelo local treinável ───────────────────────────────────────
if not executive_mode:
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
st.markdown("<div id='secao-upload'></div>", unsafe_allow_html=True)
typewriter("📷 Entrada da Imagem", tag="h3")
aba_upload, aba_camera = st.tabs(["Upload de arquivo", "Usar câmera"])

with aba_upload:
    arquivo_upload = st.file_uploader(
        "Envie a foto do terreno",
        type=["jpg", "jpeg", "png"],
        key="upload_image_input",
        on_change=handle_new_image_input,
        args=("upload_image_input", "Upload"),
    )

with aba_camera:
    foto_camera = st.camera_input(
        "Tire uma foto do terreno",
        key="camera_image_input",
        on_change=handle_new_image_input,
        args=("camera_image_input", "Câmera"),
    )
    st.caption("Permita o acesso à câmera no navegador para capturar a imagem.")

imagem_entrada = None
origem_imagem = ""
ultimo_widget = st.session_state.get("last_image_widget")
if ultimo_widget == "upload_image_input" and arquivo_upload is not None:
    imagem_entrada = arquivo_upload
    origem_imagem = "Upload"
elif ultimo_widget == "camera_image_input" and foto_camera is not None:
    imagem_entrada = foto_camera
    origem_imagem = "Câmera"
elif arquivo_upload is not None:
    imagem_entrada = arquivo_upload
    origem_imagem = "Upload"
elif foto_camera is not None:
    imagem_entrada = foto_camera
    origem_imagem = "Câmera"
else:
    st.info("Envie uma imagem ou tire uma foto pela câmera para iniciar a análise.")

if imagem_entrada:
    img_bytes = imagem_entrada.getvalue()
    sync_image_state(img_bytes, origem_imagem)
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
            f"<img src='data:image/jpeg;base64,{img_b64_upload}' style='width:{DISPLAY_W}px;border-radius:10px;box-shadow:0 8px 22px rgba(0,0,0,.16);'/>"
            f"<span class='agroscan-muted' style='font-size:12px;margin-top:4px'>Imagem carregada via {origem_imagem} ({W} x {H} px)</span></div>",
            unsafe_allow_html=True,
        )
        if st.session_state.pop("new_image_reset_notice", False):
            render_priority_notice(
                "Nova imagem detectada",
                "A etapa de análise de grids foi reiniciada automaticamente para esta nova captura.",
                level="medium",
            )
            if hasattr(st, "toast"):
                st.toast("Nova imagem recebida — a análise anterior foi resetada.", icon="🔄")

        b_f, g_f, r_f = cv2.split(img.astype("float32"))
        ndvi_global = float(np.mean((g_f - r_f) / (g_f + r_f + 1e-5)))
        total = H * W
        p_verde = np.sum(cv2.inRange(hsv, _VERDE["lo"], _VERDE["hi"]) > 0) / total
        p_amarelo = np.sum(cv2.inRange(hsv, _AMARELO["lo"], _AMARELO["hi"]) > 0) / total
        p_cinza = np.sum(cv2.inRange(hsv, _CINZA["lo"], _CINZA["hi"]) > 0) / total
        p_azul = np.sum(cv2.inRange(hsv, _AZUL["lo"], _AZUL["hi"]) > 0) / total

        if p_verde > 0.4:
            resultado_geral = "Terreno fértil"
        elif p_amarelo > 0.3:
            resultado_geral = "Terreno seco"
        elif p_cinza > 0.3:
            resultado_geral = "Terreno rochoso"
        elif p_azul > 0.2:
            resultado_geral = "Terreno úmido"
        else:
            resultado_geral = "Terreno arenoso"

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

        slider_comparacao(
            img,
            mapa,
            st.session_state.cmp_pct,
            left_label="Imagem original",
            right_label="Mapa NDVI",
            component_key="comparador_ndvi_atual",
        )

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
        render_floating_grid_legend()

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
            stage_box = st.empty()
            progress_bar = st.progress(0)
            render_analysis_progress(stage_box, progress_bar, 6, "Preparando leitura", "Validando imagem, malha e dados iniciais.")
            time.sleep(0.06)

            analises: dict = {}
            total_grids = len(coords)
            render_analysis_progress(stage_box, progress_bar, 18, "Analisando grids", f"Iniciando a varredura de {total_grids} setores do terreno.")
            for idx, (nome, (xs_, xe_, ys_, ye_)) in enumerate(coords.items(), start=1):
                analises[nome] = analisar_regiao(img[ys_:ye_, xs_:xe_])
                if idx == 1 or idx == total_grids or idx % max(1, total_grids // 6) == 0:
                    pct = 18 + int(42 * idx / total_grids)
                    render_analysis_progress(
                        stage_box,
                        progress_bar,
                        pct,
                        "Varredura do terreno",
                        f"{idx}/{total_grids} grids processados com feedback visual em tempo real.",
                    )

            score_medio = round(sum(d["score"] for d in analises.values()) / len(analises), 1)
            metricas_globais.update({
                "score_medio": score_medio,
                "total_grids": len(analises),
            })
            render_analysis_progress(stage_box, progress_bar, 72, "Cruzando clima e risco", "Gerando alertas, prioridades e resumo executivo.")

            riscos = detectar_riscos(analises, metricas_globais, clima_atual)
            resumo_executivo = gerar_resumo_executivo(resultado_geral, metricas_globais, riscos)
            area_total_ha = float((propriedade_ativa or {}).get("area_ha") or 0.0)
            localized_risks = identificar_riscos_localizados(
                analises, coords, clima_atual, rows, cols, area_total_ha
            )
            alertas_inteligentes = gerar_alertas_inteligentes(
                metricas_globais, riscos, clima_atual, localized_risks
            )
            plano_acao = montar_plano_acao_prioritario(
                analises, localized_risks, clima_atual, propriedade_ativa
            )
            risk_geo_points = estimar_coordenadas_talhoes_risco(localized_risks, propriedade_ativa)

            render_analysis_progress(stage_box, progress_bar, 88, "Preparando painel", "Organizando mapa, histórico e recomendações visuais.")
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
                    coords,
                    localized_risks,
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
                "localized_risks": localized_risks,
                "metricas_globais": metricas_globais,
                "clima_atual": clima_atual,
                "origem_imagem": origem_imagem,
                "analysis_id": analysis_id,
                "resumo_executivo": resumo_executivo,
                "propriedade_ativa": propriedade_ativa,
                "resultado_geral": resultado_geral,
                "alertas_inteligentes": alertas_inteligentes,
                "plano_acao": plano_acao,
                "risk_geo_points": risk_geo_points,
            })

            render_analysis_progress(stage_box, progress_bar, 100, "Análise concluída", "O painel foi atualizado com os novos resultados.")
            time.sleep(0.10)
            progress_bar.empty()
            stage_box.empty()
            if hasattr(st, "toast"):
                st.toast("Análise concluída e painel atualizado.", icon="✅")

            if analysis_id:
                st.success(f"Análise #{analysis_id} salva no histórico da propriedade.")
            else:
                st.info("Análise avulsa gerada com sucesso. Selecione uma propriedade apenas quando quiser salvar histórico e comparar evolução.")

# ── Exibir resultados salvos na sessão ───────────────────────────
st.markdown("<div id='secao-resultados'></div>", unsafe_allow_html=True)
if "analises" in st.session_state:
    analises = st.session_state["analises"]
    _coords = st.session_state["coords"]
    _rows = st.session_state["rows"]
    _cols = st.session_state["cols"]
    _img = st.session_state["img_orig"]
    riscos = st.session_state.get("riscos", [])
    metricas_salvas = st.session_state.get("metricas_globais", {})
    alertas_inteligentes = st.session_state.get("alertas_inteligentes", [])
    plano_acao = st.session_state.get("plano_acao", [])
    risk_geo_points = st.session_state.get("risk_geo_points", [])

    if executive_mode:
        st.markdown(
            """
            <div class='agroscan-executive-banner'>
                <strong>🎤 Apresentação executiva pronta</strong><br/>
                O foco visual fica no resumo, nos riscos priorizados e no plano de ação. A animação operacional fica disponível sob demanda.
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("🚁 Ver animação operacional dos grids", expanded=False):
            html_anim = drone_animation_component(_img, analises, _coords, _rows, _cols)
            iframe_h = int(_img.shape[0]) + 40
            st.components.v1.html(html_anim, height=iframe_h)
    else:
        typewriter("🚁 Drone em operação", tag="h3")
        html_anim = drone_animation_component(_img, analises, _coords, _rows, _cols)
        iframe_h = int(_img.shape[0]) + 40
        st.components.v1.html(html_anim, height=iframe_h)

    st.markdown("<h3 class='fade-slide'>🧾 Resumo Executivo</h3>", unsafe_allow_html=True)
    resumo_cards = [
        f"<div class='agroscan-card accent-success'>{linha}</div>"
        for linha in st.session_state.get("resumo_executivo", [])
    ]
    if resumo_cards:
        st.markdown(fade_slide_wrap(resumo_cards), unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🚨 Alertas ativos", len(alertas_inteligentes))
    k2.metric("🔴 Riscos altos", sum(1 for item in riscos if item.get("nivel") == "Alto"))
    k3.metric("🗺️ Talhões mapeados", len(risk_geo_points))
    k4.metric("📋 Ações priorizadas", len(plano_acao))

    st.markdown("<h3 class='fade-slide'>🚨 Alertas Inteligentes</h3>", unsafe_allow_html=True)
    if alertas_inteligentes:
        for alerta in alertas_inteligentes:
            nivel_ui = {"Alto": "high", "Médio": "medium", "Baixo": "low"}.get(alerta.get("nivel"), "info")
            render_priority_notice(
                f"{alerta['titulo']} — {alerta['nivel']}",
                f"<small>{alerta['janela']}</small><br/>{alerta['descricao']}<br/><strong>Ação recomendada:</strong> {alerta['acao']}",
                level=nivel_ui,
            )
    else:
        render_priority_notice(
            "Monitoramento preventivo",
            "Sem alertas críticos no momento; mantenha apenas o acompanhamento preventivo da área.",
            level="low",
        )

    st.markdown("<h3 class='fade-slide'>📈 Score Agrícola por Grid</h3>", unsafe_allow_html=True)
    score_medio = round(sum(d["score"] for d in analises.values()) / len(analises), 1)
    st.metric("Score médio do terreno", f"{score_medio}/100")

    barras = []
    for nome, d in list(analises.items())[:16]:
        cor = "#22a622" if d["score"] >= 60 else "#d4a017" if d["score"] >= 35 else "#b03030"
        barras.append(
            f"<div class='agroscan-score-row'>"
            f"<span class='agroscan-score-label'>{nome}</span>"
            f"<div class='agroscan-score-track'>"
            f"<div style='width:{d['score']}%;background:{cor};border-radius:999px;height:14px'></div></div>"
            f"<span style='font-size:12px;min-width:140px'>{d['tipo']} {d['score']}/100</span>"
            f"</div>"
        )
    st.markdown(fade_slide_wrap(barras), unsafe_allow_html=True)

    st.markdown("<h3 class='fade-slide'>⚠️ Riscos Locais</h3>", unsafe_allow_html=True)
    if riscos:
        cards_risco = []
        for risco in riscos:
            borda = "#b03030" if risco["nivel"] == "Alto" else "#d4a017"
            classe_risco = "accent-danger" if risco["nivel"] == "Alto" else "accent-warn"
            cards_risco.append(
                f"<div class='agroscan-card {classe_risco}'>"
                f"<strong>{risco['titulo']} — {risco['nivel']}</strong><br/>{risco['descricao']}<br/>"
                f"<strong>Ação:</strong> {risco['acao']}</div>"
            )
        st.markdown(fade_slide_wrap(cards_risco), unsafe_allow_html=True)
    else:
        st.success("Nenhum risco crítico foi identificado na análise automática atual.")

    st.markdown("<h3 class='fade-slide'>🛰️ Localização das Áreas de Risco</h3>", unsafe_allow_html=True)
    riscos_localizados = st.session_state.get("localized_risks", [])
    if riscos_localizados:
        filtro_niveis = st.multiselect(
            "Filtrar áreas por nível de risco",
            ["Alto", "Médio", "Baixo"],
            default=["Alto", "Médio", "Baixo"],
            key="risk_level_filter",
        )
        riscos_filtrados = [
            item for item in riscos_localizados
            if not filtro_niveis or item.get("nivel") in filtro_niveis
        ]

        rf1, rf2, rf3 = st.columns(3)
        rf1.metric("🔴 Alto", sum(1 for item in riscos_filtrados if item.get("nivel") == "Alto"))
        rf2.metric("🟠 Médio", sum(1 for item in riscos_filtrados if item.get("nivel") == "Médio"))
        rf3.metric("🟢 Baixo", sum(1 for item in riscos_filtrados if item.get("nivel") == "Baixo"))

        render_floating_grid_legend()
        html_risk_map = interactive_risk_map_component(_img, riscos_filtrados, _rows, _cols)
        st.components.v1.html(html_risk_map, height=int(_img.shape[0]) + 210)
        pontos_tabela = []
        grids_vistos = set()
        for item in riscos_filtrados:
            grid_nome = item.get("grid")
            if grid_nome in grids_vistos:
                continue
            grids_vistos.add(grid_nome)
            pontos_tabela.append({
                "Ponto": len(pontos_tabela) + 1,
                "Grid": grid_nome,
                "Localização": item["localizacao"],
                "Área estimada": f"{item['area_ha']} ha" if item.get("area_ha") is not None else f"{item.get('area_pct', 0)}%",
                "Risco": item["titulo"],
                "Nível": item["nivel"],
                "Gatilho climático": item["gatilho_clima"],
            })
            if len(pontos_tabela) >= 12:
                break
        locais_df = pd.DataFrame(pontos_tabela)
        st.dataframe(locais_df, use_container_width=True, hide_index=True)

        if risk_geo_points:
            st.markdown("#### 🗺️ Talhões críticos no mapa estimado da fazenda")
            st.caption("Posições estimadas a partir do centro da propriedade e da malha de grids. Com drone georreferenciado, elas ficam precisas em latitude/longitude.")
            geo_df = pd.DataFrame([
                {
                    "lat": item["lat"],
                    "lon": item["lon"],
                    "Grid": item["grid"],
                    "Nível": item["nivel"],
                    "Risco": item["risco"],
                    "Setor": item["localizacao"],
                }
                for item in risk_geo_points
            ])
            geo_df["label"] = geo_df["Grid"] + " — " + geo_df["Risco"]
            geo_df["info"] = "Nível: " + geo_df["Nível"] + "<br/>Setor: " + geo_df["Setor"]
            geo_df["radius"] = geo_df["Nível"].map({"Alto": 180, "Médio": 140, "Baixo": 110}).fillna(120)
            geo_df["color"] = geo_df["Nível"].map({
                "Alto": [255, 92, 92, 210],
                "Médio": [255, 176, 32, 205],
                "Baixo": [57, 217, 138, 195],
            })
            st.caption("Mapa interativo de risco — os pontos agora ficam mais estáveis e detalhados, evitando o erro visual do componente antigo.")
            render_interactive_geo_map(
                geo_df,
                center_lat=float(geo_df["lat"].mean()),
                center_lon=float(geo_df["lon"].mean()),
                zoom=13.2,
            )
            geo_label_df = geo_df[["Grid", "Nível", "Risco", "Setor"]].copy()
            st.dataframe(geo_label_df, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum ponto crítico precisou ser marcado interativamente nesta leitura do terreno.")

    st.markdown("<h3 class='fade-slide'>🧭 Plano de Ação Prioritário</h3>", unsafe_allow_html=True)
    if plano_acao:
        cards_plano = []
        for etapa in plano_acao:
            cards_plano.append(
                f"<div class='agroscan-card accent-success'>"
                f"<strong>Prioridade {etapa['prioridade']} — {etapa['objetivo']}</strong><br/>"
                f"📍 <strong>Local:</strong> {etapa['grid']} / {etapa['setor']}<br/>"
                f"⏱️ <strong>Prazo:</strong> {etapa['prazo']}<br/>"
                f"🛠️ <strong>Ação:</strong> {etapa['acao']}<br/>"
                f"📈 <strong>Impacto esperado:</strong> {etapa['impacto']}</div>"
            )
        st.markdown(fade_slide_wrap(cards_plano), unsafe_allow_html=True)
    else:
        st.info("Ainda não há ações priorizadas para esta leitura.")

    st.markdown("<h3 class='fade-slide'>📝 Aprendizado com Feedback do Usuário</h3>", unsafe_allow_html=True)
    with st.form("feedback_form", clear_on_submit=True):
        fb1, fb2 = st.columns(2)
        with fb1:
            grid_feedback = st.selectbox("Grid avaliado", list(analises.keys()))
        with fb2:
            rotulo_feedback = st.selectbox(
                "Classificação correta observada em campo",
                list(LABEL_PT.keys()),
                format_func=lambda chave: LABEL_PT.get(chave, chave),
            )
        risco_feedback = st.selectbox(
            "Status do risco em campo",
            ["Confirmado", "Parcial", "Falso positivo", "Sem risco"],
        )
        obs_feedback = st.text_area("Observação do operador", placeholder="Ex.: o grid estava mais úmido após chuva da madrugada.")
        enviar_feedback = st.form_submit_button("Salvar feedback e treinar o modelo", use_container_width=True)

    if enviar_feedback:
        ok, msg = registrar_feedback_usuario(
            current_user["id"],
            (st.session_state.get("propriedade_ativa") or propriedade_ativa or {}).get("id"),
            st.session_state.get("analysis_id"),
            grid_feedback,
            analises.get(grid_feedback, {}),
            rotulo_feedback,
            risco_feedback,
            obs_feedback,
        )
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

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
                        f"<div class='agroscan-card accent-success'>"
                        f"<strong>Opção {num} — {op['perfil']}</strong><br/>"
                        f"🌱 <strong>Culturas:</strong> {op['culturas']}<br/>"
                        f"🌿 <strong>Plantio:</strong> {op['plantio']}<br/>"
                        f"💧 <strong>Irrigação:</strong> {op['irrigacao']}<br/>"
                        f"⚙️ <strong>Ação:</strong> {op['acao']}"
                        f"</div>"
                    )
                st.markdown(fade_slide_wrap(opcoes_html), unsafe_allow_html=True)
                st.markdown(
                    f"<small class='agroscan-muted'>NDVI: {d['ndvi']} | Textura: {d['textura']} | Umidade: {d['umidade_frac']} | Modelo: {d.get('origem_modelo', 'Heurístico')}</small>",
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
                    "ibge": ibge_info,
                    "clima": st.session_state.get("clima_atual", {}),
                    "riscos": st.session_state.get("riscos", []),
                    "localized_risks": st.session_state.get("localized_risks", []),
                    "resumo_executivo": st.session_state.get("resumo_executivo", []),
                    "alertas_inteligentes": st.session_state.get("alertas_inteligentes", []),
                    "plano_acao": st.session_state.get("plano_acao", []),
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
st.markdown("<div id='secao-historico'></div>", unsafe_allow_html=True)
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
        timeline_df = hist_df.copy()
        timeline_df["Data"] = pd.to_datetime(timeline_df["Data"])
        timeline_df = timeline_df.sort_values("Data")
        th1, th2, th3 = st.columns(3)
        th1.metric("🏆 Melhor score histórico", f"{timeline_df['Score'].max():.1f}/100")
        th2.metric("🌿 Último NDVI", f"{timeline_df['NDVI'].iloc[-1]:.3f}")
        th3.metric("📈 Evolução do score", f"{timeline_df['Score'].iloc[-1] - timeline_df['Score'].iloc[0]:+.1f}")
        st.line_chart(timeline_df.set_index("Data")[["Score", "NDVI"]], use_container_width=True)
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

                    img_a_bgr = carregar_imagem_bgr(analise_a.get("image_path"))
                    img_b_bgr = carregar_imagem_bgr(analise_b.get("image_path"))
                    mapa_a_bgr = carregar_imagem_bgr(analise_a.get("map_path"))
                    mapa_b_bgr = carregar_imagem_bgr(analise_b.get("map_path"))

                    st.markdown("#### 🎚️ Comparação premium antes x depois")
                    if img_a_bgr is not None and img_b_bgr is not None:
                        tab_cmp_img, tab_cmp_ndvi = st.tabs(["📷 Imagem do terreno", "🗺️ Mapa NDVI"])
                        with tab_cmp_img:
                            slider_comparacao(
                                img_a_bgr,
                                img_b_bgr,
                                50,
                                left_label=f"Antes • #{analise_a['id']}",
                                right_label=f"Depois • #{analise_b['id']}",
                                component_key=f"historico_img_{analise_a['id']}_{analise_b['id']}",
                            )
                        with tab_cmp_ndvi:
                            if mapa_a_bgr is not None and mapa_b_bgr is not None:
                                slider_comparacao(
                                    mapa_a_bgr,
                                    mapa_b_bgr,
                                    50,
                                    left_label=f"NDVI antes • #{analise_a['id']}",
                                    right_label=f"NDVI depois • #{analise_b['id']}",
                                    component_key=f"historico_ndvi_{analise_a['id']}_{analise_b['id']}",
                                )
                            else:
                                render_priority_notice(
                                    "Mapas NDVI indisponíveis",
                                    "Algumas análises antigas ainda não possuem mapa salvo para esta comparação premium.",
                                    level="info",
                                )
                    else:
                        render_priority_notice(
                            "Comparação visual indisponível",
                            "Não foi possível carregar as imagens históricas desta propriedade para o slider premium.",
                            level="medium",
                        )

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
            render_priority_notice(
                "Histórico insuficiente",
                "Salve pelo menos 2 análises desta propriedade para liberar a comparação temporal completa.",
                level="info",
            )
    else:
        render_priority_notice(
            "Sem histórico salvo",
            "Ainda não há análises salvas para esta propriedade. Faça uma leitura e salve para acompanhar a evolução.",
            level="info",
        )
else:
    render_priority_notice(
        "Comparação temporal desativada",
        "Com 'Nenhuma' selecionado, a análise fica em modo avulso e sem comparação antes/depois. Escolha uma propriedade cadastrada quando quiser histórico completo.",
        level="info",
    )
