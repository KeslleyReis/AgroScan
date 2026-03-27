# python -m streamlit run AgroScan.py
# Dependências: pip install streamlit opencv-python-headless numpy anthropic fpdf2

import streamlit as st
import cv2
import numpy as np
import base64
import json
import io
import time

# ═══════════════════════════════════════════════════════════════
# CONSTANTES GLOBAIS
# ═══════════════════════════════════════════════════════════════

# Tamanho máximo de exibição em pixels — imagens maiores são redimensionadas
MAX_W = 720
MAX_H = 480

# Cores do mapa de terreno em BGR (OpenCV)
CORES_BGR = {
    "Fertil":  (34,  139, 34),
    "Umido":   (139, 90,  0),
    "Seco":    (30,  144, 200),
    "Arenoso": (80,  150, 210),
    "Rochoso": (80,  80,  80),
}

LABEL_PT = {
    "Fertil":  "Fertil",
    "Umido":   "Umido",
    "Seco":    "Seco",
    "Arenoso": "Arenoso",
    "Rochoso": "Rochoso",
}

# Limiares HSV fixos — nao expostos ao usuario
_VERDE   = dict(lo=(35, 40,  40),  hi=(85,  255, 255))
_AMARELO = dict(lo=(20, 100, 100), hi=(35,  255, 255))
_AZUL    = dict(lo=(90, 50,  50),  hi=(130, 255, 255))
_CINZA   = dict(lo=(0,  0,   50),  hi=(180, 50,  200))


# ═══════════════════════════════════════════════════════════════
# UTILITARIOS DE IMAGEM
# ═══════════════════════════════════════════════════════════════

def redimensionar(img: np.ndarray, max_w=MAX_W, max_h=MAX_H) -> np.ndarray:
    """Reduz a imagem para caber em max_w x max_h mantendo proporcao.
    Nunca amplia a imagem — se ela ja couber, retorna sem alteracao."""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return img
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def img_to_base64(img: np.ndarray, quality=85) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def numpy_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode(".png", img_rgb)
    return bytes(buf)


# ═══════════════════════════════════════════════════════════════
# MAPA DE TERRENO  (substitui o mapa de calor NDVI)
# ═══════════════════════════════════════════════════════════════

def gerar_mapa_terreno(img: np.ndarray) -> np.ndarray:
    """
    Mapa padrao de vegetacao usando NDVI aproximado com colormap TURBO.
    Roxo/azul = sem vegetacao | Verde = vegetacao saudavel | Amarelo/vermelho = stress.
    Simples, legivel e padrao na area de sensoriamento remoto.
    """
    b_f, g_f, r_f = cv2.split(img.astype("float32"))
    ndvi = (g_f - r_f) / (g_f + r_f + 1e-5)
    ndvi_norm = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mapa = cv2.applyColorMap(ndvi_norm, cv2.COLORMAP_VIRIDIS)
    return mapa


# ═══════════════════════════════════════════════════════════════
# ANALISE DETALHADA POR REGIAO
# ═══════════════════════════════════════════════════════════════

def analisar_regiao(img_area: np.ndarray) -> dict:
    """
    Retorna dict com: tipo, ndvi, textura, umidade_frac,
    score (0-100), culturas, irrigacao_mm_dia
    """
    b, g, r = cv2.split(img_area.astype("float32"))
    eps = 1e-5

    ndvi = float(np.mean((g - r) / (g + r + eps)))

    gray     = cv2.cvtColor(img_area, cv2.COLOR_BGR2GRAY)
    textura  = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_64F))))

    umidade_frac = float(np.mean(b > g + 10))

    hsv_area = cv2.cvtColor(img_area, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv_area[:, :, 1])) / 255.0

    # Classificacao
    if ndvi > 0.30:
        tipo = "Fertil"
    elif umidade_frac > 0.35:
        tipo = "Umido"
    elif textura > 45:
        tipo = "Rochoso"
    elif ndvi < 0.08 and sat_mean < 0.25:
        tipo = "Seco"
    else:
        tipo = "Arenoso"

    # Score agricola (0 = pessimo, 100 = excelente)
    s_ndvi    = max(0.0, min(1.0, (ndvi + 0.2) / 0.7)) * 40
    s_umidade = (1.0 - abs(umidade_frac - 0.25) / 0.25) * 25
    s_textura = max(0.0, 1.0 - textura / 80.0) * 20
    s_sat     = sat_mean * 15
    score = int(min(100, max(0, s_ndvi + s_umidade + s_textura + s_sat)))

    culturas_map = {
        "Fertil":  ["Soja", "Milho", "Feijao", "Cana-de-acucar"],
        "Umido":   ["Arroz irrigado", "Banana", "Coco", "Taro"],
        "Seco":    ["Sorgo", "Mandioca", "Palma forrageira", "Sisal"],
        "Arenoso": ["Amendoim", "Batata-doce", "Cenoura", "Melancia"],
        "Rochoso": ["Sem uso agricola direto"],
    }
    irrig_map = {
        "Fertil":  4.0,
        "Umido":   1.5,
        "Seco":    8.0,
        "Arenoso": 6.0,
        "Rochoso": 0.0,
    }

    return {
        "tipo":         tipo,
        "ndvi":         round(ndvi, 3),
        "textura":      round(textura, 1),
        "umidade_frac": round(umidade_frac, 3),
        "score":        score,
        "culturas":     culturas_map.get(tipo, ["Indefinido"]),
        "irrigacao_mm": irrig_map.get(tipo, 5.0),
    }


# ═══════════════════════════════════════════════════════════════
# RECOMENDACOES TEXTUAIS (sem IA)
# ═══════════════════════════════════════════════════════════════

def recomendacao_local(dados: dict) -> dict:
    """
    Retorna 3 recomendacoes distintas para o tipo de terreno.
    Cada opcao tem um perfil diferente: conservador, intermediario e intensivo.
    """
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
                "irrigacao": f"~{irr} mm/dia fracionado em 3x ao dia. Solo arenoso nao retém agua.",
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
# ANALISE COM IA (Claude)
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
# SLIDER COMPARATIVO  original <-> mapa
# ═══════════════════════════════════════════════════════════════

def slider_comparacao(img_orig: np.ndarray, img_mapa: np.ndarray, initial=50):
    h, w   = img_orig.shape[:2]
    b64o   = img_to_base64(img_orig)
    b64m   = img_to_base64(img_mapa)
    asp    = round((h / w) * 100, 2)
    inv    = 100 - initial

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
# ANIMACAO DO DRONE  (100% JS — zero loop Python)
# ═══════════════════════════════════════════════════════════════

def drone_animation_component(img: np.ndarray, analises: dict,
                               coords: dict, rows: int, cols: int) -> str:
    h, w    = img.shape[:2]
    img_b64 = img_to_base64(img, quality=85)

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
# EXPORTAR PDF
# ═══════════════════════════════════════════════════════════════

def gerar_pdf(img_orig: np.ndarray, img_mapa: np.ndarray,
              analises: dict, rec_texto) -> io.BytesIO | None:
    try:
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos

        def cell(pdf, w, h, txt, border=0, fill=False, align="L", newline=False):
            """Wrapper compatível com fpdf2 2.x — sem ln=True."""
            nx = XPos.LMARGIN if newline else XPos.RIGHT
            ny = YPos.NEXT    if newline else YPos.TOP
            pdf.cell(w, h, txt, border=border, fill=fill, align=align,
                     new_x=nx, new_y=ny)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Título
        pdf.set_font("Helvetica", "B", 17)
        pdf.set_text_color(34, 120, 34)
        cell(pdf, 0, 11, "AgroScan Drone - Relatorio de Analise",
             align="C", newline=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100, 100, 100)
        cell(pdf, 0, 5, f"Gerado em {time.strftime('%d/%m/%Y as %H:%M')}",
             align="C", newline=True)
        pdf.ln(4)

        # Imagens via BytesIO — sem disco
        ob = numpy_to_png_bytes(img_orig)
        mb = numpy_to_png_bytes(img_mapa)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(50, 50, 50)
        cell(pdf, 95, 6, "Imagem Original", align="C")
        cell(pdf, 95, 6, "Mapa de Terreno",  align="C", newline=True)
        y0 = pdf.get_y()
        pdf.image(io.BytesIO(ob), x=10,  y=y0, w=90, h=58)
        pdf.image(io.BytesIO(mb), x=108, y=y0, w=90, h=58)
        pdf.ln(63)

        # Tabela de grids
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(34, 120, 34)
        cell(pdf, 0, 7, "Resultado por Grid", newline=True)

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(255, 255, 255)
        pdf.set_fill_color(34, 120, 34)
        colunas = [("Grid", 42), ("Tipo", 38), ("Score", 22),
                   ("Irrig.mm/d", 35), ("Culturas", 58)]
        for i, (h_txt, w_col) in enumerate(colunas):
            ultimo = i == len(colunas) - 1
            cell(pdf, w_col, 6, h_txt, border=1, fill=True,
                 align="C", newline=ultimo)

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(30, 30, 30)
        for i, (nome, d) in enumerate(analises.items()):
            fill = i % 2 == 0
            pdf.set_fill_color(240, 248, 240) if fill else pdf.set_fill_color(255, 255, 255)
            cult = ", ".join(d["culturas"][:2])
            linha = [
                (nome,                   42),
                (d["tipo"],              38),
                (str(d["score"]),        22),
                (str(d["irrigacao_mm"]), 35),
                (cult,                   58),
            ]
            for j, (txt, w_col) in enumerate(linha):
                ultimo = j == len(linha) - 1
                cell(pdf, w_col, 5, txt, border=1, fill=fill, newline=ultimo)

        pdf.ln(5)

        # Recomendações
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(34, 120, 34)
        cell(pdf, 0, 7, "Recomendacoes", newline=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(30, 30, 30)

        def safe(txt):
            return txt.encode("latin-1", errors="replace").decode("latin-1")

        if isinstance(rec_texto, str):
            for linha in rec_texto.split("\n"):
                pdf.multi_cell(0, 4, safe(linha),
                               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            for nome, d in analises.items():
                rec = recomendacao_local(d)
                pdf.set_font("Helvetica", "B", 10)
                cell(pdf, 0, 6,
                     safe(f"{nome} - {d['tipo']} (Score {d['score']}/100)"),
                     newline=True)
                for chave_op in ("opcao_1", "opcao_2", "opcao_3"):
                    op  = rec[chave_op]
                    num = chave_op.split("_")[1]
                    pdf.set_font("Helvetica", "B", 9)
                    cell(pdf, 0, 5, safe(f"  Opcao {num} — {op['perfil']}"), newline=True)
                    pdf.set_font("Helvetica", "", 8)
                    for rotulo, chave in [
                        ("Culturas",  "culturas"),
                        ("Plantio",   "plantio"),
                        ("Irrigacao", "irrigacao"),
                        ("Acao",      "acao"),
                    ]:
                        pdf.multi_cell(0, 4,
                                       safe(f"    {rotulo}: {op[chave]}"),
                                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(3)

        buf = io.BytesIO()
        raw = pdf.output()
        buf.write(raw if isinstance(raw, bytes) else bytes(raw))
        buf.seek(0)
        return buf

    except ImportError:
        st.error("Instale fpdf2:  pip install fpdf2")
        return None
    except Exception as e:
        st.error(f"Erro ao gerar PDF: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# INTERFACE  STREAMLIT
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title="AgroScan Drone", page_icon="🌱", layout="wide")
st.title("🌱 AgroScan Drone — Análise e Recomendação de Terreno")

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuracoes")
    st.subheader("Analise com IA")
    usar_ia = st.toggle("Ativar analise com Claude IA", value=False)
    api_key = ""
    if usar_ia:
        st.caption(
            "Chave no formato sk-ant-...  \n"
            "Obtenha em console.anthropic.com  \n"
            "Cada analise consome ~$0.01."
        )
        api_key = st.text_input("Chave API Anthropic", type="password")
    st.divider()
    st.caption("AgroScan v3.0 — analise por drone simulado")

# ── Upload ────────────────────────────────────────────────────────
uploaded = st.file_uploader("Envie a foto do terreno", type=["jpg", "jpeg", "png"])

if uploaded:
    raw      = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img_full = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    # Redimensiona para tamanho fixo — resolve o problema das imagens gigantes
    img    = redimensionar(img_full, MAX_W, MAX_H)
    H, W   = img.shape[:2]
    hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Exibe a imagem centralizada com HTML puro (st.columns nao centraliza em layout wide)
    DISPLAY_W = min(W, 380)
    img_b64_upload = img_to_base64(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    st.markdown(
        f"<div style='display:flex;flex-direction:column;align-items:center;margin:0 0 12px'>"
        f"<img src='data:image/jpeg;base64,{img_b64_upload}' "
        f"style='width:{DISPLAY_W}px;border-radius:8px;'/>"
        f"<span style='font-size:12px;color:#888;margin-top:4px'>"
        f"Imagem carregada ({W} x {H} px)</span></div>",
        unsafe_allow_html=True,
    )

    # ── Metricas globais ─────────────────────────────────────────
    b_f, g_f, r_f = cv2.split(img.astype("float32"))
    ndvi_global   = float(np.mean((g_f - r_f) / (g_f + r_f + 1e-5)))

    total     = H * W
    p_verde   = np.sum(cv2.inRange(hsv, _VERDE["lo"],   _VERDE["hi"])   > 0) / total
    p_amarelo = np.sum(cv2.inRange(hsv, _AMARELO["lo"], _AMARELO["hi"]) > 0) / total
    p_cinza   = np.sum(cv2.inRange(hsv, _CINZA["lo"],   _CINZA["hi"])   > 0) / total
    p_azul    = np.sum(cv2.inRange(hsv, _AZUL["lo"],    _AZUL["hi"])    > 0) / total

    if   p_verde   > 0.4: resultado_geral = "Terreno predominantemente fertil"
    elif p_amarelo > 0.3: resultado_geral = "Terreno predominantemente seco"
    elif p_cinza   > 0.3: resultado_geral = "Terreno predominantemente rochoso"
    elif p_azul    > 0.2: resultado_geral = "Terreno predominantemente umido"
    else:                 resultado_geral = "Terreno predominantemente arenoso"

    st.subheader("📊 Resultado Geral")
    st.success(resultado_geral)
    st.metric("NDVI medio (saude da vegetacao)", f"{round(ndvi_global * 100, 1)}%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fertil",  f"{round(p_verde   * 100, 1)}%")
    c2.metric("Seco",    f"{round(p_amarelo * 100, 1)}%")
    c3.metric("Rochoso", f"{round(p_cinza   * 100, 1)}%")
    c4.metric("Umido",   f"{round(p_azul    * 100, 1)}%")

    # ── Mapa de terreno ──────────────────────────────────────────
    mapa = gerar_mapa_terreno(img)

    # ── Slider comparacao ────────────────────────────────────────
    st.subheader("🗺️ Mapa de Identificação")
    if "cmp_pct" not in st.session_state:
        st.session_state.cmp_pct = 50

    # Botões centralizados em linha compacta usando HTML + st.columns internas
    slider_w = min(W, MAX_W)

    st.markdown(
        "<div style='display:flex;justify-content:center;gap:8px;margin:0 0 8px'>",
        unsafe_allow_html=True,
    )
    bc1, bc2, bc3 = st.columns([1, 1, 1])
    with bc1:
        if st.button("Reset 50%",     use_container_width=True): st.session_state.cmp_pct = 50
    with bc2:
        if st.button("Original",      use_container_width=True): st.session_state.cmp_pct = 0
    with bc3:
        if st.button("Mapa completo", use_container_width=True): st.session_state.cmp_pct = 100
    st.markdown("</div>", unsafe_allow_html=True)

    slider_comparacao(img, mapa, st.session_state.cmp_pct)

    # Legenda vertical — itens empilhados, não espalhados
    legenda_html = "<div style='display:flex;flex-direction:column;gap:6px;margin:8px 0 16px'>"
    for chave, cor_bgr in CORES_BGR.items():
        cor_rgb = (cor_bgr[2], cor_bgr[1], cor_bgr[0])
        label   = LABEL_PT[chave]
        legenda_html += (
            f"<div style='display:flex;align-items:center;gap:8px'>"
            f"<div style='width:16px;height:16px;background:rgb{cor_rgb};"
            f"border-radius:3px;flex-shrink:0'></div>"
            f"<span style='font-size:13px'>{label}</span></div>"
        )
    legenda_html += "</div>"
    st.markdown(legenda_html, unsafe_allow_html=True)

    # ── Selecao de grids ─────────────────────────────────────────
    st.subheader("🔲 Análise por Regiões")
    num_grids = st.radio("Numero de grids:", [4, 8, 16, 32], horizontal=True)
    grid_map  = {4: (2, 2), 8: (2, 4), 16: (4, 4), 32: (4, 8)}
    rows, cols = grid_map[num_grids]

    coords: dict = {}
    for i in range(rows):
        for j in range(cols):
            nome = f"Grid {i * cols + j + 1}"
            ys_, ye_ = i * (H // rows), (i + 1) * (H // rows)
            xs_, xe_ = j * (W // cols), (j + 1) * (W // cols)
            coords[nome] = (xs_, xe_, ys_, ye_)

    # ── Iniciar analise ──────────────────────────────────────────
    if st.button("Iniciar analise de grids", type="primary"):
        analises: dict = {}
        for nome, (xs_, xe_, ys_, ye_) in coords.items():
            analises[nome] = analisar_regiao(img[ys_:ye_, xs_:xe_])

        st.session_state.update({
            "analises": analises,
            "coords":   coords,
            "rows":     rows,
            "cols":     cols,
            "img_orig": img,
            "img_mapa": mapa,
        })

    # ── Exibir resultados ────────────────────────────────────────
    if "analises" in st.session_state:
        analises = st.session_state["analises"]
        _coords  = st.session_state["coords"]
        _rows    = st.session_state["rows"]
        _cols    = st.session_state["cols"]
        _img     = st.session_state["img_orig"]

        # Animacao drone
        st.subheader("🚁 Drone em operação")
        html_anim = drone_animation_component(_img, analises, _coords, _rows, _cols)
        # Altura do iframe = altura real do canvas renderizado na largura disponível
        # A largura visível é min(W, largura_da_coluna). Usamos W como referência.
        iframe_h = int(_img.shape[0]) + 40
        st.components.v1.html(html_anim, height=iframe_h)

        # Score agricola visual
        st.subheader("📈 Score Agricola por Grid")
        score_medio = round(sum(d["score"] for d in analises.values()) / len(analises), 1)
        st.metric("Score medio do terreno", f"{score_medio}/100")

        for nome, d in list(analises.items())[:16]:
            cor = "#22a622" if d["score"] >= 60 else "#d4a017" if d["score"] >= 35 else "#b03030"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;margin:3px 0'>"
                f"<span style='min-width:70px;font-size:13px'>{nome}</span>"
                f"<div style='flex:1;background:#333;border-radius:4px;height:14px'>"
                f"<div style='width:{d['score']}%;background:{cor};"
                f"border-radius:4px;height:14px'></div></div>"
                f"<span style='font-size:12px;min-width:100px'>"
                f"{d['tipo']} {d['score']}/100</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Recomendacoes — 3 opcoes por grid
        st.subheader("Recomendações por Grid")
        if usar_ia and api_key:
            with st.spinner("Consultando Claude IA..."):
                texto_ia = analisar_com_ia(_img, analises, api_key)
            st.session_state["rec_ia"] = texto_ia
            st.markdown(texto_ia)
        else:
            for nome, d in analises.items():
                rec = recomendacao_local(d)
                with st.expander(
                    f"{nome} — {d['tipo']} | Score: {d['score']}/100 | "
                    f"Irrigacao: {d['irrigacao_mm']} mm/dia",
                    expanded=False,
                ):
                    for chave_op in ("opcao_1", "opcao_2", "opcao_3"):
                        op = rec[chave_op]
                        num = chave_op.split("_")[1]
                        st.markdown(
                            f"<div style='border-left:3px solid #22a622;padding:6px 12px;"
                            f"margin:8px 0;border-radius:0 6px 6px 0;background:#1a2e1a'>"
                            f"<strong>Opcao {num} — {op['perfil']}</strong><br/>"
                            f"🌱 <strong>Culturas:</strong> {op['culturas']}<br/>"
                            f"🌿 <strong>Plantio:</strong> {op['plantio']}<br/>"
                            f"💧 <strong>Irrigacao:</strong> {op['irrigacao']}<br/>"
                            f"⚙️ <strong>Acao:</strong> {op['acao']}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f"<small style='color:#666'>NDVI: {d['ndvi']} | "
                        f"Textura: {d['textura']} | Umidade: {d['umidade_frac']}</small>",
                        unsafe_allow_html=True,
                    )
            st.session_state.pop("rec_ia", None)

        # Exportar PDF
        st.divider()
        st.subheader("📄 Exportar Relatório")
        if st.button("Gerar PDF"):
            with st.spinner("Gerando relatorio..."):
                rec_cont = st.session_state.get("rec_ia") or analises
                pdf_buf  = gerar_pdf(
                    st.session_state["img_orig"],
                    st.session_state["img_mapa"],
                    analises,
                    rec_cont,
                )
            if pdf_buf:
                st.download_button(
                    "Baixar relatorio PDF",
                    data=pdf_buf,
                    file_name=f"agroscan_{time.strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                )
