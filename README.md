# 🌱 AgroScan

Sistema interativo de análise agrícola com **Streamlit**, focado em leitura visual de imagens, análise por **grids**, apoio climático, indicadores do **IBGE** e geração de **relatórios em PDF**.

---

## ✨ Visão geral

O `AgroScan` foi projetado para apoiar análises rápidas de terreno e imagens agrícolas, oferecendo uma experiência visual moderna e prática para demonstrações, monitoramento e tomada de decisão.

Com ele, é possível:

- fazer **upload** de imagens ou captura para análise;
- executar **análise por grids** na área observada;
- visualizar **riscos localizados** e prioridades;
- cruzar informações com **clima** e contexto territorial;
- consultar dados oficiais do **IBGE**;
- gerar **relatórios em PDF**;
- usar o app em **modo claro/escuro** com interface interativa.

---

## 🚀 Funcionalidades principais

- 🖼️ **Análise por imagem** com visualização do terreno
- 🧩 **Mapeamento por grids** para leitura setorizada
- 🚨 **Alertas visuais por prioridade**
- 🌦️ **Apoio com clima/previsão do tempo**
- 🛰️ **Localização visual de áreas de risco**
- 🌎 **Integração com IBGE**
  - município oficial
  - código do município
  - extensão territorial
  - densidade populacional
- 📄 **Exportação de relatório em PDF**
- 🌗 **Dark/Light mode**
- 📱 **UX adaptada para mobile/tablet**
- 👋 **Tour guiado na primeira abertura**

---

## 🛠️ Tecnologias utilizadas

- **Python 3.10+**
- **Streamlit**
- **OpenCV**
- **NumPy**
- **Pandas**
- **Requests**
- **Anthropic**
- **ReportLab**

---

## 📂 Estrutura do projeto

```bash
AgroScan/
├── AgroScan.py
├── requirements.txt
├── README.md
└── data/
    └── images/
```

---

## 📦 Instalação

### 1) Clone o repositório

```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd AgroScan
```

### 2) Crie e ative um ambiente virtual

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Instale as dependências

```bash
pip install -r requirements.txt
```

---

## ▶️ Como executar

```bash
streamlit run AgroScan.py
```

Se preferir, também funciona com:

```bash
python -m streamlit run AgroScan.py
```

---

## 🧪 Como usar

1. Abra o app no navegador.
2. Escolha a propriedade ativa ou use análise avulsa, se disponível.
3. Envie uma imagem do terreno/cultivo.
4. Inicie a análise por grids.
5. Explore os riscos, indicadores, clima e recomendações.
6. Gere o **PDF** ao final, se desejar.

---

## 📄 Geração de PDF

Para exportar relatórios, o projeto usa `reportlab`.

Se ocorrer erro relacionado a PDF, confira se a dependência foi instalada corretamente:

```bash
pip install reportlab
```

Ou reinstale tudo:

```bash
pip install -r requirements.txt
```

---

## ⚠️ Observações

- A precisão da análise depende da **qualidade da imagem enviada**.
- Algumas funcionalidades dependem de **conexão com internet** para consultar clima e serviços oficiais.
- Para análises geográficas realmente precisas no campo, o ideal é trabalhar com imagens **georreferenciadas**.

---

## 👨‍💻 Autor

Projeto desenvolvido por **Keslley Reis / AgroScan**.

---

## 📜 Licença

- `MIT`
- `Apache-2.0`
- `Proprietária`
