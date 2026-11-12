import re
from pathlib import Path

import streamlit as st
from src.rag_engine import LocalRAGEngine


# -----------------------------------------------------------------------------
# M&M — Assistant technique local
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="M&M",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette inspired by GE HealthCare service consoles: navy, steel blue and grey.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800;900&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --mm-ink: #101f33;
        --mm-navy: #142b49;
        --mm-blue: #1769aa;
        --mm-blue-bright: #2389d1;
        --mm-paper: #f2f5f8;
        --mm-white: #ffffff;
        --mm-line: #cbd5df;
        --mm-muted: #66798d;
    }

    .stApp { background: var(--mm-paper); color: var(--mm-ink); }
    [data-testid="stHeader"] { background: rgba(242,245,248,.96); }
    [data-testid="stSidebar"] {
        background: #e7edf3;
        border-right: 1px solid var(--mm-line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }

    .mm-sidebar-brand {
        display: flex;
        align-items: center;
        gap: 13px;
        padding: 12px 8px 22px 5px;
        border-bottom: 1px solid #c5d0db;
        margin-bottom: 22px;
    }
    .mm-logo-slot {
        width: 62px;
        height: 62px;
        min-width: 62px;
        background: #1769aa;
        border: 2px solid #0f568f;
        border-radius: 3px;
        box-shadow: 4px 4px 0 rgba(20,43,73,.12);
    }
    .mm-brand-name {
        color: var(--mm-navy);
        font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif;
        font-size: 43px;
        line-height: .82;
        font-weight: 900;
        letter-spacing: -1.2px;
        margin: 0;
    }
    .mm-brand-subtitle {
        color: #58718b;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        letter-spacing: 1.5px;
        margin-top: 8px;
        text-transform: uppercase;
    }

    .mm-hero {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        padding: 25px 28px 23px;
        background: var(--mm-navy);
        color: white;
        border-radius: 3px;
        border-bottom: 5px solid var(--mm-blue-bright);
        margin: 4px 0 23px;
    }
    .mm-hero-kicker {
        color: #8fb9dc;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .mm-hero-title {
        color: #fff;
        font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif;
        font-size: clamp(64px, 9vw, 112px);
        line-height: .82;
        letter-spacing: -2px;
        font-weight: 900;
        margin: 0;
    }
    .mm-hero-subtitle {
        color: #d4e2ef;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        letter-spacing: 1.3px;
        margin-top: 13px;
        text-transform: uppercase;
    }
    .mm-author {
        color: #ffffff;
        font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif;
        font-size: clamp(22px, 3vw, 34px);
        font-weight: 700;
        letter-spacing: 1.2px;
        text-align: right;
        white-space: nowrap;
    }
    .mm-author-caption {
        color: #8fb9dc;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        letter-spacing: 1.5px;
        text-align: right;
        margin-top: 5px;
    }
    .mm-section-title {
        color: var(--mm-navy);
        font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif;
        font-size: 29px;
        font-weight: 800;
        letter-spacing: .4px;
        text-transform: uppercase;
        margin: 4px 0 2px;
    }
    .mm-section-caption {
        color: var(--mm-muted);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        letter-spacing: .5px;
        margin-bottom: 12px;
    }
    .mm-status {
        display: inline-block;
        padding: 5px 9px;
        color: #d9edff;
        background: #1a527f;
        border: 1px solid #3f83b5;
        font: 10px 'IBM Plex Mono', monospace;
        letter-spacing: 1px;
        white-space: nowrap;
    }
    .mm-result-panel {
        background: #fff;
        border: 1px solid var(--mm-line);
        border-left: 4px solid var(--mm-blue);
        border-radius: 2px;
        padding: 18px 21px;
        margin: 10px 0 18px;
    }
    .mm-result-label {
        color: var(--mm-blue);
        font: 10px 'IBM Plex Mono', monospace;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .stTextInput input, .stTextArea textarea {
        border: 1px solid #aebdcb;
        border-radius: 2px;
        background: white;
        font-size: 15px;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--mm-blue);
        box-shadow: 0 0 0 1px var(--mm-blue);
    }
    .stButton > button, [data-testid="stFormSubmitButton"] button {
        background: var(--mm-blue);
        color: white;
        border: 1px solid #10588f;
        border-radius: 2px;
        min-height: 43px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: .7px;
        text-transform: uppercase;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
        background: #10588f;
        color: white;
        border-color: #10588f;
    }
    .mm-history-item {
        border-top: 1px solid #d8e0e8;
        padding: 14px 0;
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--mm-line);
        border-radius: 2px;
        background: rgba(255,255,255,.65);
    }
    .mm-mono {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--mm-muted);
        font-size: 10px;
        letter-spacing: .6px;
    }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

IMAGES_DIR = Path("documents/PDF")

FILTER_MAP = {
    "Tous les documents": None,
    "Procédures": "t_",
    "Dépannage": "r_ts_",
    "Théorie / Architecture": "c_",
    "Diagnostics": "r_diag_",
    "Manuels PDF": ".pdf",
}


@st.cache_resource
def load_rag():
    return LocalRAGEngine()


def find_referenced_images(sources: list[dict]) -> list[Path]:
    found_images = []
    seen = set()

    for source in sources:
        matches = re.findall(
            r"\[Image référencée\s*:\s*([^\]]+)\]",
            source.get("text", ""),
        )
        for image_name in matches:
            clean_name = Path(image_name.strip().replace("\\", "/")).name
            if clean_name in seen:
                continue

            seen.add(clean_name)
            candidate = IMAGES_DIR / clean_name
            if candidate.exists() and candidate.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".gif"
            }:
                found_images.append(candidate)

    return found_images


def render_sources_and_images(sources: list[dict]):
    images = find_referenced_images(sources)

    if images:
        st.markdown("**Schémas et images associés**")
        columns = st.columns(min(len(images), 3))
        for index, image_path in enumerate(images):
            with columns[index % 3]:
                st.image(
                    str(image_path),
                    caption=image_path.name,
                    use_container_width=True,
                )

    with st.expander("Consulter les sources documentaires"):
        for source in sources:
            page = source.get("page", -1)
            page_info = f" · Page {page}" if page != -1 else ""
            st.markdown(
                f"**{source.get('source', 'Document')}**"
                f" · Section : `{source.get('section', '—')}`{page_info}"
            )
            st.code(source.get("text", ""), language="text")


# -----------------------------------------------------------------------------
# Sidebar — only the controls needed to operate the assistant
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="mm-sidebar-brand">
            <div class="mm-logo-slot" title="Emplacement réservé au logo"></div>
            <div>
                <div class="mm-brand-name">M&amp;M</div>
                <div class="mm-brand-subtitle">TECHNICAL INTELLIGENCE</div>
            </div>
        </div>
        <div class="mm-section-title">Bibliothèque</div>
        <div class="mm-section-caption">DOCUMENTATION TECHNIQUE</div>
        """,
        unsafe_allow_html=True,
    )

    doc_filter = st.selectbox(
        "Type de document",
        options=list(FILTER_MAP.keys()),
        label_visibility="collapsed",
    )

    with st.expander("Paramètres avancés"):
        top_k = st.slider(
            "Extraits récupérés",
            min_value=3,
            max_value=12,
            value=6,
            help="Nombre de sections documentaires transmises au modèle de langage.",
        )
        use_think = st.checkbox("Réflexion approfondie Qwen3", value=False)

    st.divider()
    if st.button("Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------------------------------------------------
# Main workspace
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="mm-hero">
        <div>
            <div class="mm-hero-kicker">GE HEALTHCARE · KNOWLEDGE SYSTEM</div>
            <div class="mm-hero-title">M&amp;M</div>
            <div class="mm-hero-subtitle">Technical documentation assistant</div>
        </div>
        <div>
            <div class="mm-author">AISSAM HAMIDA</div>
            <div class="mm-author-caption">ENGINEER · CREATOR</div>
            <div style="text-align:right; margin-top:13px;">
                <span class="mm-status">LOCAL RAG SYSTEM</span>
            </div>
        </div>
    </div>
    <div class="mm-section-title">Technical workspace</div>
    <div class="mm-section-caption">SEARCH · ANALYZE · DOCUMENT</div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

rag = load_rag()

# Search form: a technical console rather than a conventional chat landing page.
with st.form("mm_query_form", clear_on_submit=True):
    prompt = st.text_input(
        "REQUÊTE TECHNIQUE",
        placeholder="Ex. : Explain the gantry reset procedure on Optima CT580...",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("LANCER LA RECHERCHE", use_container_width=True)

if submitted and prompt.strip():
    st.session_state.messages.append({"role": "user", "content": prompt.strip()})

    with st.spinner("Analyse des manuels et récupération des sections pertinentes..."):
        try:
            result = rag.ask(
                query=prompt.strip(),
                top_k=top_k,
                doc_prefix=FILTER_MAP[doc_filter],
                use_think=use_think,
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                }
            )
            st.session_state.last_result_index = len(st.session_state.messages) - 1
        except Exception as error:
            st.error(f"Erreur de génération : {error}")

# Display the latest answer as a technical report panel, not chat bubbles.
if st.session_state.messages:
    last_assistant = next(
        (item for item in reversed(st.session_state.messages)
         if item["role"] == "assistant"),
        None,
    )
    last_user = next(
        (item["content"] for item in reversed(st.session_state.messages)
         if item["role"] == "user"),
        "",
    )

    if last_assistant:
        st.markdown(
            '<div class="mm-result-panel">'
            '<div class="mm-result-label">TECHNICAL RESPONSE / DERNIÈRE ANALYSE</div>',
            unsafe_allow_html=True,
        )
        st.markdown(last_assistant["content"])
        if last_assistant.get("sources"):
            render_sources_and_images(last_assistant["sources"])
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander(f"Historique des recherches ({sum(1 for m in st.session_state.messages if m['role'] == 'user')})"):
        for index in range(0, len(st.session_state.messages), 2):
            user_message = st.session_state.messages[index]
            if user_message["role"] != "user":
                continue
            st.markdown(f"**REQUÊTE {index // 2 + 1}**")
            st.write(user_message["content"])
            if index + 1 < len(st.session_state.messages):
                answer_message = st.session_state.messages[index + 1]
                if answer_message["role"] == "assistant":
                    with st.expander("Consulter la réponse"):
                        st.markdown(answer_message["content"])
                        if answer_message.get("sources"):
                            render_sources_and_images(answer_message["sources"])

