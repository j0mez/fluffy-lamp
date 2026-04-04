import streamlit as st
import pickle
import json
import re
import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components

# ============================================================
# PAGE CONFIG — must be the very first Streamlit call
# ============================================================

st.set_page_config(
    page_title="Mental Health Discourse Analyser",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# CONFIGURATION
# ============================================================

SUBREDDIT_COLOURS = {
    'Anxiety':      '#E8A838',
    'depression':   '#4A90D9',
    'mentalhealth': '#5CB85C',
    'SuicideWatch': '#D9534F',
    'lonely':       '#9B59B6'
}

SUBREDDIT_DESCRIPTIONS = {
    'Anxiety':      'Posts resembling r/Anxiety typically feature somatic symptoms, medication names, and physiological experiences of anxiety.',
    'depression':   'Posts resembling r/depression typically feature clinical terminology, medication names, and expressions of low energy and hopelessness.',
    'SuicideWatch': 'Posts resembling r/SuicideWatch typically feature acute crisis language and expressions of wanting to end life.',
    'lonely':       'Posts resembling r/lonely typically feature relational vocabulary, references to social platforms, and expressions of social isolation.',
    'mentalhealth': 'Posts resembling r/mentalhealth typically feature broader mental health topics, diagnostic language, and information-seeking.'
}

SUBREDDIT_CLINICAL = {
    'Anxiety':      'Anxiety discourse is characterised by high arousal and external focus on triggers and bodily sensations.',
    'depression':   'Depression discourse shows high self-focus (elevated first-person pronoun rates) and low arousal.',
    'SuicideWatch': '⚠️ This text resembles crisis-level discourse. If you or someone you know is in crisis, please contact a crisis line immediately.',
    'lonely':       'Loneliness discourse is characterised by low arousal and outward orientation toward connection.',
    'mentalhealth': 'r/mentalhealth functions as a general information community rather than an acute crisis space.'
}

ROOT_CAUSE_DESCRIPTIONS = {
    'Drug And Alcohol': 'The text contains linguistic patterns associated with substance use as a root cause of mental health difficulties.',
    'Early Life':       'The text contains linguistic patterns associated with early life experiences as a root cause of mental health difficulties.',
    'Personality':      'The text contains linguistic patterns associated with personality factors as a root cause of mental health difficulties.',
    'Trauma And Stress':'The text contains linguistic patterns associated with trauma and stress as root causes of mental health difficulties.'
}

DATASET_VAD = {
    'Anxiety':      {'valence': 0.084, 'arousal': -0.021, 'dominance': -0.013},
    'SuicideWatch': {'valence': 0.085, 'arousal': -0.032, 'dominance': -0.000},
    'depression':   {'valence': 0.114, 'arousal': -0.058, 'dominance': -0.002},
    'lonely':       {'valence': 0.173, 'arousal': -0.092, 'dominance': -0.009},
    'mentalhealth': {'valence': 0.128, 'arousal': -0.054, 'dominance':  0.013}
}

EXAMPLES = {
    "Select an example...": "",
    "r/Anxiety example":     "I've been having heart palpitations all day and I can't stop worrying that something is wrong with me. My hands are shaking and I feel sick. Does anyone else get this with anxiety?",
    "r/depression example":  "I haven't left my bed in three days. I just don't see the point anymore. Everything feels grey and empty and I don't know how to explain it to anyone.",
    "r/lonely example":      "Does anyone want to chat on Discord? I've been alone all weekend and I just need someone to talk to. I feel so invisible.",
    "r/SuicideWatch example":"I am so tired of fighting every single day. I don't want to die but I don't want to keep living like this either. I just want it to stop.",
    "r/mentalhealth example":"Can anyone explain the difference between bipolar disorder and borderline personality disorder? I've been reading about both and I'm not sure which one fits my experience."
}

SECTIONS = [
    {
        "title":         "1 — What Makes Each Community Distinct?",
        "heading":       "Section 1: What Makes Each Community Distinct?",
        "method":        "Method: TF-IDF  |  Disciplinary link: Communities of practice (Lave & Wenger, 1991)",
        "colour":        "#E8A838",
        "key_finding":   "r/SuicideWatch's most distinctive terms are disturbingly specific crisis vocabulary — not general distress language.",
        "plain_english": "This asks what words each community uses that the others don't — like finding each group's dialect. The results reveal that communities don't just feel different, they talk differently in ways a computer can reliably detect.",
    },
    {
        "title":         "2 — How Do They Feel Differently?",
        "heading":       "Section 2: How Do They Feel Differently?",
        "method":        "Method: VAD Affective Scoring  |  Disciplinary link: Russell's (1980) circumplex model of affect",
        "colour":        "#E8A838",
        "key_finding":   "Arousal separates communities more than valence — r/lonely is the most emotionally flat, r/Anxiety the most activated.",
        "plain_english": "Instead of just asking 'is this positive or negative?', VAD measures three things: how positive the language is, how activated or energised, and how in-control the speaker feels. This gives a richer picture than a simple sentiment score.",
    },
    {
        "title":         "3 — How Self-Focused Is Each Community?",
        "heading":       "Section 3: How Self-Focused Is Each Community?",
        "method":        "Method: Pennebaker Pronoun Analysis  |  Disciplinary link: Clinical psycholinguistics (Pennebaker, 2011)",
        "colour":        "#E8A838",
        "key_finding":   "The pronoun hierarchy — SuicideWatch highest, Anxiety lowest — is identical in 2019 and 2022, predating COVID entirely.",
        "plain_english": "Research shows that people in distress use 'I', 'me', 'my' more often. This section counts those words across communities to measure inward attentional focus — and finds a stable ranking that hasn't shifted across the pandemic.",
    },
    {
        "title":         "4 — How Similar Are They Really?",
        "heading":       "Section 4: How Similar Are They Really?",
        "method":        "Method: Sentence Embeddings and UMAP  |  Disciplinary link: Distributional semantics and cognitive linguistics",
        "colour":        "#4A90D9",
        "key_finding":   "r/depression and r/SuicideWatch are semantically nearly identical at 0.954 — despite being different communities with different stated purposes.",
        "plain_english": "Instead of comparing words, this compares meaning. Two posts can be semantically close even if they share no words. The result is a map of how similar the communities really are at the level of ideas rather than vocabulary.",
    },
    {
        "title":         "5 — Lexical vs Semantic: Two Different Pictures",
        "heading":       "Section 5: Lexical vs Semantic — Two Different Pictures",
        "method":        "Method: Representational comparison  |  TF-IDF similarity vs embedding similarity",
        "colour":        "#4A90D9",
        "key_finding":   "Every community looks more similar in semantic space than in lexical space — meaning and vocabulary consistently tell different stories.",
        "plain_english": "This compares two maps of the same communities — one built from shared vocabulary, one from shared meaning. The gap between them reveals what surface language conceals about how close these communities really are.",
    },
    {
        "title":         "6 — The Key Finding: Depression vs SuicideWatch",
        "heading":       "Section 6: The Key Finding — Depression and SuicideWatch",
        "method":        "The most important result in the analysis — stable across the entire study period",
        "colour":        "#4A90D9",
        "key_finding":   "The 0.954 similarity between r/depression and r/SuicideWatch has been stable since April 2019. It is not a COVID effect.",
        "plain_english": "This is the most important result in the analysis. No method successfully separates depressive from suicidal discourse, and this has been true for at least three years. This is a finding about the nature of these experiences, not a failure of the method.",
    },
    {
        "title":         "7 — What Are They Talking About?",
        "heading":       "Section 7: What Are They Talking About?",
        "method":        "Method: BERTopic  |  Disciplinary link: Discourse analysis",
        "colour":        "#5CB85C",
        "key_finding":   "50% of posts resist topical categorisation — mental health discourse is highly personal and resists neat clustering.",
        "plain_english": "Topic modelling asks 'what are people actually talking about?' without being told in advance. It discovers recurring themes automatically. The high proportion of posts that resist categorisation is itself a significant finding.",
    },
    {
        "title":         "8 — Can a Classifier Tell Them Apart?",
        "heading":       "Section 8: Can a Classifier Tell Them Apart?",
        "method":        "Method: Logistic Regression on sentence embeddings  |  Subreddit prediction",
        "colour":        "#D9534F",
        "key_finding":   "62.9% accuracy — more than 3x the random baseline — but r/depression is barely above chance, confirming its overlap with r/SuicideWatch.",
        "plain_english": "This tests whether a machine learning model can correctly identify which community a post came from. The mistakes the model makes are as informative as the correct predictions.",
    },
    {
        "title":         "9 — Do Clinical Labels Work Better?",
        "heading":       "Section 9: Do Clinical Labels Work Better?",
        "method":        "Method: Logistic Regression on expert-annotated data  |  Root cause prediction",
        "colour":        "#D9534F",
        "key_finding":   "70.8% accuracy on 800 expert-labelled posts beats 62.9% on 5,000 community posts. Label quality matters more than dataset size.",
        "plain_english": "Instead of predicting which subreddit a post is from, this classifier predicts the underlying cause of distress using categories assigned by clinical experts rather than platform community labels.",
    },
    {
        "title":         "10 — Did COVID Change the Surface?",
        "heading":       "Section 10: Did COVID Change the Surface?",
        "method":        "Longitudinal analysis  |  Post volume and vocabulary stability across 2019-2022",
        "colour":        "#9B59B6",
        "key_finding":   "r/lonely grew 339.7% across the study period — the most dramatic growth in the dataset.",
        "plain_english": "This tracks how much each community posted across four April snapshots and whether the vocabulary they used stayed stable or shifted. Volume and words are the most visible and most changeable layer of discourse.",
    },
    {
        "title":         "11 — Did COVID Change the Themes?",
        "heading":       "Section 11: Did COVID Change the Themes?",
        "method":        "Longitudinal analysis  |  Topic proportions and VAD scores across 2019-2022",
        "colour":        "#9B59B6",
        "key_finding":   "Crisis language grew from 9.0% to 13.3% of posts between 2019 and 2021 — the pandemic had a measurable effect on discourse themes.",
        "plain_english": "Did the topics people discussed change across the pandemic? Did the emotional tone shift? This section tracks both over four years.",
    },
    {
        "title":         "12 — But the Deep Structure Did Not Change",
        "heading":       "Section 12: But the Deep Structure Did Not Change",
        "method":        "Longitudinal analysis  |  Cross-period classifier temporal generalisation",
        "colour":        "#9B59B6",
        "key_finding":   "Only 0.4% accuracy drop across three years of pandemic disruption — the communities' semantic identity is temporally stable.",
        "plain_english": "If vocabulary changed substantially, can a classifier trained in 2022 still read 2019 posts correctly? The answer — yes, with almost no loss — reveals that the communities' core meaning has not shifted.",
    },
    {
        "title":         "13 — Limitations and Critical Reflection",
        "heading":       "Section 13: Limitations and Critical Reflection",
        "method":        "What NLP cannot do — and what that means for mental health applications",
        "colour":        "#2C3E50",
        "key_finding":   "The most important limitation is also the most important finding — the depression/SuicideWatch boundary resists computation because it reflects a genuine clinical reality.",
        "plain_english": "Every analysis has limits. This section is honest about what NLP cannot detect, and what that means for anyone thinking about applying these methods in clinical or content moderation contexts.",
    },
]

SECTION_TITLES = [s["title"] for s in SECTIONS]

NAV_OPTIONS = [
    "🏠 Home",
    "📊 About the Data",
    "🔬 Methods and Disciplines",
    "🔍 Analyse Text",
    "⚖️ Compare Texts",
    "📚 Research Findings",
    "🔭 Synthesis",
    "🚀 Future Development",
    "⚠️ Limitations",
]

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_classifiers():
    with open('classifier_a.pkl', 'rb') as f:
        clf_a = pickle.load(f)
    with open('classifier_b.pkl', 'rb') as f:
        clf_b = pickle.load(f)
    with open('labels_a.json', 'r') as f:
        labels_a = json.load(f)
    with open('labels_b.json', 'r') as f:
        labels_b = json.load(f)
    return clf_a, clf_b, labels_a, labels_b

@st.cache_resource
def load_vad_lexicon():
    vad = pd.read_csv('NRC-VAD-Lexicon-v2.1.txt', sep='\t')
    vad.columns = ['term', 'valence', 'arousal', 'dominance']
    return dict(zip(vad['term'], zip(vad['valence'], vad['arousal'], vad['dominance'])))

@st.cache_resource
def load_topic_data():
    with open('topic_centroids.json', 'r') as f:
        centroids = json.load(f)
    with open('topic_labels.json', 'r') as f:
        labels = json.load(f)
    centroids_np = {k: np.array(v) for k, v in centroids.items()}
    return centroids_np, labels

# ============================================================
# TEXT PREPROCESSING
# ============================================================

def basic_clean(text):
    if not text:
        return ''
    text = html.unescape(str(text))
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def compute_vad(text, vad_dict):
    words = text.lower().split()
    scores = [vad_dict[w] for w in words if w in vad_dict]
    if not scores:
        return None, None, None, 0
    valence   = sum(s[0] for s in scores) / len(scores)
    arousal   = sum(s[1] for s in scores) / len(scores)
    dominance = sum(s[2] for s in scores) / len(scores)
    match_rate = len(scores) / len(words) if words else 0
    return valence, arousal, dominance, match_rate

def pronoun_rate(text):
    first_person = {'i', 'me', 'my', 'myself'}
    words = text.lower().split()
    if not words:
        return 0
    return sum(1 for w in words if w in first_person) / len(words)

def assign_topic(embedding, topic_centroids, topic_labels):
    embedding = embedding.flatten()
    best_topic = None
    best_sim = -1
    for topic_id, centroid in topic_centroids.items():
        sim = np.dot(embedding, centroid) / (
            np.linalg.norm(embedding) * np.linalg.norm(centroid) + 1e-8
        )
        if sim > best_sim:
            best_sim = sim
            best_topic = topic_id
    if best_sim < 0.3:
        return None, None, None, best_sim
    info = topic_labels[best_topic]
    return best_topic, info['label'], info['words'], best_sim

# ============================================================
# CHART FUNCTIONS
# ============================================================

def plot_classifier_probs(labels, probs, colours, title):
    sorted_pairs  = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
    labels_sorted = [p[0] for p in sorted_pairs]
    probs_sorted  = [p[1] for p in sorted_pairs]
    bar_colours   = [colours.get(l, '#4A90D9') for l in labels_sorted]
    fig = go.Figure(go.Bar(
        x=probs_sorted,
        y=[f'r/{l}' if l in colours else l for l in labels_sorted],
        orientation='h',
        marker_color=bar_colours,
        marker_opacity=0.85,
        text=[f'{p:.1%}' for p in probs_sorted],
        textposition='outside'
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, 1.1], tickformat='.0%', title='Probability'),
        plot_bgcolor='#F8F8F8',
        paper_bgcolor='#F8F8F8',
        height=280,
        margin=dict(l=10, r=60, t=40, b=10)
    )
    return fig

def plot_vad_radar(valence, arousal, dominance, predicted_subreddit,
                   name_a='Your text', colour_a='#2C3E50',
                   valence_b=None, arousal_b=None, dominance_b=None,
                   name_b=None, colour_b=None):
    categories = ['Valence', 'Arousal', 'Dominance']

    def normalise(v):
        return max(0, min(1, (v + 0.3) / 0.6))

    fig = go.Figure()
    vals_a = [normalise(valence), normalise(arousal), normalise(dominance)]
    r, g, b_int = (int(colour_a.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    fig.add_trace(go.Scatterpolar(
        r=vals_a + [vals_a[0]],
        theta=categories + [categories[0]],
        fill='toself', name=name_a,
        line_color=colour_a,
        fillcolor=f'rgba({r},{g},{b_int},0.2)'
    ))

    if valence_b is not None and name_b and colour_b:
        vals_b = [normalise(valence_b), normalise(arousal_b), normalise(dominance_b)]
        r2, g2, b2 = (int(colour_b.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        fig.add_trace(go.Scatterpolar(
            r=vals_b + [vals_b[0]],
            theta=categories + [categories[0]],
            fill='toself', name=name_b,
            line_color=colour_b,
            fillcolor=f'rgba({r2},{g2},{b2},0.2)'
        ))
    elif predicted_subreddit in DATASET_VAD:
        avg = DATASET_VAD[predicted_subreddit]
        vals_avg = [normalise(avg['valence']), normalise(avg['arousal']), normalise(avg['dominance'])]
        sc = SUBREDDIT_COLOURS[predicted_subreddit]
        ra, ga, ba = (int(sc.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        fig.add_trace(go.Scatterpolar(
            r=vals_avg + [vals_avg[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=f'r/{predicted_subreddit} average',
            line_color=sc,
            fillcolor=f'rgba({ra},{ga},{ba},0.2)'
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='VAD Affective Profile',
        plot_bgcolor='#F8F8F8',
        paper_bgcolor='#F8F8F8',
        height=350
    )
    return fig

# ============================================================
# UI HELPER FUNCTIONS
# ============================================================

def section_heading(section):
    n     = SECTIONS.index(section) + 1
    total = len(SECTIONS)
    st.markdown(f"""
    <div style='background-color:#F0F4F8;padding:16px 20px;
                border-left:4px solid {section["colour"]};border-radius:4px;
                margin:20px 0 4px 0'>
      <div style='display:flex;justify-content:space-between;align-items:center'>
        <div>
          <h2 style='margin:0;color:#2C3E50'>{section["heading"]}</h2>
          <p style='margin:4px 0 0 0;color:#666;font-size:0.9em'>{section["method"]}</p>
        </div>
        <span style='color:#999;font-size:0.85em;white-space:nowrap;padding-left:20px'>
          {n} of {total}
        </span>
      </div>
    </div>""", unsafe_allow_html=True)


def plain_english(text):
    st.markdown(f"""
    <div style='background-color:#EBF5FB;padding:14px 18px;border-radius:6px;
                border-left:3px solid #4A90D9;margin:12px 0 20px 0'>
      <p style='margin:0;color:#1A5276;font-size:0.95em'>
        <strong>In plain English:</strong> {text}
      </p>
    </div>""", unsafe_allow_html=True)


def big_stat(value, subtitle, colour):
    st.markdown(f"""
    <div style='background-color:{colour};padding:24px;border-radius:8px;
                text-align:center;margin:16px 0'>
      <h1 style='color:white;margin:0;font-size:3em'>{value}</h1>
      <p style='color:white;margin:10px 0 0 0;font-size:1.05em'>{subtitle}</p>
    </div>""", unsafe_allow_html=True)


def headline_banner(pred_a, pred_b, conf_a, conf_b):
    colour = SUBREDDIT_COLOURS.get(pred_a, '#2C3E50')
    st.markdown(f"""
    <div style='background-color:{colour};padding:22px 30px;border-radius:8px;
                margin:0 0 28px 0;display:flex;
                justify-content:space-between;align-items:center'>
      <div>
        <p style='color:rgba(255,255,255,0.8);margin:0;font-size:0.88em;
                  text-transform:uppercase;letter-spacing:0.05em'>Predicted community</p>
        <h2 style='color:white;margin:6px 0 4px 0;font-size:2em'>r/{pred_a}</h2>
        <p style='color:rgba(255,255,255,0.85);margin:0'>{conf_a:.1%} confidence</p>
      </div>
      <div style='text-align:right'>
        <p style='color:rgba(255,255,255,0.8);margin:0;font-size:0.88em;
                  text-transform:uppercase;letter-spacing:0.05em'>Predicted root cause</p>
        <h2 style='color:white;margin:6px 0 4px 0;font-size:1.5em'>{pred_b}</h2>
        <p style='color:rgba(255,255,255,0.85);margin:0'>{conf_b:.1%} confidence</p>
      </div>
    </div>""", unsafe_allow_html=True)


def about_expander():
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
**Classifiers:** Logistic Regression on 384-dimensional sentence embeddings
from `all-MiniLM-L6-v2` (sentence-transformers).

**Classifier A** — 5,000 Reddit posts (1,000 per subreddit).
Cross-validated accuracy: 62.9% (baseline: 20%).

**Classifier B** — 800 expert-annotated posts.
Cross-validated accuracy: 70.8% (baseline: 25%).

**Key limitation:** r/depression and r/SuicideWatch have a cosine similarity
of 0.954. Language alone cannot reliably separate depressive from suicidal discourse.

**VAD:** NRC Valence-Arousal-Dominance Lexicon v2.1 (Mohammad, 2025).

**Topics:** Nearest-centroid matching against BERTopic centroids
fitted on 5,000 Reddit posts.

**Data:** Reddit Mental Health Dataset (RMHD), April 2022.
Annotation: https://doi.org/10.3390/app14041547

*This tool is for research and educational purposes only.*
        """)


def prev_next_buttons(current_idx):
    st.divider()
    col_prev, col_spacer, col_next = st.columns([1, 5, 1])
    if col_prev.button("← Previous", disabled=(current_idx == 0), use_container_width=True):
        st.session_state["research_nav"] = current_idx - 1
        st.rerun()
    if col_next.button("Next →", disabled=(current_idx == len(SECTIONS) - 1), use_container_width=True):
        st.session_state["research_nav"] = current_idx + 1
        st.rerun()

# ============================================================
# SESSION STATE
# ============================================================

if "research_nav" not in st.session_state:
    st.session_state["research_nav"] = 0

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "🏠 Home"

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧠 MH Discourse Analyser")
st.sidebar.divider()

active_tab = st.sidebar.radio(
    "Navigation",
    NAV_OPTIONS,
    index=NAV_OPTIONS.index(st.session_state["active_tab"]),
    label_visibility="collapsed"
)

st.session_state["active_tab"] = active_tab

st.sidebar.divider()

if active_tab == "🔍 Analyse Text":
    st.sidebar.markdown("**Try an example**")
    selected_example = st.sidebar.selectbox(
        "Example texts", list(EXAMPLES.keys()),
        label_visibility="collapsed"
    )
    st.sidebar.divider()
    about_expander()

elif active_tab == "⚖️ Compare Texts":
    st.sidebar.markdown("**Load a comparison example**")
    load_example = st.sidebar.button(
        "Load depression vs SuicideWatch", use_container_width=True
    )
    if load_example:
        st.session_state["text_a"] = (
            "I haven't left my bed in three days. I just don't see the point anymore. "
            "Everything feels grey and empty. I'm on medication but nothing seems to help."
        )
        st.session_state["text_b"] = (
            "I am so tired of fighting every single day. I don't want to die but I can't "
            "keep living like this. I just want it all to stop."
        )
        st.rerun()
    st.sidebar.divider()
    about_expander()

elif active_tab == "📚 Research Findings":
    st.sidebar.markdown("**Jump to section**")
    selected_title = st.sidebar.selectbox(
        "Section selector", SECTION_TITLES,
        index=st.session_state["research_nav"],
        label_visibility="collapsed"
    )
    chosen_idx = SECTION_TITLES.index(selected_title)
    if chosen_idx != st.session_state["research_nav"]:
        st.session_state["research_nav"] = chosen_idx
        st.rerun()

    current_idx     = st.session_state["research_nav"]
    current_section = SECTIONS[current_idx]

    st.sidebar.markdown(
        f"<p style='color:#888;font-size:0.82em;margin:6px 0 10px 0'>"
        f"Section {current_idx + 1} of {len(SECTIONS)}</p>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        f"""<div style='background-color:#F0F4F8;padding:11px 13px;
                border-left:3px solid {current_section["colour"]};
                border-radius:4px;margin:4px 0 12px 0'>
          <p style='margin:0;font-size:0.82em;color:#2C3E50;line-height:1.5'>
            <strong>Key finding:</strong><br>{current_section["key_finding"]}
          </p>
        </div>""",
        unsafe_allow_html=True
    )
    st.sidebar.divider()
    about_expander()

else:
    about_expander()

# ============================================================
# VIEW: HOME
# ============================================================

if active_tab == "🏠 Home":

    st.markdown("""
    <div style='background-color:#2C3E50;padding:40px 30px;border-radius:8px;margin-bottom:28px'>
      <h1 style='color:white;margin:0 0 10px 0;font-size:2.4em'>
        🧠 Mental Health Discourse Analyser
      </h1>
      <p style='color:#BDC3C7;margin:0 0 16px 0;font-size:1.1em;line-height:1.7'>
        A computational analysis of 39,492 Reddit posts across five mental health
        communities, combining NLP methods from computational linguistics, affective
        psychology, clinical psycholinguistics, and discourse analysis.
      </p>
      <p style='color:#E8A838;margin:0;font-size:1em;font-weight:500'>
        Central question: can computational methods reveal structures in mental health
        communities that human readers would miss — and what are the limits of that claim?
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### The Data")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Total Posts", "39,492", help="After cleaning and deduplication")
    d2.metric("Communities", "5", help="r/Anxiety, r/depression, r/mentalhealth, r/SuicideWatch, r/lonely")
    d3.metric("Years", "4", help="April 2019, 2020, 2021, 2022")
    d4.metric("Annotated Posts", "800", help="Expert-labelled with root cause categories")
    d5.metric("NLP Methods", "5", help="TF-IDF, VAD, Embeddings, BERTopic, Classification")

    st.divider()

    st.markdown("### The Central Finding")
    st.markdown("""
    <div style='background-color:#D9534F;padding:28px;border-radius:8px;
                text-align:center;margin:0 0 28px 0'>
      <h1 style='color:white;margin:0;font-size:3.5em'>0.954</h1>
      <p style='color:white;margin:12px 0 0 0;font-size:1.15em;line-height:1.6'>
        Cosine similarity between r/depression and r/SuicideWatch in semantic space.<br>
        <strong>Stable since April 2019 — three years before the end of the dataset,
        predating COVID entirely.</strong><br>
        No NLP method applied in this analysis successfully separates these two communities.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### How to Use This Tool")
    st.markdown("Use the sidebar on the left to navigate, or click a card below to jump directly.")

    g1, g2, g3, g4 = st.columns(4)

    with g1:
        st.markdown("""
        <div style='background-color:#F0F4F8;padding:18px;border-radius:8px;
                    border-top:4px solid #4A90D9;min-height:160px'>
          <h4 style='margin:0 0 8px 0;color:#2C3E50'>📊 About the Data</h4>
          <p style='color:#555;font-size:0.88em;line-height:1.6;margin:0 0 12px 0'>
            What Reddit is, why these five communities were chosen, how the data
            was collected and the ethical considerations.
          </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to About the Data", use_container_width=True, key="nav_data"):
            st.session_state["active_tab"] = "📊 About the Data"
            st.rerun()

    with g2:
        st.markdown("""
        <div style='background-color:#F0F4F8;padding:18px;border-radius:8px;
                    border-top:4px solid #E8A838;min-height:160px'>
          <h4 style='margin:0 0 8px 0;color:#2C3E50'>🔬 Methods and Disciplines</h4>
          <p style='color:#555;font-size:0.88em;line-height:1.6;margin:0 0 12px 0'>
            How each NLP method connects to a different academic discipline —
            psychology, linguistics, discourse analysis.
          </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Methods", use_container_width=True, key="nav_methods"):
            st.session_state["active_tab"] = "🔬 Methods and Disciplines"
            st.rerun()

    with g3:
        st.markdown("""
        <div style='background-color:#F0F4F8;padding:18px;border-radius:8px;
                    border-top:4px solid #5CB85C;min-height:160px'>
          <h4 style='margin:0 0 8px 0;color:#2C3E50'>📚 Research Findings</h4>
          <p style='color:#555;font-size:0.88em;line-height:1.6;margin:0 0 12px 0'>
            Walk through all 13 sections of the full analysis with interactive
            charts and interpretations.
          </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Research Findings", use_container_width=True, key="nav_findings"):
            st.session_state["active_tab"] = "📚 Research Findings"
            st.rerun()

    with g4:
        st.markdown("""
        <div style='background-color:#F0F4F8;padding:18px;border-radius:8px;
                    border-top:4px solid #D9534F;min-height:160px'>
          <h4 style='margin:0 0 8px 0;color:#2C3E50'>⚠️ Limitations</h4>
          <p style='color:#555;font-size:0.88em;line-height:1.6;margin:0 0 12px 0'>
            A critical examination of what NLP cannot do and what that means
            for mental health applications.
          </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Limitations", use_container_width=True, key="nav_limits"):
            st.session_state["active_tab"] = "⚠️ Limitations"
            st.rerun()

    st.divider()

    st.markdown("""
    <div style='background-color:#FFF8E1;padding:20px;border-radius:8px;
                border-left:4px solid #E8A838'>
      <h4 style='margin:0 0 8px 0;color:#2C3E50'>⚠️ Ethical Note</h4>
      <p style='margin:0;color:#444;line-height:1.6'>
        This tool analyses posts from public Reddit communities. All data was collected
        from publicly accessible subreddits and used solely for research purposes. No
        individual users are identified. The posts contain sensitive content including
        expressions of suicidal ideation — this is handled with care throughout.<br><br>
        <strong>This tool is for research and educational purposes only. It does not
        provide clinical diagnosis, risk assessment, or mental health advice. If you
        or someone you know is in crisis, please contact a crisis line immediately.</strong>
      </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# VIEW: ABOUT THE DATA
# ============================================================

elif active_tab == "📊 About the Data":

    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px 30px;border-radius:8px;margin-bottom:24px'>
      <h1 style='color:white;margin:0 0 10px 0'>About the Data</h1>
      <p style='color:#BDC3C7;margin:0;line-height:1.7'>
        Understanding where the data comes from, why these communities were chosen,
        and the ethical considerations of working with mental health discourse online.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #4A90D9;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>What is Reddit?</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Reddit is a social media platform organised around communities called subreddits,
    each denoted by the prefix r/. Each subreddit is a public forum dedicated to a
    specific topic, interest, or type of content. Users post text, images, or links,
    and other users comment and vote. Subreddits are moderated by volunteer community
    members who set rules about what content is acceptable.

    Mental health subreddits function as peer support spaces where users share personal
    experiences, seek advice, and offer solidarity to others. They are not clinical
    environments — users are anonymous, there are no professional moderators, and
    there is no formal triage or safeguarding. This makes them a rich source of
    naturalistic mental health discourse but also raises important ethical questions
    about research using this data.

    Posts on Reddit are publicly accessible by default, which is why datasets like
    the RMHD can be constructed from them. However, users who post in mental health
    communities may not have anticipated that their posts would be used for research,
    even if they are technically public.
    """)

    st.divider()

    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #4A90D9;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Why These Five Communities?</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The five communities in this analysis were selected by the original RMHD study
    (Naseem et al., 2022) to represent a range of mental health experiences with
    varying degrees of clinical severity and social character.
    """)

    c1, c2, c3, c4, c5 = st.columns(5)

    for col, sub, colour, desc in [
        (c1, 'Anxiety', '#E8A838',
         'A large community focused on anxiety experiences, symptoms, and management. Posts frequently reference somatic symptoms and medication.'),
        (c2, 'depression', '#4A90D9',
         'One of the largest mental health communities on Reddit. Posts express low mood, hopelessness, and chronic suffering.'),
        (c3, 'mentalhealth', '#5CB85C',
         'A general mental health discussion community. More informational and diagnostic in character than the condition-specific communities.'),
        (c4, 'SuicideWatch', '#D9534F',
         'A community specifically for crisis support. Posts frequently express acute suicidal ideation and method-specific language.'),
        (c5, 'lonely', '#9B59B6',
         'A community focused on social isolation. Linguistically distinct from the clinical communities but semantically related to depression.'),
    ]:
        with col:
            st.markdown(f"""
            <div style='background-color:#F8F9FA;padding:14px;border-radius:6px;
                        border-top:3px solid {colour};min-height:180px'>
              <h4 style='margin:0 0 8px 0;color:{colour}'>r/{sub}</h4>
              <p style='margin:0;color:#444;font-size:0.85em;line-height:1.6'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    Together these five communities span clinical (Anxiety, depression, SuicideWatch),
    general (mentalhealth), and social (lonely) orientations toward mental health.
    """)

    st.divider()

    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #5CB85C;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Data Collection and Cleaning</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The Reddit Mental Health Dataset (RMHD) was sourced from Kaggle
    (https://doi.org/10.3390/app14041547). It contains posts scraped from the five
    subreddits between January 2019 and August 2022 using the Reddit API. This analysis
    focuses on April 2022 as the primary sample, with longitudinal extension to April
    2019, 2020, and 2021.

    **Cleaning steps applied to the raw data:**
    - Removed rows where the subreddit column contained corrupted data
    - Removed deleted posts (marked as [deleted]) and removed posts ([removed])
    - Removed empty posts with no text content
    - Removed duplicate posts based on selftext content
    - Combined post title and body text for analysis

    **After cleaning:** 48,657 total entries reduced to 39,492 posts. SuicideWatch has
    the most posts at 10,193, lonely the fewest at 4,656.

    **Preprocessing pipeline:** HTML entity decoding, contraction expansion, lowercasing,
    URL removal, tokenisation, lemmatisation. First-person pronouns were deliberately
    retained for the Pennebaker analysis.
    """)

    st.divider()

    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #9B59B6;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>The Annotation Study</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    800 posts from the dataset were manually annotated by clinical experts in the
    original RMHD study (Naseem et al., 2022) with root cause categories.
    """)

    a1, a2, a3, a4 = st.columns(4)
    for col, cat, colour, desc in [
        (a1, 'Drug and Alcohol', '#E8A838', '200 posts. Substance use as a contributing factor.'),
        (a2, 'Early Life', '#4A90D9', '200 posts. Childhood experiences or developmental factors.'),
        (a3, 'Personality', '#5CB85C', '200 posts. Personality traits or disorders.'),
        (a4, 'Trauma and Stress', '#D9534F', '200 posts. Traumatic events or chronic stress.'),
    ]:
        with col:
            st.markdown(f"""
            <div style='background-color:#F8F9FA;padding:14px;border-radius:6px;
                        border-top:3px solid {colour};min-height:120px'>
              <h4 style='margin:0 0 8px 0;color:#2C3E50'>{cat}</h4>
              <p style='margin:0;color:#444;font-size:0.85em;line-height:1.6'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    Clinical labels produce a more accurate classifier (70.8%) than platform community
    labels (62.9%) on five times fewer posts — one of the central methodological findings.
    """)

    st.divider()

    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #D9534F;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Ethical Considerations</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Working with mental health data from social media raises several ethical questions
    that deserve explicit acknowledgement.

    **Informed consent:** Users who posted in these communities did not explicitly
    consent to having their posts used for research. While posts are technically public,
    there is a reasonable expectation that a post shared in a peer support community
    is addressed to that community rather than to researchers.

    **Anonymisation:** No individual users are identified in this analysis. Usernames
    were not retained in the dataset.

    **Sensitive content:** The posts contain expressions of suicidal ideation,
    descriptions of self-harm, accounts of abuse, and other highly sensitive content.
    This content is handled with care throughout.

    **The clinical triage limitation:** This analysis explicitly argues that these
    methods are not suitable for clinical triage or automated crisis detection. A system
    that cannot reliably distinguish depressive from suicidal discourse should not be
    deployed in contexts where that distinction has clinical consequences.
    """)

    st.markdown("""
    <div style='background-color:#FFF8E1;padding:16px;border-radius:6px;
                border-left:4px solid #E8A838;margin-top:8px'>
      <p style='margin:0;color:#444;line-height:1.6'>
        <strong>This tool is for research and educational purposes only.</strong>
        It does not provide clinical diagnosis, risk assessment, or mental health advice.
        If you or someone you know is in crisis, please contact a crisis line immediately.
      </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# VIEW: METHODS AND DISCIPLINES
# ============================================================

elif active_tab == "🔬 Methods and Disciplines":

    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px 30px;border-radius:8px;margin-bottom:24px'>
      <h1 style='color:white;margin:0 0 10px 0'>Methods and Disciplines</h1>
      <p style='color:#BDC3C7;margin:0;line-height:1.7'>
        NLP methods do not exist in a vacuum. Each one operationalises a concept from
        another discipline. This section explains what each method does, where it comes
        from intellectually, and what it cannot tell us.
      </p>
    </div>
    """, unsafe_allow_html=True)

    for method in [
        {
            "title": "1. TF-IDF — Lexical Structure",
            "subtitle": "Computational linguistics → Communities of practice",
            "colour": "#E8A838",
            "body": """
**What it does:** Identifies the most distinctive terms in each community by
down-weighting words that appear across all communities and up-weighting words
characteristic of a specific one.

**Disciplinary origin:** Computational linguistics. TF-IDF is a foundational technique in information retrieval.

**Connects to:** The sociolinguistic concept of community of practice (Lave and Wenger, 1991) — groups engaged in shared activity develop shared but differentiated language use.

**What it reveals:** r/Anxiety uses somatic and pharmacological vocabulary. r/SuicideWatch uses method-specific crisis vocabulary. r/lonely uses social platform language.
            """,
            "limitation": "TF-IDF is purely statistical and cannot distinguish between semantically meaningful terms and noise. \"dan\" (a username) and \"slaveyou\" (a corrupted token) appear as the most distinctive terms in two communities."
        },
        {
            "title": "2. VAD — Affective Structure",
            "subtitle": "Affective psychology → Russell's circumplex model of affect",
            "colour": "#E8A838",
            "body": """
**What it does:** Assigns three affective dimensions to each post — Valence (positive vs negative), Arousal (activated vs calm), and Dominance (in control vs powerless) — by averaging scores from the NRC VAD Lexicon v2.1 (Mohammad, 2025).

**Disciplinary origin:** Affective psychology. Russell's (1980) circumplex model of affect.

**Connects to:** The clinical distinction between anxiety (high arousal) and depression/loneliness (low arousal) — invisible to a simple positive/negative sentiment score.

**What it reveals:** Arousal separates communities more than valence. r/lonely is the most emotionally flat (−0.092), r/Anxiety the most activated (−0.021).
            """,
            "limitation": "Averaging VAD scores dilutes signal from emotionally charged terms. Irony, sarcasm, and negation are invisible — \"I don't feel terrible\" scores positively despite its meaning."
        },
        {
            "title": "3. Pennebaker Pronouns — Psycholinguistic Structure",
            "subtitle": "Clinical psycholinguistics → First-person pronoun frequency as a marker of distress",
            "colour": "#E8A838",
            "body": """
**What it does:** Counts first-person singular pronouns (I, me, my, myself) as a proportion of total words, measuring inward attentional focus.

**Disciplinary origin:** Clinical psycholinguistics. Pennebaker (2011) demonstrated that first-person pronoun frequency is a reliable marker of depression and psychological distress.

**Connects to:** Attentional models of depression — depressive states are characterised by excessive self-focused attention.

**What it reveals:** The community ranking — SuicideWatch highest (0.1276), Anxiety lowest (0.1028) — is consistent with clinical predictions and stable since April 2019.
            """,
            "limitation": "Pronoun counting is a proxy measure. A post written in second person (\"you know when you feel...\") would score low despite describing personal distress."
        },
        {
            "title": "4. Sentence Embeddings — Semantic Structure",
            "subtitle": "Cognitive linguistics and distributional semantics → Meaning as vector geometry",
            "colour": "#4A90D9",
            "body": """
**What it does:** Represents the meaning of an entire post as a single 384-dimensional vector using all-MiniLM-L6-v2. Two posts can be close in embedding space despite sharing no words.

**Disciplinary origin:** Distributional semantics. The distributional hypothesis (Harris, 1954) — sentences occurring in similar contexts have similar meanings.

**Connects to:** Cognitive linguistic theories of meaning as contextual rather than residing in individual words.

**What it reveals:** The 0.954 cosine similarity between r/depression and r/SuicideWatch — the central finding of the analysis.
            """,
            "limitation": "The model was trained on general web text, not mental health discourse specifically. The temporal stability finding may partly reflect the model's fixed semantic space rather than genuine language stability."
        },
        {
            "title": "5. BERTopic — Latent Structure",
            "subtitle": "Discourse analysis → Latent thematic organisation of community discourse",
            "colour": "#5CB85C",
            "body": """
**What it does:** Uses sentence embeddings and HDBSCAN clustering to discover latent topics without being told what to look for. Represents topics through their most distinctive terms using c-TF-IDF.

**Disciplinary origin:** Discourse analysis. Communities organise themselves around recurring themes not always visible at the surface.

**Connects to:** Systemic functional linguistics — language choices reflect and constitute social positions and community identities.

**What it reveals:** 14 topics with striking community specificity. Topic 2 (anxiety, attack, panic) is 83% r/Anxiety. But 50% of posts resist categorisation entirely.
            """,
            "limitation": "The 50% outlier rate before reduction means half of all posts resist topical categorisation — mental health disclosure is too personal and idiosyncratic to cluster neatly."
        },
    ]:
        st.markdown(f"""
        <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid {method["colour"]};
                    border-radius:4px;margin-bottom:16px'>
          <h2 style='margin:0 0 4px 0;color:#2C3E50'>{method["title"]}</h2>
          <p style='margin:0;color:#888;font-size:0.88em'>{method["subtitle"]}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(method["body"])
        with col2:
            st.markdown(f"""
            <div style='background-color:#FFF3CD;padding:16px;border-radius:6px;
                        border:1px solid {method["colour"]}'>
              <h4 style='margin:0 0 8px 0;color:#2C3E50'>⚠️ Limitation</h4>
              <p style='margin:0;color:#444;font-size:0.9em;line-height:1.6'>
                {method["limitation"]}
              </p>
            </div>
            """, unsafe_allow_html=True)
        st.divider()

    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px;border-radius:8px;margin-top:8px'>
      <h2 style='color:white;margin:0 0 14px 0'>Why Combining Methods Matters</h2>
      <p style='color:#BDC3C7;line-height:1.7;margin:0 0 12px 0'>
        No single method tells the full story. TF-IDF shows different words. VAD shows
        different feelings. Pennebaker shows different self-focus. Embeddings show more
        similarity in meaning than vocabulary suggests. BERTopic shows latent themes —
        and how much resists categorisation.
      </p>
      <p style='color:white;line-height:1.7;margin:0;font-weight:500'>
        The most important findings emerge from convergence. r/Anxiety is the most
        linguistically distinctive community across every method simultaneously. This
        convergence is much stronger evidence than any single method could provide.
        Equally important is when methods disagree — different representational choices
        reveal different structures in the same corpus.
      </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# VIEW: ANALYSE TEXT
# ============================================================

elif active_tab == "🔍 Analyse Text":

    st.title("🧠 Mental Health Discourse Analyser")
    st.markdown("""
    This tool uses NLP classifiers trained on Reddit mental health data to analyse
    how a piece of text relates to different mental health discourse communities.
    It combines **lexical**, **affective**, and **semantic** analysis methods.

    **This is a research tool. It does not provide clinical diagnosis or advice.**
    """)

    text_input = st.text_area(
        "Enter text to analyse",
        value=EXAMPLES.get(selected_example, ""),
        height=150,
        placeholder="Type or paste text here..."
    )

    st.divider()

    if st.button("Analyse", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text to analyse.")
        else:
            with st.spinner("Analysing..."):
                embed_model                        = load_embedding_model()
                clf_a, clf_b, labels_a, labels_b   = load_classifiers()
                vad_dict                           = load_vad_lexicon()
                topic_centroids, topic_labels_data = load_topic_data()

                cleaned   = basic_clean(text_input)
                embedding = embed_model.encode([cleaned])

                probs_a = clf_a.predict_proba(embedding)[0]
                probs_b = clf_b.predict_proba(embedding)[0]
                pred_a  = labels_a[np.argmax(probs_a)]
                pred_b  = labels_b[np.argmax(probs_b)]

                valence, arousal, dominance, match_rate = compute_vad(cleaned, vad_dict)
                topic_id, topic_label, topic_words, topic_sim = assign_topic(
                    embedding, topic_centroids, topic_labels_data
                )
                fp = pronoun_rate(text_input)

            headline_banner(pred_a, pred_b, max(probs_a), max(probs_b))

            if max(probs_a) < 0.45:
                st.warning(
                    f"⚠️ Low-confidence prediction ({max(probs_a):.1%}). "
                    f"This text sits near the boundary between communities. "
                    f"Low-confidence predictions are where NLP is most unreliable and "
                    f"should not be treated as meaningful classifications."
                )

            st.subheader("Classification Results")
            col1, col2 = st.columns(2)

            with col1:
                colour = SUBREDDIT_COLOURS[pred_a]
                st.markdown(f"<h3 style='color:{colour}'>r/{pred_a}</h3>", unsafe_allow_html=True)
                st.caption(f"Confidence: {max(probs_a):.1%}")
                st.info(SUBREDDIT_DESCRIPTIONS[pred_a])
                if pred_a == 'SuicideWatch':
                    st.error(SUBREDDIT_CLINICAL[pred_a])
                elif pred_a == 'depression' and probs_a[labels_a.index('SuicideWatch')] > 0.25:
                    sw_prob = probs_a[labels_a.index('SuicideWatch')]
                    st.warning(
                        f"Note: r/depression and r/SuicideWatch are semantically "
                        f"nearly identical (cosine similarity: 0.954). "
                        f"SuicideWatch probability: {sw_prob:.1%}. "
                        f"Treat this classification with caution."
                    )
                else:
                    st.caption(SUBREDDIT_CLINICAL[pred_a])
                st.plotly_chart(
                    plot_classifier_probs(labels_a, probs_a, SUBREDDIT_COLOURS, 'Community Classification'),
                    use_container_width=True
                )

            with col2:
                st.markdown(f"### {pred_b}")
                st.caption(f"Confidence: {max(probs_b):.1%}")
                st.info(ROOT_CAUSE_DESCRIPTIONS[pred_b])
                st.plotly_chart(
                    plot_classifier_probs(labels_b, probs_b, {}, 'Root Cause Classification'),
                    use_container_width=True
                )

            st.divider()

            st.subheader("Affective Analysis — VAD")
            if valence is not None:
                vcol1, vcol2 = st.columns([1, 1])
                with vcol1:
                    st.plotly_chart(plot_vad_radar(valence, arousal, dominance, pred_a), use_container_width=True)
                    st.caption(f"VAD lexicon match rate: {match_rate:.1%} of words")
                with vcol2:
                    st.markdown("**VAD Scores**")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Valence",   f"{valence:.3f}",   help="Positive vs negative emotional tone")
                    m2.metric("Arousal",   f"{arousal:.3f}",   help="Activated vs calm")
                    m3.metric("Dominance", f"{dominance:.3f}", help="In control vs powerless")
                    st.markdown("**Community averages for comparison:**")
                    st.dataframe(pd.DataFrame(DATASET_VAD).T.round(3), use_container_width=True)
            else:
                st.info("No VAD lexicon matches found for this text.")

            st.divider()

            st.subheader("Nearest Latent Topic")
            if topic_id is not None:
                st.markdown(f"**Topic {topic_id}:** {topic_label}")
                st.caption(f"Cosine similarity to topic centroid: {topic_sim:.3f}")
                words  = [w[0] for w in topic_words[:10]]
                scores = [w[1] for w in topic_words[:10]]
                fig_topic = go.Figure(go.Bar(
                    x=scores[::-1], y=words[::-1], orientation='h',
                    marker_color='#4A90D9', marker_opacity=0.85
                ))
                fig_topic.update_layout(
                    title=f'Top Terms for Topic {topic_id} (c-TF-IDF weighted)',
                    xaxis_title='c-TF-IDF Score',
                    plot_bgcolor='#F8F8F8', paper_bgcolor='#F8F8F8',
                    height=300, margin=dict(l=10, r=20, t=40, b=10)
                )
                st.plotly_chart(fig_topic, use_container_width=True)
            else:
                st.info("This text was not assigned to any topic cluster. Approximately 50% of posts resist topical categorisation.")

            st.divider()

            st.subheader("Psycholinguistic Analysis")
            pl1, pl2, pl3 = st.columns(3)
            pl1.metric("First-Person Pronoun Rate", f"{fp:.1%}",
                       help="Proportion of words that are I, me, my, myself (Pennebaker, 2011).")
            if fp > 0.128:
                pl1.caption("Above SuicideWatch mean (0.128)")
            elif fp > 0.103:
                pl1.caption("Within dataset range")
            else:
                pl1.caption("Below Anxiety mean (0.103)")
            pl2.metric("Word Count", len(text_input.split()))
            pl3.metric("Questions", text_input.count('?'), help="Higher counts may indicate help-seeking orientation.")

# ============================================================
# VIEW: COMPARE TEXTS
# ============================================================

elif active_tab == "⚖️ Compare Texts":

    st.title("⚖️ Compare Two Texts")
    st.markdown("""
    Enter two texts to compare their linguistic profiles side by side.
    This is particularly useful for exploring the depression/SuicideWatch
    boundary — two communities that are semantically nearly identical
    (cosine similarity: 0.954).
    """)

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown("**Text A**")
        text_a = st.text_area("Text A", height=150, key="text_a",
                               placeholder="Enter first text...", label_visibility="collapsed")
    with comp_col2:
        st.markdown("**Text B**")
        text_b = st.text_area("Text B", height=150, key="text_b",
                               placeholder="Enter second text...", label_visibility="collapsed")

    if st.button("Compare", type="primary", key="compare_btn"):
        if not text_a.strip() or not text_b.strip():
            st.warning("Please enter both texts.")
        else:
            with st.spinner("Analysing both texts..."):
                embed_model                      = load_embedding_model()
                clf_a, clf_b, labels_a, labels_b = load_classifiers()
                vad_dict                         = load_vad_lexicon()

                cleaned_a = basic_clean(text_a)
                cleaned_b = basic_clean(text_b)
                emb_a     = embed_model.encode([cleaned_a])
                emb_b     = embed_model.encode([cleaned_b])

                similarity = float(np.dot(emb_a[0], emb_b[0]) / (
                    np.linalg.norm(emb_a[0]) * np.linalg.norm(emb_b[0])
                ))

                probs_a1 = clf_a.predict_proba(emb_a)[0]
                probs_a2 = clf_a.predict_proba(emb_b)[0]
                pred_a1  = labels_a[np.argmax(probs_a1)]
                pred_a2  = labels_a[np.argmax(probs_a2)]

                vad_a = compute_vad(cleaned_a, vad_dict)
                vad_b = compute_vad(cleaned_b, vad_dict)
                fp_a  = pronoun_rate(text_a)
                fp_b  = pronoun_rate(text_b)

            st.divider()

            st.subheader("Semantic Similarity")
            sim_col1, sim_col2, sim_col3 = st.columns([1, 2, 1])
            with sim_col2:
                st.metric("Cosine Similarity between texts", f"{similarity:.3f}",
                          help="1.0 = identical meaning, 0.0 = completely different.")
                if similarity > 0.9:
                    st.error("These texts are semantically nearly identical — a classifier would struggle to distinguish them.")
                elif similarity > 0.7:
                    st.warning("These texts are semantically similar.")
                else:
                    st.success("These texts are semantically distinct.")

                st.markdown("""
                <div style='background-color:#EBF5FB;padding:12px 16px;border-radius:6px;
                            border-left:3px solid #4A90D9;margin-top:12px'>
                  <p style='margin:0;color:#1A5276;font-size:0.88em;line-height:1.6'>
                    <strong>Note:</strong> This score measures similarity between these two specific texts.
                    The 0.954 figure cited elsewhere refers to the average similarity between the
                    r/depression and r/SuicideWatch community centroids — computed across thousands
                    of posts. Individual posts vary considerably around those community averages.
                  </p>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            st.subheader("Classification Comparison")
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                colour1 = SUBREDDIT_COLOURS[pred_a1]
                st.markdown(f"**Text A:** <span style='color:{colour1}'>r/{pred_a1}</span> ({max(probs_a1):.1%} confidence)", unsafe_allow_html=True)
                st.plotly_chart(plot_classifier_probs(labels_a, probs_a1, SUBREDDIT_COLOURS, 'Text A'), use_container_width=True)

            with res_col2:
                colour2 = SUBREDDIT_COLOURS[pred_a2]
                st.markdown(f"**Text B:** <span style='color:{colour2}'>r/{pred_a2}</span> ({max(probs_a2):.1%} confidence)", unsafe_allow_html=True)
                st.plotly_chart(plot_classifier_probs(labels_a, probs_a2, SUBREDDIT_COLOURS, 'Text B'), use_container_width=True)

            st.divider()

            st.subheader("Affective Profile Comparison")
            if vad_a[0] is not None and vad_b[0] is not None:
                vad_comp_col1, vad_comp_col2 = st.columns(2)
                with vad_comp_col1:
                    st.markdown("**Text A**")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Valence",   f"{vad_a[0]:.3f}")
                    m2.metric("Arousal",   f"{vad_a[1]:.3f}")
                    m3.metric("Dominance", f"{vad_a[2]:.3f}")
                with vad_comp_col2:
                    st.markdown("**Text B**")
                    m4, m5, m6 = st.columns(3)
                    m4.metric("Valence",   f"{vad_b[0]:.3f}", delta=f"{vad_b[0]-vad_a[0]:+.3f}")
                    m5.metric("Arousal",   f"{vad_b[1]:.3f}", delta=f"{vad_b[1]-vad_a[1]:+.3f}")
                    m6.metric("Dominance", f"{vad_b[2]:.3f}", delta=f"{vad_b[2]-vad_a[2]:+.3f}")

                st.plotly_chart(
                    plot_vad_radar(
                        vad_a[0], vad_a[1], vad_a[2], pred_a1,
                        name_a='Text A', colour_a='#2C3E50',
                        valence_b=vad_b[0], arousal_b=vad_b[1], dominance_b=vad_b[2],
                        name_b='Text B', colour_b='#E74C3C'
                    ),
                    use_container_width=True
                )

            st.divider()

            st.subheader("Psycholinguistic Comparison")
            penn_col1, penn_col2 = st.columns(2)
            with penn_col1:
                st.metric("Text A — Pronoun Rate", f"{fp_a:.1%}")
                st.metric("Text A — Word Count",   len(text_a.split()))
                st.metric("Text A — Questions",    text_a.count('?'))
            with penn_col2:
                st.metric("Text B — Pronoun Rate", f"{fp_b:.1%}",  delta=f"{fp_b-fp_a:+.1%}")
                st.metric("Text B — Word Count",   len(text_b.split()), delta=len(text_b.split())-len(text_a.split()))
                st.metric("Text B — Questions",    text_b.count('?'),   delta=text_b.count('?')-text_a.count('?'))

# ============================================================
# VIEW: RESEARCH FINDINGS
# ============================================================

elif active_tab == "📚 Research Findings":

    current_idx = st.session_state["research_nav"]
    section     = SECTIONS[current_idx]

    if current_idx == 0:
        st.markdown("""
        <div style='background-color:#2C3E50;padding:28px 30px;border-radius:8px;margin-bottom:24px'>
          <h1 style='color:white;margin:0 0 10px 0'>Research Findings</h1>
          <p style='color:#BDC3C7;margin:0 0 12px 0;line-height:1.7'>
            Computational analysis of 39,492 Reddit posts across five mental health
            communities — April 2019 to April 2022.
          </p>
          <p style='color:#E8A838;margin:0;font-size:0.95em;font-weight:500'>
            Central finding: The semantic boundary between depressive and suicidal
            discourse has been stable since April 2019 — predating COVID entirely.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        This section presents the full research analysis across 13 sections. The
        analytical strategy is layered — each method reveals a different aspect of
        the same corpus, and the most important findings emerge from convergence
        across methods. Sections 1-3 examine surface and psycholinguistic structure.
        Sections 4-6 examine semantic structure and introduce the central finding.
        Section 7 examines latent thematic structure. Sections 8-9 test predictive
        modelling. Sections 10-12 extend the analysis longitudinally across the
        pandemic. Section 13 reflects critically on limitations.

        Use the sidebar dropdown to jump to any section, or navigate with the
        Previous and Next buttons at the bottom of each page.
        """)

        st.divider()

    else:
        st.markdown("""
        <div style='background-color:#2C3E50;padding:28px 30px;border-radius:8px;margin-bottom:24px'>
          <h1 style='color:white;margin:0 0 10px 0'>Research Findings</h1>
          <p style='color:#BDC3C7;margin:0'>
            Computational analysis of 39,492 Reddit posts across five mental health
            communities — April 2019 to April 2022
          </p>
        </div>
        """, unsafe_allow_html=True)

    section_heading(section)
    plain_english(section["plain_english"])

    if current_idx == 0:
        st.markdown("""
        TF-IDF identifies the most distinctive terms in each subreddit by
        down-weighting words that appear commonly across all communities and
        up-weighting words characteristic of a specific one. This draws on the
        sociolinguistic concept of community of practice (Lave and Wenger, 1991).
        """)
        st.image("tfidf_bar_charts.png", use_column_width=True)
        st.caption("Most distinctive terms per subreddit weighted by TF-IDF score.")
        st.image("tfidf_wordclouds.png", use_column_width=True)
        st.caption("Word clouds showing lexical profiles weighted by TF-IDF score.")
        st.markdown("""
        r/Anxiety is dominated by somatic and pharmacological terms: nausea, palpitation,
        dizziness, propranolol, buspirone, sertraline. r/SuicideWatch's distinctive terms
        are method-specific crisis vocabulary: slit, noose, overdosing, paracetamol,
        helium, peacefully, painlessly. r/lonely's distinctive terms shifted substantially
        across the pandemic: discord, hmu, tiktok, whatsapp, friendless, ditched.

        Two artefacts illustrate a fundamental limitation of TF-IDF. In r/mentalhealth
        "dan" appears as the most distinctive term — almost certainly a username. In
        r/depression the top term is "slaveyou" — a corrupted token. TF-IDF is purely
        statistical and cannot distinguish meaningful terms from noise.
        """)

    elif current_idx == 1:
        st.markdown("""
        VAD scoring assigns three affective dimensions to each post by averaging scores
        from the NRC VAD Lexicon v2.1 (Mohammad, 2025). This draws on Russell's (1980)
        circumplex model of affect. The lexicon matched approximately 71% of tokens.
        """)
        st.image("vad_by_subreddit.png", use_column_width=True)
        st.caption("Mean VAD scores by subreddit across Valence, Arousal, and Dominance dimensions.")
        st.markdown("""
        Arousal is the most discriminating dimension. r/lonely scores lowest at minus
        0.092 — consistent with loneliness as a low-activation affective state. r/Anxiety
        scores highest at minus 0.021. Valence shows a counterintuitive pattern: r/lonely
        scores higher than r/depression at 0.173, reflecting aspirational social language
        that carries positive valence despite expressing painful absence. Dominance shows
        minimal variation — a generalised sense of low agency is shared across all five.
        """)

    elif current_idx == 2:
        st.markdown("""
        Pennebaker (2011) demonstrated that first-person singular pronoun frequency
        is a reliable psycholinguistic marker of self-focus and depression. First-person
        pronouns were deliberately retained in the preprocessing pipeline.
        """)
        st.image("pennebaker_pronouns.png", use_column_width=True)
        st.caption("Mean first-person pronoun rate by subreddit. Higher rates indicate greater self-focus.")
        st.markdown("""
        r/SuicideWatch shows the highest rate at 0.1276, followed by r/depression at
        0.1188, r/mentalhealth at 0.1087, r/lonely at 0.1044, and r/Anxiety lowest at
        0.1028. Crucially, this ranking is identical in April 2019 and April 2022 —
        a stable structural feature, not an artefact of the pandemic.
        """)

    elif current_idx == 3:
        st.markdown("""
        Sentence embeddings represent the meaning of an entire post as a single
        384-dimensional vector using all-MiniLM-L6-v2. This draws on distributional
        semantics. 1,000 posts per subreddit (5,000 total) were embedded and reduced
        using UMAP for visualisation.
        """)
        with open("semantic_umap_2d_interactive.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=750)
        st.caption("Interactive UMAP projection in 2D. Click legend items to toggle subreddits.")
        st.markdown("""
        r/Anxiety is the most semantically distinct community, with a minimum similarity
        of 0.595 with r/lonely. The most striking finding is the near-identical semantic
        profile of r/depression and r/SuicideWatch — a cosine similarity of 0.954.
        The most representative posts reveal character: r/Anxiety's centroid post is
        "Can anyone relate?" r/SuicideWatch's is "everything has gone to shit and im so tired."
        """)

    elif current_idx == 4:
        st.markdown("""
        A direct comparison of the TF-IDF similarity matrix and the embedding similarity
        matrix reveals how much the choice of representation shapes what is visible.
        TF-IDF captures shared vocabulary. Embeddings capture shared meaning.
        """)
        st.image("similarity_comparison.png", use_column_width=True)
        st.caption("Three-panel comparison: TF-IDF similarity, embedding similarity, and the difference.")
        st.markdown("""
        Every off-diagonal value in the difference matrix is positive — all communities
        are more similar in semantic space than in lexical space. The largest difference
        is between r/lonely and r/depression at 0.717. NLP systems built on lexical
        features will produce a different — and arguably less accurate — picture of
        community similarity than systems built on semantic representations.
        """)

    elif current_idx == 5:
        big_stat(
            "0.954",
            "Cosine similarity between r/depression and r/SuicideWatch<br>"
            "<strong>Stable at this level since April 2019 — three years before the end of the dataset</strong>",
            "#D9534F"
        )
        st.markdown("""
        The semantic similarity of 0.954 is the strongest finding in the entire analysis.
        No analytical method successfully separates these two communities. This has direct
        clinical implications — if automated systems cannot reliably distinguish depressive
        from suicidal discourse at the sentence level, neither can a triage algorithm.
        """)
        with open("dep_sw_similarity_over_time.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=480)
        st.caption("Depression/SuicideWatch cosine similarity across all four April periods.")
        with open("longitudinal_embedding_similarity.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=500)
        st.caption("Full pairwise similarity matrices across all four years.")
        st.markdown("""
        The similarity was 0.952 in April 2019, 0.962 in April 2020, 0.951 in April
        2021, and 0.954 in April 2022. This boundary is not a COVID effect. Depression
        is the primary risk factor for suicidal ideation and the two conditions share
        overlapping phenomenology: hopelessness, worthlessness, exhaustion, and the
        desire for relief from suffering.
        """)

    elif current_idx == 6:
        st.markdown("""
        BERTopic uses sentence embeddings and HDBSCAN clustering to discover latent
        topics without being told what to look for. This draws on discourse analysis —
        communities organise themselves around recurring themes not always visible at
        the surface.
        """)
        with open("bertopic_barchart.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=700)
        st.caption("Top terms per topic weighted by c-TF-IDF.")
        st.image("bertopic_heatmap.png", use_column_width=True)
        st.caption("Topic distribution across subreddits normalised by proportion.")
        st.markdown("""
        14 topics were identified. Topic 1 (want, die, life, kill) is dominated by
        r/SuicideWatch at 65.7%. Topic 2 (anxiety, attack, like, feel) is dominated by
        r/Anxiety at 83.0%. Topic 3 (talk, chat, anyone, someone) is dominated by
        r/lonely at 82.9%. 50% of posts resist topical categorisation — this is not a
        model failure but a finding about the idiosyncratic nature of mental health disclosure.
        """)

    elif current_idx == 7:
        big_stat("62.9%", "Cross-validated accuracy — more than 3x the 20% random baseline", "#D9534F")
        st.markdown("""
        Logistic Regression was applied to sentence embeddings to predict which subreddit
        each post belongs to. 1,000 posts per subreddit (5,000 total), with 5-fold
        stratified cross-validation.
        """)
        st.image("classifier_a_confusion.png", use_column_width=True)
        st.caption("Classifier A confusion matrix. Rows show true labels, columns show predicted labels.")
        st.markdown("""
        r/Anxiety is the most accurately classified at F1 0.78. r/depression is the hardest
        at F1 0.44 — barely above chance. The most clinically significant finding: 41
        depression posts were misclassified as SuicideWatch and 35 SuicideWatch posts as
        depression. Reading the misclassified posts confirms genuine linguistic ambiguity
        rather than model error.
        """)

    elif current_idx == 8:
        big_stat("70.8%", "Accuracy on 800 expert-labelled posts — beating 62.9% on 5,000 community posts", "#D9534F")
        st.markdown("""
        A second classifier was trained on 800 posts manually annotated with root cause
        categories by clinical annotators. This tests whether expert-guided clinical labels
        produce more learnable categories than organic community labels.
        """)
        st.image("classifier_b_confusion.png", use_column_width=True)
        st.caption("Classifier B confusion matrix. Rows show true labels, columns show predicted labels.")
        st.markdown("""
        Drug and Alcohol is the most learnable at F1 0.85. Trauma and Stress is the hardest
        at F1 0.59, most often confused with Early Life. Label quality matters more than
        dataset size.
        """)

    elif current_idx == 9:
        big_stat("339.7%", "Growth in r/lonely post volume — April 2019 to April 2022", "#9B59B6")
        st.markdown("""
        The longitudinal analysis extends the April 2022 findings to three prior April
        periods covering the full arc of the COVID-19 pandemic.
        """)
        with open("longitudinal_volume.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=580)
        st.caption("Post volume and community share of total posts across all four April periods.")
        with open("longitudinal_tfidf_similarity.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=480)
        st.caption("TF-IDF cosine similarity of each community's vocabulary to its April 2022 self.")
        st.markdown("""
        r/mentalhealth grew 148.8% and r/SuicideWatch grew 89.9%. r/depression was the
        only community to decline at 21.6%. r/lonely's TF-IDF vocabulary similarity to
        its 2022 self was only 0.335 in April 2020 — the most dramatic vocabulary shift
        in the dataset. All communities show their lowest vocabulary similarity to 2022
        in April 2020.
        """)

    elif current_idx == 10:
        st.markdown("""
        While vocabulary shifted substantially, the question remains whether the
        underlying thematic and affective structure changed. Topic proportions and VAD
        scores are tracked across all four years using the 2022 topic model applied to
        prior years via BERTopic's transform() method.
        """)
        with open("longitudinal_bertopic_heatmap.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=580)
        st.caption("Topic proportions as a percentage of assigned posts per year.")
        with open("longitudinal_bertopic_lines.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=580)
        st.caption("Topic proportions as line chart showing trends across the pandemic arc.")
        with open("longitudinal_vad.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=500)
        st.caption("VAD affective scores by subreddit across all four April periods.")
        st.markdown("""
        Crisis language (Topic 1) grew from 9.0% in 2019 to 13.3% in 2021 before settling
        at 12.0% in 2022. Social connection discourse declined from 14.8% in 2019 to 11.7%
        in 2020. All communities scored slightly lower valence in April 2020. Crucially the
        community ordering on all VAD dimensions is identical in every year.
        """)

    elif current_idx == 11:
        big_stat("0.4%", "Accuracy drop across three years of pandemic disruption — semantic structure is temporally stable", "#9B59B6")
        st.markdown("""
        A classifier trained exclusively on April 2022 data is tested on April 2019, 2020,
        and 2021 data without retraining.
        """)
        with open("cross_period_classifier.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=480)
        st.caption("Cross-period classifier accuracy: trained on 2022, tested on prior years.")
        st.markdown("""
        62.5% accuracy on 2019 data, 62.6% on 2020, 62.8% on 2021 — compared to 62.9%
        in-period. Communities changed how they talked — the specific words, platforms, and
        clinical terminology — but not what they talked about at the level of meaning.

        One important caveat: all four years were embedded using the same pre-trained model.
        The cross-period stability may partly reflect model consistency rather than language
        stability.
        """)

    elif current_idx == 12:
        st.markdown("Several limitations deserve explicit attention. The full discussion is in the **⚠️ Limitations** section.")

        for title, body in [
            ("What NLP cannot detect",
             "All five methods operate on surface text and cannot access the implicit social meaning of posting to a public community. Sarcasm, irony, and understatement are invisible."),
            ("The depression/SuicideWatch boundary",
             "The near-identical semantic profile resists computational resolution not because the methods are inadequate but because the experiences genuinely overlap. This is a finding about the nature of mental health discourse."),
            ("The embedding model caveat",
             "Temporal stability may partly reflect the consistency of the all-MiniLM-L6-v2 model rather than the language itself."),
            ("No control corpus",
             "Without a baseline against general Reddit discourse it is not possible to establish which features are specific to mental health communities."),
            ("Community labels as proxies",
             "Platform community membership is an imperfect proxy for clinical mental state. The superiority of clinical labels in Classifier B confirms this."),
        ]:
            st.markdown(f"""
            <div style='background-color:#F8F9FA;padding:16px;border-radius:6px;
                        border-left:4px solid #2C3E50;margin-bottom:10px'>
              <h4 style='margin:0 0 6px 0;color:#2C3E50'>{title}</h4>
              <p style='margin:0;color:#444;line-height:1.6;font-size:0.95em'>{body}</p>
            </div>
            """, unsafe_allow_html=True)

    prev_next_buttons(current_idx)

# ============================================================
# VIEW: SYNTHESIS
# ============================================================

elif active_tab == "🔭 Synthesis":

    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px 30px;border-radius:8px;margin-bottom:24px'>
      <h1 style='color:white;margin:0 0 10px 0'>Synthesis</h1>
      <p style='color:#BDC3C7;margin:0;line-height:1.7'>
        What the full analysis tells us — pulling together findings from all five
        methods and the longitudinal extension into a single coherent argument.
      </p>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div style='background-color:#D9534F;padding:20px;border-radius:8px;text-align:center'>
          <h2 style='color:white;margin:0;font-size:2.2em'>0.954</h2>
          <p style='color:white;margin:8px 0 0 0;font-size:0.88em'>Depression/SuicideWatch similarity — stable since 2019</p>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div style='background-color:#9B59B6;padding:20px;border-radius:8px;text-align:center'>
          <h2 style='color:white;margin:0;font-size:2.2em'>0.4%</h2>
          <p style='color:white;margin:8px 0 0 0;font-size:0.88em'>Accuracy drop across three years of pandemic disruption</p>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div style='background-color:#E8A838;padding:20px;border-radius:8px;text-align:center'>
          <h2 style='color:white;margin:0;font-size:2.2em'>339.7%</h2>
          <p style='color:white;margin:8px 0 0 0;font-size:0.88em'>Growth in r/lonely post volume — 2019 to 2022</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("## What Every Method Agrees On")

    conv1, conv2, conv3 = st.columns(3)

    with conv1:
        st.markdown("""
        <div style='background-color:#F0F4F8;padding:18px;border-radius:8px;border-top:4px solid #E8A838;min-height:220px'>
          <h4 style='margin:0 0 10px 0;color:#2C3E50'>r/Anxiety is the most distinctive community</h4>
          <p style='color:#444;font-size:0.88em;line-height:1.6;margin:0'>
            Across every method — TF-IDF, VAD, embeddings, BERTopic, and classification —
            r/Anxiety emerges as the most linguistically distinctive community. Anxiety discourse
            has a genuinely distinctive linguistic signature confirmed by five independent methods.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with conv2:
        st.markdown("""
        <div style='background-color:#F0F4F8;padding:18px;border-radius:8px;border-top:4px solid #D9534F;min-height:220px'>
          <h4 style='margin:0 0 10px 0;color:#2C3E50'>Depression and SuicideWatch cannot be separated</h4>
          <p style='color:#444;font-size:0.88em;line-height:1.6;margin:0'>
            No method successfully distinguishes r/depression from r/SuicideWatch. They share
            vocabulary, affective profiles, BERTopic topics, and embedding similarity of 0.954.
            This convergence is the strongest possible evidence that the boundary is genuinely
            unresolvable through language alone.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with conv3:
        st.markdown("""
        <div style='background-color:#F0F4F8;padding:18px;border-radius:8px;border-top:4px solid #9B59B6;min-height:220px'>
          <h4 style='margin:0 0 10px 0;color:#2C3E50'>Deep structure is temporally stable</h4>
          <p style='color:#444;font-size:0.88em;line-height:1.6;margin:0'>
            The VAD community ordering, the Pennebaker pronoun hierarchy, the semantic similarity
            matrix, and the cross-period classifier accuracy are all virtually identical across
            four years of pandemic disruption. Communities changed how they talked — not what
            they talked about at the level of meaning.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("## Surface Instability vs Deep Stability")

    surf_col, deep_col = st.columns(2)

    with surf_col:
        st.markdown("""
        <div style='background-color:#FFF3CD;padding:20px;border-radius:8px;border-left:4px solid #E8A838'>
          <h3 style='margin:0 0 12px 0;color:#2C3E50'>Surface Features Changed</h3>
          <ul style='color:#444;line-height:1.8;margin:0;padding-left:20px'>
            <li>r/lonely's vocabulary similarity to 2022 was only 0.335 in April 2020</li>
            <li>r/lonely grew 339.7% from 2019 to 2022</li>
            <li>Crisis language grew from 9.0% to 13.3% between 2019 and 2021</li>
            <li>Valence dipped measurably across all communities in April 2020</li>
            <li>Social connection discourse declined during lockdown</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with deep_col:
        st.markdown("""
        <div style='background-color:#EBF5FB;padding:20px;border-radius:8px;border-left:4px solid #4A90D9'>
          <h3 style='margin:0 0 12px 0;color:#2C3E50'>Deep Features Remained Stable</h3>
          <ul style='color:#444;line-height:1.8;margin:0;padding-left:20px'>
            <li>Semantic similarity matrix almost identical across all four years</li>
            <li>VAD community ordering unchanged throughout</li>
            <li>Pennebaker pronoun hierarchy identical in 2019 and 2022</li>
            <li>Cross-period classifier loses less than 0.4% accuracy across three years</li>
            <li>Depression/SuicideWatch similarity stable at 0.952-0.962 throughout</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style='background-color:#2C3E50;padding:30px;border-radius:8px'>
      <h2 style='color:white;margin:0 0 16px 0'>The Central Argument</h2>
      <p style='color:#BDC3C7;line-height:1.7;margin:0 0 14px 0'>
        Surface features changed across the pandemic. Deep features did not. The most
        important finding concerns the depression/SuicideWatch boundary — stable since
        April 2019, predating COVID entirely. It is a structural feature of how these
        experiences are expressed in language.
      </p>
      <p style='color:white;line-height:1.7;margin:0;font-weight:500'>
        Systems built on semantic representations can be expected to generalise across
        time even as vocabulary evolves. Systems built on lexical features will degrade
        as community language shifts. And no system — lexical or semantic — can reliably
        separate depressive from suicidal discourse, because the experiences themselves
        do not separate cleanly in language.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style='background-color:#F0F4F8;padding:24px;border-radius:8px;border-left:4px solid #5CB85C'>
      <h3 style='margin:0 0 12px 0;color:#2C3E50'>Methodological Recommendation</h3>
      <p style='color:#444;line-height:1.7;margin:0 0 12px 0'>
        For temporally robust mental health NLP, semantic representations should be
        preferred over lexical ones. For clinical applications, investment in high-quality
        clinical labelling is more valuable than collecting larger unlabelled datasets.
      </p>
      <p style='color:#2C3E50;line-height:1.7;margin:0;font-weight:500'>
        For the depression/SuicideWatch distinction specifically — no recommendation
        can be made for automated triage. The boundary resists every computational
        method applied. Clinical judgment is required at this boundary.
      </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# VIEW: FUTURE DEVELOPMENT
# ============================================================

elif active_tab == "🚀 Future Development":

    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px 30px;border-radius:8px;margin-bottom:24px'>
      <h1 style='color:white;margin:0 0 10px 0'>Future Development</h1>
      <p style='color:#BDC3C7;margin:0;line-height:1.7'>
        What this analysis makes possible — and what would need to change before
        these methods could be used responsibly in clinical or applied contexts.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## The Vision: An NLP-Assisted Mental Health Triage System")

    st.markdown("""
    <div style='background-color:#EBF5FB;padding:20px;border-radius:8px;border-left:4px solid #4A90D9;margin-bottom:20px'>
      <p style='margin:0;color:#1A5276;line-height:1.7'>
        The findings of this analysis point toward a specific applied possibility:
        an NLP system that could assist — not replace — human triage of mental health
        content at scale. Reddit receives millions of posts daily across mental health
        communities. Human moderators cannot read everything. An automated system that
        could reliably flag acute crisis content for human review would have genuine
        clinical utility. This section examines what such a system would require,
        what this analysis contributes toward it, and what significant obstacles remain.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("## What This Analysis Already Contributes")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div style='background-color:#F0F4F8;padding:18px;border-radius:8px;border-top:4px solid #5CB85C;min-height:220px'>
          <h4 style='margin:0 0 10px 0;color:#2C3E50'>Temporally robust representations</h4>
          <p style='color:#444;font-size:0.88em;line-height:1.6;margin:0'>
            The cross-period classifier loses less than 0.4% accuracy across three years.
            A system built on sentence embeddings would not need to be retrained every time
            community vocabulary evolves. This is a significant practical advantage.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style='background-color:#F0F4F8;padding:18px;border-radius:8px;border-top:4px solid #5CB85C;min-height:220px'>
          <h4 style='margin:0 0 10px 0;color:#2C3E50'>Clinical label superiority</h4>
          <p style='color:#444;font-size:0.88em;line-height:1.6;margin:0'>
            Expert-annotated labels produce 70.8% accuracy on 800 posts versus 62.9% on
            5,000 community posts. A triage system trained on clinically meaningful categories
            would be more accurate and more interpretable.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div style='background-color:#F0F4F8;padding:18px;border-radius:8px;border-top:4px solid #5CB85C;min-height:220px'>
          <h4 style='margin:0 0 10px 0;color:#2C3E50'>Community profiling at scale</h4>
          <p style='color:#444;font-size:0.88em;line-height:1.6;margin:0'>
            The longitudinal analysis demonstrates that thematic and affective structure
            can be tracked across time. A monitoring system could detect shifts in crisis
            language proportions — the growth from 9.0% to 13.3% between 2019 and 2021.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("## The Critical Obstacle")

    big_stat(
        "0.954",
        "The obstacle that no method overcomes — depression and SuicideWatch are semantically nearly identical<br>"
        "<strong>Any triage system built on language alone faces this boundary</strong>",
        "#D9534F"
    )

    st.markdown("""
    The central finding of this analysis is also its most important constraint on future
    development. A system that cannot reliably distinguish depressive from suicidal
    discourse is not safe to deploy in a clinical context without human oversight at
    every decision point. This does not mean a triage system is impossible — it means
    any responsible system must be designed around this limitation rather than in spite of it.
    """)

    st.divider()

    st.markdown("## What Would Need to Change")

    for title, colour, body in [
        ("Domain-specific fine-tuning", "#4A90D9",
         "The all-MiniLM-L6-v2 model was trained on general web text. A model fine-tuned specifically on mental health discourse — using clinical notes, crisis hotline transcripts, or annotated social media data — would produce better representations. Models like MentalBERT and MentalRoBERTa have been developed for exactly this purpose."),
        ("Larger and richer annotation", "#5CB85C",
         "800 annotated posts is sufficient to demonstrate the superiority of clinical labels. It is not sufficient to train a production triage system. A responsible system would require tens of thousands of posts annotated by clinical psychologists using validated severity scales such as the Columbia Suicide Severity Rating Scale (C-SSRS)."),
        ("Multi-modal signals", "#E8A838",
         "Language alone is insufficient. A more capable triage system would incorporate posting time, posting frequency, post history, and engagement patterns. These signals are available in the Reddit dataset but were not used in this analysis."),
        ("Human-in-the-loop architecture", "#9B59B6",
         "Any responsible deployment would use NLP as a filter rather than a decision-maker. The system would flag posts above a confidence threshold for human review. The depression/SuicideWatch finding means that threshold would need to be set conservatively, accepting a high false positive rate to minimise false negatives."),
        ("Ethical and consent framework", "#D9534F",
         "Users who post in mental health communities have not consented to algorithmic monitoring. A production system would require a transparent privacy policy, clear disclosure, and careful consideration of harms from false positives. These are not technical problems — they require engagement with clinical ethics and platform governance."),
    ]:
        st.markdown(f"""
        <div style='background-color:#F8F9FA;padding:20px;border-radius:6px;
                    border-left:4px solid {colour};margin-bottom:14px'>
          <h3 style='margin:0 0 10px 0;color:#2C3E50'>{title}</h3>
          <p style='margin:0;color:#444;line-height:1.7'>{body}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("## Immediate Extensions of This Analysis")

    e1, e2, e3, e4 = st.columns(4)

    for col, title, colour, body in [
        (e1, "Full longitudinal", "#9B59B6",
         "The full RMHD dataset spans 44 months. A monthly time series would allow event-level analysis — mapping specific events against linguistic shifts."),
        (e2, "Control corpus", "#E8A838",
         "Adding r/all or a matched general discussion subreddit would establish what is distinctive about mental health discourse versus general Reddit language."),
        (e3, "Fine-tuned model", "#4A90D9",
         "Replacing all-MiniLM-L6-v2 with MentalBERT and repeating the full analysis would test whether a domain-specific model changes the key findings."),
        (e4, "Semantic frame analysis", "#5CB85C",
         "Using an LLM to name the conceptual frames evoked by each topic cluster would add theoretical depth and connect more directly to cognitive linguistic theory."),
    ]:
        with col:
            st.markdown(f"""
            <div style='background-color:#F0F4F8;padding:16px;border-radius:8px;
                        border-top:4px solid {colour};min-height:200px'>
              <h4 style='margin:0 0 8px 0;color:#2C3E50'>{title}</h4>
              <p style='color:#444;font-size:0.85em;line-height:1.6;margin:0'>{body}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px;border-radius:8px'>
      <h2 style='color:white;margin:0 0 14px 0'>The Responsible Path Forward</h2>
      <p style='color:#BDC3C7;line-height:1.7;margin:0 0 14px 0'>
        This analysis demonstrates that NLP methods can reveal meaningful structure
        in mental health discourse — structure that is temporally stable, consistent
        across methods, and clinically interpretable. It also demonstrates clearly
        where those methods reach their limits.
      </p>
      <p style='color:white;line-height:1.7;margin:0;font-weight:500'>
        The depression/SuicideWatch finding is not a reason to abandon this work.
        It is a reason to do it carefully, with clinical expertise involved at every
        stage, and with honest acknowledgement of what language can and cannot tell
        us about the inner states of the people who use it.
      </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# VIEW: LIMITATIONS
# ============================================================

elif active_tab == "⚠️ Limitations":

    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px 30px;border-radius:8px;margin-bottom:24px'>
      <h1 style='color:white;margin:0 0 10px 0'>Limitations and Critical Reflection</h1>
      <p style='color:#BDC3C7;margin:0;line-height:1.7'>
        The most important limitation of this tool is also its most important finding.
        The inability to separate depressive from suicidal discourse computationally is
        not a bug — it reflects a genuine clinical reality that language alone cannot resolve.
      </p>
    </div>
    """, unsafe_allow_html=True)

    for lim in [
        {"title": "What NLP Cannot Detect", "colour": "#D9534F",
         "body": "All five methods operate on surface text and cannot access the implicit social meaning of posting to a public community. A post with no explicit request for help may still function as an implicit appeal. Sarcasm, irony, and understatement are invisible. The emotional weight of a post cannot be read from its VAD score alone — a post scoring low on valence may be expressing clinical despair or everyday frustration."},
        {"title": "The Depression/SuicideWatch Boundary", "colour": "#D9534F",
         "body": "The near-identical semantic profile resists computational resolution not because the methods are inadequate but because the experiences genuinely overlap. Depression is the primary risk factor for suicidal ideation. The language of hopelessness, exhaustion, and the desire for relief from suffering is shared across both conditions. This is a finding about the nature of mental health discourse, not a limitation of the analysis."},
        {"title": "The Embedding Model Caveat", "colour": "#9B59B6",
         "body": "The temporal stability finding may partly reflect the consistency of the all-MiniLM-L6-v2 model rather than the language itself. All four years were embedded using the same pre-trained model. A stronger test would use a model fine-tuned on each year's data separately."},
        {"title": "The Outlier Problem", "colour": "#9B59B6",
         "body": "Before outlier reduction, 50% of posts in April 2022 and over 65% in prior years were assigned to no BERTopic topic cluster. This reflects the genuinely idiosyncratic nature of personal mental health disclosure. The high base rate of outliers is itself a finding: mental health discourse resists topical organisation in ways that other corpora do not."},
        {"title": "No Control Corpus", "colour": "#E8A838",
         "body": "This analysis has no baseline comparison against general Reddit discourse. Without a control corpus it is not possible to establish which linguistic features are specific to mental health communities and which reflect general properties of Reddit as a platform."},
        {"title": "Community Labels as Proxies", "colour": "#E8A838",
         "body": "Platform community membership is an imperfect proxy for clinical mental state. Users self-select into communities based on perceived relevance, social norms, and moderation practices rather than clinical diagnosis. The superiority of clinical labels over community labels in Classifier B confirms this directly."},
    ]:
        st.markdown(f"""
        <div style='background-color:#F8F9FA;padding:20px;border-radius:6px;
                    border-left:4px solid {lim["colour"]};margin-bottom:14px'>
          <h3 style='margin:0 0 10px 0;color:#2C3E50'>{lim["title"]}</h3>
          <p style='margin:0;color:#444;line-height:1.7'>{lim["body"]}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("## Interactive Demonstration")
    st.markdown("The limitations above are abstract. This demonstration makes them concrete.")

    if st.button("Run the borderline case", type="primary"):
        with st.spinner("Loading models and analysing..."):
            embed_model = load_embedding_model()
            clf_a, clf_b, labels_a, labels_b = load_classifiers()

            dep_text = "I haven't left my bed in three days. I just don't see the point anymore. Everything feels grey and empty."
            sw_text  = "I am so tired of fighting every single day. I don't want to die but I can't keep living like this. I just want it all to stop."

            dep_emb = embed_model.encode([basic_clean(dep_text)])
            sw_emb  = embed_model.encode([basic_clean(sw_text)])

            similarity = float(np.dot(dep_emb[0], sw_emb[0]) / (
                np.linalg.norm(dep_emb[0]) * np.linalg.norm(sw_emb[0])
            ))

            dep_probs = clf_a.predict_proba(dep_emb)[0]
            sw_probs  = clf_a.predict_proba(sw_emb)[0]
            dep_pred  = labels_a[np.argmax(dep_probs)]
            sw_pred   = labels_a[np.argmax(sw_probs)]

        st.markdown(f"""
        <div style='background-color:#FFF3CD;padding:20px;border-radius:8px;margin:16px 0;border:1px solid #E8A838'>
          <p style='margin:0 0 8px 0;color:#444'><strong>Post A (from r/depression):</strong> "{dep_text}"</p>
          <p style='margin:0;color:#444'><strong>Post B (from r/SuicideWatch):</strong> "{sw_text}"</p>
        </div>
        """, unsafe_allow_html=True)

        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Post A classified as", f"r/{dep_pred}", f"{max(dep_probs):.1%} confidence")
        dc2.metric("Post B classified as", f"r/{sw_pred}",  f"{max(sw_probs):.1%} confidence")
        dc3.metric("Semantic similarity",  f"{similarity:.3f}",
                   help="Individual post similarity — not community centroid similarity (0.954)")

        sw_in_dep = dep_probs[labels_a.index('SuicideWatch')]
        dep_in_sw = sw_probs[labels_a.index('depression')]

        st.markdown(f"""
        <div style='background-color:#F8D7DA;padding:20px;border-radius:8px;margin:16px 0;border-left:4px solid #D9534F'>
          <h4 style='margin:0 0 10px 0;color:#2C3E50'>What this shows</h4>
          <p style='margin:0;color:#444;line-height:1.7'>
            The classifier correctly identifies both posts. But SuicideWatch receives
            {sw_in_dep:.1%} of the probability mass for Post A, and depression receives
            {dep_in_sw:.1%} for Post B. The 0.954 community-level similarity means these
            two communities occupy almost identical regions of semantic space. A classifier
            operating in that space will always be uncertain at the boundary — and the
            boundary is where clinical decisions matter most.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("## Evidence from Manual Annotation")
    st.markdown("""
    As part of this project, 30 Reddit posts were manually annotated for help-seeking
    vs help-providing orientation. The annotation process revealed that some posts
    genuinely could not be resolved as one category or the other — leading to the
    addition of a boundary category (HS/V) during coding.
    """)

    a1, a2, a3 = st.columns(3)
    for col, title, colour, body in [
        (a1, "The Poem", "#D9534F",
         "A poem used rhetorical help-seeking questions throughout but ended with an offer of solidarity. It simultaneously asked and offered — impossible to code as either."),
        (a2, '"Does this happen to anyone else?"', "#E8A838",
         "A single question that simultaneously seeks validation and implicitly tells other sufferers they are not alone. One sentence, two communicative functions."),
        (a3, "The Deafening Loneliness Post", "#9B59B6",
         "Explicitly solicited input about the writer's own situation AND explicitly invited others to share their problems. The most balanced mixed case in the dataset."),
    ]:
        with col:
            st.markdown(f"""
            <div style='background-color:#F8F9FA;padding:16px;border-radius:6px;border-top:3px solid {colour};min-height:180px'>
              <h4 style='margin:0 0 8px 0;color:#2C3E50'>{title}</h4>
              <p style='margin:0;color:#444;font-size:0.88em;line-height:1.6'>{body}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color:#EBF5FB;padding:16px;border-radius:6px;border-left:3px solid #4A90D9;margin-top:16px'>
      <p style='margin:0;color:#1A5276;line-height:1.7'>
        <strong>The key insight:</strong> if a trained human annotator with a clear coding scheme
        cannot reliably classify some posts, we should not expect a machine classifier to do so
        either. NLP can identify patterns at scale, but it cannot resolve genuine communicative
        ambiguity that resists even careful human judgment.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px;border-radius:8px'>
      <h2 style='color:white;margin:0 0 14px 0'>What This Means for Clinical Deployment</h2>
      <p style='color:#BDC3C7;line-height:1.7;margin:0 0 14px 0'>
        The methods demonstrated in this tool have genuine utility for mental health research
        at scale. They are not suitable, as currently implemented, for clinical triage or
        automated crisis detection.
      </p>
      <p style='color:white;line-height:1.7;margin:0;font-weight:500'>
        The goal of this tool is not to replace clinical judgment. It is to demonstrate
        what computational methods can reveal about language at scale, to show where those
        methods reach their limits, and to be transparent about what those limits mean.
        Knowing what NLP cannot do is as important as knowing what it can.
      </p>
    </div>
    """, unsafe_allow_html=True)
