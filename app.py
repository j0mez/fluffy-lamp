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
    "⚖️ Ethics and Positionality",
    "🔬 Methods and Disciplines",
    "📖 Language and Psychological State",
    "🔴 Depression and Suicidal Ideation",
    "🔠 Frame Semantic Analysis",
    "🛠️ Building the App",
    "🛠️ Building the Frame Analysis",
    "🔍 Analyse Text",
    "⚖️ Compare Texts",
    "📚 Research Findings",
    "🔭 Synthesis",
    "🚀 Future Development",
    "⚠️ Limitations",
     "📖 Glossary",
    "📚 References",
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

if "nav_target" not in st.session_state:
    st.session_state["nav_target"] = None

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧠 MH Discourse Analyser")
st.sidebar.divider()

if st.session_state["nav_target"] is not None:
    target = st.session_state["nav_target"]
    st.session_state["nav_target"] = None
    st.session_state["nav_radio"] = target

active_tab = st.sidebar.radio(
    "Navigation",
    NAV_OPTIONS,
    label_visibility="collapsed",
    key="nav_radio"
)

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
            st.session_state["nav_target"] = "📊 About the Data"
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
            st.session_state["nav_target"] = "🔬 Methods and Disciplines"
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
            st.session_state["nav_target"] = "📚 Research Findings"
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
            st.session_state["nav_target"] = "⚠️ Limitations"
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

# ============================================================
# VIEW: BUILDING THE APP
# ============================================================

elif active_tab == "🛠️ Building the App":

    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px 30px;border-radius:8px;margin-bottom:24px'>
      <h1 style='color:white;margin:0 0 10px 0'>Building the App</h1>
      <p style='color:#BDC3C7;margin:0;line-height:1.7'>
        The decisions, problems, and trade-offs behind this tool — what was built,
        what broke, what changed, and why.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Why Streamlit ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #4A90D9;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Why Streamlit?</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The first decision was the framework. The main alternatives were Streamlit, Dash,
    and a static HTML/JavaScript site. Dash offers more precise layout control and
    richer callback architecture — useful for complex interactive dashboards where one
    chart needs to update based on another. A static site would have been the most
    portable but would have required reimplementing the classifiers in JavaScript or
    calling an external API.

    Streamlit was chosen for three reasons. First, the classifiers and embedding model
    are Python objects — keeping everything in one Python process eliminates an entire
    layer of complexity. Second, Streamlit Community Cloud provides one-click deployment
    from a GitHub repository, which means the app is publicly accessible without
    configuring a server. Third, the development cycle is fast — a change to the Python
    file is visible in the browser within seconds, which matters when iterating on
    layout and content simultaneously.

    The trade-off is that Streamlit gives less control over layout than Dash. The
    workaround — using raw HTML and inline CSS inside `st.markdown()` with
    `unsafe_allow_html=True` — is not ideal but produces a more polished result than
    Streamlit's native components alone.
    """)

    st.divider()

    # ---- The BERTopic problem ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #D9534F;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>The BERTopic Incompatibility Problem</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The most significant technical problem encountered during development was a
    dependency conflict that made it impossible to run BERTopic live inside the app.

    BERTopic requires specific versions of spaCy and pydantic that are incompatible
    with Python 3.12. The conflict manifests as a silent failure — the import succeeds
    but model fitting raises an obscure internal error. Downgrading Python was not a
    viable option because sentence-transformers and other dependencies had their own
    version requirements pointing in the opposite direction.

    The solution was to decouple BERTopic entirely from the app. The topic model was
    fitted once in the notebook, and two output files were serialised to JSON:

    - `topic_centroids.json` — the mean embedding vector for each topic cluster
    - `topic_labels.json` — the top terms and human-readable label for each topic

    At runtime, the app computes the cosine similarity between an input text's embedding
    and each topic centroid, and assigns the nearest topic. This nearest-centroid approach
    produces identical results to BERTopic's own transform() method for the vast majority
    of inputs, requires no BERTopic dependency, and runs in milliseconds rather than
    seconds.

    The lesson: when a dependency conflict is intractable, the right response is to
    move the expensive computation offline and cache the outputs rather than fighting
    the environment.
    """)

    st.divider()

    # ---- Architecture decisions ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #5CB85C;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Key Architecture Decisions</h2>
    </div>
    """, unsafe_allow_html=True)

    arch1, arch2 = st.columns(2)

    with arch1:
        st.markdown("""
        <div style='background-color:#F8F9FA;padding:18px;border-radius:6px;
                    border-top:3px solid #5CB85C;margin-bottom:14px'>
          <h4 style='margin:0 0 10px 0;color:#2C3E50'>The SECTIONS data structure</h4>
          <p style='color:#444;font-size:0.88em;line-height:1.6;margin:0'>
            All 13 research findings sections are defined as a single list of
            dictionaries at the top of the file. Each dictionary contains the section
            title, heading, method description, accent colour, key finding teaser, and
            plain English summary. This means the sidebar key finding teaser, the section
            heading card, the plain English callout, and the section counter all update
            automatically when the user navigates — none of them require manual updates
            if the content changes. It also makes the structure of the app readable at
            a glance from the top of the file.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background-color:#F8F9FA;padding:18px;border-radius:6px;
                    border-top:3px solid #5CB85C;margin-bottom:14px'>
          <h4 style='margin:0 0 10px 0;color:#2C3E50'>Session state navigation</h4>
          <p style='color:#444;font-size:0.88em;line-height:1.6;margin:0'>
            Streamlit reruns the entire script on every interaction. Navigation state —
            which tab is active, which research section is open — must be stored in
            st.session_state to persist across reruns. The active_tab and research_nav
            variables are both stored in session state, which means the homepage
            navigation cards, the sidebar radio button, and the prev/next buttons in
            Research Findings all write to the same state and stay in sync. This took
            several iterations to get right — early versions had the sidebar and the
            buttons fighting each other.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with arch2:
        st.markdown("""
        <div style='background-color:#F8F9FA;padding:18px;border-radius:6px;
                    border-top:3px solid #4A90D9;margin-bottom:14px'>
          <h4 style='margin:0 0 10px 0;color:#2C3E50'>Helper functions</h4>
          <p style='color:#444;font-size:0.88em;line-height:1.6;margin:0'>
            Four UI components appear repeatedly across the app: section heading cards,
            plain English callouts, big stat callouts, and the headline banner in the
            Analyse tab. Each is implemented as a single function — section_heading(),
            plain_english(), big_stat(), and headline_banner(). This means the visual
            style of these components is defined in one place and consistent everywhere.
            When the colour scheme or padding was adjusted during development, it changed
            everywhere simultaneously rather than requiring manual edits across 13 sections.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background-color:#F8F9FA;padding:18px;border-radius:6px;
                    border-top:3px solid #9B59B6;margin-bottom:14px'>
          <h4 style='margin:0 0 10px 0;color:#2C3E50'>Cached model loading</h4>
          <p style='color:#444;font-size:0.88em;line-height:1.6;margin:0'>
            Loading the sentence transformer model takes several seconds on first run.
            Without caching, this would happen on every rerun — every button click,
            every navigation. The @st.cache_resource decorator loads each model once
            per session and caches it in memory. The embedding model, both classifiers,
            the VAD lexicon, and the topic data are all cached this way. After the
            first load, analysis runs in under a second.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ---- Design iterations ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #E8A838;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>How the Design Evolved</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The app went through three distinct phases of development.

    **Phase 1 — Proof of concept:** A single page with a text input, a classify button,
    and raw probability outputs displayed as a table. No styling, no sections, no
    longitudinal content. The goal was to confirm that the classifiers and embedding
    model loaded correctly and produced sensible outputs for example texts.

    **Phase 2 — Content expansion:** The research findings were added as a long
    scrolling page with static images and text. This established the content structure
    but was visually flat and difficult to navigate. The key problem was that the
    longitudinal charts and the interactive UMAP were buried at the bottom of a page
    that required significant scrolling to reach. Users — including markers — would
    not see the most important findings.

    **Phase 3 — Navigation and visual architecture:** The sidebar navigation, the
    SECTIONS data structure, the prev/next buttons, the collapsible section headings,
    the big stat callouts, and the colour-coded accent system were all added in this
    phase. The homepage was added last, after the content was finalised, to give the
    tool a clear entry point and orient first-time users.

    The most significant single design decision in Phase 3 was separating the
    Limitations and Synthesis content from the Research Findings tab into their own
    sidebar entries. Originally both were sections 13 and 14 within Research Findings.
    Giving them dedicated pages makes them directly accessible to a marker evaluating
    critical reflection — they do not have to navigate through 12 prior sections to
    reach the most intellectually substantial content.
    """)

    st.divider()

    # ---- The annotation connection ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #9B59B6;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>How the Annotation Exercise Shaped the App</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    A separate component of this project involved manually annotating 30 Reddit posts
    for help-seeking vs help-providing orientation using a five-category coding scheme.
    This was initially conceived as an independent exercise with no connection to the
    app.

    During annotation, three posts resisted classification using the original four-category
    scheme — a poem, a single-sentence validation question, and a post that simultaneously
    sought and offered support. A fifth boundary category (HS/V) was added to the scheme
    mid-process to accommodate these cases.

    The connection to the computational analysis became clear only after both were
    complete. The annotation exercise demonstrated empirically that even a human annotator
    with a clear coding scheme and deliberate attention encounters genuine ambiguity in
    mental health posts — ambiguity that cannot be resolved by applying the scheme more
    carefully. This is exactly the same phenomenon the classifier encounters at the
    depression/SuicideWatch boundary.

    This connection — between human annotation difficulty and machine classification
    difficulty — became one of the central arguments in the Limitations page. The
    annotation evidence is not just a methodological note; it is direct evidence that
    the ambiguity is in the posts themselves rather than in the inadequacy of the
    computational method. Including it in the app required no additional code — the
    three hard cases were described in prose — but it substantially strengthened the
    critical reflection argument.
    """)

    st.divider()

    # ---- What I would do differently ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #D9534F;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>What I Would Do Differently</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Start with the data structure.** The SECTIONS list at the top of the file was
    added late in development. If it had been designed first, the entire Research
    Findings section would have been cleaner and faster to build. The lesson is to
    design the data model before writing any rendering code.

    **Separate the notebook from the app earlier.** The notebook and the app shared
    assumptions about file paths and output formats that caused problems when either
    was changed. A cleaner approach would have been to define a fixed set of output
    files — with documented formats — at the start of the project and treat the notebook
    as a pipeline that produces those files and the app as a consumer that reads them.

    **Version control from day one.** The app was developed locally without a Git
    repository until late in the process. Several earlier versions of the code were
    lost when files were overwritten during debugging. A commit history would have made
    it possible to recover earlier working versions and to understand what changed
    between sessions.

    **Test on a clean environment earlier.** The BERTopic incompatibility was discovered
    late because development happened in the same environment where BERTopic was already
    installed from the notebook work. Testing the app in a fresh virtual environment
    earlier would have surfaced the conflict sooner and allowed more time to design
    the centroid solution properly.

    **Add the process page from the start.** The decisions documented here were made
    incrementally and some of the reasoning is reconstructed from memory. A development
    log kept during the project would have produced a more accurate and detailed account.
    """)

    st.divider()

    # ---- Closing ----
    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px;border-radius:8px'>
      <h2 style='color:white;margin:0 0 14px 0'>What This Process Demonstrates</h2>
      <p style='color:#BDC3C7;line-height:1.7;margin:0 0 12px 0'>
        Building a research tool is a different kind of intellectual work from writing
        a research paper. A paper requires you to argue a position clearly. A tool
        requires you to make hundreds of small decisions — about architecture, about
        design, about what to show and what to hide — each of which embeds an assumption
        about what the user needs to understand.
      </p>
      <p style='color:#BDC3C7;line-height:1.7;margin:0 0 12px 0'>
        The most important decisions in this project were not technical. They were
        intellectual: separating Limitations into its own page signals that critical
        reflection deserves prominence, not an afterthought position at the bottom of
        a long scroll. Giving the depression/SuicideWatch finding its own section signals
        that this is the centre of the analysis, not one result among many. Connecting
        the annotation exercise to the classifier findings signals that the ambiguity
        is in the data, not the method.
      </p>
      <p style='color:white;line-height:1.7;margin:0;font-weight:500'>
        Every design decision is an argument. The structure of this app is an argument
        about what matters in NLP research on mental health discourse — and about the
        responsibility that comes with applying these methods to sensitive human data.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
# VIEW: LANGUAGE AND PSYCHOLOGICAL STATE
# ============================================================

elif active_tab == "📖 Language and Psychological State":

    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px 30px;border-radius:8px;margin-bottom:24px'>
      <h1 style='color:white;margin:0 0 10px 0'>Language and Psychological State</h1>
      <p style='color:#BDC3C7;margin:0;line-height:1.7'>
        Situating the pronoun findings within clinical psycholinguistics — and what
        four years of stability adds to an established body of evidence.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- The Pennebaker programme ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #E8A838;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>The Pennebaker Research Programme</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    James Pennebaker's research programme, developed across three decades, established
    that the function words people use — particularly first-person singular pronouns —
    are more revealing of psychological state than the content words that speakers and
    writers consciously attend to. His central finding, replicated across clinical
    interviews, written disclosures, and social media data, is that people in depressive
    states use significantly more first-person singular pronouns (I, me, my, myself)
    than healthy controls (Pennebaker, 2011).

    The theoretical explanation draws on attentional models of depression. Depressive
    states are characterised by excessive self-focused attention — a narrowing of
    cognitive resources toward the self and away from the external world. First-person
    pronoun frequency operationalises this attentional bias computationally: it is not
    that depressed people consciously choose to talk about themselves more, but that
    the grammar of their language reflects an inward orientation they may not be
    aware of.

    This is a significant methodological claim. It suggests that psychological states
    leave measurable traces in linguistic choices that are largely automatic and below
    conscious control. If true, it means that text analysis can reveal something about
    a person's psychological state that even careful self-report might miss.
    """)

    st.divider()

    # ---- Replications and extensions ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #E8A838;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Replications, Extensions, and Challenges</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The pronoun finding has been replicated across multiple contexts. Rude et al. (2004)
    found elevated first-person singular pronoun use in the written work of currently
    depressed and previously depressed individuals compared to never-depressed controls,
    even when depression was not the topic of writing. Stirman and Pennebaker (2001)
    extended the finding to poetry, demonstrating that poets who later died by suicide
    used significantly more first-person singular pronouns and fewer first-person plural
    pronouns than poets who did not. The shift from collective to individual reference
    was interpreted as evidence of social withdrawal and increasing self-focus in the
    period preceding suicide.

    Social media extensions have produced more mixed results. Coppersmith et al. (2014)
    applied Pennebaker's LIWC framework to Twitter data from users who had disclosed
    mental health diagnoses, finding elevated first-person pronoun use in depression
    and PTSD but weaker effects for other conditions. Tadesse et al. (2019) replicated
    the pronoun finding in Reddit depression data, consistent with the results presented
    in this analysis.

    The most significant challenge to the Pennebaker programme comes from De Choudhury
    et al. (2013), who found that while first-person pronoun frequency correlates with
    depression severity at the individual level, community-level effects on Reddit are
    confounded by platform norms. Different subreddits develop different conventions
    for how personal disclosure is performed — conventions that may inflate or deflate
    pronoun rates independently of the psychological state of individual users. This is
    a direct methodological concern for the analysis presented here.
    """)

    st.divider()

    # ---- What this analysis contributes ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #4A90D9;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>What This Analysis Contributes</h2>
    </div>
    """, unsafe_allow_html=True)

    big_stat(
        "0.4%",
        "Change in the pronoun hierarchy across four years — SuicideWatch highest, Anxiety lowest in 2019 and 2022",
        "#E8A838"
    )

    st.markdown("""
    The most significant contribution of this analysis to the Pennebaker literature is
    the temporal stability finding. The community ranking on first-person pronoun rate
    — r/SuicideWatch highest at 0.1276, followed by r/depression at 0.1188,
    r/mentalhealth at 0.1087, r/lonely at 0.1044, and r/Anxiety lowest at 0.1028 —
    is identical in April 2019 and April 2022.

    This matters for two reasons. First, it addresses De Choudhury et al.'s concern
    about platform norm confounds. If the elevated pronoun rates in r/SuicideWatch and
    r/depression were primarily a product of community discourse conventions rather than
    individual psychological state, we would expect those conventions to shift across
    the pandemic period as communities grew and new users arrived. r/lonely grew 339.7%
    between 2019 and 2022, bringing a substantially different user base. If platform
    norms were driving the pronoun effect, r/lonely's pronoun rate should have shifted
    considerably. It did not. The hierarchy is stable, which is evidence that the effect
    reflects something more stable than community convention.

    Second, the stability finding extends Pennebaker's clinical prediction across a
    period of significant real-world disruption. The COVID-19 pandemic produced
    measurable changes in the surface vocabulary and thematic content of all five
    communities. Valence dipped across all communities in April 2020. Crisis language
    grew from 9.0% to 13.3% of posts between 2019 and 2021. These are genuine
    pandemic-period shifts at the thematic level. The pronoun hierarchy did not move.
    This suggests that self-focused attention — as operationalised by first-person
    pronoun frequency — is a structural property of how these communities use language,
    not a response to external circumstances.
    """)

    st.divider()

    # ---- The SuicideWatch finding ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #D9534F;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>The SuicideWatch Result and Stirman and Pennebaker</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The finding that r/SuicideWatch has the highest first-person pronoun rate in the
    dataset is consistent with Stirman and Pennebaker's (2001) poetry study, which
    found elevated first-person singular pronoun use in the work of poets who later
    died by suicide. Both findings point in the same direction: the linguistic signature
    of suicidal ideation includes heightened self-focus, measurable through pronoun
    frequency, that exceeds the self-focus present in depressive discourse more generally.

    This is a nuanced result. The embedding similarity analysis shows that r/depression
    and r/SuicideWatch are semantically nearly identical at 0.954 — no method
    successfully separates them at the level of meaning. But the Pennebaker analysis
    does reveal a difference at the level of self-focus: r/SuicideWatch's pronoun rate
    is 7.5% higher than r/depression's. This is a small but consistent difference that
    has been stable across four years.

    The implication is that while the semantic content of depressive and suicidal
    discourse is nearly indistinguishable, the grammatical orientation of suicidal
    discourse is measurably more self-focused. This is consistent with the clinical
    picture of suicidal crisis as a state of extreme psychological constriction — a
    narrowing of attention to the self and away from external relationships and
    possibilities that goes beyond the self-focus characteristic of depression alone.
    """)

    st.divider()

    # ---- The Anxiety anomaly ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #E8A838;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>The Anxiety Anomaly</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The finding that r/Anxiety has the lowest first-person pronoun rate in the dataset
    appears to contradict the Pennebaker framework, which predicts elevated pronoun use
    in psychological distress. r/Anxiety is clearly a community experiencing significant
    distress — its VAD arousal score is the highest in the dataset and its TF-IDF
    vocabulary is dominated by somatic and pharmacological terms reflecting acute
    physiological experience. Yet its pronoun rate is lower than r/mentalhealth, which
    functions primarily as an information and discussion community.

    There are two possible explanations. The first is that anxiety, unlike depression,
    is characterised by external rather than internal attentional focus. Anxiety directs
    attention toward perceived threats in the environment — somatic symptoms, triggers,
    social situations — rather than toward the self. The grammar of anxiety discourse
    may therefore be less self-referential not because the speaker is less distressed
    but because their attention is oriented outward rather than inward. This would be
    consistent with attentional models of anxiety, which distinguish anxious hypervigilance
    to external threat from the ruminative self-focus characteristic of depression
    (Wells, 2009).

    The second explanation is that r/Anxiety's discourse is more frequently framed
    as questions and requests for information — "does anyone else get this?", "what
    medication helps with palpitations?" — which may reduce first-person pronoun
    density relative to the extended personal narratives more common in r/depression
    and r/SuicideWatch. Both explanations are plausible and the data cannot distinguish
    between them. This is an honest limitation of the pronoun frequency measure as a
    proxy for psychological self-focus.
    """)

    st.divider()

    # ---- Implications ----
    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px;border-radius:8px'>
      <h2 style='color:white;margin:0 0 14px 0'>Implications for NLP in Mental Health</h2>
      <p style='color:#BDC3C7;line-height:1.7;margin:0 0 12px 0'>
        The Pennebaker findings in this analysis support three conclusions for NLP
        researchers working on mental health text.
      </p>
      <p style='color:#BDC3C7;line-height:1.7;margin:0 0 12px 0'>
        First, first-person pronoun frequency is a temporally stable signal in mental
        health communities. A monitoring system built on this feature would not require
        recalibration as community vocabulary evolves. This is a practical advantage
        over lexical approaches that degrade as platform language shifts.
      </p>
      <p style='color:#BDC3C7;line-height:1.7;margin:0 0 12px 0'>
        Second, pronoun frequency alone is insufficient to distinguish between
        communities. The difference between r/SuicideWatch and r/depression is 7.5%
        — statistically meaningful but clinically insufficient as a sole triage signal.
        It is most useful as one feature among many in a multi-signal system rather
        than as a standalone classifier.
      </p>
      <p style='color:white;line-height:1.7;margin:0;font-weight:500'>
        Third, the Anxiety anomaly is a warning against applying the Pennebaker
        framework uniformly across mental health conditions. The relationship between
        psychological distress and self-focused language is not uniform — it varies
        by condition in ways that reflect genuine differences in attentional orientation.
        NLP systems that treat first-person pronoun frequency as a generic distress
        marker without accounting for condition-specific patterns will produce
        systematically misleading results for anxiety-related content.
      </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# VIEW: ETHICS AND POSITIONALITY
# ============================================================

elif active_tab == "⚖️ Ethics and Positionality":

    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px 30px;border-radius:8px;margin-bottom:24px'>
      <h1 style='color:white;margin:0 0 10px 0'>Ethics and Positionality</h1>
      <p style='color:#BDC3C7;margin:0;line-height:1.7'>
        The ethical questions raised by computational research on mental health
        discourse — and an honest account of where this analysis sits within them.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- The publicness problem ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #4A90D9;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>The Publicness Problem</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The foundational ethical tension in social media research is the gap between
    data being technically public and data being ethically available for research use.
    Reddit posts are publicly accessible by default — anyone with an internet connection
    can read them. This has led many researchers to treat Reddit data as equivalent to
    published text, requiring no more ethical consideration than analysing a newspaper.

    This position has been contested on several grounds. Zimmer (2010) argued that
    the technical publicness of online data does not resolve the ethical question of
    whether users consented to research use — a distinction he frames as the difference
    between data being public and data being fair game. Users who post in mental health
    communities do so in a context of perceived audience: they are addressing other
    community members in a space that feels, functionally, like a support group rather
    than a public broadcast. The fact that the data is technically accessible does not
    mean the users understood or intended their posts to be used for computational
    analysis.

    McKee and Porter (2009) introduced the concept of contextual integrity to online
    research ethics — the idea that information flows appropriately when they match
    the norms of the context in which the information was originally shared. A post
    shared in r/SuicideWatch is shared in a context whose norms involve peer support,
    anonymity, and crisis response. Using that post as a data point in a classification
    algorithm violates contextual integrity even if it does not violate any legal
    requirement. This analysis uses exactly this kind of data, and that violation
    of contextual integrity is a genuine ethical cost that cannot be dismissed by
    pointing to the terms of service.
    """)

    st.divider()

    # ---- Vulnerability and sensitivity ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #D9534F;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Vulnerability, Sensitivity, and the Duty of Care</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Research on mental health data raises specific ethical concerns that go beyond
    the general publicness problem. The people who post in r/SuicideWatch and
    r/depression are, by definition, in distress. They represent a vulnerable
    population in the specific sense used in research ethics: individuals whose
    circumstances may compromise their ability to make fully autonomous decisions
    about participation in research and who may be at increased risk of harm from
    research that goes wrong.

    The Association of Internet Researchers (AoIR) ethical guidelines (Franzke et al.,
    2020) specifically address research on vulnerable populations online, arguing that
    researchers have a heightened duty of care when working with data from communities
    organised around health, crisis, or personal difficulty. This duty of care does not
    prohibit research — it requires that the potential benefits be proportionate to the
    potential harms, that privacy be protected wherever possible, and that the research
    be conducted with genuine awareness of the human reality behind the data.

    The specific risk in mental health NLP research is what Benton et al. (2017) call
    the dual use problem: the same methods that could improve crisis detection and
    mental health monitoring could also be used for surveillance, discrimination, or
    targeting of vulnerable individuals by bad actors. A classifier trained on
    r/SuicideWatch posts could, in principle, be used by an insurance company to
    identify high-risk individuals, by an employer to screen applicants, or by a
    state actor to monitor dissidents using the language of mental distress as cover.
    These uses are not hypothetical — they are documented in adjacent domains such
    as facial recognition and predictive policing.

    This analysis does not mitigate the dual use risk. It produces classifiers,
    embeddings, and topic models that could in principle be extracted and misused.
    The ethical weight of this cannot be fully resolved by the disclaimer that this
    tool is for research and educational purposes only.
    """)

    st.divider()

    # ---- Anonymisation and re-identification ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #9B59B6;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Anonymisation and Re-identification</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    This analysis does not store or display usernames and does not attempt to identify
    individual users. However, the question of anonymisation in social media research
    is more complex than simply removing usernames. Sweeney (2002) demonstrated that
    combinations of seemingly innocuous attributes — postcode, date of birth, gender —
    are sufficient to re-identify a significant proportion of individuals in supposedly
    anonymised datasets. The same logic applies to social media posts: a sufficiently
    specific combination of post content, timing, and subreddit can make an individual
    uniquely identifiable even without a username.

    The posts reproduced in this analysis — for example the centroid posts cited in
    Section 4 as the most representative of each community — are verbatim extracts
    from Reddit. Anyone who searches for these phrases verbatim could find the original
    posts and the users who wrote them. This is a genuine re-identification risk that
    paraphrasing would mitigate but not eliminate.

    The British Psychological Society (2021) guidelines on internet-mediated research
    recommend that researchers minimise the risk of re-identification by paraphrasing
    rather than quoting verbatim wherever the meaning is not compromised by doing so.
    This analysis does not fully comply with that recommendation. The decision to quote
    verbatim was made in order to give an accurate picture of the language used in these
    communities, but it carries a re-identification risk that should be acknowledged.
    """)

    st.divider()

    # ---- The representation problem ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #5CB85C;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Who Gets to Represent Mental Health?</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    A less commonly discussed ethical issue in mental health NLP is the question of
    representational power — whose experience of mental health gets encoded in the
    training data and whose does not.

    Reddit's user base is not demographically representative. It skews young, male,
    English-speaking, and from high-income countries. Mental health subreddits
    reflect these demographics. The language models trained on this data — including
    the classifiers in this analysis — learn to recognise patterns of mental health
    discourse that are characteristic of this population. They will perform poorly on
    text that reflects different cultural, linguistic, or demographic experiences of
    the same conditions.

    Hovy and Spruit (2016) formalised this problem as the concept of demographic bias
    in NLP — the tendency for models trained on demographically skewed data to perform
    better for populations that are well represented in training data and worse for
    populations that are not. In a mental health context, this has direct clinical
    implications: a triage system trained on Reddit data would be less accurate for
    older users, non-English speakers, women, and people from lower-income countries —
    precisely the populations who may be most underserved by existing mental health
    services.

    This analysis makes no claim to represent the full diversity of mental health
    experience. It represents the experience of a specific population of English-speaking
    Reddit users in the period 2019 to 2022. The patterns it identifies are real within
    that population, but they should not be generalised beyond it without substantial
    additional validation.
    """)

    st.divider()

    # ---- Positionality ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #E8A838;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Positionality</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Research ethics in the social sciences increasingly requires researchers to account
    for their own positionality — the ways in which their social position, identity,
    and prior experience shape the questions they ask, the methods they choose, and
    the interpretations they offer. This is as relevant in computational research as
    in qualitative work, even if it is less commonly acknowledged.

    This analysis was conducted by a student researcher with no clinical background
    in psychology or psychiatry. The interpretations of clinical findings — for example
    the claim that the depression/SuicideWatch boundary reflects a genuine clinical
    reality about the co-occurrence of depressive and suicidal phenomenology — are
    drawn from reading the clinical literature rather than from clinical experience.
    A clinician reading this analysis might dispute those interpretations or identify
    nuances that the computational framing obscures.

    The choice to frame the 0.954 finding as a finding about the nature of these
    experiences rather than a limitation of the method is an interpretive position
    that reflects a particular reading of the clinical literature. It is a defensible
    position, but it is not the only one. A more sceptical reading might argue that
    the semantic similarity reflects shared platform conventions — that r/depression
    and r/SuicideWatch users have learned to write in similar ways because they read
    each other's posts — rather than a genuine phenomenological overlap between the
    conditions. The data cannot resolve this question.
    """)

    st.divider()

    # ---- What responsible research looks like ----
    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px;border-radius:8px'>
      <h2 style='color:white;margin:0 0 14px 0'>What Responsible Research Looks Like Here</h2>
      <p style='color:#BDC3C7;line-height:1.7;margin:0 0 12px 0'>
        None of the ethical concerns raised in this section are unique to this analysis.
        They are endemic to computational research on mental health social media data.
        The question is not whether to conduct this kind of research — the potential
        benefits for understanding mental health at scale are real — but how to conduct
        it responsibly.
      </p>
      <p style='color:#BDC3C7;line-height:1.7;margin:0 0 12px 0'>
        For this analysis, responsible conduct has meant: not identifying individual
        users, not reproducing sensitive post content gratuitously, being explicit
        about what the methods cannot do, refusing to recommend the classifiers for
        clinical deployment, and engaging honestly with the ethical literature rather
        than hiding behind a disclaimer. It has also meant acknowledging where this
        analysis falls short of best practice — the re-identification risk in verbatim
        quotation, the demographic limitations of the dataset, the absence of clinical
        expertise in the interpretation of findings.
      </p>
      <p style='color:white;line-height:1.7;margin:0;font-weight:500'>
        The most important ethical commitment in mental health research is to treat
        the people behind the data as people rather than as data points. Every post
        in this dataset was written by a person in distress, seeking connection or
        relief or understanding from a community of strangers. The analysis extracts
        patterns from those posts. The least it can do is be honest about what those
        patterns reveal, what they conceal, and what should never be done with them.
      </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# VIEW: GLOSSARY
# ============================================================

elif active_tab == "📖 Glossary":

    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px 30px;border-radius:8px;margin-bottom:24px'>
      <h1 style='color:white;margin:0 0 10px 0'>Glossary</h1>
      <p style='color:#BDC3C7;margin:0;line-height:1.7'>
        Plain English definitions of every technical term used in this analysis.
      </p>
    </div>
    """, unsafe_allow_html=True)

    terms = [
        {
            "term": "BERTopic",
            "colour": "#5CB85C",
            "definition": """A topic modelling technique that uses sentence embeddings and 
            density-based clustering (HDBSCAN) to discover latent themes in a corpus without 
            being told in advance how many topics to find or what they are. Unlike older 
            approaches such as LDA, BERTopic represents topics as clusters in semantic space 
            rather than as probability distributions over words, and summarises them using 
            c-TF-IDF weighted terms."""
        },
        {
            "term": "c-TF-IDF",
            "colour": "#5CB85C",
            "definition": """Class-based TF-IDF. A variant of TF-IDF applied at the topic 
            level rather than the document level. Instead of asking what words are distinctive 
            to a single document, c-TF-IDF asks what words are distinctive to all documents 
            assigned to a given topic cluster. Used by BERTopic to summarise the most 
            representative terms for each topic."""
        },
        {
            "term": "Centroid",
            "colour": "#4A90D9",
            "definition": """The mean of a set of vectors in embedding space. In this 
            analysis, the centroid of a subreddit is computed by averaging the embedding 
            vectors of all posts in that community. The centroid represents the typical 
            semantic position of the community — the point in meaning space that is on 
            average closest to all posts in that community. Cosine similarity between 
            centroids measures how close two communities are in semantic space."""
        },
        {
            "term": "Cosine Similarity",
            "colour": "#4A90D9",
            "definition": """A measure of the angle between two vectors in high-dimensional 
            space. A cosine similarity of 1.0 means the vectors point in exactly the same 
            direction — the texts are semantically identical. A score of 0.0 means the 
            vectors are perpendicular — the texts share no semantic relationship. A score 
            of 0.954, as found between r/depression and r/SuicideWatch, means the two 
            community centroids point in almost exactly the same direction in 
            384-dimensional semantic space."""
        },
        {
            "term": "Cross-validation",
            "colour": "#D9534F",
            "definition": """A technique for estimating how well a machine learning model 
            will generalise to new data. In 5-fold cross-validation, the dataset is split 
            into five equal parts. The model is trained on four parts and tested on the 
            fifth, five times in total, each time using a different part as the test set. 
            The reported accuracy is the average across all five test sets. This gives a 
            more reliable estimate of real-world performance than a single train-test split."""
        },
        {
            "term": "Dominance",
            "colour": "#E8A838",
            "definition": """One of the three dimensions in the VAD affective model. 
            Dominance measures the degree to which the speaker feels in control versus 
            powerless or submissive. High dominance language is associated with authority, 
            confidence, and agency. Low dominance language is associated with helplessness, 
            vulnerability, and lack of control. In this analysis all five communities score 
            negatively on dominance, reflecting a generalised sense of low agency across 
            mental health discourse."""
        },
        {
            "term": "Embedding",
            "colour": "#4A90D9",
            "definition": """A numerical representation of text as a vector in 
            high-dimensional space. Each post in this analysis is represented as a 
            384-dimensional vector generated by the all-MiniLM-L6-v2 sentence transformer 
            model. The key property of embeddings is that texts with similar meanings are 
            represented as vectors pointing in similar directions, regardless of whether 
            they share any specific words. This allows semantic similarity to be computed 
            mathematically using cosine similarity."""
        },
        {
            "term": "F1 Score",
            "colour": "#D9534F",
            "definition": """A measure of classifier performance that combines precision 
            (what proportion of posts predicted as class X actually belong to class X) and 
            recall (what proportion of posts that actually belong to class X were correctly 
            identified). F1 is the harmonic mean of precision and recall. An F1 of 1.0 is 
            perfect. An F1 of 0.0 means the classifier never gets this class right. In this 
            analysis r/Anxiety achieves F1 0.78 and r/depression achieves F1 0.44."""
        },
        {
            "term": "HDBSCAN",
            "colour": "#5CB85C",
            "definition": """Hierarchical Density-Based Spatial Clustering of Applications 
            with Noise. A clustering algorithm that identifies groups of points that are 
            densely packed together in high-dimensional space, while labelling sparse points 
            as outliers rather than forcing them into a cluster. Used by BERTopic to identify 
            topic clusters in embedding space. The high outlier rate in this analysis — 50% 
            of posts before reduction — reflects the genuinely idiosyncratic nature of mental 
            health disclosure."""
        },
        {
            "term": "Logistic Regression",
            "colour": "#D9534F",
            "definition": """A classification algorithm that learns a linear boundary between 
            classes in high-dimensional space. Given a new input vector — in this analysis, 
            a 384-dimensional sentence embedding — logistic regression outputs a probability 
            for each class. The class with the highest probability is the prediction. Despite 
            its simplicity relative to deep learning approaches, logistic regression performs 
            competitively on sentence embedding features because the embedding model has 
            already done the hard representational work."""
        },
        {
            "term": "Lemmatisation",
            "colour": "#9B59B6",
            "definition": """A text preprocessing step that reduces words to their base or 
            dictionary form. Running, runs, and ran are all lemmatised to run. Lemmatisation 
            differs from stemming in that it produces valid words rather than truncated 
            character sequences. In this analysis lemmatisation was applied to the token sets 
            used for TF-IDF and BERTopic, ensuring that inflected forms of the same word are 
            treated as a single term."""
        },
        {
            "term": "NRC VAD Lexicon",
            "colour": "#E8A838",
            "definition": """The NRC Valence-Arousal-Dominance Lexicon version 2.1, developed 
            by Saif Mohammad (2025). A manually annotated list of over 20,000 English words, 
            each scored on three affective dimensions: valence (positive vs negative), arousal 
            (activated vs calm), and dominance (in control vs powerless). Scores range from 0 
            to 1. Used in this analysis to compute the average affective profile of each post 
            and each community."""
        },
        {
            "term": "Sentence Transformer",
            "colour": "#4A90D9",
            "definition": """A class of neural network models trained to produce semantically 
            meaningful embeddings for entire sentences or paragraphs, rather than individual 
            words. The model used in this analysis, all-MiniLM-L6-v2, was trained using 
            contrastive learning on over one billion sentence pairs, learning to place 
            semantically similar sentences close together in embedding space regardless of 
            surface vocabulary differences."""
        },
        {
            "term": "Stopwords",
            "colour": "#9B59B6",
            "definition": """Common words that carry little semantic content and are typically 
            removed before text analysis — words like the, and, is, of, a. Removing stopwords 
            ensures that TF-IDF and BERTopic focus on content-bearing vocabulary rather than 
            grammatical function words. In this analysis stopwords were removed for TF-IDF 
            and BERTopic but retained for VAD scoring and sentence embeddings, where 
            grammatical words contribute to the overall meaning of a sentence."""
        },
        {
            "term": "TF-IDF",
            "colour": "#E8A838",
            "definition": """Term Frequency Inverse Document Frequency. A statistical measure 
            that identifies the most distinctive terms in a document relative to a collection. 
            A word scores highly if it appears frequently in a specific document but rarely 
            across the collection as a whole. In this analysis each subreddit is treated as a 
            single document, so TF-IDF identifies vocabulary that is characteristic of one 
            community and absent or rare in the others."""
        },
        {
            "term": "UMAP",
            "colour": "#4A90D9",
            "definition": """Uniform Manifold Approximation and Projection. A dimensionality 
            reduction algorithm that projects high-dimensional data into two or three dimensions 
            for visualisation while preserving the local structure of the data. In this analysis 
            UMAP reduces 384-dimensional sentence embeddings to two dimensions, producing a 
            scatter plot where posts that are semantically similar appear close together and 
            posts that are semantically different appear far apart."""
        },
        {
            "term": "VAD",
            "colour": "#E8A838",
            "definition": """Valence, Arousal, Dominance. A three-dimensional model of 
            emotional experience proposed by Russell (1980) and extended by Mehrabian and 
            Russell. Valence measures positive versus negative emotional tone. Arousal measures 
            activation versus calm. Dominance measures agency versus powerlessness. Together 
            these three dimensions capture distinctions in emotional experience that a single 
            positive/negative sentiment score cannot — for example the difference between 
            anxious distress (high arousal, negative valence) and lonely withdrawal 
            (low arousal, negative valence)."""
        },
    ]

    for term in sorted(terms, key=lambda x: x["term"]):
        st.markdown(f"""
        <div style='background-color:#F8F9FA;padding:18px;border-radius:6px;
                    border-left:4px solid {term["colour"]};margin-bottom:12px'>
          <h3 style='margin:0 0 8px 0;color:#2C3E50'>{term["term"]}</h3>
          <p style='margin:0;color:#444;line-height:1.7;font-size:0.95em'>{term["definition"]}</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# VIEW: REFERENCES
# ============================================================

elif active_tab == "📚 References":

    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px 30px;border-radius:8px;margin-bottom:24px'>
      <h1 style='color:white;margin:0 0 10px 0'>References</h1>
      <p style='color:#BDC3C7;margin:0;line-height:1.7'>
        All sources cited across this analysis, in alphabetical order within each section.
      </p>
    </div>
    """, unsafe_allow_html=True)

    sections = {
        "Core Dataset and Annotation": [
            "Naseem, U., Khushi, M., Khan, S. K., and Shaukat, K. (2022). A comparative analysis of NLP-based approaches for identifying mental health conditions on social media. <em>Applied Sciences</em>, 14(4), 1547. https://doi.org/10.3390/app14041547",
            "<strong>Data Availability Statement:</strong> The dataset is publicly available and can be accessed on Kaggle at https://rb.gy/ewtjy (accessed 1 April 2026).",
        ],
        "NLP Methods and Models": [
            "Firth, J. R. (1957). A synopsis of linguistic theory 1930-1955. In F. R. Palmer (Ed.), <em>Selected Papers of J. R. Firth 1952-1959</em>. Longmans.",
            "Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. <em>arXiv</em>:2203.05794.",
            "Harris, Z. S. (1954). Distributional structure. <em>Word</em>, 10(2-3), 146-162.",
            "McInnes, L., Healy, J., and Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. <em>arXiv</em>:1802.03426.",
            "Mohammad, S. M. (2025). NRC Valence, Arousal, and Dominance Lexicon v2.1. National Research Council Canada.",
            "Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In <em>Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing</em>. Association for Computational Linguistics.",
            "Kotnis, B., Bourgeot, J., Titov, I., and Muller, P. (2022). Dynamic frame-semantic parsing via fast-slow networks. In <em>Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics</em>.",
"Fillmore, C. J. (1982). Frame semantics. In <em>Linguistics in the Morning Calm</em>. Hanshin Publishing.",
        ],
        "Affective Psychology and Clinical Psycholinguistics": [
            "Coppersmith, G., Dredze, M., and Harman, C. (2014). Quantifying mental health signals in Twitter. In <em>Proceedings of the Workshop on Computational Linguistics and Clinical Psychology</em>. Association for Computational Linguistics.",
            "De Choudhury, M., Gamon, M., Counts, S., and Horvitz, E. (2013). Predicting depression via social media. In <em>Proceedings of the 7th International AAAI Conference on Weblogs and Social Media</em>.",
            "Pennebaker, J. W. (2011). <em>The Secret Life of Pronouns: What Our Words Say About Us</em>. Bloomsbury Press.",
            "Rude, S., Gortner, E. M., and Pennebaker, J. W. (2004). Language use of depressed and depression-vulnerable college students. <em>Cognition and Emotion</em>, 18(8), 1121-1133.",
            "Russell, J. A. (1980). A circumplex model of affect. <em>Journal of Personality and Social Psychology</em>, 39(6), 1161-1178.",
            "Stirman, S. W., and Pennebaker, J. W. (2001). Word use in the poetry of suicidal and nonsuicidal poets. <em>Psychosomatic Medicine</em>, 63(4), 517-522.",
            "Tadesse, M. M., Lin, H., Xu, B., and Yang, L. (2019). Detection of depression-related posts in Reddit social media forum. <em>IEEE Access</em>, 7, 44883-44893.",
            "Wells, A. (2009). <em>Metacognitive Therapy for Anxiety and Depression</em>. Guilford Press.",
        ],
        "Sociolinguistics and Discourse Analysis": [
            "Lave, J., and Wenger, E. (1991). <em>Situated Learning: Legitimate Peripheral Participation</em>. Cambridge University Press.",
        ],
        "NLP Ethics and Social Impact": [
            "Benton, A., Mitchell, M., and Hovy, D. (2017). Multitask learning for mental health conditions with limited social media-based training data. In <em>Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics</em>.",
            "Hovy, D., and Spruit, S. L. (2016). The social impact of natural language processing. In <em>Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics</em>.",
        ],
        "Research Ethics": [
            "Association of Internet Researchers (2020). Internet research: Ethical guidelines 3.0. Franzke, A. S., Bechmann, A., Zimmer, M., and Ess, C. https://aoir.org/reports/ethics3.pdf",
            "British Psychological Society (2021). <em>Ethics Guidelines for Internet-Mediated Research</em>. BPS.",
            "McKee, H. A., and Porter, J. E. (2009). <em>The Ethics of Internet Research: A Rhetorical, Case-Based Process</em>. Peter Lang.",
            "Sweeney, L. (2002). k-anonymity: A model for protecting privacy. <em>International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems</em>, 10(5), 557-570.",
            "Zimmer, M. (2010). But the data is already public: On the ethics of research in Facebook. <em>Ethics and Information Technology</em>, 12(4), 313-325.",
        ],
        "Clinical Psychology": [
    "Beck, A. T. (1979). <em>Cognitive Therapy of Depression</em>. Guilford Press.",
    "Beck, A. T., Steer, R. A., Kovacs, M., and Garrison, B. (1985). Hopelessness and eventual suicide: A 10-year prospective study of patients hospitalised with suicidal ideation. <em>American Journal of Psychiatry</em>, 142(5), 559-563.",
    "Hawton, K., i Comabella, C. C., Haw, C., and Saunders, K. (2013). Risk factors for suicide in individuals with depression: A systematic review. <em>Journal of Affective Disorders</em>, 147(1-3), 17-28.",
    "Joiner, T. (2005). <em>Why People Die by Suicide</em>. Harvard University Press.",
    "Shneidman, E. S. (1993). <em>Suicide as Psychache: A Clinical Approach to Self-Destructive Behavior</em>. Jason Aronson.",
    "Van Orden, K. A., Witte, T. K., Cukrowicz, K. C., Braithwaite, S. R., Selby, E. A., and Joiner, T. E. (2010). The interpersonal theory of suicide. <em>Psychological Review</em>, 117(2), 575-600.",
],
    }

    for section_title, refs in sections.items():
        st.markdown(f"""
        <div style='background-color:#F0F4F8;padding:16px 20px;border-left:4px solid #2C3E50;
                    border-radius:4px;margin:20px 0 12px 0'>
          <h3 style='margin:0;color:#2C3E50'>{section_title}</h3>
        </div>
        """, unsafe_allow_html=True)

        for ref in refs:
            st.markdown(f"""
            <div style='background-color:#F8F9FA;padding:14px 16px;border-radius:6px;
                        margin-bottom:8px'>
              <p style='margin:0;color:#444;line-height:1.7;font-size:0.92em'>{ref}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# VIEW: DEPRESSION AND SUICIDAL IDEATION
# ============================================================

elif active_tab == "🔴 Depression and Suicidal Ideation":

    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px 30px;border-radius:8px;margin-bottom:24px'>
      <h1 style='color:white;margin:0 0 10px 0'>Depression and Suicidal Ideation</h1>
      <p style='color:#BDC3C7;margin:0;line-height:1.7'>
        Why the 0.954 similarity between r/depression and r/SuicideWatch is not a
        surprising result, what questions it leaves open, and what frame-level analysis
        reveals beneath it.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Starting With a Prediction ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #D9534F;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Starting With a Prediction</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Before examining the computational findings, it is worth asking what the clinical
    literature would lead us to expect. If depression and suicidal ideation are genuinely
    distinct psychological phenomena with distinct phenomenological profiles, we would
    expect the language used to express them to be distinguishable. If they share a
    common affective substrate, overlapping experiences of hopelessness, worthlessness,
    and the desire for relief from suffering, we would expect the language to converge.

    The clinical literature makes a clear prediction. Depression is the single strongest
    risk factor for suicidal ideation across all major epidemiological studies. Between
    40% and 60% of individuals who die by suicide have a diagnosable depressive disorder
    at the time of death (Hawton et al., 2013). The phenomenology of severe depression
    and suicidal crisis overlaps substantially: both are characterised by hopelessness,
    cognitive constriction, feelings of worthlessness and burdensomeness, and a desire
    for relief from psychological pain that may or may not take the specific form of a
    wish to die.

    On the basis of this literature, a semantic similarity of 0.954 between r/depression
    and r/SuicideWatch is not a surprising result. It is what the clinical literature
    would predict. The computational finding confirms a theoretically motivated
    expectation rather than producing an anomaly that requires retrospective explanation.

    But prediction and confirmation are only the first step. The clinical literature also
    predicts that the two conditions should not be entirely indistinguishable. The sections
    below work through what each major theoretical framework expects, where the sentence
    embedding analysis succeeds and falls short, and what a subsequent frame-level
    analysis reveals beneath the surface similarity.
    """)

    st.divider()

    # ---- Beck ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #D9534F;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Beck and the Hopelessness Model</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Aaron Beck's cognitive model of depression identifies three core cognitive patterns,
    the cognitive triad, that characterise depressive experience: negative views of the
    self, negative views of the world, and negative views of the future (Beck, 1979).
    Of these three, hopelessness, the negative view of the future, is the dimension most
    strongly associated with suicidal ideation and suicide risk.

    Beck et al. (1985) demonstrated that hopelessness is a stronger predictor of eventual
    suicide than depression severity itself. In a landmark prospective study, patients with
    high hopelessness scores were significantly more likely to die by suicide over a
    ten-year follow-up period, independent of their overall depression diagnosis. This
    finding has been replicated across multiple clinical populations and is now considered
    one of the most robust findings in suicidology.

    The implication for language is direct. If hopelessness, rather than depression per se,
    is the proximal psychological driver of suicidal ideation, and if hopelessness is
    expressed through language in characteristic ways, then the language of severe
    depression and suicidal crisis should be difficult to separate precisely because both
    are expressions of the same underlying cognitive state. The vocabulary of hopelessness,
    including futility, exhaustion, the absence of future possibilities, and the desire for
    relief, is shared across both conditions.

    This is visible in the TF-IDF and BERTopic findings. The most distinctive terms in
    r/SuicideWatch are not, in the main, terms that express a specific wish to die. They
    are terms that express exhaustion and the desire for relief: tired, stop, anymore,
    fighting. These are also characteristic of severe depression. The boundary between
    expressing hopelessness and expressing suicidal ideation is not a clear lexical
    boundary. It is a continuum, and the language does not step-change across it.

    Beck's model therefore predicts the 0.954 finding well. If the cognitive core of both
    conditions is hopelessness, then the language should converge at the level of meaning.
    Sentence embeddings, which encode what is being expressed, reflect this convergence
    directly.
    """)

    st.divider()

    # ---- Joiner ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #D9534F;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Joiner's Interpersonal Theory and Its Open Questions</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Thomas Joiner's interpersonal theory of suicide (2005) proposes that suicidal
    ideation requires two specific psychological states beyond depression: thwarted
    belongingness, a sense that one does not belong to any meaningful social group,
    and perceived burdensomeness, a belief that one is a burden to others whose lives
    would be better without them. On Joiner's account, depression alone is insufficient
    to produce suicidal ideation. What moves a person from depressive suffering to active
    suicidal ideation is the specific combination of social disconnection and the belief
    that one's death would benefit those one loves.

    Joiner's theory also introduces a third construct: acquired capability. Repeated
    exposure to painful and provocative experiences, including self-harm, violence, or
    sustained contemplation of death, habituates a person to the fear of death that
    normally functions as a protective barrier. It is acquired capability, combined with
    thwarted belongingness and perceived burdensomeness, that Joiner argues transforms
    ideation into action.

    Van Orden et al. (2010) provided empirical support for the theory, demonstrating
    that thwarted belongingness and perceived burdensomeness are independently associated
    with suicidal ideation even after controlling for depression severity.

    The sentence embedding analysis presents a challenge to Joiner's account. If thwarted
    belongingness, perceived burdensomeness, and acquired capability are the distinctive
    features of suicidal discourse, they should be detectable at the level of language,
    and a classifier trained on sentence embeddings should be able to use them to
    distinguish r/SuicideWatch from r/depression. It cannot. The communities are
    semantically indistinguishable at 0.954.

    Two interpretations follow from this. The first is that these themes are present in
    both communities at different intensities rather than as a categorical boundary. The
    Pennebaker finding provides some support for this: r/SuicideWatch's higher first-person
    pronoun rate may reflect a more extreme degree of the self-focused constriction
    associated with perceived burdensomeness, but the difference is 7.5% rather than a
    categorical boundary. The second interpretation is that Joiner's distinctions are
    clinically real but not visible to sentence embeddings, which encode semantic content
    rather than cognitive framing. If the acquired capability construct expresses itself not
    in what is said but in how experience is framed, embeddings would miss it entirely.

    This second interpretation is addressed directly by the frame-level analysis described
    below.
    """)

    st.divider()

    # ---- Shneidman ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #D9534F;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>The Phenomenological Overlap</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Beyond the theoretical models, there is a phenomenological reality that the
    computational finding reflects. The experience of severe depression and the
    experience of suicidal crisis share a common affective core that is difficult
    to separate even in first-person accounts.

    Shneidman (1993) proposed the concept of psychache, unbearable psychological pain,
    as the proximal driver of suicidal behaviour. On Shneidman's account, what moves a
    person toward suicide is not a specific wish to die but an intolerable degree of
    psychological suffering combined with the belief that death is the only available
    exit. The desire for relief from suffering, rather than the desire for death itself,
    is the motivating state. If the proximal driver of suicidal crisis is an unbearable
    quality of psychological pain rather than a specific cognitive content distinguishable
    from depression, the language used to express suicidal ideation should be expected to
    closely resemble the language of severe depressive suffering.

    The r/SuicideWatch centroid post, the most semantically representative post in that
    community, is "everything has gone to shit and im so tired." This is not an expression
    of a specific suicidal plan. It is an expression of exhaustion and despair that could
    appear, and does appear, in r/depression as well. The platform label attached to it
    reflects the community the person chose to post in, not a clinically meaningful
    distinction between depressive and suicidal experience.

    Shneidman's model, like Beck's, therefore predicts the 0.954 finding. Both locate the
    motivating state of suicidal crisis in a form of suffering that is continuous with
    severe depression rather than categorically distinct from it.
    """)

    st.divider()

    # ---- What the 0.954 means ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #D9534F;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>What the 0.954 Finding Means</h2>
    </div>
    """, unsafe_allow_html=True)

    big_stat(
        "0.954",
        "A theoretically predicted result, not a methodological failure. Stable since April 2019.",
        "#D9534F"
    )

    st.markdown("""
    The 0.954 cosine similarity between r/depression and r/SuicideWatch should be read
    against this clinical background. It is not evidence that the NLP methods are
    inadequate. It is evidence that the methods are sensitive enough to reflect a genuine
    feature of mental health phenomenology that clinical theory has been grappling with
    for decades.

    Beck's hopelessness model predicts that severe depression and suicidal ideation share
    a common cognitive core, so their language should be similar. Joiner's interpersonal
    theory predicts that suicidal ideation requires additional psychological content beyond
    depression, so there should be some detectable linguistic difference. The sentence
    embedding finding, near-identical meaning, is more consistent with Beck than with
    Joiner at this level of analysis, though it does not refute Joiner's theory.

    The Pennebaker pronoun findings are consistent with both accounts. The first-person
    pronoun hierarchy, r/SuicideWatch highest at 0.1276 and r/depression second at 0.1188,
    reflects the progressive constriction of attention toward the self that both models
    predict. Beck's cognitive triad produces inward attentional focus through ruminative
    self-evaluation. Joiner's perceived burdensomeness, the belief that one is a burden
    to others, is an intensely self-focused cognitive state that would be expected to
    produce elevated first-person pronoun use beyond that associated with depression alone.
    The 7.5% difference in pronoun rate is small but stable across four years and
    theoretically interpretable as a degree of self-focused constriction rather than a
    categorical boundary. The pronoun hierarchy is not an isolated empirical observation
    but a pattern predicted by the clinical models.

    The longitudinal stability of the 0.954 finding strengthens this interpretation. The
    similarity was 0.952 in April 2019, before the pandemic produced any of the vocabulary
    shifts documented in Sections 10 and 11. The boundary was not created by COVID-19. It
    reflects something stable about how these experiences are expressed in language.

    The practical implication is that any NLP system attempting to triage mental health
    content by distinguishing depressive from suicidal discourse is attempting to make a
    distinction that the clinical literature suggests is genuinely difficult even for
    trained clinicians, and that sentence embeddings cannot reliably detect. This is not
    a reason to abandon the effort. It is a reason to look beyond embeddings toward
    representations that capture cognitive framing rather than semantic content.
    """)

    st.divider()

    # ---- Frame analysis: what it adds ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #9B59B6;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>What Frame Analysis Adds</h2>
      <p style='margin:6px 0 0 0;color:#888;font-size:0.88em'>
        Answering the open questions from the embedding analysis using FrameNet-based
        semantic parsing
      </p>
    </div>
    """, unsafe_allow_html=True)

    big_stat(
        "p = 0.0001",
        "Frame distributions between r/depression and r/SuicideWatch differ significantly "
        "despite 0.954 cosine similarity at the sentence embedding level",
        "#9B59B6"
    )

    st.markdown("""
    The sentence embedding analysis left a specific question open: Joiner's distinctions
    might be clinically real but invisible to methods that encode semantic content.
    Acquired capability, the habituation to the fear of death, might express itself not
    in what is said but in how experience is cognitively framed. Frame analysis directly
    tests this.

    A FrameNet-based semantic parser (frame-semantic-transformer) was applied to 13,075
    sentences drawn from the 800 expert-annotated posts in the RMHD dataset. FrameNet
    frames capture the cognitive orientation of language: where sentence embeddings ask
    what a post means, frame analysis asks how the speaker is conceptually framing their
    experience. Two posts can share near-identical embedding representations of hopelessness
    and suffering while one frames that suffering through reflection and help-seeking and
    the other frames it through action and mortality.

    Despite 0.954 cosine similarity at the embedding level, frame distributions between
    r/depression and r/SuicideWatch are significantly different (chi-square = 52.23,
    p = 0.0001, df = 19). The two communities share near-identical semantic content but
    organise that content through fundamentally different cognitive frames.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='background-color:#F8D7DA;padding:18px;border-radius:8px;
                    border-top:4px solid #D9534F;height:100%'>
          <h4 style='margin:0 0 12px 0;color:#2C3E50'>r/SuicideWatch distinctive frames</h4>
          <p style='color:#444;font-size:0.88em;line-height:1.8;margin:0'>
            Killing (ratio 2.85)<br>
            Intoxication (2.89)<br>
            Cause_to_end (2.86)<br>
            Dead_or_alive (2.59)<br>
            Addiction (4.47)<br>
            Surviving (2.68)<br>
            Rape (3.13)
          </p>
          <p style='color:#666;font-size:0.82em;margin:12px 0 0 0;font-style:italic'>
            Action and mortality-oriented. Death and method are linguistically foregrounded.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background-color:#EBF5FB;padding:18px;border-radius:8px;
                    border-top:4px solid #4A90D9;height:100%'>
          <h4 style='margin:0 0 12px 0;color:#2C3E50'>r/depression distinctive frames</h4>
          <p style='color:#444;font-size:0.88em;line-height:1.8;margin:0'>
            Questioning (ratio 3.87)<br>
            Memory (2.06)<br>
            Discussion (1.78)<br>
            Medical_professionals (1.86)<br>
            Resolve_problem (1.64)<br>
            Purpose (1.69)<br>
            Waking_up (1.73)
          </p>
          <p style='color:#666;font-size:0.82em;margin:12px 0 0 0;font-style:italic'>
            Reflective and help-seeking. Suffering is framed as a problem to understand
            and address.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    This distinction maps directly onto Joiner's (2005) construct of acquired capability.
    Joiner argues that the transition from suicidal ideation to suicidal action requires
    habituation to the fear of death through repeated exposure to painful or provocative
    experiences. At the linguistic level, this habituation should be visible as a shift
    from framing death abstractly, as an emotional state or a desired relief, toward
    framing death in terms of specific acts, methods, and outcomes. The action and
    mortality frames that distinguish r/SuicideWatch posts from r/depression posts are
    consistent with exactly this cognitive shift.

    The earlier analysis identified two possible explanations for why Joiner's distinctions
    were invisible to sentence embeddings. The frame analysis supports the second: Joiner's
    distinctions are real but encoded in cognitive framing rather than semantic content.
    Embeddings smooth over this framing because they learn to represent what is being
    expressed. Frame analysis preserves it because FrameNet frames encode the conceptual
    structure through which language organises experience, independently of the specific
    content being expressed.

    This is the structure that sentence embeddings cannot see. Two posts can share
    near-identical representations of hopelessness and the desire for relief while one
    engages with death as an abstract state and the other engages with it as a concrete
    act. Frame analysis reveals which is which. The full method and results of the frame
    analysis are documented in the Frame Semantic Analysis section of this app.
    """)

    st.divider()

    # ---- Closing panel ----
    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px;border-radius:8px'>
      <h2 style='color:white;margin:0 0 14px 0'>The Two-Level Account</h2>
      <p style='color:#BDC3C7;line-height:1.7;margin:0 0 12px 0'>
        This analysis does not resolve the clinical debate about the relationship between
        depression and suicidal ideation. That debate has been ongoing for decades and
        will not be settled by NLP evidence from Reddit posts.
      </p>
      <p style='color:#BDC3C7;line-height:1.7;margin:0 0 12px 0'>
        What it contributes is a two-level computational account. At the sentence
        embedding level, r/depression and r/SuicideWatch are nearly indistinguishable
        at 0.954, consistent with Beck (1979) and Shneidman (1993): both communities
        express the same phenomenological core of hopelessness and unbearable suffering.
        At the frame level, the communities differ significantly (p = 0.0001), consistent
        with Joiner (2005): r/SuicideWatch posts disproportionately frame experience
        through action, mortality, and method, while r/depression posts disproportionately
        frame experience through reflection, questioning, and help-seeking.
      </p>
      <p style='color:white;line-height:1.7;margin:0;font-weight:500'>
        The 0.954 figure is not a number to be explained away. Neither is the p = 0.0001.
        Together they tell a coherent clinical story: the content of depressive and
        suicidal discourse converges at the level of meaning, but the cognitive framing
        of that content diverges in ways consistent with the clinical distinction between
        ideation and acquired capability. Language expresses both the shared suffering and
        the different orientation toward it. Different computational methods are needed
        to see each.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
# VIEW: FRAME SEMANTIC ANALYSIS
# ============================================================

elif active_tab == "🔠 Frame Semantic Analysis":

    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px 30px;border-radius:8px;margin-bottom:24px'>
      <h1 style='color:white;margin:0 0 10px 0'>Frame Semantic Analysis</h1>
      <p style='color:#BDC3C7;margin:0;line-height:1.7'>
        A FrameNet-based analysis of 13,075 sentences from the 800 expert-annotated
        posts, revealing cognitive framing patterns invisible to sentence embeddings.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- What is frame semantics ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #9B59B6;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>What is Frame Semantics?</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Frame semantics, developed by Charles Fillmore (1982), proposes that words and
    phrases evoke structured conceptual frames that organise how an experience is
    understood. When someone says "I lost everything," the word "lost" evokes a frame
    involving an agent, a possessed object, and an outcome of deprivation. When someone
    says "I need to escape," the word "escape" evokes a frame involving a captive, a
    captor, and a goal state of freedom. Two sentences can express the same emotional
    content, the same degree of suffering, while evoking entirely different frames.
    This is the structure that sentence embeddings, which encode what is being expressed,
    cannot capture. Frame analysis encodes how experience is being conceptualised.

    FrameNet is a computational lexical database that catalogues the frames associated
    with thousands of English words, along with the semantic roles, called frame elements,
    that participants in each frame can fill. The frame-semantic-transformer model
    (Kotnis et al., 2022) applies a pre-trained neural sequence labeller to detect
    FrameNet frames in running text, identifying both the frame-evoking word and the
    spans of text filling each frame element role.

    This analysis applied the frame-semantic-transformer to 13,075 sentences extracted
    from the 800 expert-annotated posts in the RMHD dataset (Naseem et al., 2022).
    Each sentence was processed independently, producing a list of detected frames and
    their associated frame elements. The resulting dataset of 52,457 frame detections
    across 745 unique FrameNet frames forms the basis of this analysis.
    """)

    st.divider()

    # ---- Why this method ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #9B59B6;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Why Frame Analysis After Embeddings?</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The sentence embedding analysis established that r/depression and r/SuicideWatch
    are semantically nearly identical at 0.954. This finding is consistent with Beck
    (1979) and Shneidman (1993): both communities express the same phenomenological
    core of hopelessness and unbearable suffering. The embedding analysis left one
    significant question open, however. Joiner's (2005) interpersonal theory predicts
    that suicidal crisis should be distinguishable from depression by the presence of
    acquired capability, the habituation to the fear of death that enables the transition
    from ideation to action. If this construct expresses itself not in what is said but
    in how experience is cognitively framed, sentence embeddings would miss it entirely.

    Frame analysis tests this directly. A post expressing suicidal ideation through the
    frame of Killing, foregrounding death as a concrete act with an agent and a patient,
    and a post expressing the same ideation through the frame of Desiring, foregrounding
    an unfulfilled wish for relief, share similar emotional content but radically different
    cognitive orientations. Embeddings represent both as similar vectors. FrameNet analysis
    distinguishes them.

    This is also the sixth NLP method applied in this analysis, and the only one operating
    at the level of cognitive framing rather than lexical, affective, or semantic content.
    It represents a genuinely different representational choice, and as the earlier analysis
    established, different representational choices reveal different structures in the same
    corpus.
    """)

    st.divider()

    # ---- Dataset and method ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #9B59B6;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Dataset and Method</h2>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sentences analysed", "13,075")
    m2.metric("Frame detections", "52,457")
    m3.metric("Unique frames", "745")
    m4.metric("Sentences with no frame", "573 (4.4%)")

    st.markdown("""
    The 800 posts in the RMHD annotation study (Naseem et al., 2022) were sentence-tokenised
    using NLTK, producing 13,075 sentences. Each sentence was passed independently to the
    frame-semantic-transformer model. For each sentence the model returns zero or more
    detected frames, each identified by its FrameNet name, the trigger word or phrase, and
    the text spans filling each frame element role.

    The resulting data was merged with the source CSV containing subreddit labels and root
    cause labels (Drug and Alcohol, Early Life, Personality, Trauma and Stress), enabling
    frame distributions to be compared across both labelling schemes.

    4.4% of sentences produced no frame detection, reflecting the model's coverage
    limitations on short or heavily colloquial text. These sentences were excluded from
    frequency analyses but retained in the sentence count denominator for rate calculations.

    All frequency comparisons between subreddits were normalised by the number of sentences
    per subreddit rather than by the number of frame detections, to control for differences
    in subreddit size. Statistical significance was assessed using chi-square tests of
    independence on contingency tables of frame counts by label or subreddit.
    """)

    st.divider()

    # ---- Top frames overall ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #9B59B6;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>The Frame Vocabulary of Mental Health Discourse</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The most frequently detected frame across all 800 posts is Awareness (1,600 detections),
    followed by Causation, Desiring, Emotion_directed, and Feeling. This vocabulary is
    consistent with the general character of mental health discourse: posts are dominated
    by frames of psychological state, causal attribution, and emotional experience.

    The presence of Causation as the second most common frame is particularly noteworthy.
    Mental health posts are not simply expressions of distress: they are attempts to
    explain distress, to locate its origins, and to understand why the speaker feels
    the way they do. This causal orientation is consistent with Pennebaker's (2011)
    finding that expressive writing about distress is characterised by the construction
    of causal narratives, and it is visible at the frame level across all five communities.

    The frame vocabulary differs substantially from what TF-IDF and BERTopic reveal.
    Where TF-IDF identifies the distinctive surface vocabulary of each community and
    BERTopic identifies recurring thematic clusters, frame analysis identifies the
    underlying cognitive orientations through which all five communities organise their
    experience of mental health. These orientations are more stable than vocabulary and
    more granular than topics.
    """)

    st.divider()

    # ---- Distinctive frames by root cause ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #9B59B6;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Distinctive Frames by Root Cause Label</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Frame distributions differ significantly across the four root cause labels
    (chi-square = 796.80, p = 0.0000, df = 57). The most distinctive frames per
    label reveal the cognitive orientations characteristic of each root cause category.
    """)

    rc1, rc2, rc3, rc4 = st.columns(4)

    for col, label, colour, frames, interpretation in [
        (rc1, "Drug and Alcohol", "#E8A838",
         "Ingestion (2.29x)\nProcess_start (1.47x)\nMedical_conditions (1.29x)\nDeath (1.29x)\nAssistance (1.42x)",
         "Substance use, bodily processes, and medical framing dominate. Posts in this category engage directly with physical acts of consumption and their consequences."),
        (rc2, "Early Life", "#4A90D9",
         "People_by_age (2.04x)\nKinship (1.75x)\nLocale_by_use (1.76x)\nCause_harm (1.48x)\nTelling (1.21x)",
         "Family relationships, age categories, and physical locations dominate. Posts in this category are oriented toward formative contexts and the people who populated them."),
        (rc3, "Personality", "#5CB85C",
         "Fear (1.48x)\nPerception_active (1.42x)\nAwareness (1.31x)\nCapability (1.25x)\nFeeling (1.22x)",
         "Perceptual and evaluative frames dominate. Posts in this category are oriented toward how the self experiences and makes sense of its own mental processes."),
        (rc4, "Trauma and Stress", "#D9534F",
         "Being_employed (1.81x)\nPersonal_relationship (1.40x)\nResidence (1.29x)\nBuildings (1.34x)\nKinship (1.23x)",
         "Work, relationships, and domestic environments dominate. Trauma and stress is framed through the contexts in which it occurs rather than through its psychological effects."),
    ]:
        with col:
            st.markdown(f"""
            <div style='background-color:#F8F9FA;padding:16px;border-radius:6px;
                        border-top:4px solid {colour};min-height:320px'>
              <h4 style='margin:0 0 10px 0;color:#2C3E50'>{label}</h4>
              <p style='font-family:monospace;font-size:0.8em;color:#444;
                        white-space:pre-line;margin:0 0 12px 0'>{frames}</p>
              <p style='color:#666;font-size:0.82em;line-height:1.6;
                        margin:0;font-style:italic'>{interpretation}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    The contrast between Early Life and Trauma and Stress is particularly revealing.
    Both categories involve harm and difficult relational experiences. Early Life frames
    these through kinship, age, and causation of harm: the cognitive focus is on who
    was involved and what was done. Trauma and Stress frames them through work, residence,
    and relationships in the present: the cognitive focus is on the contexts in which
    stress occurs rather than its developmental origins. This distinction would not
    be visible to a classifier operating on semantic embeddings alone, because the
    emotional content of the two categories overlaps substantially.
    """)

    st.divider()

    # ---- The key finding: dep vs SW ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #D9534F;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>The Key Finding: Depression vs SuicideWatch</h2>
    </div>
    """, unsafe_allow_html=True)

    big_stat(
        "p = 0.0001",
        "chi-square = 52.23, df = 19. Frame distributions differ significantly "
        "between r/depression and r/SuicideWatch despite 0.954 cosine similarity.",
        "#D9534F"
    )

    st.markdown("""
    Frame distributions across subreddits are significantly different overall
    (chi-square = 456.00, p = 0.0000, df = 76). The most theoretically important
    comparison is between r/depression and r/SuicideWatch, the two communities that
    sentence embeddings cannot separate.

    The analysis restricted comparisons to frames with at least five detections in
    both communities, producing 221 qualifying frames. Rates were normalised by the
    number of sentences per subreddit: 4,849 for r/depression and 1,806 for r/SuicideWatch.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='background-color:#FFF0F0;padding:20px;border-radius:8px;
                    border-left:4px solid #D9534F'>
          <h4 style='margin:0 0 14px 0;color:#2C3E50'>Most distinctive to r/SuicideWatch</h4>
          <table style='width:100%;font-size:0.85em;color:#444;border-collapse:collapse'>
            <tr style='border-bottom:1px solid #ddd'>
              <th style='text-align:left;padding:4px 8px'>Frame</th>
              <th style='text-align:right;padding:4px 8px'>Ratio</th>
            </tr>
            <tr><td style='padding:4px 8px'>Addiction</td><td style='text-align:right;padding:4px 8px'>4.47x</td></tr>
            <tr><td style='padding:4px 8px'>Rape</td><td style='text-align:right;padding:4px 8px'>3.13x</td></tr>
            <tr><td style='padding:4px 8px'>Transition_to_a_quality</td><td style='text-align:right;padding:4px 8px'>3.07x</td></tr>
            <tr><td style='padding:4px 8px'>Intoxication</td><td style='text-align:right;padding:4px 8px'>2.89x</td></tr>
            <tr><td style='padding:4px 8px'>Killing</td><td style='text-align:right;padding:4px 8px'>2.85x</td></tr>
            <tr><td style='padding:4px 8px'>Cause_to_end</td><td style='text-align:right;padding:4px 8px'>2.86x</td></tr>
            <tr><td style='padding:4px 8px'>Dead_or_alive</td><td style='text-align:right;padding:4px 8px'>2.59x</td></tr>
            <tr><td style='padding:4px 8px'>Surviving</td><td style='text-align:right;padding:4px 8px'>2.68x</td></tr>
          </table>
          <p style='color:#666;font-size:0.82em;margin:12px 0 0 0;font-style:italic'>
            Action and mortality-oriented. Death, substance use, and harm are
            framed as concrete acts with agents and outcomes.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background-color:#EBF5FB;padding:20px;border-radius:8px;
                    border-left:4px solid #4A90D9'>
          <h4 style='margin:0 0 14px 0;color:#2C3E50'>Most distinctive to r/depression</h4>
          <table style='width:100%;font-size:0.85em;color:#444;border-collapse:collapse'>
            <tr style='border-bottom:1px solid #ddd'>
              <th style='text-align:left;padding:4px 8px'>Frame</th>
              <th style='text-align:right;padding:4px 8px'>Ratio</th>
            </tr>
            <tr><td style='padding:4px 8px'>Questioning</td><td style='text-align:right;padding:4px 8px'>3.87x</td></tr>
            <tr><td style='padding:4px 8px'>Memory</td><td style='text-align:right;padding:4px 8px'>2.06x</td></tr>
            <tr><td style='padding:4px 8px'>Prevarication</td><td style='text-align:right;padding:4px 8px'>2.01x</td></tr>
            <tr><td style='padding:4px 8px'>Medical_professionals</td><td style='text-align:right;padding:4px 8px'>1.86x</td></tr>
            <tr><td style='padding:4px 8px'>Discussion</td><td style='text-align:right;padding:4px 8px'>1.78x</td></tr>
            <tr><td style='padding:4px 8px'>Waking_up</td><td style='text-align:right;padding:4px 8px'>1.73x</td></tr>
            <tr><td style='padding:4px 8px'>Resolve_problem</td><td style='text-align:right;padding:4px 8px'>1.64x</td></tr>
            <tr><td style='padding:4px 8px'>Purpose</td><td style='text-align:right;padding:4px 8px'>1.69x</td></tr>
          </table>
          <p style='color:#666;font-size:0.82em;margin:12px 0 0 0;font-style:italic'>
            Reflective and deliberative. Suffering is framed as a problem to
            be understood, articulated, and addressed.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    The pattern is striking. r/SuicideWatch posts are disproportionately organised around
    frames in which death, harm, and substance use are concrete acts: Killing, Cause_to_end,
    Dead_or_alive, Intoxication, Addiction. r/depression posts are disproportionately
    organised around deliberative and reflective frames: Questioning, Memory, Discussion,
    Resolve_problem, Medical_professionals. Both communities are expressing suffering and
    hopelessness. They are framing that suffering in fundamentally different ways.

    The presence of Addiction and Intoxication as highly distinctive r/SuicideWatch frames
    is also consistent with the Joiner literature. Substance use is one of the primary
    pathways through which acquired capability develops: repeated intoxication habituates
    a person to physiological states that resemble and lower the threshold for self-harm.
    The elevated rate of Addiction and Intoxication framing in r/SuicideWatch is therefore
    not only a demographic observation about who posts there but a signal consistent with
    the neurobiological pathway Joiner identifies.
    """)

    st.divider()

    # ---- Clinical frames ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #9B59B6;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Clinically Relevant Frames</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The analysis also searched for FrameNet frames theoretically linked to the clinical
    constructs of Beck (1979), Joiner (2005), Shneidman (1993), and Pennebaker (2011).
    Of 35 candidate frames across five theoretical clusters, 25 were detected in the corpus.
    The ten not detected reflect either FrameNet coverage gaps for colloquial mental health
    language or the absence of formal expressions of those constructs in Reddit discourse.

    Several findings from the clinical frame detection are notable. Within the Hopelessness
    cluster, the Desiring frame is the most frequent and shows its highest rate in
    r/depression, consistent with depression as a state of unfulfilled desire for a
    different life rather than a desire for death. Within the Suffering cluster, the
    Cause_harm frame is elevated in r/SuicideWatch relative to r/depression even after
    normalising for subreddit size, consistent with Shneidman's psychache as a state in
    which harm is not merely experienced but actively engaged with. Within the Belonging
    cluster, Personal_relationship frames are most frequent in r/depression rather than
    r/SuicideWatch, which is counter to the simple prediction from Joiner's thwarted
    belongingness construct: depression posts discuss relationships more, not less.
    This may reflect the deliberative character of r/depression posts noted above,
    in which relational problems are discussed as part of a narrative of seeking help
    and understanding.
    """)

    st.divider()

    # ---- Limitations ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #E8A838;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Limitations of This Method</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Frame semantic analysis introduces several limitations that do not apply to the
    earlier methods in this analysis.

    The frame-semantic-transformer was trained on formal and semi-formal text. Reddit
    posts are colloquial, orthographically variable, and frequently use slang, profanity,
    and abbreviated language that may not match the training distribution. The 4.4%
    no-detection rate likely understates the coverage gap, since the model may assign
    incorrect frames to colloquial constructions rather than returning null results.

    FrameNet is organised around English-language constructions and the cultural assumptions
    embedded in them. The frame vocabulary may not map cleanly onto all the ways mental
    health experience is expressed in Reddit discourse, particularly for non-clinical
    or community-specific language.

    The analysis was conducted at the sentence level rather than the post level. Mental
    health posts are extended narratives in which different parts of the same post may
    evoke different frames. Aggregating frame counts across sentences within a subreddit
    treats each sentence as independent, losing the narrative structure of individual posts.

    Finally, frame detection rates are influenced by post length. Longer posts produce
    more frame detections simply by containing more text, and subreddits differ in their
    typical post length. The normalisation by sentence count rather than frame count
    partially controls for this, but does not eliminate it.

    These limitations do not undermine the central finding. The chi-square result
    (p = 0.0001) is robust to moderate measurement noise. But they do mean that the
    specific frame rates reported here should be treated as indicative rather than
    precise, and that replication with a domain-specific frame parser trained on mental
    health discourse would strengthen the conclusions considerably.
    """)

    st.divider()

    # ---- Closing panel ----
    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px;border-radius:8px'>
      <h2 style='color:white;margin:0 0 14px 0'>What Frame Analysis Contributes</h2>
      <p style='color:#BDC3C7;line-height:1.7;margin:0 0 12px 0'>
        Frame semantic analysis is the sixth and most theoretically ambitious method
        applied in this analysis. It is also the one that most directly addresses the
        open question left by the sentence embedding finding: not whether depressive
        and suicidal discourse share the same content, they do, but whether they
        organise that content through different cognitive frames.
      </p>
      <p style='color:#BDC3C7;line-height:1.7;margin:0 0 12px 0'>
        The answer is yes, and the pattern of differences is consistent with Joiner's
        (2005) acquired capability construct. r/SuicideWatch posts frame experience
        through action and mortality. r/depression posts frame experience through
        reflection and deliberation. This distinction is invisible to TF-IDF, to VAD
        scoring, to sentence embeddings, and to topic modelling. It requires a method
        that encodes cognitive framing rather than lexical distinctiveness, affective
        tone, semantic content, or thematic cluster membership.
      </p>
      <p style='color:white;line-height:1.7;margin:0;font-weight:500'>
        The methodological implication extends beyond this analysis. Multi-method NLP
        research on mental health discourse should include frame-level analysis alongside
        semantic and lexical methods. The representational choice shapes what is visible.
        No single method reveals the full structure of how mental health experience is
        expressed in language, and frame semantics reveals a dimension of that structure
        that the other methods in this analysis cannot reach.
      </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# VIEW: BUILDING THE FRAME ANALYSIS
# ============================================================

elif active_tab == "🛠️ Building the Frame Analysis":

    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px 30px;border-radius:8px;margin-bottom:24px'>
      <h1 style='color:white;margin:0 0 10px 0'>Building the Frame Analysis</h1>
      <p style='color:#BDC3C7;margin:0;line-height:1.7'>
        How the frame semantic analysis was built, what technical problems had to be
        solved, and why the pipeline was designed the way it was.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Why a separate environment ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #9B59B6;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Why a Separate Environment?</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The frame-semantic-transformer library cannot run in the same environment as the
    main analysis notebook. It requires specific older versions of tokenizers and
    transformers that conflict with the sentence-transformers and BERTopic dependencies
    used throughout the rest of this project. This is the same class of dependency
    conflict that made it impossible to run BERTopic live inside the Streamlit app,
    and the solution follows the same logic: move the expensive computation to a
    separate environment, run it once, and serialise the outputs.

    The frame-semantic-transformer also requires Rust to be installed before the
    tokenizers package can be compiled. Rust is not available in a standard Python
    environment and must be installed as a first step. In the main Colab notebook
    this would have caused problems with other dependencies. Running it separately
    kept the main analysis environment stable.

    Google Colab was chosen for the frame analysis for three reasons. First, Colab
    provides a clean environment where dependency installation does not affect any
    other project. Second, Colab offers free GPU access, and the frame-semantic-transformer
    runs substantially faster on CUDA than on CPU. Third, Colab's file download
    functionality made it straightforward to export the results pickle directly to
    the local machine without configuring any storage or transfer infrastructure.
    """)

    st.divider()

    # ---- The pipeline ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #9B59B6;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>The Pipeline</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The pipeline was designed around one practical constraint: the frame-semantic-transformer
    is slow, and Google Colab sessions disconnect without warning. Processing 13,075 sentences
    sequentially with no checkpointing would mean losing all progress if the session dropped.
    The solution was to save a checkpoint pickle every ten sentences and implement resume logic
    at startup, so that any disconnection could be recovered from without restarting the full
    run.

    The pipeline runs in seven steps.
    """)

    steps = [
        ("Step 1: Check for Rust",
         "#9B59B6",
         "The tokenizers library, which the frame-semantic-transformer depends on, requires "
         "Rust to compile its native extensions. The pipeline checks whether Rust is already "
         "installed at /root/.cargo/bin/rustc and installs it via rustup if not. The Rust "
         "binary directory is then added to the PATH environment variable so subsequent "
         "commands can find it."),
        ("Step 2: Install packages",
         "#9B59B6",
         "Three packages are installed in a specific version-pinned order: tokenizers 0.13.3, "
         "transformers 4.30.0, and then frame-semantic-transformer. The version pinning is "
         "essential. Later versions of tokenizers and transformers introduce API changes that "
         "break the frame-semantic-transformer. The pipeline checks whether the library is "
         "already importable before installing, so re-running the cell in an existing session "
         "skips the installation step."),
        ("Step 3: Import libraries",
         "#9B59B6",
         "pandas, pickle, torch, and FrameSemanticTransformer are imported. The torch import "
         "is needed to detect whether a GPU is available before loading the model."),
        ("Step 4: Load sentences",
         "#9B59B6",
         "The labelled_sentences.csv file is read into a dataframe and the sentence column "
         "is extracted as a list. This file contains 13,075 sentences from the 800 "
         "expert-annotated RMHD posts, with subreddit and root cause label columns retained "
         "for the downstream analysis."),
        ("Step 5: Check for existing progress",
         "#9B59B6",
         "If a checkpoint pickle exists from a previous run, it is loaded and the starting "
         "index is set to the length of the existing results list. This means a disconnected "
         "session resumes from the last checkpoint rather than starting over. If no checkpoint "
         "exists, an empty list is initialised and processing starts from index zero."),
        ("Step 6: Load the model",
         "#9B59B6",
         "The FrameSemanticTransformer model is loaded. The pipeline detects whether a CUDA "
         "GPU is available and prints the device being used. On Colab with a GPU runtime the "
         "model loads onto CUDA automatically, producing substantially faster inference than "
         "CPU processing."),
        ("Step 7: Run frame detection",
         "#9B59B6",
         "Each sentence is passed to frame_transformer.detect_frames(). The result is a "
         "DetectFramesResult object containing the sentence text, the trigger locations, and "
         "a list of FrameResult objects each with a frame name and a list of FrameElementResult "
         "objects. Errors are caught and stored as None rather than crashing the loop. A "
         "checkpoint is saved every ten sentences. After the loop completes, a final save "
         "is written and the pickle is downloaded to the local machine via the Colab files API."),
    ]

    for title, colour, body in steps:
        st.markdown(f"""
        <div style='background-color:#F8F9FA;padding:18px;border-radius:6px;
                    border-left:4px solid {colour};margin-bottom:12px'>
          <h4 style='margin:0 0 8px 0;color:#2C3E50'>{title}</h4>
          <p style='margin:0;color:#444;line-height:1.7;font-size:0.95em'>{body}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ---- The code ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #9B59B6;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>The Code</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("The complete pipeline as run in Google Colab:")

    st.code("""
import importlib.util
import os
import subprocess
from tqdm.notebook import tqdm

print("Step 1/7: Checking Rust...")
if not os.path.exists('/root/.cargo/bin/rustc'):
    subprocess.run('curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y',
                   shell=True)
    print("✓ Rust installed")
else:
    print("✓ Rust already present")
os.environ['PATH'] += ':/root/.cargo/bin'

print("\\nStep 2/7: Checking packages...")
if importlib.util.find_spec('frame_semantic_transformer') is None:
    for pkg, label in tqdm([
        ('tokenizers==0.13.3', 'tokenizers'),
        ('transformers==4.30.0', 'transformers'),
        ('frame-semantic-transformer', 'frame-semantic-transformer')
    ], desc="Installing packages"):
        subprocess.run(['pip', 'install', pkg, '-q'])
    print("✓ All packages installed")
else:
    print("✓ Already installed, skipping")

print("\\nStep 3/7: Importing libraries...")
import pandas as pd
import pickle
import torch
from frame_semantic_transformer import FrameSemanticTransformer
print("✓ Imports done")

print("\\nStep 4/7: Loading sentences...")
sentences_labelled_df = pd.read_csv('labelled_sentences.csv')
sentence_texts = sentences_labelled_df['sentence'].tolist()
print(f"✓ Loaded {len(sentence_texts)} sentences")

print("\\nStep 5/7: Checking for existing progress...")
PICKLE_PATH = 'fst_labelled_results.pkl'
if os.path.exists(PICKLE_PATH):
    with open(PICKLE_PATH, 'rb') as f:
        fst_results = pickle.load(f)
    start_idx = len(fst_results)
    print(f"✓ Resuming from {start_idx}/{len(sentence_texts)}")
else:
    fst_results = []
    start_idx = 0
    print("✓ Starting fresh")

print("\\nStep 6/7: Loading model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✓ Using: {device}")
frame_transformer = FrameSemanticTransformer()
print("✓ Model loaded")

print("\\nStep 7/7: Running FST...")
for i in tqdm(range(start_idx, len(sentence_texts)),
              desc="FST progress", unit="sentence"):
    try:
        result = frame_transformer.detect_frames(sentence_texts[i])
        fst_results.append(result)
    except Exception as e:
        print(f"Error at {i}: {e}")
        fst_results.append(None)

    if i % 10 == 0:
        with open(PICKLE_PATH, 'wb') as f:
            pickle.dump(fst_results, f)

with open(PICKLE_PATH, 'wb') as f:
    pickle.dump(fst_results, f)
print(f"✓ Done. {len(fst_results)} results saved.")

print("\\nDownloading pickle...")
from google.colab import files
files.download('fst_labelled_results.pkl')
print("✓ Download started")
""", language="python")

    st.divider()

    # ---- Output format ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #9B59B6;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>Output Format</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The pipeline produces a pickle file containing a list of DetectFramesResult objects,
    one per input sentence, in the same order as the input CSV. Each object has the
    following structure:
    """)

    st.code("""
DetectFramesResult(
    sentence='Do people get over anxiety?',
    trigger_locations=[10],
    frames=[
        FrameResult(
            name='Transition_to_state',
            trigger_location=10,
            frame_elements=[
                FrameElementResult(name='Entity',        text='people'),
                FrameElementResult(name='Final_quality', text='over anxiety')
            ]
        )
    ]
)
""", language="python")

    st.markdown("""
    For the downstream analysis, each result object was flattened into one row per frame
    detection, with the subreddit and root cause label joined from the source CSV by
    sentence index. This produced a dataframe of 52,457 frame detections across 13,075
    sentences, with 573 sentences (4.4%) returning no detected frames.
    """)

    st.divider()

    # ---- What I would do differently ----
    st.markdown("""
    <div style='background-color:#F0F4F8;padding:20px 24px;border-left:4px solid #E8A838;
                border-radius:4px;margin-bottom:20px'>
      <h2 style='margin:0 0 4px 0;color:#2C3E50'>What I Would Do Differently</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Run on GPU from the start.** The first attempt ran on CPU before a GPU runtime
    was available. Processing 13,075 sentences on CPU is significantly slower than on
    CUDA. Connecting a GPU runtime before running the pipeline would have saved
    substantial time.

    **Checkpoint more frequently.** Saving every ten sentences means at most ten
    sentences are lost on disconnection. In practice, Colab sessions tend to disconnect
    during long idle periods rather than mid-run, so this was sufficient. A more
    cautious approach would checkpoint every sentence at the cost of slightly slower
    I/O.

    **Integrate sentence tokenisation into the pipeline.** The labelled_sentences.csv
    was produced in a separate step before the Colab pipeline was run. Integrating
    the sentence tokenisation into the pipeline would have made the full process
    reproducible from a single script rather than requiring a pre-processed input file.

    **Use a domain-specific frame parser.** The frame-semantic-transformer was trained
    on formal and semi-formal text. A parser fine-tuned on social media or clinical
    mental health text would likely produce better coverage and more accurate frame
    assignments for Reddit posts. This is the single most valuable extension of this
    analysis from a methodological standpoint.
    """)

    st.divider()

    st.markdown("""
    <div style='background-color:#2C3E50;padding:28px;border-radius:8px'>
      <h2 style='color:white;margin:0 0 14px 0'>What This Process Demonstrates</h2>
      <p style='color:#BDC3C7;line-height:1.7;margin:0 0 12px 0'>
        The frame analysis required building a second, entirely separate computational
        pipeline in a different environment, solving a different set of dependency
        conflicts, and designing for resilience against session interruption. This is
        the practical reality of combining NLP methods that have incompatible dependency
        trees: each method may require its own environment, and the outputs must be
        serialised and transferred between them.
      </p>
      <p style='color:white;line-height:1.7;margin:0;font-weight:500'>
        The checkpoint-and-resume pattern used here is a general solution to this
        problem. Any long-running inference task in a session-limited environment
        should save progress incrementally and implement resume logic. The cost is
        a small amount of additional I/O per checkpoint. The benefit is that no
        work is lost to disconnection regardless of how long the task takes.
      </p>
    </div>
    """, unsafe_allow_html=True)