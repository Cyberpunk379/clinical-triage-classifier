"""Streamlit demo for the Clinical NLP Triage Classifier."""

from __future__ import annotations

import streamlit as st

from src.classifier import DEFAULT_MODEL, get_classifier
from src.extractor import SYMPTOM_CATEGORIES, get_extractor
from src.data import EXAMPLE_NOTES


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Clinical NLP Triage Classifier",
    page_icon="🩺",
    layout="wide",
)

TIER_COLORS = {
    "ROUTINE": "#22c55e",
    "SOON":    "#f59e0b",
    "URGENT":  "#ef4444",
}

TIER_DESCRIPTIONS = {
    "ROUTINE": "Schedule a normal appointment. No time-sensitive concerns identified.",
    "SOON":    "Should be seen within 24-48 hours. Symptoms warrant clinician evaluation.",
    "URGENT":  "Seek immediate care. Symptoms consistent with a potentially serious condition.",
}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🩺 Clinical Triage")
    st.markdown(
        "Zero-shot triage classification and symptom extraction for free-text "
        "clinical notes. **Not a medical device.** See README for limitations."
    )

    st.subheader("Example notes")
    example_idx = st.selectbox(
        "Pick a bundled example to populate the input box:",
        options=range(len(EXAMPLE_NOTES)),
        format_func=lambda i: EXAMPLE_NOTES[i].label,
        index=None,
        placeholder="— select an example —",
    )

    st.subheader("Settings")
    use_nli_extractor = st.toggle("Use NLI for implicit symptom finding", value=True)
    implicit_threshold = st.slider(
        "Implicit-finding threshold", min_value=0.3, max_value=0.9, value=0.55, step=0.05
    )

    st.subheader("Model")
    AVAILABLE_MODELS = {
        "valhalla/distilbart-mnli-12-3": "DistilBART-MNLI · 300MB · fast (default)",
        "facebook/bart-large-mnli": "BART-large-MNLI · 1.6GB · more accurate",
    }
    selected_model = st.selectbox(
        "Triage model",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda k: AVAILABLE_MODELS[k],
        index=0,
        help="Swap between models live. Larger models give better accuracy at the cost of memory and first-load time.",
    )
    if selected_model != DEFAULT_MODEL:
        st.caption("⚠️ First use of this model downloads ~1.6GB and requires more RAM.")
    st.caption("Extraction: hybrid lexicon + NLI, with NegEx negation")


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("Clinical NLP Triage Classifier")
st.caption(
    "Paste a clinical note or patient-reported symptom narrative. "
    "The classifier returns a triage tier with confidence, plus the symptom "
    "categories it detected."
)

default_text = EXAMPLE_NOTES[example_idx].text if example_idx is not None else ""
note = st.text_area(
    "Clinical note",
    value=default_text,
    height=220,
    placeholder="e.g. 62-year-old male with sudden onset crushing chest pressure...",
)

run = st.button("Classify", type="primary", disabled=not note.strip())

if run:
    with st.spinner("Loading model (first run downloads weights)..."):
        classifier = get_classifier(selected_model)
        extractor = get_extractor(selected_model)
        extractor.use_nli = use_nli_extractor

    with st.spinner("Classifying..."):
        triage = classifier.classify(note)
        symptoms = extractor.extract(note)

    # ---- Triage card ---------------------------------------------------------
    color = TIER_COLORS[triage.tier]
    st.markdown(
        f"""
        <div style="
            padding: 1rem 1.25rem;
            border-left: 6px solid {color};
            background: {color}1F;
            border-radius: 4px;
            margin: 1rem 0;
            color: inherit;
        ">
          <div style="font-size: 0.8rem; opacity: 0.65; text-transform: uppercase; letter-spacing: 0.05em;">
            Predicted tier
          </div>
          <div style="font-size: 2rem; font-weight: 700; color: {color}; line-height: 1.2;">
            {triage.tier}
          </div>
          <div style="font-size: 0.95rem; opacity: 0.85; margin-top: 0.25rem;">
            {TIER_DESCRIPTIONS[triage.tier]}
          </div>
          <div style="font-size: 0.85rem; opacity: 0.7; margin-top: 0.5rem;">
            <em>{triage.rationale}</em>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Distribution + symptoms in columns ---------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Tier distribution")
        for tier, score in sorted(triage.distribution.items(), key=lambda kv: -kv[1]):
            bar_color = TIER_COLORS[tier]
            st.markdown(
                f"""
                <div style="margin-bottom: 0.6rem;">
                  <div style="display: flex; justify-content: space-between; font-size: 0.9rem; opacity: 0.9;">
                    <span>{tier}</span><span>{score:.1%}</span>
                  </div>
                  <div style="background: rgba(128,128,128,0.2); border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="width: {score*100:.1f}%; height: 100%; background: {bar_color};"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_right:
        st.subheader("Symptom categories")
        all_cats = symptoms.all_categories(implicit_threshold=implicit_threshold)

        if not all_cats:
            st.info("No symptom categories identified above threshold.")
        else:
            for cat in sorted(all_cats):
                explicit_terms = [
                    m.term for m in symptoms.explicit if m.category == cat
                ]
                implicit_score = symptoms.implicit_scores.get(cat, 0.0)
                is_explicit = bool(explicit_terms)

                tag = "explicit" if is_explicit else "implicit"
                tag_color = "#2E5C8A" if is_explicit else "#888"

                terms_text = (
                    ", ".join(sorted(set(explicit_terms)))
                    if explicit_terms
                    else f"NLI score: {implicit_score:.0%}"
                )

                st.markdown(
                    f"""
                    <div style="padding: 0.6rem 0.85rem; background: rgba(128,128,128,0.15);
                                border-radius: 6px; margin-bottom: 0.4rem;">
                      <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="text-transform: capitalize; color: inherit; font-size: 0.95rem;">{cat}</strong>
                        <span style="font-size: 0.72rem; color: white;
                                     background: {tag_color}; padding: 0.15rem 0.55rem;
                                     border-radius: 10px; font-weight: 600;">{tag}</span>
                      </div>
                      <div style="font-size: 0.85rem; opacity: 0.75; margin-top: 0.3rem; color: inherit;">
                        {terms_text}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ---- Disclaimer ----------------------------------------------------------
    st.divider()
    st.caption(
        "⚠️ Research demonstrator only. Not validated for clinical use. "
        "Do not enter PHI into hosted versions. See README for full limitations."
    )

else:
    st.info("Paste a note above (or pick an example) and click **Classify**.")
