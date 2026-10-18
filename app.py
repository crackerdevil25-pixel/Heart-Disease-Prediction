"""
Heart Disease Risk Assessment
-----------------------------
A Streamlit web app that wraps a Logistic Regression model trained on the
UCI Cleveland Heart Disease dataset (university Machine Learning assignment).

Model input contract (exactly 4 features, in this order):
    age     - age in years
    sex     - 1 = male, 0 = female
    cp      - chest pain type (UCI encoding 1-4)
    thalach - maximum heart rate achieved

Run locally with:
    streamlit run app.py
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# The model file must sit in the same folder as this script.
MODEL_PATH = Path(__file__).parent / "heart_disease_model.pkl"

# Column names and order the model was trained with. Do not change these.
FEATURE_COLUMNS = ["age", "sex", "cp", "thalach"]

# Test accuracy reported in the assignment (not a per-person guarantee).
TEST_ACCURACY = "83.33%"

# Dropdown label -> numeric value the model expects.
SEX_OPTIONS = {"Female": 0, "Male": 1}
CP_OPTIONS = {
    "1 — Typical angina": 1,
    "2 — Atypical angina": 2,
    "3 — Non-anginal pain": 3,
    "4 — Asymptomatic": 4,
}

# Palette: navy, teal and white.
NAVY = "#0B2545"
NAVY_SOFT = "#13315C"
TEAL = "#14A3A3"
TEAL_DARK = "#0E7C7B"
MIST = "#EAF6F6"


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------

def inject_css() -> None:
    """Add custom CSS for the header, result card and footer."""
    st.markdown(
        f"""
        <style>
        .block-container {{
            max-width: 920px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}

        /* ---------- Header ---------- */
        .hero {{
            background: linear-gradient(135deg, {NAVY} 0%, {NAVY_SOFT} 100%);
            border-radius: 20px;
            padding: 1.75rem 1.75rem;
            display: flex;
            align-items: center;
            gap: 1.25rem;
            border-bottom: 4px solid {TEAL};
            margin-bottom: 1.5rem;
        }}
        .hero-icon {{
            flex: 0 0 auto;
            width: 64px;
            height: 64px;
            border-radius: 18px;
            background: {TEAL};
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .hero h1 {{
            color: #FFFFFF;
            font-size: 2rem;
            line-height: 1.2;
            margin: 0;
            padding: 0;
        }}
        .hero p {{
            color: #CFE3EE;
            margin: 0.4rem 0 0 0;
            font-size: 1rem;
            line-height: 1.5;
        }}

        /* ---------- Section titles ---------- */
        .section-title {{
            color: {TEAL_DARK};
            font-weight: 700;
            font-size: 1.15rem;
            margin: 0.25rem 0 0.25rem 0;
        }}

        /* ---------- Result card ---------- */
        .result-card {{
            border-radius: 20px;
            padding: 1.5rem 1.75rem;
            margin: 0.5rem 0 1rem 0;
            background: #FFFFFF;
            border: 1px solid #D5E3EA;
            border-left: 10px solid var(--accent);
        }}
        .result-card .label {{
            color: #4B6478;
            font-size: 0.95rem;
            margin: 0 0 0.35rem 0;
        }}
        .result-card .headline {{
            color: {NAVY};
            font-size: 1.6rem;
            font-weight: 800;
            line-height: 1.25;
            margin: 0;
        }}
        .result-card .detail {{
            color: #35506A;
            margin: 0.75rem 0 0 0;
            line-height: 1.55;
        }}

        /* ---------- Summary chips ---------- */
        .chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.25rem;
        }}
        .chip {{
            background: {MIST};
            color: {NAVY};
            border: 1px solid #BFE3E3;
            border-radius: 999px;
            padding: 0.3rem 0.8rem;
            font-size: 0.9rem;
        }}

        /* ---------- Footer ---------- */
        .footer {{
            margin-top: 2.5rem;
            padding: 1rem 1.25rem;
            border-radius: 14px;
            background: {NAVY};
            color: #CFE3EE;
            text-align: center;
            font-size: 0.9rem;
            line-height: 1.5;
        }}

        /* ---------- Mobile ---------- */
        @media (max-width: 640px) {{
            .hero {{ flex-direction: column; align-items: flex-start; padding: 1.25rem; }}
            .hero h1 {{ font-size: 1.5rem; }}
            .result-card .headline {{ font-size: 1.3rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Show the page title, heart icon (inline SVG, no external files) and subtitle."""
    st.markdown(
        """
        <div class="hero">
            <div class="hero-icon">
                <svg width="38" height="38" viewBox="0 0 24 24" fill="none"
                     stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"
                     stroke-linejoin="round" aria-hidden="true">
                    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
                    <polyline points="6 12 9 12 10.5 9 13 15 14.5 12 18 12"/>
                </svg>
            </div>
            <div>
                <h1>Heart Disease Risk Assessment</h1>
                <p>An educational machine-learning demo. Enter four values and see what a
                Logistic Regression model trained on the UCI Cleveland dataset predicts.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Show the educational-use disclaimer."""
    st.markdown(
        """
        <div class="footer">
            This tool is for <strong>educational purposes only</strong> and is not a
            substitute for professional medical advice, diagnosis or treatment.
            Always consult a qualified healthcare provider about your health.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the pickled model once and cache it.

    Returns a tuple (model, error_message). Exactly one of them is None.
    Errors are returned (not raised) so the app can show a friendly message.
    """
    if not MODEL_PATH.exists():
        return None, (
            f"The model file `{MODEL_PATH.name}` was not found. Make sure it is in "
            "the same folder as `app.py` (the root of your GitHub repository)."
        )
    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
    except Exception as exc:  # noqa: BLE001 - we want to show any load problem
        return None, (
            f"The model file could not be loaded ({type(exc).__name__}: {exc}). "
            "This usually means the scikit-learn version in `requirements.txt` "
            "does not match the version used to train the model."
        )

    if not hasattr(model, "predict"):
        return None, "The loaded file is not a valid scikit-learn model (no `predict` method)."
    return model, None


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------

def make_prediction(model, age: int, sex: int, cp: int, thalach: int) -> int:
    """Build the one-row DataFrame the model expects and return 0 or 1."""
    input_df = pd.DataFrame(
        [[age, sex, cp, thalach]],
        columns=FEATURE_COLUMNS,  # same names and order used in training
    )
    return int(model.predict(input_df)[0])


def render_result(prediction: int) -> None:
    """Show the prominent result card for a prediction of 0 or 1."""
    if prediction == 1:
        accent = "#D98324"  # amber: draws attention without sounding like a diagnosis
        headline = "Heart disease detected by the model"
    else:
        accent = TEAL
        headline = "No heart disease detected by the model"

    st.markdown(
        f"""
        <div class="result-card" style="--accent: {accent};">
            <p class="label">Model prediction</p>
            <p class="headline">{headline}</p>
            <p class="detail">
                This is a machine-learning prediction based on only four inputs.
                It is <strong>not a medical diagnosis</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if prediction == 1:
        st.warning(
            "The model's output is not a diagnosis. If you have any health concerns, "
            "please speak with a qualified healthcare professional."
        )
    else:
        st.success(
            "The model did not flag heart disease for these inputs. This does not "
            "mean a person is medically healthy, since the model uses just four values."
        )


# ----------------------------------------------------------------------------
# Page sections
# ----------------------------------------------------------------------------

def render_sidebar() -> None:
    """Sidebar with background information about the model."""
    with st.sidebar:
        st.markdown("### About this model")
        st.markdown(
            "- **Algorithm:** Logistic Regression\n"
            "- **Dataset:** UCI Cleveland Heart Disease\n"
            "- **Records used:** 297 (303 minus rows with missing values)\n"
            "- **Inputs:** age, sex, chest pain type, maximum heart rate"
        )
        st.info(
            f"Test accuracy from the assignment: about **{TEST_ACCURACY}**. "
            "This describes performance on a held-out test set. It is not a "
            "guarantee for any individual prediction."
        )
        st.caption("The model has not been clinically validated.")


def render_input_form():
    """
    Show the four input widgets inside a form.

    Returns (submitted, age, sex_label, cp_label, thalach).
    Nothing is predicted until the user presses the button.
    """
    st.markdown('<p class="section-title">Patient information</p>', unsafe_allow_html=True)

    with st.form("patient_form", border=True):
        col_left, col_right = st.columns(2)  # columns stack automatically on mobile

        with col_left:
            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=55,
                step=1,
                help="Age in whole years (1–120).",
            )
            cp_label = st.selectbox(
                "Chest pain type",
                options=list(CP_OPTIONS.keys()),
                index=0,
                help="Uses the original UCI Cleveland dataset coding (1–4).",
            )

        with col_right:
            sex_label = st.selectbox(
                "Sex",
                options=list(SEX_OPTIONS.keys()),
                index=0,
                help="Sex as recorded in the dataset (Female = 0, Male = 1).",
            )
            thalach = st.number_input(
                "Maximum heart rate achieved (bpm)",
                min_value=50,
                max_value=250,
                value=150,
                step=1,
                help="Highest heart rate reached, in beats per minute (50–250).",
            )

        submitted = st.form_submit_button(
            "Predict",
            type="primary",
            width="stretch",
        )

    return submitted, int(age), sex_label, cp_label, int(thalach)


def render_summary_chips(age: int, sex_label: str, cp_label: str, thalach: int) -> None:
    """Show the values that were sent to the model."""
    st.markdown(
        f"""
        <div class="chip-row">
            <span class="chip">Age: {age}</span>
            <span class="chip">Sex: {sex_label}</span>
            <span class="chip">Chest pain: {cp_label}</span>
            <span class="chip">Max heart rate: {thalach} bpm</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Heart Disease Risk Assessment",
        page_icon="❤️",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    render_header()
    render_sidebar()

    # Load the model first so a problem is reported straight away.
    model, error = load_model()
    if error:
        st.error(error)
        render_footer()
        return

    submitted, age, sex_label, cp_label, thalach = render_input_form()

    st.markdown('<p class="section-title">Prediction</p>', unsafe_allow_html=True)

    if not submitted:
        st.info("Fill in the form above and press **Predict** to see the model's output.")
    else:
        # Friendly heads-up if values are unusual compared with typical adult data.
        # (The Cleveland dataset covers roughly ages 29–77.)
        if age < 29 or age > 77:
            st.info(
                "This age is outside the range covered by most records in the "
                "Cleveland dataset (about 29–77 years), so the model has seen little "
                "similar data."
            )
        try:
            prediction = make_prediction(
                model,
                age=age,
                sex=SEX_OPTIONS[sex_label],
                cp=CP_OPTIONS[cp_label],
                thalach=thalach,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(
                f"The model could not make a prediction ({type(exc).__name__}: {exc}). "
                "Please check that the model file and scikit-learn version match."
            )
        else:
            render_result(prediction)
            st.markdown("**Inputs sent to the model**")
            render_summary_chips(age, sex_label, cp_label, thalach)

    render_footer()


if __name__ == "__main__":
    main()
