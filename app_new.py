
import json
from io import BytesIO
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from utils import ROOT, load_checkpoint, predict_image


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="WasteVision AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = ROOT / "models" / "wastevision_stage3.pth"
METRICS_PATH = ROOT / "results" / "stage4" / "metrics.json"
CONFUSION_PATH = ROOT / "results" / "stage4" / "confusion_matrix.png"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

GUIDES = {
    "cardboard": "Keep cardboard clean and dry. Flatten boxes before recycling.",
    "glass": "Separate glass carefully. Handle broken glass with extra care.",
    "metal": "Rinse metal containers and separate them for recycling.",
    "paper": "Keep paper dry and clean before placing it in recycling.",
    "plastic": "Empty and rinse plastic containers. Follow local recycling rules.",
    "trash": "This category needs manual sorting. Check local disposal guidance.",
}


# --------------------------------------------------
# ANIMATED DESIGN SYSTEM
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #08130f;
    --panel: #11221a;
    --panel2: #172c21;
    --green: #a3ffb0;
    --green2: #45dc91;
    --muted: #9aafa2;
    --white: #effff4;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(48, 160, 100, .15), transparent 35%),
        radial-gradient(ellipse at 90% 20%, rgba(40, 120, 90, .12), transparent 35%),
        var(--bg);
    color: var(--white);
    font-family: 'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--white);
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10251b, #091710);
    border-right: 1px solid rgba(163,255,176,.12);
}

section[data-testid="stSidebar"] * {
    color: #eaffef;
}

/* Hero */

.hero {
    position: relative;
    overflow: hidden;
    padding: 38px;
    border-radius: 25px;
    background: linear-gradient(120deg, #163b28, #10251b 55%, #164331);
    border: 1px solid rgba(163,255,176,.23);
    box-shadow: 0 15px 55px rgba(0,0,0,.25);
    animation: fadeUp .8s ease both;
}

.hero:before {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    right: -65px;
    top: -115px;
    border-radius: 50%;
    background: rgba(133,255,164,.13);
    filter: blur(2px);
    animation: floatGlow 6s ease-in-out infinite;
}

.hero:after {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    right: 180px;
    bottom: -125px;
    border-radius: 50%;
    background: rgba(69,220,145,.12);
    animation: floatGlow 8s ease-in-out infinite reverse;
}

.hero-label {
    color: var(--green);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(34px, 5vw, 58px);
    line-height: 1.08;
    font-weight: 800;
    margin: 15px 0;
    color: white;
}

.hero-title span {
    color: var(--green);
}

.hero-description {
    color: #c0d7c8;
    font-size: 15px;
    max-width: 650px;
    line-height: 1.8;
}

.hero-pill {
    display: inline-block;
    margin-top: 18px;
    padding: 8px 14px;
    border: 1px solid rgba(163,255,176,.3);
    border-radius: 30px;
    background: rgba(163,255,176,.08);
    color: var(--green);
    font-size: 12px;
}

/* Metric cards */

.metric-card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(145deg, #193526, #10251b);
    border: 1px solid rgba(163,255,176,.13);
    border-radius: 18px;
    padding: 22px;
    min-height: 145px;
    transition: transform .3s ease, border .3s ease, box-shadow .3s ease;
    animation: fadeUp .7s ease both;
}

.metric-card:hover {
    transform: translateY(-7px);
    border-color: rgba(163,255,176,.55);
    box-shadow: 0 12px 35px rgba(69,220,145,.12);
}

.metric-card:after {
    content: "";
    position: absolute;
    width: 100px;
    height: 100px;
    right: -35px;
    bottom: -45px;
    border-radius: 50%;
    background: rgba(163,255,176,.08);
}

.metric-label {
    color: #a8c2b0;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

.metric-value {
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 34px;
    font-weight: 800;
    margin-top: 12px;
}

.metric-note {
    color: var(--green);
    font-size: 11px;
    margin-top: 5px;
}

/* Content panels */

.panel {
    background: rgba(17,34,26,.88);
    border: 1px solid rgba(163,255,176,.12);
    border-radius: 20px;
    padding: 23px;
    transition: border .3s ease, transform .3s ease;
    animation: fadeUp .8s ease both;
}

.panel:hover {
    border-color: rgba(163,255,176,.3);
}

.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 21px;
    font-weight: 700;
    color: white;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #9aafa2;
    font-size: 13px;
    margin-bottom: 20px;
}

.green-text {
    color: var(--green);
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #54ef9b;
    border-radius: 50%;
    margin-right: 8px;
    box-shadow: 0 0 12px #54ef9b;
    animation: pulse 1.7s infinite;
}

/* Buttons */

.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(163,255,176,.3);
    background: linear-gradient(110deg, #a3ffb0, #45dc91);
    color: #07150d;
    font-weight: 800;
    padding: 10px 20px;
    transition: all .25s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(69,220,145,.3);
    border-color: #a3ffb0;
    color: #07150d;
}

/* Inputs */

[data-testid="stFileUploader"] {
    background: #11251b;
    border: 1px dashed rgba(163,255,176,.35);
    border-radius: 18px;
    padding: 15px;
}

/* Animations */

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes floatGlow {
    0%, 100% { transform: translateY(0) scale(1); }
    50% { transform: translateY(25px) scale(1.12); }
}

@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 12px #54ef9b; }
    50% { opacity: .45; box-shadow: 0 0 3px #54ef9b; }
}

div[data-testid="stDataFrame"] {
    border: 1px solid rgba(163,255,176,.15);
    border-radius: 14px;
    overflow: hidden;
}

hr {
    border-color: rgba(163,255,176,.12);
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource(max_entries=1)
def cached_checkpoint(path, device):
    torch.set_num_threads(4)
    return load_checkpoint(path, device)


metrics = load_metrics()

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------
# REUSABLE UI COMPONENTS
# --------------------------------------------------

def hero(title, highlight, description, label="AI POWERED WASTE INTELLIGENCE"):
    st.markdown(f"""
    <div class="hero">
        <div class="hero-label">{label}</div>
        <div class="hero-title">{title}<br><span>{highlight}</span></div>
        <div class="hero-description">{description}</div>
        <div class="hero-pill">♻️ SMARTER SORTING · CLEANER FUTURE</div>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value, note):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def section_title(title, subtitle=""):
    st.markdown(f"""
    <div class="section-heading">{title}</div>
    <div class="section-subtitle">{subtitle}</div>
    """, unsafe_allow_html=True)


def chart_layout(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d8eadf", family="DM Sans"),
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(163,255,176,.15)",
    )
    fig.update_yaxes(
        gridcolor="rgba(163,255,176,.10)",
        zeroline=False,
    )
    return fig


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="padding:15px 5px 25px 5px;">
        <div style="font-size:35px;">♻️</div>
        <div style="font-family:Space Grotesk;font-size:25px;font-weight:800;">
            WasteVision<span style="color:#a3ffb0;"> AI</span>
        </div>
        <div style="color:#9aafa2;font-size:11px;letter-spacing:2px;">
            INTELLIGENT WASTE SORTING
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Classify Waste",
            "Analytics",
            "Prediction History",
            "Model Performance",
        ],
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown(
        '<span class="status-dot"></span> AI SYSTEM READY',
        unsafe_allow_html=True,
    )
    st.caption("WasteVision AI · Project Dashboard")
    st.caption("ResNet18 · Image Classification")


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    hero(
        "See Waste.",
        "Think Smarter.",
        "An intelligent image classification system that identifies waste categories "
        "and helps make waste segregation easier.",
    )

    st.write("")
    st.write("")

    if metrics:
        accuracy = metrics.get("accuracy", 0) * 100
        macro = metrics.get("macro_average", {})
        precision = macro.get("precision", 0) * 100
        recall = macro.get("recall", 0) * 100
        f1 = macro.get("f1-score", 0) * 100
        test_count = metrics.get("test_images", 0)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("Test Accuracy", f"{accuracy:.1f}%", "Held-out test dataset")

        with c2:
            metric_card("Macro Precision", f"{precision:.1f}%", "Average across classes")

        with c3:
            metric_card("Macro Recall", f"{recall:.1f}%", "Average across classes")

        with c4:
            metric_card("Macro F1 Score", f"{f1:.1f}%", "Balanced performance")

        st.write("")
        st.write("")

        left, right = st.columns([1.3, 1])

        with left:
            
            section_title("Class-wise intelligence", "How the model performs for each waste category")

            report = metrics.get("classification_report", {})
            rows = []

            for category in CLASSES:
                if category in report:
                    item = report[category]
                    rows.append({
                        "Category": category.title(),
                        "Precision": round(item["precision"] * 100, 1),
                        "Recall": round(item["recall"] * 100, 1),
                        "F1 Score": round(item["f1-score"] * 100, 1),
                        "Images": int(item["support"]),
                    })

            if rows:
                st.dataframe(
                    pd.DataFrame(rows),
                    hide_index=True,
                    width="stretch",
                )

             

        with right:
            
            section_title("Dataset composition", "Images in the held-out test set")

            distribution = [
                {"Category": category.title(),
                 "Images": int(report[category]["support"])}
                for category in CLASSES if category in report
            ]

            fig = px.pie(
                pd.DataFrame(distribution),
                names="Category",
                values="Images",
                hole=.68,
                color_discrete_sequence=[
                    "#a3ffb0", "#45dc91", "#1b9d72",
                    "#d5e9a0", "#4cbdab", "#557b68"
                ],
            )
            fig.update_traces(
                textposition="inside",
                textinfo="percent",
                marker=dict(line=dict(color="#11221a", width=3)),
            )
            fig.update_layout(showlegend=True)
            st.plotly_chart(chart_layout(fig, 320), width="stretch")

             

        st.caption(
            f"Evaluation based on {test_count} held-out test images. "
            "These are model evaluation results, not live waste collection statistics."
        )

    else:
        st.warning("Model metrics were not found. Check results/stage4/metrics.json.")

    st.write("")
    
    section_title("Recent predictions", "Your latest image classifications in this session")

    if st.session_state.history:
        recent = st.session_state.history[-5:][::-1]
        st.dataframe(
            pd.DataFrame([
                {
                    "Time": item["Time"],
                    "Category": item["Category"].title(),
                    "Confidence": f'{item["Confidence"]:.1f}%',
                }
                for item in recent
            ]),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No images classified in this session yet. Visit Classify Waste to begin.")

     


# --------------------------------------------------
# CLASSIFY WASTE
# --------------------------------------------------

elif page == "Classify Waste":

    hero(
        "AI Waste",
        "Recognition.",
        "Upload an image and let the trained ResNet18 model identify its most likely waste category.",
        "INTELLIGENT IMAGE CLASSIFICATION",
    )

    st.write("")
    st.write("")

    left, right = st.columns([1, 1], gap="large")

    with left:
        
        section_title("Upload your image", "Supported formats: JPG, JPEG and PNG")

        uploaded = st.file_uploader(
            "Choose a waste image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded:
            try:
                image = Image.open(BytesIO(uploaded.getvalue())).convert("RGB")
                st.image(image, caption="Image ready for analysis", width="stretch")

                analyze = st.button(
                    "✦  ANALYZE IMAGE",
                    width="stretch",
                )

                if analyze:
                    if not MODEL_PATH.exists():
                        st.error("Model checkpoint not found. Check the models folder.")
                    else:
                        with st.spinner("AI is analyzing your image..."):
                            result = predict_image(
                                image,
                                model_path=MODEL_PATH,
                                checkpoint_loader=cached_checkpoint,
                            )

                        predicted_class = result["predicted_class"]
                        confidence = float(result["score"]) * 100
                        top_three = result["top_three"]

                        entry = {
                            "Time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                            "Category": predicted_class,
                            "Confidence": confidence,
                            "Filename": uploaded.name,
                        }

                        st.session_state.history.append(entry)
                        st.session_state.latest_result = {
                            "category": predicted_class,
                            "confidence": confidence,
                            "top_three": top_three,
                        }

                        st.success("Image analysis completed!")

            except (UnidentifiedImageError, OSError):
                st.error("This image could not be opened. Try another image.")

         

    with right:
        
        section_title("AI analysis result", "Your latest classification")

        latest = st.session_state.get("latest_result")

        if latest:
            category = latest["category"]
            confidence = latest["confidence"]

            st.markdown(
                f"""
                <div style="text-align:center;padding:22px 5px;">
                    <div style="font-size:50px;">♻️</div>
                    <div style="color:#9aafa2;font-size:12px;letter-spacing:2px;">
                        DETECTED CATEGORY
                    </div>
                    <div style="font-family:Space Grotesk;font-size:36px;
                                font-weight:800;color:#a3ffb0;margin:12px 0;">
                        {category.title()}
                    </div>
                    <div style="font-size:15px;color:#c4d9ca;">
                        Model confidence: <b>{confidence:.1f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(min(max(confidence / 100, 0.0), 1.0))

            if confidence < 60:
                st.warning("Lower confidence: please verify the predicted category manually.")
            else:
                st.success("The model has a relatively strong prediction score.")

            st.markdown("**Top three predictions**")

            top_rows = []
            for item in latest["top_three"]:
                top_rows.append({
                    "Category": item["class"].title(),
                    "Score": f'{float(item["score"]) * 100:.1f}%',
                })

            st.dataframe(
                pd.DataFrame(top_rows),
                hide_index=True,
                width="stretch",
            )

            st.markdown("**Disposal guidance**")
            st.info(GUIDES.get(category, "Check local waste disposal guidance."))

        else:
            st.markdown("""
            <div style="text-align:center;padding:65px 10px;color:#9aafa2;">
                <div style="font-size:55px;">✧</div>
                <div style="font-size:16px;color:#eaffef;font-weight:700;">
                    Waiting for an image
                </div>
                <div style="font-size:13px;margin-top:8px;">
                    Your AI result will appear here.
                </div>
            </div>
            """, unsafe_allow_html=True)

         


# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------

elif page == "Analytics":

    hero(
        "Data Meets",
        "Sustainability.",
        "Explore the model's test-set performance and understand how it recognizes different waste categories.",
        "MODEL ANALYTICS",
    )

    if metrics:
        report = metrics.get("classification_report", {})

        rows = []
        for category in CLASSES:
            if category in report:
                item = report[category]
                rows.append({
                    "Category": category.title(),
                    "F1 Score": item["f1-score"] * 100,
                    "Precision": item["precision"] * 100,
                    "Recall": item["recall"] * 100,
                    "Images": item["support"],
                })

        df = pd.DataFrame(rows)

        st.write("")
        col1, col2 = st.columns(2)

        with col1:
            
            section_title("F1 score by category", "Higher values indicate a better balance of precision and recall")

            fig = px.bar(
                df,
                x="Category",
                y="F1 Score",
                text="F1 Score",
                color="F1 Score",
                color_continuous_scale=["#24553b", "#45dc91", "#b6ffbd"],
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_yaxes(range=[0, 110], title="F1 Score (%)")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(chart_layout(fig), width="stretch")

             

        with col2:
            
            section_title("Precision vs recall", "Compare prediction quality across categories")

            long_df = df.melt(
                id_vars="Category",
                value_vars=["Precision", "Recall"],
                var_name="Metric",
                value_name="Score",
            )

            fig = px.bar(
                long_df,
                x="Category",
                y="Score",
                color="Metric",
                barmode="group",
                color_discrete_sequence=["#a3ffb0", "#45a9a0"],
            )
            fig.update_yaxes(range=[0, 110], title="Score (%)")
            st.plotly_chart(chart_layout(fig), width="stretch")

             

        st.write("")
        
        section_title("Test-set image distribution", "Number of images evaluated per waste category")

        fig = px.bar(
            df,
            x="Category",
            y="Images",
            text="Images",
            color="Category",
            color_discrete_sequence=[
                "#a3ffb0", "#45dc91", "#1b9d72",
                "#d5e9a0", "#4cbdab", "#557b68"
            ],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart_layout(fig), width="stretch")

        st.caption("This chart shows the test dataset composition, not real-world waste volumes.")
         

    else:
        st.warning("Evaluation metrics are unavailable.")


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

elif page == "Prediction History":

    hero(
        "Every Scan.",
        "One Record.",
        "Review the images classified during your current application session and export the prediction log.",
        "PREDICTION ACTIVITY",
    )

    st.write("")
    
    section_title("Prediction history", "Session-based classification records")

    history = st.session_state.history

    if history:
        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            hide_index=True,
            width="stretch",
        )

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "↓  Download prediction history",
            data=csv,
            file_name="wastevision_prediction_history.csv",
            mime="text/csv",
        )

        if st.button("Clear session history"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info("No predictions yet. Classify an image to start building your history.")

    


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    hero(
        "Inside the",
        "AI Engine.",
        "A closer look at the trained ResNet18 model, its evaluation results and its classification performance.",
        "MODEL INTELLIGENCE",
    )

    st.write("")

    if metrics:
        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "Selected Epoch",
                str(metrics.get("selected_epoch", "N/A")),
                "Best checkpoint by validation loss",
            )

        with c2:
            metric_card(
                "Validation Loss",
                f'{metrics.get("validation_loss", 0):.4f}',
                "Selected model checkpoint",
            )

        with c3:
            metric_card(
                "Test Images",
                str(metrics.get("test_images", 0)),
                "Held-out evaluation samples",
            )

        st.write("")
        left, right = st.columns([1.2, 1])

        with left:
            
            section_title("Confusion matrix", "Actual versus predicted categories")

            if CONFUSION_PATH.exists():
                st.image(
                    str(CONFUSION_PATH),
                    width="stretch",
                )
            else:
                st.info("Confusion matrix image not found.")

            

        with right:
            
            section_title("Model configuration", "Training setup")

            details = {
                "Architecture": "ResNet18",
                "Learning approach": "Transfer Learning",
                "Backbone": "Frozen pretrained layers",
                "Optimizer": "Adam",
                "Learning rate": "0.001",
                "Random seed": metrics.get("seed", "N/A"),
                "Checkpoint": metrics.get("checkpoint", "N/A"),
                "Selection metric": metrics.get("selection_metric", "N/A"),
            }

            for key, value in details.items():
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;
                                padding:10px 0;border-bottom:1px solid
                                rgba(163,255,176,.10);gap:10px;">
                        <span style="color:#9aafa2;">{key}</span>
                        <b style="color:#eaffef;text-align:right;">{value}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            

        st.write("")
        
        section_title("Detailed classification report", "Per-category test-set evaluation")

        report = metrics.get("classification_report", {})
        rows = []

        for category in CLASSES:
            if category in report:
                item = report[category]
                rows.append({
                    "Category": category.title(),
                    "Precision": round(item["precision"] * 100, 2),
                    "Recall": round(item["recall"] * 100, 2),
                    "F1 Score": round(item["f1-score"] * 100, 2),
                    "Support": int(item["support"]),
                })

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
        )

        

    else:
        st.warning("Model metrics are unavailable.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div style="text-align:center;padding:35px 10px 5px;
            color:#6f8b79;font-size:12px;">
    ♻️ <b style="color:#a3ffb0;">WASTEVISION AI</b>
    <br><br>
    Intelligent Waste Classification · Built with Python, PyTorch and Streamlit
    <br>
    A machine learning project for smarter waste segregation.
</div>
""", unsafe_allow_html=True)
