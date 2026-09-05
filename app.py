from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.create_demo_data import create_demo_data
from scripts.train_models import train
from src.anomaly_detection import anomaly_risk_score
from src.config import ARTIFACT_PATH, METRICS_PATH, SAMPLE_DATA_PATH
from src.explainability import attach_explanations, explain_transaction
from src.feature_engineering import engineer_features
from src.fraud_model import load_artifact, predict_fraud_probability
from src.preprocessing import expected_columns, validate_transactions
from src.risk_scoring import build_risk_report
from src.utils import read_json


st.set_page_config(page_title="RiskLens", page_icon="RL", layout="wide")

RISK_COLORS = {"LOW": "#15803d", "MEDIUM": "#b45309", "HIGH": "#b91c1c"}
RISK_COLORS_DARK = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#f87171"}
ACTION_COLORS = {
    "APPROVE": "#15803d",
    "REVIEW / MONITOR": "#b45309",
    "HOLD FOR MANUAL REVIEW": "#c2410c",
    "BLOCK": "#b91c1c",
}

DISPLAY_COLUMNS = [
    "transaction_id",
    "amount",
    "risk_score",
    "risk_level",
    "fraud_probability",
    "anomaly_score",
    "recommended_action",
    "top_risk_factors",
]


@st.cache_resource
def get_artifact():
    if not ARTIFACT_PATH.exists():
        create_demo_data()
        train()
    return load_artifact()


@st.cache_data
def load_demo_data() -> pd.DataFrame:
    if not SAMPLE_DATA_PATH.exists():
        create_demo_data()
    return pd.read_csv(SAMPLE_DATA_PATH)


def analyze(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    validation = validate_transactions(df, require_target=False)
    if validation.errors:
        return pd.DataFrame(), validation.errors
    artifact = get_artifact()
    engineered = engineer_features(validation.data)
    fraud_probability = predict_fraud_probability(artifact["fraud_model"], validation.data)
    anomaly_score = anomaly_risk_score(artifact["anomaly_model"], validation.data)
    report = build_risk_report(engineered, fraud_probability, anomaly_score)
    return attach_explanations(report), validation.warnings


def read_uploaded_csv(uploaded) -> tuple[pd.DataFrame | None, str | None]:
    if uploaded is None:
        return None, None
    try:
        return pd.read_csv(uploaded), None
    except pd.errors.EmptyDataError:
        return None, "The uploaded CSV is empty."
    except pd.errors.ParserError:
        return None, "The uploaded file could not be parsed as a valid CSV."
    except UnicodeDecodeError:
        return None, "The uploaded CSV encoding is not supported. Please upload a UTF-8 CSV."


def render_metric(label: str, value: str, helper: str = "") -> None:
    st.markdown(
        metric_markup(label, value, helper),
        unsafe_allow_html=True,
    )


def metric_markup(label: str, value: str, helper: str = "") -> str:
    return (
        '<div class="rl-metric">'
        f'<div class="rl-metric-label">{label}</div>'
        f'<div class="rl-metric-value">{value}</div>'
        f'<div class="rl-metric-helper">{helper}</div>'
        "</div>"
    )


def render_metric_grid(items: list[tuple[str, str | int, str]]) -> None:
    cards = "".join(metric_markup(label, str(value), helper) for label, value, helper in items)
    st.markdown(f"<div class='rl-metric-grid'>{cards}</div>", unsafe_allow_html=True)


def render_badge(text: str, color: str) -> str:
    return f"<span class='rl-badge' style='background:{color}1a;color:{color};border-color:{color}55'>{text}</span>"


def plot_layout(fig, mode: str):
    dark = mode == "Dark"
    paper = "#111827" if dark else "#ffffff"
    plot = "#111827" if dark else "#ffffff"
    text = "#f8fafc" if dark else "#172033"
    grid = "#263244" if dark else "#e7ebf0"
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=paper,
        plot_bgcolor=plot,
        font_color=text,
        title_font_color=text,
        margin=dict(l=18, r=18, t=58, b=32),
        legend_title_text="",
        height=380,
    )
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid)
    return fig


def display_table(df: pd.DataFrame) -> None:
    visible = df[DISPLAY_COLUMNS].copy()
    visible["amount"] = visible["amount"].map(lambda value: f"Rs. {value:,.2f}")
    visible["fraud_probability"] = visible["fraud_probability"].map(lambda value: f"{value:.1%}")
    visible["anomaly_score"] = visible["anomaly_score"].map(lambda value: f"{value:.1%}")
    st.dataframe(
        visible,
        use_container_width=True,
        hide_index=True,
        height=360,
        column_config={
            "transaction_id": st.column_config.TextColumn("Transaction ID", width="small"),
            "amount": st.column_config.TextColumn("Amount", width="small"),
            "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, width="small"),
            "risk_level": st.column_config.TextColumn("Risk Level", width="small"),
            "fraud_probability": st.column_config.TextColumn("Fraud Probability", width="small"),
            "anomaly_score": st.column_config.TextColumn("Anomaly Score", width="small"),
            "recommended_action": st.column_config.TextColumn("Action", width="medium"),
            "top_risk_factors": st.column_config.TextColumn("Top Risk Factors", width="large"),
        },
    )


st.sidebar.title("RiskLens")
st.sidebar.caption("Financial risk command center")
theme_mode = st.sidebar.toggle("Light mode", value=False)
theme = "Light" if theme_mode else "Dark"
risk_palette = RISK_COLORS if theme == "Light" else RISK_COLORS_DARK

sections = ["Dashboard", "Transaction Analysis", "Transaction Details", "Analyst Queue", "What-if Simulator"]
section = st.sidebar.radio("Navigate", sections)

if theme == "Dark":
    theme_css = """
    :root {
        --rl-ink: #f8fafc;
        --rl-muted: #a8b3c4;
        --rl-line: #263244;
        --rl-panel: #111827;
        --rl-panel-soft: #0f172a;
        --rl-bg: #090f1c;
        --rl-sidebar: #050b16;
        --rl-button: #38bdf8;
        --rl-button-text: #06111f;
    }
    """
else:
    theme_css = """
    :root {
        --rl-ink: #172033;
        --rl-muted: #637083;
        --rl-line: #dfe5ec;
        --rl-panel: #ffffff;
        --rl-panel-soft: #f8fafc;
        --rl-bg: #f6f8fb;
        --rl-sidebar: #101827;
        --rl-button: #0f172a;
        --rl-button-text: #ffffff;
    }
    """

style_block = """
    <style>
    __THEME_CSS__
    .stApp { background: var(--rl-bg); color: var(--rl-ink); overflow-x: hidden; }
    .block-container {
        padding: 1.6rem 2.2rem 2rem;
        max-width: min(100%, calc(100vw - 360px)) !important;
        width: min(100%, calc(100vw - 360px)) !important;
        box-sizing: border-box;
        overflow-x: hidden;
    }
    h1, h2, h3, p, li, label, span, div { color: var(--rl-ink); }
    h1 { font-size: 2.2rem; margin-bottom: 0.2rem; }
    h2, h3 { margin-top: 0.7rem; }
    [data-testid="stSidebar"] { background: var(--rl-sidebar); border-right: 1px solid #243044; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div { color: #f8fafc !important; }
    .rl-page-kicker {
        color: #38bdf8;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0;
        font-size: 0.78rem;
        margin-bottom: 0.25rem;
    }
    .rl-subtitle { color: var(--rl-muted); margin-top: -0.6rem; margin-bottom: 1.25rem; }
    .rl-metric {
        background: var(--rl-panel);
        border: 1px solid var(--rl-line);
        border-radius: 8px;
        padding: 12px 14px;
        min-height: 96px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .rl-metric-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .rl-metric-label { color: var(--rl-muted); font-size: 0.84rem; font-weight: 700; }
    .rl-metric-value { color: var(--rl-ink); font-size: 1.72rem; line-height: 1.2; font-weight: 800; margin-top: 0.25rem; }
    .rl-metric-helper { color: var(--rl-muted); font-size: 0.78rem; margin-top: 0.2rem; min-height: 1rem; }
    .rl-panel {
        background: var(--rl-panel);
        border: 1px solid var(--rl-line);
        border-radius: 8px;
        padding: 18px;
        margin: 0.75rem 0 1rem;
    }
    .rl-badge {
        display: inline-flex;
        align-items: center;
        border: 1px solid;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 0.78rem;
        font-weight: 800;
    }
    .rl-action {
        background: var(--rl-panel-soft);
        border: 1px solid var(--rl-line);
        color: var(--rl-ink);
        border-radius: 8px;
        padding: 14px 16px;
        font-weight: 700;
    }
    .stButton > button, .stDownloadButton > button {
        background: var(--rl-button);
        color: var(--rl-button-text);
        border-radius: 8px;
        border: 1px solid #0f172a;
        font-weight: 700;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #0ea5e9;
        border-color: #0ea5e9;
        color: #06111f;
    }
    [data-testid="stFileUploader"] section {
        background: var(--rl-panel);
        border: 1px dashed #a7b1c2;
        border-radius: 8px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--rl-line);
        border-radius: 8px;
        overflow: hidden;
    }
    div[data-testid="stHorizontalBlock"] { gap: 0.8rem; }
    .stPlotlyChart { background: var(--rl-panel); border: 1px solid var(--rl-line); border-radius: 8px; overflow: hidden; }
    @media (max-width: 900px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .rl-metric-value { font-size: 1.45rem; }
        .rl-metric-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 620px) {
        .rl-metric-grid { grid-template-columns: 1fr; }
    }
    </style>
    """.replace("__THEME_CSS__", theme_css)

st.markdown(style_block, unsafe_allow_html=True)

artifact = get_artifact()
metrics = read_json(METRICS_PATH, artifact.get("metrics", {}))
demo_df = load_demo_data()
report, warnings = analyze(demo_df.drop(columns=["is_fraud"], errors="ignore"))

st.markdown("<div class='rl-page-kicker'>AI Risk Manager</div>", unsafe_allow_html=True)
st.title("RiskLens")
st.markdown(
    "<div class='rl-subtitle'>Fraud probability, anomaly detection, interpretable risk scoring, and analyst-ready recommendations.</div>",
    unsafe_allow_html=True,
)

if section == "Dashboard":
    render_metric_grid([
        ("Total Transactions", f"{len(report):,}", "Synthetic demo batch"),
        ("High Risk", int((report["risk_level"] == "HIGH").sum()), "Requires hold or block"),
        ("Medium Risk", int((report["risk_level"] == "MEDIUM").sum()), "Review or monitor"),
        ("Low Risk", int((report["risk_level"] == "LOW").sum()), "Eligible for approval"),
        ("Fraud Alerts", int((report["fraud_probability"] >= 0.5).sum()), "Probability >= 50%"),
        ("Average Risk Score", f"{report['risk_score'].mean():.1f}", "0-100 scale"),
        ("Model Precision", f"{metrics.get('precision', 0):.2f}", "False alarm control"),
        ("Model Recall", f"{metrics.get('recall', 0):.2f}", "Missed fraud control"),
    ])
    fig = px.histogram(
        report,
        x="risk_score",
        color="risk_level",
        nbins=24,
        title="Risk Score Distribution",
        color_discrete_map=risk_palette,
    )
    st.plotly_chart(plot_layout(fig, theme), use_container_width=True)
    counts = report["recommended_action"].value_counts().reset_index()
    counts.columns = ["action", "count"]
    fig = px.bar(
        counts,
        x="action",
        y="count",
        title="Recommended Actions",
        color="action",
        color_discrete_map=ACTION_COLORS,
    )
    st.plotly_chart(plot_layout(fig, theme), use_container_width=True)

elif section == "Transaction Analysis":
    st.subheader("Analyze Transactions")
    st.markdown(
        "<div class='rl-panel'>Upload a CSV or use the included synthetic demo data. "
        f"Expected columns: <strong>{', '.join(expected_columns(False))}</strong></div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
    st.download_button(
        "Download sample CSV",
        demo_df.drop(columns=["is_fraud"], errors="ignore").head(75).to_csv(index=False),
        "risklens_sample_upload.csv",
        "text/csv",
    )
    uploaded_df, upload_error = read_uploaded_csv(uploaded)
    if upload_error:
        st.error(upload_error)
        source_df = pd.DataFrame()
    else:
        source_df = uploaded_df if uploaded_df is not None else demo_df.drop(columns=["is_fraud"], errors="ignore")
    analyzed, messages = analyze(source_df) if not source_df.empty else (pd.DataFrame(), [])
    for message in messages:
        st.warning(message)
    if analyzed.empty:
        if not upload_error:
            st.error("No valid transactions to analyze.")
    else:
        render_metric_grid([
            ("Analyzed", f"{len(analyzed):,}", "Rows scored"),
            ("High Risk", int((analyzed["risk_level"] == "HIGH").sum()), "Immediate attention"),
            ("Review Queue", int(analyzed["recommended_action"].isin(["REVIEW / MONITOR", "HOLD FOR MANUAL REVIEW"]).sum()), "Human analyst flow"),
            ("Blocks", int((analyzed["recommended_action"] == "BLOCK").sum()), "Critical threshold"),
        ])
        risk_filter = st.multiselect("Risk level", ["LOW", "MEDIUM", "HIGH"], default=["LOW", "MEDIUM", "HIGH"])
        search = st.text_input("Search transaction ID")
        sort_order = st.selectbox("Sort by", ["Highest risk first", "Lowest risk first", "Amount high to low"])
        view = analyzed[analyzed["risk_level"].isin(risk_filter)]
        if search:
            view = view[view["transaction_id"].astype(str).str.contains(search, case=False, na=False)]
        if sort_order == "Lowest risk first":
            view = view.sort_values("risk_score", ascending=True)
        elif sort_order == "Amount high to low":
            view = view.sort_values("amount", ascending=False)
        else:
            view = view.sort_values("risk_score", ascending=False)
        st.caption(f"Showing {len(view):,} of {len(analyzed):,} analyzed transactions")
        st.download_button("Download full risk report CSV", analyzed.to_csv(index=False), "risklens_report.csv", "text/csv")
        display_table(view)

elif section == "Transaction Details":
    st.subheader("Transaction Details")
    selected = st.selectbox("Select transaction", report.sort_values("risk_score", ascending=False)["transaction_id"])
    row = report[report["transaction_id"] == selected].iloc[0]
    render_metric_grid([
        ("Risk Score", f"{row['risk_score']}/100", "Blended risk engine"),
        ("Risk Level", row["risk_level"], "Decision band"),
        ("Fraud Probability", f"{row['fraud_probability']:.2%}", "Supervised model"),
        ("Anomaly Score", f"{row['anomaly_score']:.2%}", "Isolation Forest"),
    ])
    st.markdown(
        f"<div class='rl-action'>Recommended Action: {row['recommended_action']}<br>Reason: {row['action_reason']}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("### Top Risk Factors")
    for reason in explain_transaction(row):
        st.write(f"- {reason}")
    st.markdown("### Transaction Snapshot")
    st.markdown(f"Risk level: {render_badge(row['risk_level'], risk_palette[row['risk_level']])}", unsafe_allow_html=True)
    st.write(f"Amount: Rs. {row['amount']:,.2f}")
    st.write(f"Customer average amount: Rs. {row['avg_customer_amount']:,.2f}")
    st.write(f"Transactions in last 24h: {int(row['transactions_last_24h'])}")
    st.write(f"Merchant category: {row['merchant_category']}")
    detail = row.to_frame("value")
    st.dataframe(detail, use_container_width=True, height=420)

elif section == "Analyst Queue":
    st.subheader("Analyst Queue")
    queue = report[report["recommended_action"].isin(["BLOCK", "HOLD FOR MANUAL REVIEW", "REVIEW / MONITOR"])].copy()
    queue["priority"] = queue["recommended_action"].map({"BLOCK": 1, "HOLD FOR MANUAL REVIEW": 2, "REVIEW / MONITOR": 3})
    queue = queue.sort_values(["priority", "risk_score"], ascending=[True, False])

    render_metric_grid([
        ("Open Cases", f"{len(queue):,}", "Needs analyst attention"),
        ("Critical Blocks", int((queue["recommended_action"] == "BLOCK").sum()), "Highest priority"),
        ("Manual Holds", int((queue["recommended_action"] == "HOLD FOR MANUAL REVIEW").sum()), "Review before approval"),
        ("Estimated Workload", f"{max(1, round(len(queue) * 3 / 60))}h", "Assumes 3 min/case"),
    ])

    st.markdown("<div class='rl-panel'>Prioritized analyst worklist for the highest-risk transactions.</div>", unsafe_allow_html=True)
    display_table(queue.head(100))
    st.download_button("Download analyst queue", queue.to_csv(index=False), "risklens_analyst_queue.csv", "text/csv")

elif section == "What-if Simulator":
    st.subheader("What-if Simulator")
    st.markdown("<div class='rl-panel'>Change transaction conditions and see how RiskLens updates the score and recommendation.</div>", unsafe_allow_html=True)
    base_txn = report.sort_values("risk_score", ascending=False).iloc[0].copy()
    amount = st.number_input("Amount", min_value=0.0, value=float(base_txn["amount"]), step=500.0)
    avg_amount = st.number_input("Customer average amount", min_value=1.0, value=float(base_txn["avg_customer_amount"]), step=100.0)
    velocity = st.slider("Transactions in last 24h", 0, 25, int(base_txn["transactions_last_24h"]))
    hour = st.slider("Hour of day", 0, 23, int(base_txn["hour"]))
    day = st.slider("Day of week", 0, 6, int(base_txn["day_of_week"]))
    foreign = st.toggle("Foreign transaction", value=bool(base_txn["is_foreign"]))
    merchant = st.selectbox("Merchant category", sorted(report["merchant_category"].unique()), index=sorted(report["merchant_category"].unique()).index(base_txn["merchant_category"]))

    scenario = pd.DataFrame([{
        "transaction_id": "SIM-001",
        "customer_id": "SIM-CUSTOMER",
        "amount": amount,
        "merchant_category": merchant,
        "hour": hour,
        "day_of_week": day,
        "transactions_last_24h": velocity,
        "avg_customer_amount": avg_amount,
        "is_foreign": int(foreign),
    }])
    simulated, sim_messages = analyze(scenario)
    for message in sim_messages:
        st.warning(message)
    if not simulated.empty:
        sim = simulated.iloc[0]
        render_metric_grid([
            ("Risk Score", f"{sim['risk_score']}/100", "Simulated"),
            ("Risk Level", sim["risk_level"], "Updated band"),
            ("Fraud Probability", f"{sim['fraud_probability']:.1%}", "Model output"),
            ("Anomaly Score", f"{sim['anomaly_score']:.1%}", "Pattern deviation"),
        ])
        st.markdown(f"<div class='rl-action'>Recommended Action: {sim['recommended_action']}<br>Reason: {sim['action_reason']}</div>", unsafe_allow_html=True)
        st.markdown("### Why this changed")
        for reason in explain_transaction(sim):
            st.write(f"- {reason}")

else:
    st.info("Select a section from the sidebar.")
