"""TimesFM Examples — streamlit run timesfm-forecasting/ui/app.py"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
UI_DIR    = Path(__file__).parent
EXAMPLES  = UI_DIR.parent / "examples"
REPO_ROOT = UI_DIR.parent.parent

GLOBAL_TEMP     = EXAMPLES / "global-temperature"
ANOMALY         = EXAMPLES / "anomaly-detection"
COVARIATES      = EXAMPLES / "covariates-forecasting"
FINETUNING      = EXAMPLES / "finetuning"
BUSINESS        = EXAMPLES / "business-forecast"

# TimesFM requires Python 3.10+; prefer python3.11 if available
_PY = shutil.which("python3.11") or shutil.which("python3") or "python3"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TimesFM",
    page_icon="⏱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Body */
[data-testid="stAppViewContainer"] {
    background: #ffffff;
}
[data-testid="stMain"] > div {
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    font-size: 0.88rem;
    padding: 6px 0;
    cursor: pointer;
    transition: color 0.15s;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    color: #fff !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1e293b !important;
}

/* Typography */
h1 { font-size: 1.6rem !important; font-weight: 700 !important; color: #0f172a !important; letter-spacing: -0.5px; }
h2 { font-size: 1.15rem !important; font-weight: 600 !important; color: #1e293b !important; margin-top: 2rem !important; }
h3 { font-size: 0.95rem !important; font-weight: 600 !important; color: #334155 !important; }

/* Run button */
[data-testid="stButton"] > button {
    background: #0f172a;
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 0.45rem 1.2rem;
    font-size: 0.85rem;
    font-weight: 500;
    transition: background 0.15s, transform 0.1s;
}
[data-testid="stButton"] > button:hover {
    background: #1e293b;
    color: #fff;
}
[data-testid="stButton"] > button:active { transform: scale(0.97); }

/* Metrics */
[data-testid="stMetric"] {
    background: #f8fafc;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #64748b !important; }
[data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: 700 !important; color: #0f172a !important; }

/* Divider */
hr { border-color: #f1f5f9 !important; margin: 1.2rem 0 !important; }

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #f1f5f9 !important;
    border-radius: 8px !important;
}
summary { font-size: 0.85rem !important; color: #475569 !important; }

/* Code blocks */
code { font-size: 0.8rem !important; }
pre  { background: #f8fafc !important; border-radius: 8px !important; border: 1px solid #e2e8f0 !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* Info/warning/success */
[data-testid="stAlert"] { border-radius: 8px !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1.5rem 1rem 0.5rem 1rem;">
        <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.3px;">TimesFM</div>
        <div style="font-size: 0.75rem; color: #475569; margin-top: 2px;">Google Research · ICML 2024</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Examples",
        options=["Global Temperature", "Anomaly Detection", "Covariates", "Fine-tuning", "Business Forecast"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("""
    <div style="padding: 0 1rem; font-size: 0.73rem; color: #334155; line-height: 1.8;">
        <div style="margin-bottom: 0.5rem; color: #475569; font-weight: 500;">Model</div>
        TimesFM 1.0 / 2.5<br>
        200M parameters<br>
        Zero-shot forecasting
        <br><br>
        <a href="https://arxiv.org/abs/2310.10688" style="color: #6366f1; text-decoration: none;">↗ Paper</a>
        &nbsp;·&nbsp;
        <a href="https://huggingface.co/google/timesfm-1.0-200m-pytorch" style="color: #6366f1; text-decoration: none;">↗ HuggingFace</a>
    </div>
    """, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_script(script: Path, cwd: Path) -> tuple[bool, str]:
    r = subprocess.run(
        [_PY, str(script)],
        cwd=str(cwd), capture_output=True, text=True, timeout=300,
    )
    out = r.stdout + ("\n" + r.stderr if r.stderr else "")
    return r.returncode == 0, out.strip()


def show_image(path: Path, caption: str = "") -> None:
    if path.exists():
        st.image(Image.open(path), caption=caption or None, use_container_width=True)
    else:
        st.caption("Run the example to generate this image.")


PLOTLY_LAYOUT = dict(
    height=380,
    margin=dict(l=0, r=0, t=10, b=0),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font=dict(family="Inter, sans-serif", size=12, color="#475569"),
    xaxis=dict(showgrid=False, linecolor="#e2e8f0", tickcolor="#e2e8f0"),
    yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
    hovermode="x unified",
)


# ════════════════════════════════════════════════════════════════════════════
# Global Temperature
# ════════════════════════════════════════════════════════════════════════════
if page == "Global Temperature":

    st.title("Global Temperature Forecast")
    st.caption("36 months of NOAA GISTEMP anomaly data → 12-month zero-shot forecast with prediction intervals")
    st.markdown("---")

    # Run button
    col_l, col_r = st.columns([5, 1])
    with col_r:
        run = st.button("Run", use_container_width=True)
    if run:
        with st.spinner("Running forecast…"):
            ok1, o1 = run_script(GLOBAL_TEMP / "run_forecast.py", GLOBAL_TEMP)
            ok2, o2 = run_script(GLOBAL_TEMP / "visualize_forecast.py", GLOBAL_TEMP)
        if ok1 and ok2:
            st.success("Done.")
        else:
            st.error("Something went wrong.")
        with st.expander("Output log"):
            st.code(o1 + "\n" + o2, language="text")

    # Load data
    json_path = GLOBAL_TEMP / "output" / "forecast_output.json"
    ctx_path  = GLOBAL_TEMP / "temperature_anomaly.csv"

    if json_path.exists():
        d   = json.loads(json_path.read_text())
        s   = d["summary"]
        inp = d["input"]
        fc  = d["forecast"]

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Forecast mean",  f"{s['forecast_mean_c']:.2f} °C")
        c2.metric("Forecast high",  f"{s['forecast_max_c']:.2f} °C")
        c3.metric("Forecast low",   f"{s['forecast_min_c']:.2f} °C")
        c4.metric("vs 2024 avg",    f"{s['vs_last_year_mean']:+.2f} °C")

        st.markdown("---")

        # Chart
        ctx_df = pd.read_csv(ctx_path, parse_dates=["date"]).sort_values("date")
        ctx_x  = ctx_df["date"].dt.strftime("%Y-%m").tolist()
        ctx_y  = ctx_df["anomaly_c"].tolist()

        dates  = fc["dates"]
        point  = fc["point"]
        q10    = fc["quantiles"]["10%"]
        q90    = fc["quantiles"]["90%"]
        q20    = fc["quantiles"]["20%"]
        q80    = fc["quantiles"]["80%"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ctx_x, y=ctx_y,
            name="Observed",
            line=dict(color="#6366f1", width=2),
            mode="lines+markers", marker=dict(size=4),
        ))
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=q90 + q10[::-1],
            fill="toself", fillcolor="rgba(251,146,60,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="80 % PI",
        ))
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=q80 + q20[::-1],
            fill="toself", fillcolor="rgba(251,146,60,0.22)",
            line=dict(color="rgba(0,0,0,0)"), name="60 % PI",
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=point,
            name="Forecast",
            line=dict(color="#f97316", width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=5),
        ))
        fig.add_vline(x=ctx_x[-1], line_dash="dot", line_color="#cbd5e1")
        fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Anomaly (°C)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Table
        st.markdown("**Monthly breakdown**")
        tbl = pd.DataFrame({
            "Month":         dates,
            "Point (°C)":   [f"{v:.3f}" for v in point],
            "10th pct":     [f"{v:.3f}" for v in q10],
            "90th pct":     [f"{v:.3f}" for v in q90],
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.markdown("---")
    show_image(GLOBAL_TEMP / "output" / "forecast_visualization.png")

    with st.expander("Code"):
        st.code("""\
import timesfm

model = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(horizon_len=12),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    ),
)
point, quantiles = model.forecast([values], freq=[0])
# point.shape     → (1, 12)
# quantiles.shape → (1, 12, 10)   index 1 = q10, index 9 = q90
""", language="python")


# ════════════════════════════════════════════════════════════════════════════
# Anomaly Detection
# ════════════════════════════════════════════════════════════════════════════
elif page == "Anomaly Detection":

    st.title("Anomaly Detection")
    st.caption("Two phases: Z-score on the historical window, then quantile prediction intervals on the forecast")
    st.markdown("---")

    col_l, col_r = st.columns([5, 1])
    with col_r:
        run = st.button("Run", use_container_width=True)
    if run:
        with st.spinner("Running…"):
            ok, out = run_script(ANOMALY / "detect_anomalies.py", ANOMALY)
        if ok:
            st.success("Done.")
        else:
            st.error("Something went wrong.")
        with st.expander("Output log"):
            st.code(out, language="text")

    json_path = ANOMALY / "output" / "anomaly_detection.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        cs   = data["context_summary"]
        fs   = data["forecast_summary"]

        # Summary
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Context critical", cs["critical"])
        c2.metric("Context warning",  cs["warning"])
        c3.metric("Context normal",   cs["normal"])
        c4.metric("Forecast critical", fs["critical"])
        c5.metric("Forecast warning",  fs["warning"])
        c6.metric("Forecast normal",   fs["normal"])

        st.markdown("---")

        # Chart
        ctx = data["context_detections"]
        fcd = data["forecast_detections"]
        res_std = cs["res_std"]

        ctx_dates  = [r["date"]     for r in ctx]
        ctx_values = [r["value"]    for r in ctx]
        ctx_trend  = [r["trend"]    for r in ctx]
        fc_dates   = [r["date"]     for r in fcd]
        fc_actual  = [r["actual"]   for r in fcd]
        fc_point   = [r["forecast"] for r in fcd]
        fc_q10     = [r["q10"]      for r in fcd]
        fc_q90     = [r["q90"]      for r in fcd]

        fig = go.Figure()

        # ±2σ band
        fig.add_trace(go.Scatter(
            x=ctx_dates + ctx_dates[::-1],
            y=[t + 2*res_std for t in ctx_trend] + [t - 2*res_std for t in ctx_trend[::-1]],
            fill="toself", fillcolor="rgba(99,102,241,0.08)",
            line=dict(color="rgba(0,0,0,0)"), name="±2σ", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=ctx_dates, y=ctx_trend, mode="lines",
            line=dict(color="#cbd5e1", dash="dash", width=1.5), name="Trend",
        ))
        fig.add_trace(go.Scatter(
            x=ctx_dates, y=ctx_values, mode="lines+markers",
            line=dict(color="#6366f1", width=2), marker=dict(size=4), name="Observed",
        ))

        # Anomaly markers (context)
        crit_x = [r["date"] for r in ctx if r["severity"] == "CRITICAL"]
        crit_y = [r["value"] for r in ctx if r["severity"] == "CRITICAL"]
        warn_x = [r["date"] for r in ctx if r["severity"] == "WARNING"]
        warn_y = [r["value"] for r in ctx if r["severity"] == "WARNING"]
        if crit_x:
            fig.add_trace(go.Scatter(x=crit_x, y=crit_y, mode="markers",
                marker=dict(symbol="diamond", size=10, color="#ef4444"), name="Critical", showlegend=True))
        if warn_x:
            fig.add_trace(go.Scatter(x=warn_x, y=warn_y, mode="markers",
                marker=dict(symbol="diamond", size=10, color="#f59e0b"), name="Warning", showlegend=True))

        # Divider
        fig.add_vline(x=ctx_dates[-1], line_dash="dot", line_color="#cbd5e1")

        # Forecast PI + point + actuals
        fig.add_trace(go.Scatter(
            x=fc_dates + fc_dates[::-1],
            y=fc_q90 + fc_q10[::-1],
            fill="toself", fillcolor="rgba(249,115,22,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="80 % PI",
        ))
        fig.add_trace(go.Scatter(
            x=fc_dates, y=fc_point, mode="lines",
            line=dict(color="#f97316", width=2, dash="dash"), name="Forecast",
        ))
        fig.add_trace(go.Scatter(
            x=fc_dates, y=fc_actual, mode="lines+markers",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
            marker=dict(size=4), name="Synthetic truth",
        ))

        fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Temperature anomaly (°C)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Tables side by side
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Context anomalies**")
            flagged = [r for r in ctx if r["severity"] != "NORMAL"]
            if flagged:
                df = pd.DataFrame(flagged)[["date", "value", "z_score", "severity"]]
                df.columns = ["Month", "°C", "Z-score", "Severity"]
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.caption("None detected.")
        with t2:
            st.markdown("**Forecast anomalies**")
            flagged_fc = [r for r in fcd if r["severity"] != "NORMAL"]
            if flagged_fc:
                df2 = pd.DataFrame(flagged_fc)[["date", "actual", "forecast", "severity", "was_injected"]]
                df2.columns = ["Month", "Actual", "Forecast", "Severity", "Injected"]
                st.dataframe(df2, use_container_width=True, hide_index=True)
            else:
                st.caption("None detected.")

    st.markdown("---")
    show_image(ANOMALY / "output" / "anomaly_detection.png")

    with st.expander("Code"):
        st.code("""\
# Phase 1 — Z-score on context
trend = np.polyval(np.polyfit(np.arange(n), values, 1), np.arange(n))
z     = (values - trend) / (values - trend).std()
# |z| >= 3.0 → CRITICAL,  |z| >= 2.0 → WARNING

# Phase 2 — quantile PI on forecast
_, q = model.forecast([values], freq=[0])
q10, q90 = q[0, :, 1], q[0, :, 9]    # index 1 = q10, index 9 = q90
outside = (actual < q10) | (actual > q90)
""", language="python")


# ════════════════════════════════════════════════════════════════════════════
# Covariates
# ════════════════════════════════════════════════════════════════════════════
elif page == "Covariates":

    st.title("Covariates Forecasting")
    st.caption("Retail sales across 3 stores with price, promotion, and holiday effects — using TimesFM 2.5 XReg")
    st.markdown("---")

    col_l, col_r = st.columns([5, 1])
    with col_r:
        run = st.button("Run", use_container_width=True)
    if run:
        with st.spinner("Generating data…"):
            ok, out = run_script(COVARIATES / "demo_covariates.py", COVARIATES)
        if ok:
            st.success("Done.")
        else:
            st.error("Something went wrong.")
        with st.expander("Output log"):
            st.code(out, language="text")

    csv_path  = COVARIATES / "output" / "sales_with_covariates.csv"
    meta_path = COVARIATES / "output" / "covariates_metadata.json"

    if csv_path.exists():
        df   = pd.read_csv(csv_path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        # Store metrics
        if meta:
            st_data = meta.get("stores", {})
            c1, c2, c3 = st.columns(3)
            for col, sid in zip([c1, c2, c3], ["store_A", "store_B", "store_C"]):
                s = st_data.get(sid, {})
                col.metric(
                    f"{sid}  ·  {s.get('type', '')}",
                    f"{s.get('mean_sales_context', 0):.0f} units / wk",
                    delta=s.get("region", ""),
                )

        st.markdown("---")

        # Sales chart
        colors = {"store_A": "#6366f1", "store_B": "#10b981", "store_C": "#f97316"}
        fig = go.Figure()
        for sid, color in colors.items():
            sub = df[df["store_id"] == sid].sort_values("week")
            ctx = sub[sub["split"] == "context"]
            hor = sub[sub["split"] == "horizon"]
            fig.add_trace(go.Scatter(
                x=ctx["week"], y=ctx["sales"], mode="lines",
                line=dict(color=color, width=2.5), name=sid,
            ))
            fig.add_trace(go.Scatter(
                x=hor["week"], y=hor["sales"], mode="lines",
                line=dict(color=color, width=1.5, dash="dash"),
                showlegend=False, opacity=0.6,
            ))
        fig.add_vline(x=23.5, line_dash="dot", line_color="#cbd5e1",
                      annotation_text="horizon", annotation_font_color="#94a3b8")
        fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Weekly sales (units)", xaxis_title="Week")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Effect magnitudes
        if meta.get("effect_magnitudes"):
            st.markdown("**Covariate effects on Store A**")
            e = meta["effect_magnitudes"]
            e1, e2, e3 = st.columns(3)
            e1.metric("Holiday weeks",   e.get("holiday", "—"))
            e2.metric("Promotion weeks", e.get("promotion", "—"))
            e3.metric("Price elasticity", e.get("price", "—"))

        st.markdown("---")

        with st.expander("Raw data  (first 24 rows)"):
            st.dataframe(df.head(24), use_container_width=True, hide_index=True)

    show_image(COVARIATES / "output" / "covariates_data.png")

    with st.expander("Code"):
        st.code("""\
# Requires: pip install timesfm[xreg]
point, quantiles = model.forecast_with_covariates(
    inputs=[sales_a, sales_b, sales_c],
    dynamic_numerical_covariates={"price":   [price_a, price_b, price_c]},
    dynamic_categorical_covariates={"holiday": [hol_a,   hol_b,   hol_c]},
    static_categorical_covariates={"store_type": ["premium", "standard", "discount"]},
    xreg_mode="xreg + timesfm",
)
""", language="python")


# ════════════════════════════════════════════════════════════════════════════
# Fine-tuning
# ════════════════════════════════════════════════════════════════════════════
elif page == "Fine-tuning":

    st.title("LoRA Fine-tuning")
    st.caption("Parameter-efficient fine-tuning of TimesFM 2.5 on retail sales with HuggingFace PEFT")
    st.markdown("---")

    st.warning(
        "Fine-tuning downloads the full model (~800 MB) and trains for multiple epochs. "
        "Run from the terminal — configure parameters below and copy the command.",
        icon="⚠️",
    )

    st.markdown("---")
    st.markdown("**Training parameters**")

    c1, c2 = st.columns(2)
    with c1:
        context_len  = st.slider("Context length (weeks)",   32,  512,  64, step=32)
        horizon_len  = st.slider("Forecast horizon (weeks)",  4,   52,  13)
        epochs       = st.slider("Epochs",                    1,   50,  10)
        batch_size   = st.selectbox("Batch size", [8, 16, 32, 64], index=2)
    with c2:
        lr           = st.select_slider("Learning rate", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3], value=1e-4)
        lora_r       = st.selectbox("LoRA rank",  [2, 4, 8, 16], index=1)
        lora_alpha   = st.selectbox("LoRA alpha", [4, 8, 16, 32], index=1)
        num_samples  = st.slider("Training windows", 1000, 10000, 5000, step=500)

    # Memory estimate
    total_gb = 0.8 + 0.5 + 0.0002 * num_samples * context_len / 1000
    m1, m2 = st.columns(2)
    m1.metric("Estimated RAM", f"{total_gb:.1f} GB")
    m2.metric("Trainable params", f"~{lora_r * 2 * 12 / 1000:.0f}K  ({lora_r * 2 * 12 * 100 / 200_000:.2f}% of 200M)")
    if total_gb > 16:
        st.error("Exceeds 16 GB — reduce context length or training windows.")
    elif total_gb > 8:
        st.warning("Tight on 8 GB RAM.")

    st.markdown("---")

    script = (FINETUNING / "finetune_lora.py").relative_to(REPO_ROOT)
    cmd = (
        f"python {script} \\\n"
        f"  --context_len {context_len} --horizon_len {horizon_len} \\\n"
        f"  --epochs {epochs} --batch_size {batch_size} --lr {lr} \\\n"
        f"  --lora_r {lora_r} --lora_alpha {lora_alpha} \\\n"
        f"  --num_samples {num_samples}"
    )
    st.markdown("**Terminal command**")
    st.code(cmd, language="bash")

    st.markdown("---")
    st.markdown("**Install dependencies first**")
    st.code("pip install transformers accelerate peft pandas pyarrow scikit-learn", language="bash")

    with st.expander("Code"):
        st.code("""\
from peft import LoraConfig, get_peft_model
from transformers import TimesFm2_5ModelForPrediction

model = TimesFm2_5ModelForPrediction.from_pretrained(
    "google/timesfm-2.5-200m-transformers",
    torch_dtype=torch.bfloat16,
)
model = get_peft_model(model, LoraConfig(
    r=4, lora_alpha=8,
    target_modules="all-linear",
    lora_dropout=0.05,
))

outputs = model(past_values=context, future_values=targets)
outputs.loss.backward()
""", language="python")

    readme = FINETUNING / "README.md"
    if readme.exists():
        with st.expander("README"):
            st.markdown(readme.read_text())


# ════════════════════════════════════════════════════════════════════════════
# Business Forecast
# ════════════════════════════════════════════════════════════════════════════
elif page == "Business Forecast":

    st.title("Business Plan Forecast")
    st.caption(
        "36 months of synthetic B2B SaaS P&L history → 12-month TimesFM forecast "
        "with pessimistic / base / optimistic scenarios"
    )
    st.markdown("---")

    col_l, col_r = st.columns([5, 1])
    with col_r:
        run = st.button("Run", use_container_width=True)
    if run:
        with st.spinner("Running forecast (downloads model on first run)…"):
            ok, out = run_script(BUSINESS / "business_forecast.py", BUSINESS)
        if ok:
            st.success("Done.")
        else:
            st.error("Something went wrong.")
        with st.expander("Output log"):
            st.code(out, language="text")

    json_path = BUSINESS / "output" / "business_forecast.json"
    csv_path  = BUSINESS / "output" / "business_data.csv"

    if json_path.exists():
        d   = json.loads(json_path.read_text())
        ins = d["insights"]
        fc  = d["forecasts"]
        fc_dates = d["forecast_dates"]

        # ── Key insights ───────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current ARR",     f"${ins['arr_current_usd']/1e6:.2f}M")
        c2.metric("Forecast ARR",    f"${ins['arr_forecast_usd']/1e6:.2f}M",
                  delta=f"+{(ins['arr_forecast_usd']-ins['arr_current_usd'])/ins['arr_current_usd']*100:.1f}%")
        c3.metric("MRR YoY growth",  f"{ins['mrr_yoy_growth_pct']:+.1f}%")
        c4.metric("Year-end HC",     f"{ins['headcount_eoy']:.0f} people")

        st.markdown("---")

        # ── MRR chart ─────────────────────────────────────────────────────
        st.markdown("**Monthly Recurring Revenue**")
        if csv_path.exists():
            hist_df = pd.read_csv(csv_path)
            hist = hist_df[hist_df["split"] == "history"]
            hist_x = hist["date"].tolist()
            hist_mrr = hist["mrr"].tolist()
        else:
            hist_x, hist_mrr = [], []

        fig = go.Figure()
        if hist_x:
            fig.add_trace(go.Scatter(
                x=hist_x, y=hist_mrr, name="Historical",
                line=dict(color="#6366f1", width=2.5), mode="lines",
            ))

        # Scenario band
        fig.add_trace(go.Scatter(
            x=fc_dates + fc_dates[::-1],
            y=fc["mrr"]["optimistic"] + fc["mrr"]["pessimistic"][::-1],
            fill="toself", fillcolor="rgba(249,115,22,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="Scenario range", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=fc_dates, y=fc["mrr"]["pessimistic"], mode="lines",
            line=dict(color="#ef4444", width=1.5, dash="dot"), name="Pessimistic",
        ))
        fig.add_trace(go.Scatter(
            x=fc_dates, y=fc["mrr"]["base"], mode="lines+markers",
            line=dict(color="#f97316", width=2.5), marker=dict(size=5), name="Base",
        ))
        fig.add_trace(go.Scatter(
            x=fc_dates, y=fc["mrr"]["optimistic"], mode="lines",
            line=dict(color="#22c55e", width=1.5, dash="dot"), name="Optimistic",
        ))
        if hist_x:
            fig.add_vline(x=hist_x[-1], line_dash="dot", line_color="#cbd5e1")
        fig.update_layout(
            **PLOTLY_LAYOUT,
            yaxis_title="MRR (USD)",
            yaxis_tickformat="$,.0f",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Net income + expenses side by side ────────────────────────────
        left, right = st.columns(2)

        with left:
            st.markdown("**Net Income**")
            fig2 = go.Figure()
            if hist_x and csv_path.exists():
                ni_hist = hist["net_income"].tolist()
                colors_bar = ["#22c55e" if v >= 0 else "#ef4444" for v in ni_hist]
                fig2.add_trace(go.Bar(
                    x=hist_x, y=ni_hist, name="Historical",
                    marker_color=colors_bar, opacity=0.75,
                ))
            fig2.add_trace(go.Scatter(
                x=fc_dates + fc_dates[::-1],
                y=fc["net_income"]["optimistic"] + fc["net_income"]["pessimistic"][::-1],
                fill="toself", fillcolor="rgba(249,115,22,0.12)",
                line=dict(color="rgba(0,0,0,0)"), name="Range",
            ))
            fig2.add_trace(go.Scatter(
                x=fc_dates, y=fc["net_income"]["base"], mode="lines",
                line=dict(color="#f97316", width=2), name="Base forecast",
            ))
            fig2.add_hline(y=0, line_color="#cbd5e1", line_dash="solid", line_width=1)
            layout2 = {**PLOTLY_LAYOUT, "height": 300}
            fig2.update_layout(**layout2, yaxis_title="Net income (USD)", yaxis_tickformat="$,.0f")
            st.plotly_chart(fig2, use_container_width=True)

        with right:
            st.markdown("**Operating Expenses**")
            fig3 = go.Figure()
            if hist_x and csv_path.exists():
                fig3.add_trace(go.Scatter(
                    x=hist_x, y=hist["opex"].tolist(), name="Historical",
                    line=dict(color="#6366f1", width=2.5), mode="lines",
                ))
            fig3.add_trace(go.Scatter(
                x=fc_dates + fc_dates[::-1],
                y=fc["opex"]["optimistic"] + fc["opex"]["pessimistic"][::-1],
                fill="toself", fillcolor="rgba(249,115,22,0.10)",
                line=dict(color="rgba(0,0,0,0)"), name="Range",
            ))
            fig3.add_trace(go.Scatter(
                x=fc_dates, y=fc["opex"]["base"], mode="lines",
                line=dict(color="#f97316", width=2), name="Base forecast",
            ))
            layout3 = {**PLOTLY_LAYOUT, "height": 300}
            fig3.update_layout(**layout3, yaxis_title="OpEx (USD)", yaxis_tickformat="$,.0f")
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")

        # ── Headcount ─────────────────────────────────────────────────────
        st.markdown("**Headcount**")
        fig4 = go.Figure()
        if hist_x and csv_path.exists():
            fig4.add_trace(go.Scatter(
                x=hist_x, y=hist["headcount"].tolist(), name="Historical",
                line=dict(color="#8b5cf6", width=2.5), mode="lines",
                fill="tozeroy", fillcolor="rgba(139,92,246,0.08)",
            ))
        fig4.add_trace(go.Scatter(
            x=fc_dates + fc_dates[::-1],
            y=fc["headcount"]["optimistic"] + fc["headcount"]["pessimistic"][::-1],
            fill="toself", fillcolor="rgba(249,115,22,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name="Range",
        ))
        fig4.add_trace(go.Scatter(
            x=fc_dates, y=fc["headcount"]["base"], mode="lines+markers",
            line=dict(color="#f97316", width=2), marker=dict(size=5), name="Base forecast",
        ))
        if hist_x:
            fig4.add_vline(x=hist_x[-1], line_dash="dot", line_color="#cbd5e1")
        fig4.update_layout(**{**PLOTLY_LAYOUT, "height": 280}, yaxis_title="Headcount")
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")

        # ── Scenario comparison table ──────────────────────────────────────
        st.markdown("**12-Month Scenario Summary**")
        ni = d["forecasts"]["net_income"]
        mrr_f = d["forecasts"]["mrr"]
        tbl = pd.DataFrame({
            "Scenario":    ["Pessimistic", "Base", "Optimistic"],
            "Final MRR":   [
                f"${mrr_f['pessimistic'][-1]/1e3:.0f}K",
                f"${mrr_f['base'][-1]/1e3:.0f}K",
                f"${mrr_f['optimistic'][-1]/1e3:.0f}K",
            ],
            "Cumulative Net Income": [
                f"${ins['net_income_range']['pessimistic']/1e3:.0f}K",
                f"${ins['net_income_range']['base']/1e3:.0f}K",
                f"${ins['net_income_range']['optimistic']/1e3:.0f}K",
            ],
            "Year-end HC": [
                f"{fc['headcount']['pessimistic'][-1]:.0f}",
                f"{fc['headcount']['base'][-1]:.0f}",
                f"{fc['headcount']['optimistic'][-1]:.0f}",
            ],
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.markdown("---")
    show_image(BUSINESS / "output" / "business_forecast.png")

    with st.expander("Code"):
        st.code("""\
# Batch-forecast multiple business metrics at once
metrics = ["mrr", "opex", "net_income", "headcount"]
inputs  = [df[m].values.astype(np.float32) for m in metrics]

point, quantiles = model.forecast(inputs, freq=[0] * len(inputs))

# Three scenarios per metric
base        = point[i]                  # q50 point forecast
pessimistic = quantiles[i, :, 2]       # q20
optimistic  = quantiles[i, :, 8]       # q80
""", language="python")
