#!/usr/bin/env python3
"""
TimesFM Business Plan Forecasting Example

Simulates 3 years (36 months) of historical P&L data for a B2B SaaS company
and uses TimesFM to forecast the next 12 months across four key metrics:

  - Monthly Recurring Revenue (MRR)
  - Operating Expenses (OpEx)
  - Net Income (MRR - OpEx)
  - Headcount

Produces three forecast scenarios per metric:
  - Pessimistic  → q20 quantile
  - Base         → point forecast (q50)
  - Optimistic   → q80 quantile

Outputs
-------
  output/business_forecast.png   — 2×2 panel visualization
  output/business_forecast.json  — structured forecast + business insights
  output/business_data.csv       — full dataset (history + forecast rows)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "output"
HORIZON    = 12
SEED       = 42

# Quantile index mapping: 0=mean, 1=q10 … 9=q90
IDX_Q20, IDX_Q50, IDX_Q80 = 2, 5, 8


# ── Data generation ────────────────────────────────────────────────────────

def generate_business_data(seed: int = SEED) -> pd.DataFrame:
    """
    Generate 36 months of synthetic B2B SaaS P&L data (Jan 2022 – Dec 2024).

    Company profile
    ---------------
    - Starts at $45K MRR, grows to ~$210K MRR over 3 years
    - Monthly growth rate decelerates: 4–6% early → 1–2% late (S-curve)
    - Revenue has mild Q4 seasonality (+8%) and soft Q1 (−5%)
    - OpEx grows in discrete steps (hiring events) with a budget-cycle pattern
    - Headcount grows from 6 → 28 with two hiring surges (months 10, 24)
    """
    rng    = np.random.default_rng(seed)
    n      = 36
    months = pd.date_range("2022-01-01", periods=n, freq="MS")

    # ── Revenue (S-curve growth + seasonality + noise) ─────────────────────
    t            = np.linspace(0, 1, n)
    s_curve      = 1 / (1 + np.exp(-8 * (t - 0.45)))           # logistic
    mrr_base     = 45_000 + 165_000 * s_curve                   # $45K → $210K
    seasonality  = np.array([
        -0.05, -0.03, 0.01, 0.02, 0.03, 0.04,
         0.03,  0.02, 0.04, 0.06, 0.07, 0.08,
    ] * 3)
    mrr = mrr_base * (1 + seasonality) + rng.normal(0, 3_000, n)
    mrr = np.maximum(mrr, 10_000).astype(np.float32)

    # ── Headcount (step increases at months 9 and 23) ─────────────────────
    hc = np.ones(n, dtype=np.float32) * 6
    for i in range(n):
        if   i < 9:  hc[i] = 6  + i * 0.3
        elif i < 23: hc[i] = 10 + (i - 9) * 0.55
        else:        hc[i] = 18 + (i - 23) * 0.45
    hc += rng.normal(0, 0.4, n)
    hc  = np.maximum(hc, 1).astype(np.float32)

    # ── OpEx (headcount-driven + infrastructure + marketing) ──────────────
    salary_cost  = hc * 9_500                                   # ~$9.5K/hc/mo
    infra_cost   = 3_000 + mrr * 0.05                          # 5% of MRR
    marketing    = mrr * 0.12 + rng.normal(0, 1_000, n)        # 12% of MRR
    opex         = salary_cost + infra_cost + marketing
    opex        += rng.normal(0, 4_000, n)
    opex         = np.maximum(opex, 20_000).astype(np.float32)

    # ── Net income ─────────────────────────────────────────────────────────
    net_income = (mrr - opex).astype(np.float32)

    return pd.DataFrame({
        "date":       months,
        "mrr":        mrr,
        "opex":       opex,
        "net_income": net_income,
        "headcount":  hc,
    })


# ── Forecasting ────────────────────────────────────────────────────────────

def run_forecast(df: pd.DataFrame):
    """Load TimesFM and batch-forecast all four metrics."""
    print("\n  Loading TimesFM 1.0…")
    import timesfm

    hparams    = timesfm.TimesFmHparams(horizon_len=HORIZON)
    checkpoint = timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    )
    model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

    metrics = ["mrr", "opex", "net_income", "headcount"]
    inputs  = [df[m].values.astype(np.float32) for m in metrics]

    print("  Running batch forecast…")
    point_out, quant_out = model.forecast(inputs, freq=[0] * len(inputs))

    results = {}
    for i, m in enumerate(metrics):
        results[m] = {
            "point": point_out[i],               # (HORIZON,)
            "quant": quant_out[i],               # (HORIZON, 10)
        }
    return results


# ── Visualization ──────────────────────────────────────────────────────────

def _fmt_money(ax) -> None:
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1_000:.0f}K" if abs(x) >= 1000 else f"${x:.0f}"
    ))


def plot_results(df: pd.DataFrame, fc: dict, forecast_dates: list) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    hist_x = df["date"].tolist()
    fc_x   = forecast_dates

    PALETTE = {
        "hist":    "#4f46e5",
        "base":    "#f97316",
        "opt":     "#22c55e",
        "pess":    "#ef4444",
        "band":    "rgba(249,115,22,0.12)",
        "zero":    "#94a3b8",
        "hc":      "#8b5cf6",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.32})
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle("Business Plan Forecast  ·  TimesFM 12-Month Outlook",
                 fontsize=14, fontweight="bold", color="#0f172a", y=1.01)

    def divider(ax):
        ax.axvline(hist_x[-1], color="#cbd5e1", lw=1.2, ls=":")
        ax.axvspan(hist_x[-1], fc_x[-1], alpha=0.04, color="#94a3b8", zorder=0)

    def style(ax):
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#e2e8f0")
        ax.tick_params(colors="#64748b", labelsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(8))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.grid(axis="y", color="#f1f5f9", lw=1)

    # ── Panel (0,0): MRR ────────────────────────────────────────────────────
    ax = axes[0, 0]
    m  = "mrr"
    q  = fc[m]["quant"]
    pt = fc[m]["point"]
    ax.plot(hist_x, df[m], color=PALETTE["hist"], lw=2, label="Historical")
    ax.fill_between(fc_x, q[:, IDX_Q20], q[:, IDX_Q80], alpha=0.18, color=PALETTE["base"])
    ax.plot(fc_x, q[:, IDX_Q80], color=PALETTE["opt"],  lw=1.5, ls="--", label="Optimistic")
    ax.plot(fc_x, pt,            color=PALETTE["base"], lw=2.2, ls="-",  label="Base")
    ax.plot(fc_x, q[:, IDX_Q20], color=PALETTE["pess"], lw=1.5, ls="--", label="Pessimistic")
    divider(ax); style(ax); _fmt_money(ax)
    ax.set_title("Monthly Recurring Revenue (MRR)", fontsize=11, fontweight="600", color="#1e293b")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)

    # ── Panel (0,1): OpEx ───────────────────────────────────────────────────
    ax = axes[0, 1]
    m  = "opex"
    q  = fc[m]["quant"]
    pt = fc[m]["point"]
    ax.plot(hist_x, df[m], color=PALETTE["hist"], lw=2, label="Historical")
    ax.fill_between(fc_x, q[:, IDX_Q20], q[:, IDX_Q80], alpha=0.18, color=PALETTE["base"])
    ax.plot(fc_x, q[:, IDX_Q80], color=PALETTE["pess"], lw=1.5, ls="--", label="High scenario")
    ax.plot(fc_x, pt,            color=PALETTE["base"], lw=2.2, ls="-",  label="Base")
    ax.plot(fc_x, q[:, IDX_Q20], color=PALETTE["opt"],  lw=1.5, ls="--", label="Low scenario")
    divider(ax); style(ax); _fmt_money(ax)
    ax.set_title("Operating Expenses (OpEx)", fontsize=11, fontweight="600", color="#1e293b")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)

    # ── Panel (1,0): Net Income ─────────────────────────────────────────────
    ax = axes[1, 0]
    m  = "net_income"
    q  = fc[m]["quant"]
    pt = fc[m]["point"]
    ax.axhline(0, color=PALETTE["zero"], lw=1.2, ls="-", alpha=0.6, zorder=1)
    # Color positive vs negative history
    hist_y = df[m].values
    ax.fill_between(hist_x, hist_y, 0,
                    where=(hist_y >= 0), alpha=0.15, color="#22c55e", interpolate=True)
    ax.fill_between(hist_x, hist_y, 0,
                    where=(hist_y < 0),  alpha=0.15, color="#ef4444", interpolate=True)
    ax.plot(hist_x, hist_y, color=PALETTE["hist"], lw=2, label="Historical")
    ax.fill_between(fc_x, q[:, IDX_Q20], q[:, IDX_Q80], alpha=0.18, color=PALETTE["base"])
    ax.plot(fc_x, q[:, IDX_Q80], color=PALETTE["opt"],  lw=1.5, ls="--", label="Optimistic")
    ax.plot(fc_x, pt,            color=PALETTE["base"], lw=2.2, ls="-",  label="Base")
    ax.plot(fc_x, q[:, IDX_Q20], color=PALETTE["pess"], lw=1.5, ls="--", label="Pessimistic")
    divider(ax); style(ax); _fmt_money(ax)
    ax.set_title("Net Income  (MRR − OpEx)", fontsize=11, fontweight="600", color="#1e293b")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)

    # ── Panel (1,1): Headcount ──────────────────────────────────────────────
    ax = axes[1, 1]
    m  = "headcount"
    q  = fc[m]["quant"]
    pt = fc[m]["point"]
    ax.fill_between(hist_x, df[m], alpha=0.12, color=PALETTE["hc"], step="mid")
    ax.step(hist_x, df[m], color=PALETTE["hc"], lw=2, where="mid", label="Historical")
    ax.fill_between(fc_x, q[:, IDX_Q20], q[:, IDX_Q80], alpha=0.15, color=PALETTE["base"], step="mid")
    ax.step(fc_x, pt,            color=PALETTE["base"], lw=2.2, where="mid", label="Base")
    ax.step(fc_x, q[:, IDX_Q80], color=PALETTE["opt"],  lw=1.5, where="mid", ls="--", label="Optimistic")
    ax.step(fc_x, q[:, IDX_Q20], color=PALETTE["pess"], lw=1.5, where="mid", ls="--", label="Pessimistic")
    divider(ax); style(ax)
    ax.set_title("Headcount", fontsize=11, fontweight="600", color="#1e293b")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)

    # Shared date label on bottom panels
    for ax in axes[1]:
        ax.set_xlabel("Month", fontsize=9, color="#64748b")

    out = OUTPUT_DIR / "business_forecast.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close()
    print(f"\n  Saved: {out}")


# ── Business insights ──────────────────────────────────────────────────────

def compute_insights(df: pd.DataFrame, fc: dict, forecast_dates: list) -> dict:
    """Derive plain-language insights from the forecast."""
    mrr_hist  = df["mrr"].values
    mrr_base  = fc["mrr"]["point"]
    opex_base = fc["opex"]["point"]
    ni_base   = fc["net_income"]["point"]
    hc_base   = fc["headcount"]["point"]

    # MRR growth
    mrr_yoy = (mrr_base[-1] - mrr_hist[-12]) / mrr_hist[-12] * 100

    # Profit months
    profitable_months = int((ni_base > 0).sum())

    # Break-even: first forecast month with positive net income
    be_month = next(
        (forecast_dates[i].strftime("%Y-%m") for i, v in enumerate(ni_base) if v > 0),
        "Not reached in forecast window",
    )

    # ARR
    arr_current  = float(mrr_hist[-1]) * 12
    arr_forecast = float(mrr_base[-1]) * 12

    return {
        "arr_current_usd":    round(arr_current),
        "arr_forecast_usd":   round(arr_forecast),
        "mrr_yoy_growth_pct": round(float(mrr_yoy), 1),
        "profitable_months_in_horizon": profitable_months,
        "break_even_month":   be_month,
        "headcount_eoy":      round(float(hc_base[-1]), 1),
        "net_income_range": {
            "pessimistic": round(float(fc["net_income"]["quant"][:, IDX_Q20].sum())),
            "base":        round(float(ni_base.sum())),
            "optimistic":  round(float(fc["net_income"]["quant"][:, IDX_Q80].sum())),
        },
    }


# ── Export ─────────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame, fc: dict, forecast_dates: list) -> None:
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "date":       row["date"].strftime("%Y-%m"),
            "split":      "history",
            "mrr":        round(float(row["mrr"]), 2),
            "opex":       round(float(row["opex"]), 2),
            "net_income": round(float(row["net_income"]), 2),
            "headcount":  round(float(row["headcount"]), 1),
            "mrr_pess":   None, "mrr_base":  None, "mrr_opt":  None,
        })
    for i, dt in enumerate(forecast_dates):
        rows.append({
            "date":       dt.strftime("%Y-%m"),
            "split":      "forecast",
            "mrr":        None,
            "opex":       None,
            "net_income": None,
            "headcount":  None,
            "mrr_pess":   round(float(fc["mrr"]["quant"][i, IDX_Q20]), 2),
            "mrr_base":   round(float(fc["mrr"]["point"][i]), 2),
            "mrr_opt":    round(float(fc["mrr"]["quant"][i, IDX_Q80]), 2),
        })
    out = OUTPUT_DIR / "business_data.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Saved: {out}")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("  TIMESFM BUSINESS PLAN FORECASTING")
    print("=" * 62)

    print("\n  Generating 36-month P&L history…")
    df = generate_business_data()
    print(f"  Date range : {df['date'].iloc[0].strftime('%Y-%m')} – {df['date'].iloc[-1].strftime('%Y-%m')}")
    print(f"  MRR range  : ${df['mrr'].min()/1e3:.0f}K – ${df['mrr'].max()/1e3:.0f}K/mo")
    print(f"  Final HC   : {df['headcount'].iloc[-1]:.0f} employees")

    forecast_dates = pd.date_range(
        df["date"].iloc[-1] + pd.DateOffset(months=1),
        periods=HORIZON, freq="MS",
    ).tolist()

    fc = run_forecast(df)

    print("\n  Generating visualization…")
    plot_results(df, fc, forecast_dates)

    insights = compute_insights(df, fc, forecast_dates)
    print("\n  Business Insights:")
    print(f"    Current ARR         : ${insights['arr_current_usd']/1e6:.2f}M")
    print(f"    Forecast ARR (12mo) : ${insights['arr_forecast_usd']/1e6:.2f}M")
    print(f"    MRR YoY growth      : {insights['mrr_yoy_growth_pct']:+.1f}%")
    print(f"    Break-even month    : {insights['break_even_month']}")
    print(f"    Profitable months   : {insights['profitable_months_in_horizon']}/12")
    print(f"    Year-end headcount  : {insights['headcount_eoy']:.0f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = {
        "model":          "TimesFM 1.0 (200M) PyTorch",
        "history_months": len(df),
        "horizon_months": HORIZON,
        "scenarios":      {"pessimistic": "q20", "base": "point (q50)", "optimistic": "q80"},
        "metrics":        ["mrr", "opex", "net_income", "headcount"],
        "forecast_dates": [d.strftime("%Y-%m") for d in forecast_dates],
        "forecasts": {
            m: {
                "base":       fc[m]["point"].tolist(),
                "pessimistic": fc[m]["quant"][:, IDX_Q20].tolist(),
                "optimistic":  fc[m]["quant"][:, IDX_Q80].tolist(),
            }
            for m in ["mrr", "opex", "net_income", "headcount"]
        },
        "insights": insights,
    }
    json_out = OUTPUT_DIR / "business_forecast.json"
    with open(json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {json_out}")

    export_csv(df, fc, forecast_dates)

    print("\n" + "=" * 62)
    print("  COMPLETE")
    print("=" * 62)


if __name__ == "__main__":
    main()
