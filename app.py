# app.py (Gradio)
import joblib
import pandas as pd
from download_models import ensure_models
import gradio as gr
import os

# Ensure models are present (downloads if missing)
classifier_path, regressor_path = ensure_models()

# Load models
print("Loading models...")
clf_model = joblib.load(classifier_path)
reg_model = joblib.load(regressor_path)
print("Models loaded.")

# Required feature columns (from your pipeline)
FEATURE_COLUMNS = [
    "current_bank_balance","monthly_expense","monthly_revenue","runway_months",
    "recurring_obligations","cash_inflows_next_30d","startup_age_months",
    "burn_variability_index","cash_utilization_rate",
    "sector","investment_style","compliance_flag","has_funding_round"
]

def predict_single(current_bank_balance, monthly_expense, monthly_revenue,
                   runway_months, recurring_obligations, cash_inflows_next_30d,
                   startup_age_months, burn_variability_index, cash_utilization_rate,
                   sector, investment_style, compliance_flag, has_funding_round):
    row = {
        "current_bank_balance": float(current_bank_balance),
        "monthly_expense": float(monthly_expense),
        "monthly_revenue": float(monthly_revenue),
        "runway_months": float(runway_months),
        "recurring_obligations": float(recurring_obligations),
        "cash_inflows_next_30d": float(cash_inflows_next_30d),
        "startup_age_months": float(startup_age_months),
        "burn_variability_index": float(burn_variability_index),
        "cash_utilization_rate": float(cash_utilization_rate),
        "sector": str(sector),
        "investment_style": str(investment_style),
        "compliance_flag": str(compliance_flag),
        "has_funding_round": str(has_funding_round)
    }
    df = pd.DataFrame([row])
    suggested = clf_model.predict(df)[0]
    expected = reg_model.predict(df)[0]
    return {"Suggested Action": str(suggested), "Expected Return": float(expected)}

def predict_csv(file_obj):
    try:
        df = pd.read_csv(file_obj.name)
    except Exception:
        df = pd.read_csv(file_obj)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        return f"CSV missing required columns: {missing}"
    df_out = df.copy()
    df_out["Suggested Action"] = clf_model.predict(df_out)
    df_out["Expected Return"] = reg_model.predict(df_out)
    return df_out

# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# AlphaCash Engine")
    with gr.Tab("Manual Input"):
        with gr.Row():
            current_bank_balance = gr.Number(value=250000, label="current_bank_balance")
            monthly_expense = gr.Number(value=80000, label="monthly_expense")
            monthly_revenue = gr.Number(value=120000, label="monthly_revenue")
        with gr.Row():
            runway_months = gr.Number(value=6, label="runway_months")
            recurring_obligations = gr.Number(value=15000, label="recurring_obligations")
            cash_inflows_next_30d = gr.Number(value=100000, label="cash_inflows_next_30d")
        with gr.Row():
            startup_age_months = gr.Number(value=24, label="startup_age_months")
            burn_variability_index = gr.Number(value=0.3, label="burn_variability_index")
            cash_utilization_rate = gr.Number(value=0.75, label="cash_utilization_rate")
        with gr.Row():
            sector = gr.Textbox(value="fintech", label="sector")
            investment_style = gr.Textbox(value="equity", label="investment_style")
            compliance_flag = gr.Dropdown(choices=["yes","no"], value="yes", label="compliance_flag")
            has_funding_round = gr.Dropdown(choices=["yes","no"], value="no", label="has_funding_round")
        btn = gr.Button("Predict")
        out = gr.JSON()
        btn.click(fn=predict_single, inputs=[
            current_bank_balance, monthly_expense, monthly_revenue,
            runway_months, recurring_obligations, cash_inflows_next_30d,
            startup_age_months, burn_variability_index, cash_utilization_rate,
            sector, investment_style, compliance_flag, has_funding_round
        ], outputs=out)

    with gr.Tab("CSV Batch"):
        csv_in = gr.File(label="Upload CSV (must contain all required columns)")
        csv_btn = gr.Button("Run CSV")
        csv_out = gr.Dataframe()
        csv_btn.click(fn=predict_csv, inputs=csv_in, outputs=csv_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
