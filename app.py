# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, precision_recall_curve, auc, average_precision_score
)
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide", page_title="Gym Workout Health Risk Warning System")

# --- Load model + scaler ---
MODEL_FILE = "best_model.pkl"
SCALER_FILE = "scaler.save"
DATA_FILE = "fitbit_processed.csv"  # used for evaluation only

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# ----- Sidebar / Inputs -----
st.title("🏋️ Gym Workout Health Risk Warning System")

with st.sidebar:
    st.header("Demo controls")
    show_eval = st.checkbox("Show model evaluation (confusion matrix, ROC/PR)", value=False)
    st.write("Tip: evaluation runs on the test split of fitbit_processed.csv")

# ----- Main interactive input -----
col1, col2 = st.columns([2, 1])
with col1:
    heart_rate = st.slider("Heart Rate (bpm)", 60, 200, 120)
    steps = st.slider("Steps per Hour", 0, 15000, 5000)
    calories = st.slider("Calories Burned (per hour)", 0, 2000, 400)
with col2:
    st.markdown("### Quick info")
    st.write("Model file:", MODEL_FILE)
    st.write("Scaler file:", SCALER_FILE)

# Prepare input for model
raw_feat = np.array([[heart_rate, steps, calories]])
scaled_feat = scaler.transform(raw_feat)
pred = model.predict(scaled_feat)[0]

# Rule-based check (same rule used in preprocessing)
rule_risk = (heart_rate > 160) or (steps > 8000) or (calories > 900)

# Results display
st.write("---")
st.markdown("### Live Prediction")
c1, c2 = st.columns(2)
with c1:
    st.subheader("ML Model")
    if pred == 1:
        st.error("⚠️ Risk (Model)")
    else:
        st.success("✅ Safe (Model)")
with c2:
    st.subheader("Rule-based")
    if rule_risk:
        st.error("⚠️ Risk (Rule)")
    else:
        st.success("✅ Safe (Rule)")

if pred == 1 or rule_risk:
    st.error("⚠️ Final Decision: Risky Workout Detected!")
else:
    st.success("✅ Final Decision: Safe Workout")

# ------------------------
# Evaluation panel (optional)
# ------------------------
if show_eval:
    st.write("---")
    st.header("Model Evaluation (test split)")

    # Load processed dataset and build test split
    df = pd.read_csv(DATA_FILE)
    X = df[['heart_rate','steps','calories']].values
    y = df['risk'].values

    # Use stratified split where possible
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

    # Predictions & probabilities/scores
    y_pred = model.predict(X_test)
    try:
        y_score = model.predict_proba(X_test)[:, 1]
    except Exception:
        try:
            y_score = model.decision_function(X_test)
        except Exception:
            y_score = y_pred  # fallback

    # Classification report
    st.subheader("Classification report")
    report_text = classification_report(y_test, y_pred, zero_division=0)
    st.text(report_text)

    # Confusion matrices
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig1, ax1 = plt.subplots(figsize=(4,3))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Safe(0)','Risk(1)'])
    disp.plot(ax=ax1, values_format='d')
    ax1.set_title("Confusion Matrix (counts)")
    st.pyplot(fig1)
    plt.close(fig1)

    cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
    fig2, ax2 = plt.subplots(figsize=(4,3))
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['Safe(0)','Risk(1)'])
    disp2.plot(ax=ax2, cmap="Blues")
    ax2.set_title("Confusion Matrix (normalized by true class)")
    st.pyplot(fig2)
    plt.close(fig2)

    # ROC & PR curves
    st.subheader("ROC & Precision-Recall Curves")
    # ROC
    try:
        roc_auc = roc_auc_score(y_test, y_score) if len(np.unique(y_test)) > 1 else float('nan')
        fpr, tpr, _ = roc_curve(y_test, y_score)
        fig3, ax3 = plt.subplots(figsize=(5,3))
        ax3.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        ax3.plot([0,1],[0,1],'--', color='gray')
        ax3.set_xlabel("False Positive Rate")
        ax3.set_ylabel("True Positive Rate")
        ax3.set_title("ROC Curve")
        ax3.legend(loc="lower right")
        ax3.grid(alpha=0.3)
        st.pyplot(fig3)
        plt.close(fig3)
    except Exception as e:
        st.write("ROC could not be computed:", e)

    # PR curve
    try:
        prec, recall, _ = precision_recall_curve(y_test, y_score)
        pr_auc = auc(recall, prec)
        avg_prec = average_precision_score(y_test, y_score) if len(np.unique(y_test)) > 1 else float('nan')
        fig4, ax4 = plt.subplots(figsize=(5,3))
        ax4.plot(recall, prec, label=f'PR (AUC = {pr_auc:.3f}, AP = {avg_prec:.3f})')
        ax4.set_xlabel("Recall")
        ax4.set_ylabel("Precision")
        ax4.set_title("Precision-Recall Curve")
        ax4.legend(loc='lower left')
        ax4.grid(alpha=0.3)
        st.pyplot(fig4)
        plt.close(fig4)
    except Exception as e:
        st.write("PR curve could not be computed:", e)

    st.info("Note: if your dataset had very few real positives and you used upsampling, evaluation may appear overly optimistic. Mention this in your report.")
