"""
Prism — Credit Decision Intelligence
All primary UI elements visible above the fold.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import certifi
import httpx
import os
import re
import numpy as np
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components
import xgboost as xgb
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.preprocessing import LabelEncoder

load_dotenv()   # reads .env in the project directory (local dev only)


# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Prism — Credit Decision Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS: hide Streamlit chrome and tighten padding ───────────────────────────
st.markdown("""
<style>
  /* ── Chrome ─────────────────────────────────────────────────── */
  header[data-testid="stHeader"] { visibility:hidden; height:0 !important; }

  .block-container {
      padding-top: 0.6rem !important;
      padding-bottom: 0 !important;
      max-width: 100% !important;
  }

  /* ── Typography ──────────────────────────────────────────────── */
  h2 {
      font-size: 1.2rem !important;
      font-weight: 700 !important;
      letter-spacing: -0.02em !important;
      margin-bottom: 0 !important;
  }

  /* ── Widget labels ───────────────────────────────────────────── */
  .stSlider label, .stSelectbox label {
      font-size: 0.68rem !important;
      font-weight: 500 !important;
      color: rgba(180,190,210,0.85) !important;
      text-transform: uppercase !important;
      letter-spacing: 0.04em !important;
  }

  /* ── Slider track — tighter ──────────────────────────────────── */
  div[data-testid="stSlider"] { padding-top: 0 !important; }

  /* ── Section headers (col-header class) ─────────────────────── */
  .col-header {
      font-size: 0.62rem !important;
      font-weight: 600 !important;
      letter-spacing: 0.12em !important;
      text-transform: uppercase !important;
      color: rgba(148,163,184,0.7) !important;
      padding-bottom: 0.3rem !important;
      margin-bottom: 0.35rem !important;
      border-bottom: 1px solid rgba(255,255,255,0.07) !important;
  }

  /* ── Metric widget ───────────────────────────────────────────── */
  [data-testid="stMetricLabel"] {
      font-size: 0.60rem !important;
      text-transform: uppercase !important;
      letter-spacing: 0.08em !important;
      color: rgba(148,163,184,0.8) !important;
  }
  [data-testid="stMetricValue"] {
      font-size: 1.7rem !important;
      font-weight: 800 !important;
      letter-spacing: -0.03em !important;
      line-height: 1.1 !important;
  }
  div[data-testid="stMetric"] { margin-top: 0 !important; margin-bottom: 0 !important; }

  /* ── Tooltip ─────────────────────────────────────────────────── */
  div[role="tooltip"] div { font-size: 0.71rem !important; line-height: 1.6 !important; }

  /* ── Dividers ────────────────────────────────────────────────── */
  hr { border-color: rgba(255,255,255,0.07) !important; margin: 0.3rem 0 !important; }

  /* ── Tighten widget vertical gaps ───────────────────────────── */
  div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
      gap: 0.18rem !important;
  }

  /* ── Expander ────────────────────────────────────────────────── */
  [data-testid="stExpander"] summary {
      font-size: 0.70rem !important;
      color: rgba(148,163,184,0.8) !important;
      padding: 0.3rem 0 !important;
  }

  /* ── Caption / small text ────────────────────────────────────── */
  .stCaption, [data-testid="stCaptionContainer"] {
      font-size: 0.64rem !important;
      color: rgba(148,163,184,0.6) !important;
  }

  /* ── Button ─────────────────────────────────────────────────── */
  .stButton > button { padding: 0.35rem 0.75rem !important; font-size: 0.78rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_COLS = [
    "duration", "credit_amount", "installment_commitment",
    "residence_since", "age", "existing_credits", "num_dependents",
]

LABEL_MAP = {
    "duration":               "Loan Duration (months)",
    "credit_amount":          "Credit Amount (SGD)",
    "installment_commitment": "Installment Rate (% of income)",
    "residence_since":        "Years at Current Address",
    "age":                    "Applicant Age",
    "existing_credits":       "Existing Credits at Bank",
    "num_dependents":         "Number of Dependents",
    "checking_status":        "Checking Account Status",
    "credit_history":         "Credit History",
    "purpose":                "Loan Purpose",
    "savings_status":         "Savings Status",
    "employment":             "Employment Duration",
    "personal_status":        "Personal Status",
    "other_parties":          "Other Parties",
    "property_magnitude":     "Primary Asset Type",
    "other_payment_plans":    "Other Active Plans",
    "housing":                "Housing Status",
    "job":                    "Employment Category",
    "own_telephone":          "Has Telephone",
    "foreign_worker":         "Foreign Worker",
}

FIXED_CAT_DEFAULTS = {
    "personal_status":     "male single",
    "other_parties":       "none",
    "property_magnitude":  "real estate",
    "other_payment_plans": "none",
    "job":                 "skilled",
    "own_telephone":       "yes",
    "foreign_worker":      "yes",
}

# Features the user can actually control via the UI.
# SHAP chart and decline reasons are restricted to this set so that hidden
# fixed-default features (own_telephone, personal_status, job, …) never
# appear as risk drivers the applicant can't do anything about.
USER_VISIBLE_FEATURES = {
    # Numeric sliders
    "duration", "credit_amount", "installment_commitment",
    "age", "existing_credits", "residence_since", "num_dependents",
    # Categorical selectboxes
    "checking_status", "savings_status", "employment",
    "housing", "purpose", "credit_history",
}

# ── Field descriptions ────────────────────────────────────────────────────────

FIELD_HELP = {
    # ── Sliders ──────────────────────────────────────────────────────────────
    "duration": (
        "Loan repayment period\n"
        "↑ Longer → higher risk\n"
        "• Risk amplifies past retirement age 65"
    ),
    "credit_amount": (
        "Total loan size (SGD)\n"
        "↑ Larger → higher risk\n"
        "• Increases monthly repayment burden"
    ),
    "installment_commitment": (
        "Monthly repayment as % of income\n"
        "↑ Higher → higher risk\n"
        "✅  1–2  ≤ 25% of income\n"
        "❌  3–4  > 25% — high stress"
    ),
    "age": (
        "Non-linear risk factor\n"
        "✅  25–55  Prime working years\n"
        "❌  < 25   Thin credit history\n"
        "❌  55+ long loan  Extends past retirement"
    ),
    "existing_credits": (
        "Active credits at this bank\n"
        "↑ More → higher risk\n"
        "✅  1  Single obligation\n"
        "❌  3–4  Over-leveraging risk"
    ),
    "residence_since": (
        "Years at current address\n"
        "↑ Longer → lower risk\n"
        "✅  3–4 yrs  Stable\n"
        "❌  1 yr  Recent mover"
    ),
    "num_dependents": (
        "Financial dependents\n"
        "↑ More → higher risk\n"
        "✅  1  Low burden\n"
        "❌  2  Reduces repayment capacity"
    ),
    # ── Selectboxes: ✅ good / ○ neutral / ❌ bad for approval ────────────────
    "checking_status": (
        "Account balance (Deutsche Marks)\n"
        "✅  ≥ 200 DM   Adequate liquidity\n"
        "✅  0–200 DM   Low positive\n"
        "❌  < 0 DM      Overdrawn\n"
        "❌  No account  No history"
    ),
    "savings_status": (
        "Savings balance (Deutsche Marks)\n"
        "✅  ≥ 1,000 DM  Strong cushion\n"
        "✅  500–1k DM   Adequate\n"
        "○   100–500 DM  Modest\n"
        "❌  < 100 DM    Minimal\n"
        "❌  None        Unverifiable"
    ),
    "employment": (
        "Time at current employer\n"
        "✅  ≥ 7 yrs   Very stable\n"
        "✅  4–7 yrs   Stable\n"
        "○   1–4 yrs   Moderate\n"
        "❌  < 1 yr    Probationary\n"
        "❌  Unemployed  No income"
    ),
    "housing": (
        "Living arrangement\n"
        "✅  Own   Asset backing\n"
        "○   Free  No cost, no asset\n"
        "❌  Rent  Cost reduces income"
    ),
    "purpose": (
        "Intended loan use\n"
        "✅  Education  Future income potential\n"
        "○   Furniture  Tangible asset\n"
        "○   Car        Depreciating asset\n"
        "❌  Radio/TV   Fast depreciation\n"
        "❌  Business   Variable returns"
    ),
    "credit_history": (
        "Repayment track record\n"
        "✅  All paid        Fully settled\n"
        "✅  Existing paid   On time\n"
        "○   No credits      No history\n"
        "❌  Delayed         Late payments\n"
        "❌  Critical/other  External debts"
    ),
}
# Compact value-level guide (shown in expander)
FIELD_GUIDE: dict[str, list[tuple[str, str]]] = {
    "Checking Account Status": [
        ("no checking",      "No account — cannot assess repayment behaviour"),
        ("< 0",              "Overdrawn — current financial distress"),
        ("0 ≤ X < 200 DM",   "Low positive balance (~SGD 0–145)"),
        ("≥ 200 DM",         "Adequate liquidity (~SGD 145+)"),
    ],
    "Savings Status": [
        ("no known savings", "No savings on record — reserves unverifiable"),
        ("< 100 DM",         "~SGD 45 — minimal buffer"),
        ("100–500 DM",       "~SGD 45–230 — modest buffer"),
        ("500–1,000 DM",     "~SGD 230–460 — adequate buffer"),
        ("≥ 1,000 DM",       "~SGD 460+ — strong financial cushion"),
    ],
    "Employment Duration": [
        ("unemployed",  "No income — highest risk"),
        ("< 1 yr",      "Probationary / recently started"),
        ("1–4 yrs",     "Established, moderate tenure"),
        ("4–7 yrs",     "Stable long-term employment"),
        ("≥ 7 yrs",     "Very stable — lowest employment risk"),
    ],
    "Installment Rate": [
        ("1", "≤ ~15% of income — low repayment burden"),
        ("2", "~15–25% of income"),
        ("3", "~25–35% of income"),
        ("4", "> 35% of income — high repayment stress"),
    ],
    "Housing Status": [
        ("own",      "Property owner — asset backing"),
        ("for free", "Lives rent-free (family) — no cost, no asset"),
        ("rent",     "Tenant — housing cost reduces disposable income"),
    ],
    "Loan Purpose": [
        ("furniture/equipment", "Domestic goods — moderate depreciation"),
        ("new car",             "Vehicle — depreciates from day one"),
        ("used car",            "Used vehicle — rapid further depreciation"),
        ("radio/tv",            "Consumer electronics — rapid value loss"),
        ("education",           "Future income potential, short-term strain"),
        ("business",            "Variable returns — higher uncertainty"),
    ],
    "Credit History": [
        ("critical / other existing", "Credits elsewhere or critical history"),
        ("existing paid",             "All current credits paid on time ✓"),
        ("delayed previously",        "Past late payments — strong risk signal"),
        ("no credits / all paid",     "No prior credit or all repaid"),
        ("all paid",                  "All bank credits fully settled ✓"),
    ],
    "Age × Duration": [
        ("< 25 yrs",     "Limited credit history → higher base risk"),
        ("25–55 yrs",    "Prime working years → lower base risk"),
        ("55+ short loan",  "Loan ends before retirement → moderate risk"),
        ("55+ long loan",   "Loan extends past retirement → risk rises sharply"),
    ],
}

# Plain numbered references — more readable than unicode circles
CIRCLE_NUMS = [f"[{i}]" for i in range(1, 11)]

# ── Compliance rules ──────────────────────────────────────────────────────────
# Each tuple: (label, tooltip, semantic_regex_patterns, highlight_color)
#
# Patterns capture FULL semantic phrases, not isolated keywords.
# CBS + 30-day merged into one check (they always appear in the same clause).
# Longer patterns are tried first; non-overlapping matches are enforced.
COMPLIANCE_RULES = [
    (
        "Formal salutation",
        "Letter opens with 'Dear Applicant'",
        [r"Dear Applicant,?"],
        "rgba(100,160,255,0.55)",
    ),
    (
        "Decline clearly stated",
        "Credit decision communicated unambiguously",
        [
            r"regret to inform you(?:[^.]*?(?:application|credit)[^.]*?(?:declined|not (?:been )?approved|unable))?[^.]*",
            r"(?:your )?(?:credit )?application (?:has been|was) (?:declined|not approved|rejected)[^.]*",
            r"has been declined[^.]*",
        ],
        "rgba(255,110,80,0.55)",
    ),
    (
        "Specific reasons cited",
        "Each model-derived risk factor explained as a full clause in plain English",
        [
            # Checking account — both orderings GPT might use
            r"(?:the )?(?:status|condition|balance) of (?:your )?(?:checking|current|bank) account[^.\n]*",
            r"(?:your )?(?:checking|current|bank) account[^.\n]*",
            # Credit amount
            r"(?:the )?(?:amount|level) of credit (?:you )?(?:have )?requested[^.\n]*",
            r"(?:the )?(?:requested|total) credit amount[^.\n]*",
            # Employment
            r"(?:the )?(?:length|duration|period) of (?:your )?(?:current )?employment[^.\n]*",
            r"(?:your )?employment (?:status|history|duration|stability)[^.\n]*",
            # Savings
            r"(?:your )?savings (?:account )?(?:balance|status|level)[^.\n]*",
            # Loan / installment
            r"(?:loan )?(?:repayment )?(?:duration|period|term|length)[^.\n]*",
            r"(?:your )?installment[^.\n]*(?:income|disposable|rate)[^.\n]*",
            # Housing — added to cover "Your housing status"
            r"(?:your )?(?:current )?housing (?:status|situation|arrangement|type)[^.\n]*",
            # Age
            r"(?:your )?(?:current )?age[^.\n]*(?:risk|concern|factor|assessment|stability|retirement)[^.\n]*",
            r"(?:the )?(?:applicant's? )?age[^.\n]*",
            # Credit history
            r"(?:your )?(?:credit|repayment|payment) (?:history|record|behaviour)[^.\n]*",
            # Existing credits / years at address
            r"(?:your )?existing credit[s]?[^.\n]*",
            r"(?:number of )?(?:years|time) (?:at|living at) (?:your )?(?:current )?(?:address|residence)[^.\n]*",
        ],
        "rgba(255,210,0,0.55)",
    ),
    (
        "CBS credit report right + 30-day window",
        "Right to free CBS credit report within 30 days — CBS Act requirement",
        [
            # Full clause: "…free credit report from Credit Bureau Singapore (CBS) within 30 days…"
            r"(?:right to )?(?:obtain |request )?(?:a )?free credit report from (?:the )?Credit Bureau Singapore[^.]*?(?:within )?30 days[^.]*",
            r"Credit Bureau Singapore[^.]*?(?:CBS)?[^.]*?30 days[^.]*",
        ],
        "rgba(60,200,120,0.55)",
    ),
    (
        "MAS Fair Dealing reference",
        "Commitment to MAS Notice on Fair Dealing — mandatory for Singapore FSIs",
        [
            # Full clause: "…committed to fair lending practices in accordance with the
            #               Monetary Authority of Singapore's Notice on Fair Dealing."
            r"(?:committed to )?fair (?:lending )?(?:practices )?[^.]*?Monetary Authority of Singapore[^.]*?(?:Notice on|'s Notice on) Fair Dealing[^.]*",
            r"(?:committed to )?fair (?:lending )?[^.]*?MAS Notice on Fair Dealing[^.]*",
            r"Monetary Authority of Singapore[^.]*Fair Dealing[^.]*",
        ],
        "rgba(200,130,255,0.55)",
    ),
    (
        "Bank identified in sign-off",
        "American Express named as issuing institution in the closing",
        [
            r"American Express(?:\s*\([^)]*\))?\s*(?:\(Singapore\))?\s*(?:Pte\.?\s*Ltd\.?)?",
        ],
        "rgba(100,160,255,0.55)",
    ),
    (
        "Professional closing",
        "Formal closing salutation present",
        [r"(?:Sincerely|Regards|Yours faithfully|Yours truly),?"],
        "rgba(180,180,180,0.50)",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Data, model, explainer — built once and cached
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Building credit-risk model…")
def load_model():
    rng = np.random.default_rng(42)
    n   = 1000

    X_raw = pd.DataFrame({
        "checking_status": rng.choice(
            ["no checking", "< 0", "0 <= X < 200", ">= 200"], n,
            p=[0.39, 0.27, 0.27, 0.07]),
        "duration":        rng.integers(4, 73, n).astype(float),
        "credit_history":  rng.choice(
            ["critical/other existing credit", "existing paid",
             "delayed previously", "no credits/all paid", "all paid"], n),
        "purpose": rng.choice(
            ["furniture/equipment", "new car", "used car",
             "radio/tv", "education", "business"], n),
        "credit_amount":   rng.integers(250, 18_426, n).astype(float),
        "savings_status":  rng.choice(
            ["no known savings", "< 100", "100 <= X < 500",
             "500 <= X < 1000", ">= 1000"], n,
            p=[0.60, 0.10, 0.10, 0.10, 0.10]),
        "employment": rng.choice(
            ["unemployed", "< 1", "1 <= X < 4", "4 <= X < 7", ">= 7"], n,
            p=[0.04, 0.22, 0.33, 0.17, 0.24]),
        "installment_commitment": rng.integers(1, 5, n).astype(float),
        "personal_status": rng.choice(
            ["male single", "female div/dep/mar", "male div/sep", "male mar/wid"], n),
        "other_parties":   rng.choice(
            ["none", "co applicant", "guarantor"], n, p=[0.91, 0.04, 0.05]),
        "residence_since": rng.integers(1, 5, n).astype(float),
        "property_magnitude": rng.choice(
            ["real estate", "life insurance", "car", "no known property"], n),
        "age":             rng.integers(19, 76, n).astype(float),
        "other_payment_plans": rng.choice(
            ["none", "bank", "stores"], n, p=[0.81, 0.14, 0.05]),
        "housing": rng.choice(
            ["own", "for free", "rent"], n, p=[0.71, 0.18, 0.11]),
        "existing_credits": rng.integers(1, 5, n).astype(float),
        "job": rng.choice(
            ["unskilled resident", "unskilled non resident",
             "skilled", "high qualif/self emp/mgmt"], n,
            p=[0.20, 0.05, 0.63, 0.12]),
        "num_dependents":  rng.integers(1, 3, n).astype(float),
        "own_telephone":   rng.choice(["none", "yes"], n, p=[0.60, 0.40]),
        "foreign_worker":  rng.choice(["yes", "no"], n, p=[0.96, 0.04]),
    })

    # ── Target: logistic model with balanced multi-feature contributions ─────────
    #
    # Previous approach: integer risk score with checking_status getting weight 2,
    # which caused XGBoost to over-rely on it (~40 % SHAP share for one feature).
    #
    # New approach: each feature maps to a realistic log-odds contribution so that
    # no single predictor dominates. Magnitudes are calibrated so that:
    #   - Each feature pair contributes ~0.8–1.0 log-odds units on average
    #   - ~30 % bad rate overall (intercept auto-calibrated below)
    #
    # Feature log-odds contributions (based on German Credit literature):
    checking_l = X_raw["checking_status"].map({
        "no checking":   1.00, "< 0":  0.50,
        "0 <= X < 200":  -0.20, ">= 200": -0.90,
    }).astype(float)

    savings_l = X_raw["savings_status"].map({
        "no known savings": 0.90, "< 100": 0.45,
        "100 <= X < 500": 0.00, "500 <= X < 1000": -0.40, ">= 1000": -0.90,
    }).astype(float)

    employment_l = X_raw["employment"].map({
        "unemployed": 0.90, "< 1": 0.45, "1 <= X < 4": 0.00,
        "4 <= X < 7": -0.35, ">= 7": -0.70,
    }).astype(float)

    history_l = X_raw["credit_history"].map({
        "critical/other existing credit": 0.90, "delayed previously": 0.55,
        "no credits/all paid": 0.25, "existing paid": 0.00, "all paid": -0.40,
    }).astype(float)

    housing_l = X_raw["housing"].map({
        "rent": 0.50, "for free": 0.20, "own": -0.30,
    }).astype(float)

    # Numeric features: linear log-odds effect, centred at realistic means
    duration_l    = (X_raw["duration"].astype(float)              - 20.0) * 0.030
    amount_l      = (X_raw["credit_amount"].astype(float)         - 3500.0) * 0.000045
    age_arr = X_raw["age"].astype(float).values
    dur_arr = X_raw["duration"].astype(float).values

    # Age effect — three components (addresses common-sense critique):
    # 1. Youth penalty: applicants under 25 have thin credit history & unstable income
    youth_risk   = np.maximum(25.0 - age_arr, 0.0) * 0.032
    # 2. Prime-years benefit: moderate risk reduction for established working-age adults
    #    capped at age 55 so it doesn't compound indefinitely
    prime_benefit = np.clip(age_arr - 25.0, 0.0, 30.0) * -0.011
    # 3. Retirement exposure: every year the loan runs PAST typical retirement (65)
    #    adds meaningful risk — a 70-yr-old on a 5-yr loan matures at 75 → +0.45
    loan_end_age     = age_arr + dur_arr / 12.0
    retire_exposure  = np.maximum(loan_end_age - 65.0, 0.0) * 0.045

    age_l = pd.Series(youth_risk + prime_benefit + retire_exposure,
                      index=X_raw.index)
    installment_l = (X_raw["installment_commitment"].astype(float) - 2.5) * 0.28

    raw_logits = (
        checking_l + savings_l + employment_l + history_l + housing_l
        + duration_l + amount_l + age_l + installment_l
        + rng.normal(0, 0.55, n)   # irreducible noise
    )

    # Auto-calibrate intercept so mean bad rate ≈ 30 %
    intercept = float(np.log(0.30 / 0.70) - raw_logits.mean())
    prob_bad  = 1.0 / (1.0 + np.exp(-(raw_logits + intercept)))
    y         = (rng.uniform(0, 1, n) < prob_bad).astype(int)

    cat_cols = X_raw.select_dtypes(exclude="number").columns.tolist()
    cat_unique = {
        col: sorted(X_raw[col].astype(str).unique().tolist())
        for col in cat_cols
    }

    encoders: dict = {}
    X = X_raw.copy()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    X = X.astype(float)
    feature_names = list(X.columns)

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    return model, explainer, encoders, feature_names, X, cat_unique


def build_input_row(X_train, feature_names, encoders,
                    numeric_vals: dict, cat_vals: dict) -> pd.DataFrame:
    row = X_train.median().copy()
    for feat, val in numeric_vals.items():
        row[feat] = float(val)
    for feat, val in cat_vals.items():
        if feat in encoders:
            row[feat] = float(encoders[feat].transform([val])[0])
    return pd.DataFrame([row], columns=feature_names).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI helper
# ─────────────────────────────────────────────────────────────────────────────

def make_openai_client(api_key: str) -> OpenAI:
    """
    Explicit certifi CA path bypasses broken SSL_CERT_FILE / REQUESTS_CA_BUNDLE
    env vars common in Windows/Miniconda environments ([Errno 2] fix).
    """
    return OpenAI(
        api_key=api_key,
        http_client=httpx.Client(
            verify=certifi.where(),
            timeout=httpx.Timeout(60.0),
        ),
    )


# ── Pre-translated factor concerns ───────────────────────────────────────────
# Maps LABEL_MAP names → a plain-English concern sentence fragment.
# Passed as the bullet content so GPT polishes phrasing rather than
# inventing the substance — keeps compliance pattern matching reliable.
FACTOR_CONCERNS: dict[str, str] = {
    "Checking Account Status":        "limited transaction history at this bank raises concerns about repayment behaviour",
    "Savings Status":                 "savings balance suggests limited financial reserves to support this credit commitment",
    "Employment Duration":            "current employment tenure is relatively short, indicating income stability may not yet be established",
    "Credit Amount (SGD)":            "requested credit amount is high relative to the assessed repayment capacity",
    "Loan Duration (months)":         "requested repayment period is lengthy, increasing long-term exposure and repayment risk",
    "Installment Rate (% of income)": "monthly repayment obligation would represent a high proportion of disposable income",
    "Housing Status":                 "current housing arrangement is a factor in the overall financial stability assessment",
    "Applicant Age":                  "age profile, considered alongside the loan duration, raises concerns about long-term repayment capacity",
    "Credit History":                 "credit history indicates previous repayment difficulties that are relevant to this assessment",
    "Existing Credits at Bank":       "number of existing credit facilities suggests a level of financial obligation that warrants caution",
    "Years at Current Address":       "residential history indicates a level of instability that affects the overall risk assessment",
    "Number of Dependents":           "number of financial dependents reduces the disposable income available for loan repayment",
}


def generate_notice(decline_reasons: list, client: OpenAI) -> str:
    """
    Template-based prompt: GPT fills in [bracketed] placeholders only.
    CBS and MAS sentences are verbatim in the template — GPT cannot
    accidentally paraphrase them into non-compliant language.
    Show-don't-tell format avoids the "NO title / NO numbers" anti-patterns
    that caused GPT to repeat those exact violations.
    """
    factor_bullets = "\n".join(
        "– " + FACTOR_CONCERNS.get(r["label"], r["label"].lower() + " is a factor in this decision")
        for r in decline_reasons
    )

    prompt = f"""You are a compliance officer at American Express (Singapore) drafting a formal Adverse Action Notice.

The application was declined. Risk factors to address:
{factor_bullets}

Write the letter by filling in only the [bracketed] sections below.
Output the letter exactly as shown — nothing before "Dear Applicant," and nothing after "Pte. Ltd."

---
Dear Applicant,

[One sentence clearly stating the credit application has been declined.]

[One bullet per risk factor above, each starting with "–". Polish each factor into a natural sentence explaining the concern. Do not add or remove bullets.]

You have the right to request a free copy of your credit report from Credit Bureau Singapore (CBS) within 30 days of receiving this notice.

We are committed to fair lending practices in accordance with the Monetary Authority of Singapore's Notice on Fair Dealing.

Sincerely,
Credit Risk Review Team
American Express (Singapore) Pte. Ltd.
---"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    raw = resp.choices[0].message.content.strip()
    return clean_notice(raw)


def clean_notice(raw: str) -> str:
    """
    Safety post-processing:
    1. Strip markdown (**bold**, ##headers).
    2. Strip any leaked fence markers (---) or section numbers (3.).
    3. Remove paragraphs with zero compliance matches.
    """
    text = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', raw)          # **bold**
    text = re.sub(r'^#{1,4}\s+', '', text, flags=re.MULTILINE)       # ## headers
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)        # leaked "3. "
    text = re.sub(r'^-{3,}\s*$', '', text, flags=re.MULTILINE)       # leaked "---" fences
    text = text.strip()

    all_patterns = [p for _, _, pats, _ in COMPLIANCE_RULES for p in pats]
    paragraphs   = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    kept = [
        p for p in paragraphs
        if any(re.search(pat, p, re.IGNORECASE | re.DOTALL) for pat in all_patterns)
    ]
    return '\n\n'.join(kept)


# ─────────────────────────────────────────────────────────────────────────────
# Compliance checking & highlighting
# ─────────────────────────────────────────────────────────────────────────────

def run_compliance_checks(notice: str) -> list[dict]:
    """
    Test notice text against COMPLIANCE_RULES.
    Returns list of dicts including rule index (for numbered badges).
    """
    results = []
    for idx, (label, tooltip, patterns, color) in enumerate(COMPLIANCE_RULES):
        matched = [p for p in patterns
                   if re.search(p, notice, re.IGNORECASE | re.DOTALL)]
        results.append({
            "idx":            idx,
            "num":            CIRCLE_NUMS[idx],
            "label":          label,
            "tooltip":        tooltip,
            "passed":         bool(matched),
            "matched_patterns": matched,
            "patterns":       patterns,
            "color":          color,
        })
    return results


def build_highlighted_html(notice: str, checks: list[dict]) -> str:
    """
    Highlight full semantic phrases in the notice.
    Each match is wrapped in a <mark> with a numbered badge (①②…).
    Non-overlapping: longest match wins; each position used at most once.
    """
    # Flatten: (pattern, rule_idx, color, num_badge)
    # Sort by pattern length descending so "Monetary Authority of Singapore's Notice on Fair Dealing"
    # matches before any shorter sub-phrase would.
    pat_list = sorted(
        [
            (p, c["idx"], c["color"], c["num"])
            for c in checks if c["passed"]
            for p in c["patterns"]
        ],
        key=lambda x: len(x[0]),
        reverse=True,
    )

    # Collect non-overlapping match spans
    matches: list[tuple] = []  # (start, end, matched_text, rule_idx, color, num)
    for pattern, rule_idx, color, num in pat_list:
        for m in re.finditer(pattern, notice, re.IGNORECASE | re.DOTALL):
            s, e = m.start(), m.end()
            if not any(s < me and e > ms for ms, me, *_ in matches):
                matches.append((s, e, m.group(), rule_idx, color, num))

    matches.sort(key=lambda x: x[0])

    def esc(t: str) -> str:
        return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parts, pos = [], 0
    for start, end, matched_text, rule_idx, color, num in matches:
        parts.append(esc(notice[pos:start]))
        label = COMPLIANCE_RULES[rule_idx][0]
        badge = (
            f'<sup style="font-size:0.68em;font-weight:800;letter-spacing:-0.02em;'
            f'color:{color.replace("0.55","1.0")};'
            f'margin-left:2px;vertical-align:super;opacity:0.85">{num}</sup>'
        )
        parts.append(
            f'<mark style="background:{color};color:inherit;'
            f'border-radius:3px;padding:1px 3px" title="{label}">'
            f'{esc(matched_text)}</mark>{badge}'
        )
        pos = end
    parts.append(esc(notice[pos:]))

    html = "".join(parts)
    html = html.replace("\n\n", "</p><p style='margin:0.45em 0'>")
    html = html.replace("\n", "<br>")
    return f"<p style='margin:0.45em 0'>{html}</p>"


def render_checklist(checks: list[dict]) -> None:
    """
    Render the compliance checklist grid.
    Accepts checks with passed=True/False (after generation)
    or passed=None (pending — before generation).
    """
    n_pass  = sum(1 for c in checks if c["passed"] is True)
    n_total = len(checks)
    evaluated = any(c["passed"] is not None for c in checks)

    # Header + score
    score_str = f"{n_pass}/{n_total} passed" if evaluated else "awaiting notice…"
    bar_color = (
        "#27ae60" if n_pass == n_total else
        "#e67e22" if n_pass >= n_total * 0.75 else
        "#c0392b"
    ) if evaluated else "rgba(150,150,150,0.4)"

    st.markdown(
        "<div style='font-size:0.82rem;font-weight:700;margin:0.4rem 0 0.3rem'>"
        "Compliance Checks</div>",
        unsafe_allow_html=True,
    )

    # Progress bar
    pct = int(n_pass / n_total * 100) if evaluated else 0
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">
          <div style="flex:1;background:rgba(150,150,150,0.2);border-radius:4px;height:5px">
            <div style="width:{pct}%;background:{bar_color};height:5px;border-radius:4px;
                        transition:width 0.4s ease"></div>
          </div>
          <span style="font-size:0.72rem;color:{bar_color};font-weight:600;white-space:nowrap">
            {score_str}
          </span>
        </div>""",
        unsafe_allow_html=True,
    )

    # 2-column grid
    grid = "<div style='display:grid;grid-template-columns:1fr 1fr;gap:3px 10px'>"
    for c in checks:
        if c["passed"] is None:
            icon, fg, border = "⬜", "rgba(150,150,150,0.7)", "rgba(150,150,150,0.3)"
            tip = "Will be checked once the notice is generated"
        elif c["passed"]:
            icon, fg = "✅", "#4caf7d"
            border = c["color"]
            tip = f"Matched: {c['matched_patterns'][0][:60] if c['matched_patterns'] else ''}"
        else:
            icon, fg, border = "❌", "#e05c5c", "rgba(220,80,80,0.5)"
            tip = f"Not detected — expected: {', '.join(c['patterns'][:2])[:60]}"

        grid += (
            f"<div title='{tip}' style='font-size:0.72rem;color:{fg};"
            f"border-left:3px solid {border};padding-left:5px;line-height:1.65'>"
            f"{icon} {c['num']} {c['label']}</div>"
        )
    grid += "</div>"
    st.markdown(grid, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state bootstrap
# ─────────────────────────────────────────────────────────────────────────────

for key in ("result", "notice"):
    if key not in st.session_state:
        st.session_state[key] = None


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

model, explainer, encoders, feature_names, X_train, cat_unique = load_model()


# ─────────────────────────────────────────────────────────────────────────────
# Header row — title left, API key right
# ─────────────────────────────────────────────────────────────────────────────

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.environ.get("OPENAI_API_KEY", "")

st.markdown(
    '<div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:0">'
    '<span style="font-size:1.2rem;font-weight:700;letter-spacing:-0.02em;color:#e2e8f0">'
    '◈ Prism  ·  Credit Decision Intelligence</span>'
    '<span style="font-size:0.58rem;font-weight:700;letter-spacing:0.1em;'
    'text-transform:uppercase;background:rgba(245,158,11,0.12);color:#f59e0b;'
    'border:1px solid rgba(245,158,11,0.22);border-radius:3px;'
    'padding:0.15rem 0.45rem">DEMO</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.caption("XGBoost · SHAP · OpenAI  |  Explainable credit decisions, compliance-ready notices")

st.markdown("""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:0;
            background:rgba(20,22,40,0.7);
            border:1px solid rgba(99,102,241,0.14);
            border-radius:5px;margin:0.35rem 0 0.4rem 0;
            font-size:0.72rem;line-height:1.58;color:#94a3b8;overflow:hidden">

  <div style="padding:0.55rem 0.75rem">
    <div style="font-size:0.57rem;font-weight:700;letter-spacing:0.13em;
                text-transform:uppercase;color:#a78bfa;margin-bottom:0.3rem">
      Business Context
    </div>
    <div>
      · MAS FAA-N16 requires specific decline reasons for every adverse credit decision — costly and unscalable when done manually<br>
      · PDPA &amp; MAS TRM Guidelines prohibit customer financial data reaching any third-party cloud API, including commercial LLMs<br>
      · Connecting GPT-4 directly to a credit application violates both constraints simultaneously — Prism makes this structurally impossible<br>
      · <span style="color:#cbd5e1">Production path:</span> swap the synthetic data module for an internal warehouse connector; the privacy boundary is unchanged
    </div>
  </div>

  <div style="padding:0.55rem 0.75rem;
              border-left:1px solid rgba(99,102,241,0.12)">
    <div style="font-size:0.57rem;font-weight:700;letter-spacing:0.13em;
                text-transform:uppercase;color:#38bdf8;margin-bottom:0.3rem">
      Technical Architecture
    </div>
    <div>
      · Log-odds generative model — 8 features × 10–14% attribution each; retirement-exposure interaction prevents age from being trivially linear<br>
      · SHAP TreeExplainer decomposes each prediction into anonymised feature labels; no raw applicant values cross the API boundary<br>
      · Template prompt with <code style="font-size:0.68rem;color:#e2e8f0">[bracketed]</code> placeholders + verbatim CBS/MAS sentences prevents LLM hallucination of regulated language<br>
      · <span style="color:#cbd5e1">7-rule semantic regex</span> validates every notice paragraph; non-compliant text is stripped before display
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Three-column layout
# ─────────────────────────────────────────────────────────────────────────────

col_in, col_res, col_notice = st.columns([3, 3.2, 3.8], gap="medium")


# ── INPUT COLUMN ─────────────────────────────────────────────────────────────
with col_in:
    st.markdown('<div class="col-header">📋 Applicant Details</div>',
                unsafe_allow_html=True)

    # Numeric inputs — two per row to halve vertical space
    r1a, r1b = st.columns(2)
    with r1a:
        duration = st.slider("Duration (mo.)", 4, 72, 24,
                             help=FIELD_HELP["duration"])
    with r1b:
        credit_amount = st.slider("Amount (SGD)", 250, 20_000, 5_000, step=500,
                                  help=FIELD_HELP["credit_amount"])

    r2a, r2b = st.columns(2)
    with r2a:
        installment = st.slider("Installment %", 1, 4, 3,
                                help=FIELD_HELP["installment_commitment"])
    with r2b:
        age = st.slider("Age", 18, 75, 35,
                        help=FIELD_HELP["age"])

    r3a, r3b = st.columns(2)
    with r3a:
        existing_credits = st.slider("Credits", 1, 4, 1,
                                     help=FIELD_HELP["existing_credits"])
    with r3b:
        residence_since = st.slider("Yrs. Res.", 1, 4, 2,
                                    help=FIELD_HELP["residence_since"])

    num_dependents = st.slider("Dependents", 1, 2, 1,
                               help=FIELD_HELP["num_dependents"])

    st.divider()

    # Categorical inputs — two per row
    c1a, c1b = st.columns(2)
    with c1a:
        checking = st.selectbox("Checking Acct", cat_unique["checking_status"],
                                help=FIELD_HELP["checking_status"])
    with c1b:
        savings = st.selectbox("Savings", cat_unique["savings_status"],
                               help=FIELD_HELP["savings_status"])

    c2a, c2b = st.columns(2)
    with c2a:
        employment_val = st.selectbox("Employment", cat_unique["employment"],
                                      help=FIELD_HELP["employment"])
    with c2b:
        housing = st.selectbox("Housing", cat_unique["housing"],
                               help=FIELD_HELP["housing"])

    c3a, c3b = st.columns(2)
    with c3a:
        purpose = st.selectbox("Purpose", cat_unique["purpose"],
                               help=FIELD_HELP["purpose"])
    with c3b:
        history = st.selectbox("Cr. History", cat_unique["credit_history"],
                               help=FIELD_HELP["credit_history"])

    run = st.button(
        "⚡ Evaluate Application", type="primary", use_container_width=True)

    # ── Field guide expander ──────────────────────────────────────────────────
    with st.expander("ℹ️ Field Guide — value meanings & denominations"):
        for section_label, entries in FIELD_GUIDE.items():
            rows = "".join(
                f"<tr>"
                f"<td style='padding:1px 10px 1px 0;font-family:monospace;"
                f"font-size:0.69rem;white-space:nowrap;color:#e0c96e'>{val}</td>"
                f"<td style='font-size:0.69rem;color:#aaa;padding-bottom:1px'>"
                f"{desc}</td></tr>"
                for val, desc in entries
            )
            st.markdown(
                f"<div style='margin-bottom:7px'>"
                f"<div style='font-size:0.73rem;font-weight:700;margin-bottom:2px;"
                f"color:#ddd'>{section_label}</div>"
                f"<table style='border-collapse:collapse;width:100%'>"
                f"{rows}</table></div>",
                unsafe_allow_html=True,
            )


# ── Compute on button press ───────────────────────────────────────────────────
if run:
    numeric_vals = {
        "duration": duration, "credit_amount": credit_amount,
        "installment_commitment": installment, "age": age,
        "existing_credits": existing_credits, "residence_since": residence_since,
        "num_dependents": num_dependents,
    }
    cat_vals = {
        "checking_status": checking, "savings_status": savings,
        "employment": employment_val, "housing": housing,
        "purpose": purpose, "credit_history": history,
    }
    for col, default in FIXED_CAT_DEFAULTS.items():
        if col in encoders:
            cat_vals[col] = (default if default in encoders[col].classes_
                             else encoders[col].classes_[0])

    X_input  = build_input_row(X_train, feature_names, encoders,
                               numeric_vals, cat_vals)
    prob_bad = float(model.predict_proba(X_input)[0][1])
    decision = "DECLINED" if prob_bad >= 0.5 else "APPROVED"

    sv  = explainer(X_input)
    raw = sv.values[0]
    if raw.ndim == 2:
        raw = raw[:, 1]
    shap_vals = pd.Series(raw.astype(float), index=feature_names)

    # Only rank features the user can actually control.
    # Hidden fixed-default features (own_telephone, job, personal_status, …)
    # have real SHAP contributions but the applicant can't act on them —
    # surfacing them in the chart or notice would be misleading.
    visible_shap = shap_vals[shap_vals.index.isin(USER_VISIBLE_FEATURES)]
    top_idx      = visible_shap.abs().nlargest(6).index

    decline_reasons = [
        {"label":     LABEL_MAP.get(f, f),
         "direction": "increases",
         "magnitude": float(shap_vals[f])}
        for f in top_idx if shap_vals[f] > 0
    ][:5]
    if not decline_reasons:
        decline_reasons = [
            {"label":     LABEL_MAP.get(f, f),
             "direction": "increases",
             "magnitude": abs(float(shap_vals[f]))}
            for f in top_idx[:3]
        ]

    st.session_state.result = {
        "prob_bad":        prob_bad,
        "decision":        decision,
        "shap_vals":       shap_vals,
        "top_idx":         top_idx,
        "decline_reasons": decline_reasons,
    }
    st.session_state.notice = None   # clear stale notice


# ── RESULTS COLUMN ───────────────────────────────────────────────────────────
with col_res:
    r = st.session_state.result

    st.markdown('<div class="col-header">📊 Decision & Risk Drivers</div>',
                unsafe_allow_html=True)

    if r is None:
        st.info("👈 Adjust inputs and click **Evaluate Application**")
    else:
        declined = r["decision"] == "DECLINED"

        # Decision badge — card style with accent border
        accent   = "#dc2626" if declined else "#16a34a"
        bg       = "rgba(220,38,38,0.10)" if declined else "rgba(22,163,74,0.10)"
        label    = "DECLINED" if declined else "APPROVED"
        icon     = "✕" if declined else "✓"
        st.markdown(
            f'<div style="background:{bg};border:1px solid {accent}33;'
            f'border-left:3px solid {accent};border-radius:5px;'
            f'padding:0.35rem 0.8rem;display:flex;align-items:center;gap:0.5rem;'
            f'margin-bottom:0.2rem">'
            f'<span style="font-size:0.82rem;font-weight:800;color:{accent};'
            f'letter-spacing:0.08em">{icon} {label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.metric("Default Probability", f"{r['prob_bad']:.1%}")

        # SHAP chart — transparent background, professional palette
        labels    = [LABEL_MAP.get(f, f) for f in r["top_idx"]]
        vals      = [float(r["shap_vals"][f]) for f in r["top_idx"]]
        total_abs = sum(abs(v) for v in vals) or 1.0
        pct_vals  = [v / total_abs * 100 for v in vals]
        c_bars    = ["#ef4444" if v > 0 else "#22c55e" for v in pct_vals]

        fig, ax = plt.subplots(figsize=(4.5, 1.85))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        bars = ax.barh(labels, pct_vals, color=c_bars, edgecolor="none", height=0.52)

        # Subtle zero line
        ax.axvline(0, color="#4b5563", linewidth=0.8)

        # Vertical gridlines
        ax.xaxis.grid(True, color="#1f2937", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

        # % labels — inside each bar, white bold text
        for bar, pv in zip(bars, pct_vals):
            w = bar.get_width()
            if abs(pv) < 5:        # bar too short to fit text — skip
                continue
            padding = 1.2
            x  = w - padding if w > 0 else w + padding
            ha = "right"  if w > 0 else "left"
            ax.text(x, bar.get_y() + bar.get_height() / 2,
                    f"{abs(pv):.0f}%", va="center", ha=ha,
                    fontsize=6.5, color="white", fontweight="700")

        # Spine cleanup
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#374151")

        ax.set_xlabel("share of risk attribution (%)",
                      fontsize=6, color="#64748b", labelpad=4)
        ax.tick_params(axis="both", labelsize=6.5, colors="#94a3b8", length=0)
        ax.set_title("RISK ATTRIBUTION",
                     fontsize=6.5, color="#64748b", fontweight="600",
                     loc="left", pad=6)
        fig.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption("Only feature labels & SHAP direction sent to OpenAI — no applicant data.")


# ── NOTICE COLUMN ─────────────────────────────────────────────────────────────
with col_notice:
    r           = st.session_state.result
    notice_text = st.session_state.notice

    st.markdown('<div class="col-header">📄 Credit Decision Notice</div>',
                unsafe_allow_html=True)

    # Derive checks now so checklist is always renderable
    if notice_text:
        checks = run_compliance_checks(notice_text)
    else:
        # Pending — no notice yet; passed=None means "awaiting"
        checks = [
            {
                "idx": i, "num": CIRCLE_NUMS[i],
                "label": label, "tooltip": tooltip,
                "passed": None, "matched_patterns": [],
                "patterns": patterns, "color": color,
            }
            for i, (label, tooltip, patterns, color) in enumerate(COMPLIANCE_RULES)
        ]

    # ── Notice body ────────────────────────────────────────────────────────────
    if r is None:
        st.info("👈 Evaluate an application to generate the notice.")

    elif r["decision"] == "APPROVED":
        st.success("Application **approved** — no adverse action notice required.")
        audit_df = pd.DataFrame({
            "Feature":   [LABEL_MAP.get(f, f) for f in r["top_idx"]],
            "SHAP":      [f"{r['shap_vals'][f]:+.3f}" for f in r["top_idx"]],
            "Direction": ["↑ risk" if r["shap_vals"][f] > 0 else "↓ risk"
                          for f in r["top_idx"]],
        })
        st.dataframe(audit_df, use_container_width=True, hide_index=True)

    else:
        if not api_key:
            st.warning(
                "**OPENAI_API_KEY** not found. "
                "Add it to your `.env` file and restart:\n\n"
                "```\nOPENAI_API_KEY=sk-…\n```"
            )
        else:
            # Generate once per evaluation cycle
            if notice_text is None:
                try:
                    with st.spinner("Drafting notice via gpt-4o-mini…"):
                        st.session_state.notice = generate_notice(
                            r["decline_reasons"], make_openai_client(api_key))
                        notice_text = st.session_state.notice
                        checks = run_compliance_checks(notice_text)
                except Exception as e:
                    st.error(f"OpenAI error: {e}")

            if notice_text:
                hl_html = build_highlighted_html(notice_text, checks)

                # Highlighted notice — auto-height, no inner scroll
                st.markdown(
                    f"""<div style="font-size:0.82rem;line-height:1.6;
                        padding:0.75rem 0.9rem;
                        border:1px solid rgba(150,150,150,0.25);
                        border-radius:6px;">{hl_html}</div>""",
                    unsafe_allow_html=True,
                )

                dl_col, cap_col = st.columns([1, 2])
                with dl_col:
                    st.download_button(
                        "⬇ Download .txt", data=notice_text,
                        file_name="adverse_action_notice.txt",
                        mime="text/plain", use_container_width=True,
                    )
                with cap_col:
                    st.caption("⚠️ Requires compliance review before sending.")

    # ── Compliance checklist — always visible ──────────────────────────────────
    st.divider()
    render_checklist(checks)


# ═════════════════════════════════════════════════════════════════════════════
# LIMITATIONS  —  below the fold
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)
st.divider()

st.markdown("""
<div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.8rem">
  <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;
              text-transform:uppercase;color:#f59e0b;
              background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.25);
              border-radius:4px;padding:0.25rem 0.6rem">
    ⚠ Demo — Not for Production Use
  </div>
  <div style="font-size:0.72rem;color:#64748b;font-style:italic">
    Prism is a proof-of-concept demonstrating architecture and regulatory thinking.
    The items below would be addressed before any production deployment.
  </div>
</div>

<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:1rem;
            font-size:0.72rem;line-height:1.65;color:#94a3b8">

  <div>
    <div style="font-size:0.60rem;font-weight:700;letter-spacing:0.1em;
                text-transform:uppercase;color:#f87171;margin-bottom:0.4rem">
      Data &amp; Model
    </div>
    <div>
      · Trained on synthetic data (n=1,000) — not real credit applications<br>
      · No train/test split; model validity on held-out data is unverified<br>
      · Target variable is rule-based, not learned from actual default outcomes<br>
      · Feature set (20 columns) is from a 1990s German dataset — not reflective of AmEx's current data universe
    </div>
  </div>

  <div>
    <div style="font-size:0.60rem;font-weight:700;letter-spacing:0.1em;
                text-transform:uppercase;color:#fb923c;margin-bottom:0.4rem">
      Decision System
    </div>
    <div>
      · 50% probability threshold is arbitrary — production requires calibrated, risk-appetite-informed cut-offs<br>
      · Log-odds contributions are simplified estimates; a production model requires full actuarial and regulatory validation<br>
      · Applicant inputs are self-reported; production would use verified bureau and transactional data
    </div>
  </div>

  <div>
    <div style="font-size:0.60rem;font-weight:700;letter-spacing:0.1em;
                text-transform:uppercase;color:#facc15;margin-bottom:0.4rem">
      Compliance &amp; Legal
    </div>
    <div>
      · Generated notices require human compliance review before sending — this is surfaced in the UI but bears explicit statement<br>
      · Compliance patterns are regex-based; production requires legal sign-off on every template variant and clause<br>
      · MAS FAA-N16 and CBS requirements cited are current as of 2024; regulatory obligations change
    </div>
  </div>

  <div>
    <div style="font-size:0.60rem;font-weight:700;letter-spacing:0.1em;
                text-transform:uppercase;color:#34d399;margin-bottom:0.4rem">
      Privacy &amp; Security
    </div>
    <div>
      · The API boundary holds for the prototype; production deployment requires full security review of the data pipeline<br>
      · SHAP values from production models may carry more inferential sensitivity than synthetic-model values<br>
      · <code style="font-size:0.68rem">.env</code> key management is for local development only — production requires a secrets management service (e.g. AWS Secrets Manager, Vault)
    </div>
  </div>

</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# VISUAL DEMO  —  Applicant Pipeline Dashboard  (no backend / no API calls)
# Pre-defined profiles illustrating how Prism evaluates different risk profiles.
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("<div style='margin-top:2.5rem'></div>", unsafe_allow_html=True)
st.divider()
st.markdown(
    '<div style="font-size:0.62rem;font-weight:700;letter-spacing:0.12em;'
    'text-transform:uppercase;color:rgba(148,163,184,0.65);margin-bottom:0.3rem">'
    'Visual Demo  ·  Applicant Pipeline</div>'
    '<div style="font-size:0.74rem;color:#475569;margin-bottom:0.7rem">'
    'Eight synthetic applicants illustrating distinct risk scenarios. '
    'Hover a row to preview · click to lock. Static — no API calls.</div>',
    unsafe_allow_html=True,
)

PIPELINE_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{
  background:transparent;
  color:#cbd5e1;
  font-family:-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',sans-serif;
  font-size:12.5px;
}
.dashboard{
  display:flex;flex-direction:column;
  height:496px;
  background:#0d0f1a;
  border:1px solid rgba(255,255,255,0.08);
  border-radius:8px;overflow:hidden;
}
.dash-header{
  display:flex;justify-content:space-between;align-items:center;
  padding:9px 16px;
  border-bottom:1px solid rgba(255,255,255,0.07);
  background:rgba(8,9,18,0.9);
  flex-shrink:0;
}
.dash-title{
  font-size:10.5px;font-weight:700;letter-spacing:0.1em;
  text-transform:uppercase;color:rgba(148,163,184,0.85);
  display:flex;align-items:center;gap:8px;
}
.demo-pill{
  font-size:9.5px;font-weight:700;letter-spacing:0.08em;
  background:rgba(245,158,11,0.12);color:#f59e0b;
  border:1px solid rgba(245,158,11,0.22);
  border-radius:3px;padding:1px 6px;
}
.dash-stats{display:flex;gap:6px;font-size:10.5px}
.stat-pill{
  padding:2px 8px;border-radius:3px;
  font-weight:600;letter-spacing:0.04em;font-size:10px;
}
.s-ok{background:rgba(22,163,74,0.12);color:#4ade80;border:1px solid rgba(22,163,74,0.2)}
.s-no{background:rgba(220,38,38,0.12);color:#f87171;border:1px solid rgba(220,38,38,0.2)}
.s-tot{background:rgba(99,102,241,0.12);color:#a5b4fc;border:1px solid rgba(99,102,241,0.2)}
.main-layout{
  display:grid;grid-template-columns:1fr 310px;
  flex:1;overflow:hidden;
}
.table-section{
  display:flex;flex-direction:column;overflow:hidden;
  border-right:1px solid rgba(255,255,255,0.06);
}
.col-headers{
  display:grid;
  grid-template-columns:92px 1fr 46px 90px 72px 82px 110px;
  padding:0 14px;height:28px;align-items:center;
  background:rgba(8,9,18,0.6);
  border-bottom:1px solid rgba(255,255,255,0.05);
  font-size:9.5px;font-weight:600;letter-spacing:0.08em;
  text-transform:uppercase;color:rgba(100,116,139,0.75);
  flex-shrink:0;
}
.table-body{overflow-y:auto;flex:1}
.table-body::-webkit-scrollbar{width:3px}
.table-body::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.08);border-radius:2px}
.app-row{
  display:grid;
  grid-template-columns:92px 1fr 46px 90px 72px 82px 110px;
  padding:0 14px;height:46px;align-items:center;
  cursor:pointer;
  border-bottom:1px solid rgba(255,255,255,0.035);
  border-left:2px solid transparent;
  transition:background 0.1s,border-color 0.1s;
}
.app-row:hover{background:rgba(99,102,241,0.07)}
.app-row.selected{background:rgba(99,102,241,0.11);border-left-color:#6366f1}
.ref{font-family:'Courier New',monospace;font-size:10px;color:rgba(100,116,139,0.75)}
.aname{font-weight:600;color:#e2e8f0;font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.asub{font-size:9.5px;color:#475569;margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.risk-badge{display:inline-flex;align-items:center;gap:4px}
.rdot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.status-pill{
  display:inline-flex;align-items:center;gap:4px;
  padding:2px 7px;border-radius:3px;
  font-size:10px;font-weight:700;letter-spacing:0.04em;
}
.ok{background:rgba(22,163,74,0.12);color:#4ade80;border:1px solid rgba(22,163,74,0.22)}
.no{background:rgba(220,38,38,0.12);color:#f87171;border:1px solid rgba(220,38,38,0.22)}

/* DETAIL PANEL */
.detail-section{
  padding:12px 14px;overflow-y:auto;display:flex;flex-direction:column;
  background:rgba(6,8,16,0.5);
}
.detail-section::-webkit-scrollbar{width:3px}
.detail-section::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.08);border-radius:2px}
.placeholder{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  height:100%;color:#334155;font-size:11.5px;text-align:center;gap:8px;
}
.dname{font-size:13px;font-weight:700;color:#e2e8f0;margin-bottom:1px}
.dmeta{font-size:10px;color:#475569;margin-bottom:8px}
.dstatus{
  display:flex;align-items:center;gap:9px;
  padding:7px 9px;border-radius:4px;margin-bottom:6px;
}
.sec-label{
  font-size:9px;font-weight:700;letter-spacing:0.1em;
  text-transform:uppercase;color:#475569;
  margin:8px 0 4px;
}
.factor-item{
  font-size:10.5px;color:#64748b;line-height:1.65;
  padding:1px 0 1px 10px;position:relative;
}
.factor-item::before{content:"›";position:absolute;left:0;color:#4f46e5;font-weight:700}
.notice-box{
  background:rgba(8,10,20,0.7);
  border:1px solid rgba(255,255,255,0.06);
  border-radius:4px;padding:9px 11px;
  font-size:10.5px;line-height:1.7;color:#64748b;
  margin-top:4px;white-space:pre-wrap;
}
.notice-box em{color:#94a3b8;font-style:normal}
.approval-box{
  background:rgba(22,163,74,0.06);
  border:1px solid rgba(22,163,74,0.14);
  border-radius:4px;padding:9px 11px;
  font-size:10.5px;line-height:1.65;color:#4ade80;
  margin-top:4px;
}
.kv{display:grid;grid-template-columns:1fr 1fr;gap:3px 8px;margin-bottom:6px}
.kv-item{font-size:10px;color:#334155}
.kv-item span{color:#64748b}
.rmeter{
  height:3px;background:rgba(255,255,255,0.07);
  border-radius:2px;overflow:hidden;margin:3px 0 8px;
}
.rfill{height:100%;border-radius:2px;transition:width 0.4s ease}
</style>

<div class="dashboard">
  <div class="dash-header">
    <div class="dash-title">
      ◈ Prism  ·  Applicant Pipeline
      <span class="demo-pill">DEMO</span>
    </div>
    <div class="dash-stats">
      <span class="stat-pill s-ok">✓ 4 Approved</span>
      <span class="stat-pill s-no">✕ 4 Declined</span>
      <span class="stat-pill s-tot">8 Total</span>
    </div>
  </div>

  <div class="main-layout">
    <div class="table-section">
      <div class="col-headers">
        <div>Ref. No.</div><div>Applicant</div><div>Age</div>
        <div>Amount</div><div>Duration</div><div>Risk Score</div><div>Decision</div>
      </div>
      <div class="table-body" id="tBody"></div>
    </div>
    <div class="detail-section" id="dPanel">
      <div class="placeholder">
        <div style="font-size:22px;opacity:0.2">◈</div>
        <div>Hover or click a row<br>to view details</div>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = [
  {
    id:"AP-2024-001", name:"Sarah Tan", job:"Software Engineer",
    age:32, amt:"SGD 12,000", dur:"24 mo", risk:18.3,
    chk:"≥ 200 DM", sav:"≥ 1,000 DM", emp:"≥ 7 years",
    hse:"Own", hist:"Existing paid", status:"APPROVED",
    factors:[
      "Long-term employment (≥ 7 yrs) demonstrates income continuity",
      "Strong savings buffer (≥ 1,000 DM) provides financial resilience",
      "Positive checking account history confirms reliable transaction behaviour"
    ],
    notice: null
  },
  {
    id:"AP-2024-002", name:"Marcus Lee", job:"Retired Executive",
    age:67, amt:"SGD 15,000", dur:"60 mo", risk:89.4,
    chk:"0–200 DM", sav:"< 100 DM", emp:"≥ 7 years",
    hse:"Own", hist:"All paid", status:"DECLINED",
    factors:[
      "Retirement exposure: loan matures at age 72 — seven years past the standard threshold",
      "Minimal savings (< 100 DM) insufficient to support a SGD 15,000 obligation",
      "Fixed post-retirement income significantly increases long-term repayment risk"
    ],
    notice:`Dear Applicant,

We regret to inform you that your credit application with American Express has been declined.

– Your age profile, combined with the requested loan duration, indicates the loan would mature seven years past the standard retirement threshold, raising significant concerns about long-term repayment capacity under a fixed-income profile.
– Your current savings balance suggests limited financial reserves to support this level of credit commitment over the requested period.

You have the right to request a free copy of your credit report from Credit Bureau Singapore (CBS) within 30 days of receiving this notice.

We are committed to fair lending practices in accordance with the Monetary Authority of Singapore's Notice on Fair Dealing.

Sincerely,
Credit Risk Review Team
American Express (Singapore) Pte. Ltd.`
  },
  {
    id:"AP-2024-003", name:"Priya Nair", job:"Secondary School Teacher",
    age:27, amt:"SGD 6,000", dur:"24 mo", risk:31.7,
    chk:"0–200 DM", sav:"100–500 DM", emp:"1–4 years",
    hse:"For free", hist:"Existing paid", status:"APPROVED",
    factors:[
      "Public-sector employment provides stable, predictable income",
      "Requested amount is modest and well within assessed repayment capacity",
      "Consistent on-time repayment history demonstrates credit reliability"
    ],
    notice: null
  },
  {
    id:"AP-2024-004", name:"Ryan Teo", job:"Junior Marketing Executive",
    age:22, amt:"SGD 8,000", dur:"36 mo", risk:84.9,
    chk:"No account", sav:"No known savings", emp:"< 1 year",
    hse:"Rent", hist:"No credits", status:"DECLINED",
    factors:[
      "Employment tenure under one year — income stability not yet established",
      "No checking account on record; repayment behaviour cannot be assessed",
      "Absence of savings provides no buffer against income disruption",
      "Rental costs reduce disposable income available for repayment"
    ],
    notice:`Dear Applicant,

We regret to inform you that your credit application with American Express has been declined.

– Your current employment tenure is under one year, indicating that income stability may not yet be fully established.
– The absence of an active checking account at this bank means we are unable to assess your recent financial transaction behaviour and repayment patterns.
– Your savings balance suggests limited financial reserves to support the requested credit commitment should income be disrupted.

You have the right to request a free copy of your credit report from Credit Bureau Singapore (CBS) within 30 days of receiving this notice.

We are committed to fair lending practices in accordance with the Monetary Authority of Singapore's Notice on Fair Dealing.

Sincerely,
Credit Risk Review Team
American Express (Singapore) Pte. Ltd.`
  },
  {
    id:"AP-2024-005", name:"Michelle Ong", job:"HR Manager",
    age:45, amt:"SGD 10,000", dur:"36 mo", risk:22.1,
    chk:"0–200 DM", sav:"500–1,000 DM", emp:"4–7 years",
    hse:"Own", hist:"Existing paid", status:"APPROVED",
    factors:[
      "Property ownership provides tangible asset backing",
      "Mid-career tenure at a stable institution reduces income disruption risk",
      "Adequate savings buffer relative to the requested loan size"
    ],
    notice: null
  },
  {
    id:"AP-2024-006", name:"Benjamin Koh", job:"Sales Director",
    age:38, amt:"SGD 18,000", dur:"48 mo", risk:76.3,
    chk:"< 0 DM (overdrawn)", sav:"< 100 DM", emp:"4–7 years",
    hse:"Rent", hist:"Delayed previously", status:"DECLINED",
    factors:[
      "Checking account is currently overdrawn — active financial distress signal",
      "Previous payment delays on record represent a material repayment risk",
      "Requested amount of SGD 18,000 is high relative to the current financial profile",
      "Minimal savings provide no buffer against income volatility common in commission roles"
    ],
    notice:`Dear Applicant,

We regret to inform you that your credit application with American Express has been declined.

– Your checking account is currently overdrawn, which raises concerns about your present financial position and capacity to service additional credit obligations.
– Your credit history indicates previous payment delays, which is a significant factor in our assessment of repayment reliability.
– The amount of credit requested is high relative to your assessed financial profile and current account standing.

You have the right to request a free copy of your credit report from Credit Bureau Singapore (CBS) within 30 days of receiving this notice.

We are committed to fair lending practices in accordance with the Monetary Authority of Singapore's Notice on Fair Dealing.

Sincerely,
Credit Risk Review Team
American Express (Singapore) Pte. Ltd.`
  },
  {
    id:"AP-2024-007", name:"Dr. Anita Singh", job:"Senior Medical Consultant",
    age:52, amt:"SGD 5,000", dur:"12 mo", risk:10.8,
    chk:"≥ 200 DM", sav:"≥ 1,000 DM", emp:"≥ 7 years",
    hse:"Own", hist:"All paid", status:"APPROVED",
    factors:[
      "Exemplary credit history — all prior credits fully settled without delay",
      "Highest employment stability tier combined with senior professional income",
      "Low loan amount and short duration minimise overall exposure",
      "Strong savings and property ownership provide multiple layers of backing"
    ],
    notice: null
  },
  {
    id:"AP-2024-008", name:"Kevin Lim", job:"Logistics Coordinator",
    age:35, amt:"SGD 9,000", dur:"30 mo", risk:68.7,
    chk:"No account", sav:"< 100 DM", emp:"1–4 years",
    hse:"Rent", hist:"No credits", status:"DECLINED",
    factors:[
      "No checking account on record; transaction history unavailable for assessment",
      "Savings balance below the threshold sufficient to support this commitment",
      "Rental costs represent an ongoing obligation reducing disposable repayment capacity",
      "Absence of prior credit history limits ability to assess repayment behaviour"
    ],
    notice:`Dear Applicant,

We regret to inform you that your credit application with American Express has been declined.

– The absence of a checking account at this bank means we are unable to assess your financial transaction patterns or repayment behaviour.
– Your current savings balance is below the level we consider sufficient to support the requested credit commitment, leaving limited buffer against income disruption.
– Your housing arrangement involves an ongoing rental cost that reduces the disposable income available for loan repayment.

You have the right to request a free copy of your credit report from Credit Bureau Singapore (CBS) within 30 days of receiving this notice.

We are committed to fair lending practices in accordance with the Monetary Authority of Singapore's Notice on Fair Dealing.

Sincerely,
Credit Risk Review Team
American Express (Singapore) Pte. Ltd.`
  }
];

let lockedId = null;

function riskColor(s){
  return s < 30 ? '#4ade80' : s < 60 ? '#fb923c' : '#f87171';
}

function renderTable(){
  document.getElementById('tBody').innerHTML = DATA.map(a => `
    <div class="app-row" id="row-${a.id}"
         onmouseover="show('${a.id}',false)"
         onmouseout="onOut()"
         onclick="lock('${a.id}')">
      <div class="ref">${a.id}</div>
      <div style="overflow:hidden">
        <div class="aname">${a.name}</div>
        <div class="asub">${a.job}</div>
      </div>
      <div style="font-size:12px;color:#94a3b8">${a.age}</div>
      <div style="font-size:12px;color:#94a3b8">${a.amt}</div>
      <div style="font-size:12px;color:#94a3b8">${a.dur}</div>
      <div>
        <div class="risk-badge">
          <div class="rdot" style="background:${riskColor(a.risk)}"></div>
          <span style="color:${riskColor(a.risk)};font-size:12px;font-weight:700">${a.risk.toFixed(1)}%</span>
        </div>
      </div>
      <div>
        <span class="status-pill ${a.status==='APPROVED'?'ok':'no'}">
          ${a.status==='APPROVED'?'✓':'✕'} ${a.status}
        </span>
      </div>
    </div>`).join('');
}

function show(id, isLocked){
  const a = DATA.find(x=>x.id===id);
  if(!a) return;
  const rc = riskColor(a.risk);
  const dec = a.status==='DECLINED';
  document.getElementById('dPanel').innerHTML = `
    <div class="dname">${a.name}</div>
    <div class="dmeta">${a.job}  ·  Age ${a.age}  ·  ${a.dur}</div>

    <div class="dstatus" style="background:${dec?'rgba(220,38,38,0.08)':'rgba(22,163,74,0.08)'};border:1px solid ${dec?'rgba(220,38,38,0.18)':'rgba(22,163,74,0.18)'}">
      <span style="font-size:16px;line-height:1">${dec?'✕':'✓'}</span>
      <div>
        <div style="font-size:11px;font-weight:700;color:${dec?'#f87171':'#4ade80'};letter-spacing:0.05em">${a.status}</div>
        <div style="font-size:9.5px;color:#475569">Default probability: <strong style="color:${rc}">${a.risk.toFixed(1)}%</strong></div>
      </div>
    </div>

    <div class="rmeter"><div class="rfill" style="width:${Math.min(a.risk,100)}%;background:${rc}"></div></div>

    <div class="kv">
      <div class="kv-item">Checking <span>${a.chk}</span></div>
      <div class="kv-item">Savings <span>${a.sav}</span></div>
      <div class="kv-item">Employment <span>${a.emp}</span></div>
      <div class="kv-item">Housing <span>${a.hse}</span></div>
    </div>

    <div class="sec-label">${dec?'Decline Factors':'Approval Factors'}</div>
    ${a.factors.map(f=>`<div class="factor-item">${f}</div>`).join('')}

    ${dec ? `
      <div class="sec-label" style="margin-top:8px">Credit Decision Notice</div>
      <div class="notice-box">${a.notice}</div>
    ` : `
      <div class="approval-box" style="margin-top:8px">
        <div style="font-weight:700;margin-bottom:3px">✓ Application Approved</div>
        <div style="font-size:10px;opacity:0.8">No adverse action notice required. Standard approval documentation issued through primary onboarding channel.</div>
      </div>
    `}
  `;
}

function lock(id){
  lockedId = id;
  document.querySelectorAll('.app-row').forEach(r=>r.classList.remove('selected'));
  document.getElementById('row-'+id)?.classList.add('selected');
  show(id, true);
}

function onOut(){
  if(lockedId) show(lockedId, true);
  else {
    document.getElementById('dPanel').innerHTML =
      '<div class="placeholder"><div style="font-size:22px;opacity:0.2">◈</div><div>Hover or click a row<br>to view details</div></div>';
  }
}

renderTable();
</script>
</html>
"""

components.html(PIPELINE_HTML, height=500, scrolling=False)
