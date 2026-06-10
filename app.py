"""
Credit Decision Intelligence
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
    page_title="Credit Decision Intelligence",
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

  /* ── Ghost placeholder animation ─────────────────────── */
  @keyframes gp{0%,100%{opacity:.55}50%{opacity:.3}}
  .gbar{animation:gp 2.6s ease-in-out infinite;border-radius:2px}
  .gred{background:rgba(239,68,68,0.28)}
  .ggrn{background:rgba(34,197,94,0.28)}
  .gneu{background:rgba(148,163,184,0.15)}

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
    "duration": (
        "Loan repayment period\n"
        "\n"
        "\u2191 Longer \u2192 higher risk\n"
        "\n"
        "\u2022 Risk amplifies when loan matures past retirement age 65"
    ),
    "credit_amount": (
        "Total loan size (SGD)\n"
        "\n"
        "\u2191 Larger \u2192 higher risk\n"
        "\n"
        "\u2022 Increases monthly repayment burden"
    ),
    "installment_commitment": (
        "Monthly repayment as % of income\n"
        "\n"
        "\u2705\u2705  1  \u2014  \u2264 15%  (very low burden)\n"
        "\n"
        "\u2705  2  \u2014  15\u201325%\n"
        "\n"
        "\u274c  3  \u2014  25\u201335%\n"
        "\n"
        "\u274c\u274c  4  \u2014  > 35%  (high stress)"
    ),
    "age": (
        "Non-linear risk factor\n"
        "\n"
        "\u2705\u2705  30\u201350  \u2014  peak earning years\n"
        "\n"
        "\u2705  25\u201330 / 50\u201355  \u2014  stable\n"
        "\n"
        "\u274c  < 25  \u2014  thin credit history\n"
        "\n"
        "\u274c\u274c  55+ with long loan  \u2014  retirement exposure"
    ),
    "existing_credits": (
        "Active credit facilities at this bank\n"
        "\n"
        "\u2191 More \u2192 higher risk\n"
        "\n"
        "\u2705\u2705  1  \u2014  single obligation\n"
        "\n"
        "\u2705  2  \u2014  manageable\n"
        "\n"
        "\u274c  3  \u2014  elevated\n"
        "\n"
        "\u274c\u274c  4  \u2014  over-leveraged"
    ),
    "residence_since": (
        "Years at current address\n"
        "\n"
        "\u2191 Longer \u2192 lower risk\n"
        "\n"
        "\u2705\u2705  4 years  \u2014  very stable\n"
        "\n"
        "\u2705  3 years  \u2014  stable\n"
        "\n"
        "\u274c  2 years  \u2014  moderate\n"
        "\n"
        "\u274c\u274c  1 year  \u2014  recent mover"
    ),
    "num_dependents": (
        "Number of financial dependents\n"
        "\n"
        "\u2705\u2705  1  \u2014  low burden\n"
        "\n"
        "\u2705  2  \u2014  manageable\n"
        "\n"
        "\u25cb   3  \u2014  moderate\n"
        "\n"
        "\u274c  4  \u2014  high burden\n"
        "\n"
        "\u274c\u274c  5+  \u2014  significantly reduces repayment capacity"
    ),
    "checking_status": (
        "Checking account balance\n"
        "(Values in Deutsche Marks)\n"
        "\n"
        "\u2705\u2705  \u2265 200 DM  \u2014  adequate liquidity\n"
        "\n"
        "\u2705  0\u2013200 DM  \u2014  low positive balance\n"
        "\n"
        "\u274c  < 0 DM  \u2014  overdrawn\n"
        "\n"
        "\u274c\u274c  no account  \u2014  no history available"
    ),
    "savings_status": (
        "Savings / bond balance\n"
        "(Values in Deutsche Marks)\n"
        "\n"
        "\u2705\u2705  \u2265 1,000 DM  \u2014  strong cushion\n"
        "\n"
        "\u2705  500\u20131,000  \u2014  adequate\n"
        "\n"
        "\u25cb   100\u2013500  \u2014  modest\n"
        "\n"
        "\u274c  < 100 DM  \u2014  minimal buffer\n"
        "\n"
        "\u274c\u274c  none  \u2014  unverifiable"
    ),
    "employment": (
        "Time at current employer\n"
        "\n"
        "\u2705\u2705  \u2265 7 yrs  \u2014  very stable income\n"
        "\n"
        "\u2705  4\u20137 yrs  \u2014  stable\n"
        "\n"
        "\u25cb   1\u20134 yrs  \u2014  moderate\n"
        "\n"
        "\u274c  < 1 yr  \u2014  probationary\n"
        "\n"
        "\u274c\u274c  unemployed  \u2014  no income"
    ),
    "housing": (
        "Current living arrangement\n"
        "\n"
        "\u2705\u2705  own  \u2014  property owner, asset backing\n"
        "\n"
        "\u25cb   for free  \u2014  no cost, no asset\n"
        "\n"
        "\u274c\u274c  rent  \u2014  ongoing cost reduces income"
    ),
    "purpose": (
        "Intended use of the loan\n"
        "\n"
        "\u2705\u2705  education  \u2014  future income potential\n"
        "\n"
        "\u2705  furniture  \u2014  tangible asset\n"
        "\n"
        "\u25cb   new / used car  \u2014  depreciating asset\n"
        "\n"
        "\u274c  radio / tv  \u2014  fast depreciation\n"
        "\n"
        "\u274c\u274c  business  \u2014  variable returns"
    ),
    "credit_history": (
        "Past repayment behaviour\n"
        "\n"
        "\u2705\u2705  all paid  \u2014  fully settled\n"
        "\n"
        "\u2705  existing paid  \u2014  on time\n"
        "\n"
        "\u25cb   no credits  \u2014  no prior history\n"
        "\n"
        "\u274c  delayed  \u2014  late payments\n"
        "\n"
        "\u274c\u274c  critical / other  \u2014  credits elsewhere"
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
        "Issuing institution named in the sign-off",
        [
            r"Credit Decision Intelligence(?:\s*(?:Pte\.?\s*Ltd\.?)?)?",
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
        "num_dependents":  rng.integers(1, 6, n).astype(float),
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

    prompt = f"""You are a compliance officer drafting a formal Credit Decision Notice.

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
Credit Decision Intelligence
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


def render_checklist(checks: list[dict], ghost: bool = False) -> None:
    """
    Render the compliance checklist grid.
    ghost=True → pending pre-evaluation state with dimmed styling.
    ghost=False + passed=None → normal pending (same visual but not wrapped).
    """
    n_pass  = sum(1 for c in checks if c["passed"] is True)
    n_total = len(checks)
    evaluated = any(c["passed"] is not None for c in checks)

    opacity = "0.35" if ghost else "1"

    # Header + score
    score_str = "awaiting evaluation…" if ghost else (
        f"{n_pass}/{n_total} passed" if evaluated else "awaiting notice…"
    )
    bar_color = "rgba(150,150,150,0.25)" if ghost else (
        (
            "#27ae60" if n_pass == n_total else
            "#e67e22" if n_pass >= n_total * 0.75 else
            "#c0392b"
        ) if evaluated else "rgba(150,150,150,0.4)"
    )

    st.markdown(
        f"<div style='font-size:0.82rem;font-weight:700;margin:0.4rem 0 0.3rem;"
        f"opacity:{opacity}'>Compliance Checks</div>",
        unsafe_allow_html=True,
    )

    pct = int(n_pass / n_total * 100) if (evaluated and not ghost) else 0
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;opacity:{opacity}">
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

    grid = f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:3px 10px;opacity:{opacity}'>"
    for c in checks:
        if ghost or c["passed"] is None:
            icon  = "&#9633;"   # hollow square — clearly pending
            fg    = "rgba(150,150,150,0.5)"
            border= "rgba(150,150,150,0.2)"
            tip   = "Evaluated after notice is generated"
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
    '◈ Credit Decision Intelligence</span>'
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
      · <span style="color:#cbd5e1">The problem:</span> Banks must explain every credit decline in specific, plain-language terms — a regulatory requirement that is slow and inconsistent to fulfil manually at scale.<br>
      · <span style="color:#cbd5e1">The constraint:</span> Privacy regulations prohibit sending customer financial data to external AI services, ruling out a straightforward ChatGPT integration.<br>
      · <span style="color:#cbd5e1">The solution:</span> This system sends only anonymised risk signals — not applicant data — to the AI, which uses them to draft a compliant, specific decline notice automatically.<br>
      · <span style="color:#cbd5e1">The result:</span> A privacy-safe pipeline that produces a regulator-ready credit decision notice in seconds, without a human drafting each one from scratch.
    </div>
  </div>

  <div style="padding:0.55rem 0.75rem;
              border-left:1px solid rgba(99,102,241,0.12)">
    <div style="font-size:0.57rem;font-weight:700;letter-spacing:0.13em;
                text-transform:uppercase;color:#38bdf8;margin-bottom:0.3rem">
      Technical Architecture
    </div>
    <div>
      · <span style="color:#cbd5e1">Risk assessment:</span> The applicant's profile is scored by a credit risk model that identifies the likelihood of default and the factors that drove that outcome.<br>
      · <span style="color:#cbd5e1">Explainability:</span> Each decision is broken down into its contributing factors — showing not just the score, but why. This is what makes the notice specific rather than generic.<br>
      · <span style="color:#cbd5e1">Privacy boundary:</span> Only the risk factors are passed to the AI — never the applicant's personal details. The explanation is generated without the underlying data ever leaving the system.<br>
      · <span style="color:#cbd5e1">Compliance guardrails:</span> The AI writes within a structured template that enforces required regulatory language. Every output is automatically validated before it is shown.
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
        residence_since = st.select_slider(
            "Years of Residence",
            options=[1, 2, 3, 4],
            format_func=lambda x: f"{x} year" if x == 1 else f"{x} years",
            value=2,
            help=FIELD_HELP["residence_since"],
        )

    num_dependents = st.select_slider(
        "Dependents",
        options=list(range(1, 6)),
        format_func=lambda x: "5+" if x == 5 else str(x),
        value=1,
        help=FIELD_HELP["num_dependents"],
    )

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
        st.markdown(
            '<div style="background:rgba(100,116,139,0.08);border:1px solid rgba(100,116,139,0.18);'
            'border-left:3px solid rgba(100,116,139,0.3);border-radius:5px;'
            'padding:0.35rem 0.8rem;margin-bottom:0.3rem">'
            '<span style="font-size:0.8rem;font-weight:800;color:rgba(148,163,184,0.38);'
            'letter-spacing:0.08em;text-transform:uppercase">&#9711;  Awaiting Evaluation</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.metric("Default Probability", "\u2014 %")
        ghost_labels = [
            "Employment Duration", "Checking Account",
            "Applicant Age", "Credit Amount (SGD)",
            "Credit History", "Savings Status",
        ]
        ghost_vals   = [40, 25, 18, 14, -12, -9]
        ghost_colors = [
            (0.94, 0.27, 0.27, 0.18) if v > 0 else (0.13, 0.77, 0.37, 0.18)
            for v in ghost_vals
        ]
        fig, ax = plt.subplots(figsize=(4.5, 1.85))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.barh(ghost_labels, ghost_vals, color=ghost_colors, edgecolor="none", height=0.52)
        ax.axvline(0, color="#374151", linewidth=0.8)
        for spine in ("top", "right", "left", "bottom"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", labelsize=6, colors="#475569", length=0)
        ax.set_xlabel("share of risk attribution (%)", fontsize=6, color="#4b5563", labelpad=4)
        ax.set_title("RISK ATTRIBUTION", fontsize=6.5, color="#64748b",
                     fontweight="600", loc="left", pad=6)
        fig.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Evaluate an application to see real risk drivers.")
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
        st.markdown(
            '<div style="font-size:0.77rem;line-height:1.9;color:rgba(148,163,184,0.42);'
            'border:1px solid rgba(255,255,255,0.06);border-radius:5px;'
            'padding:0.75rem 1rem;margin-bottom:0.6rem">'
            'Dear Applicant,<br><br>'
            'We regret to inform you that your credit application has been declined.<br><br>'
            '<span style="opacity:.6">\u2013 &nbsp;</span>'
            '<span style="background:rgba(148,163,184,0.12);border-radius:3px;padding:1px 60px">&nbsp;</span><br>'
            '<span style="opacity:.6">\u2013 &nbsp;</span>'
            '<span style="background:rgba(148,163,184,0.12);border-radius:3px;padding:1px 45px">&nbsp;</span><br>'
            '<span style="opacity:.6">\u2013 &nbsp;</span>'
            '<span style="background:rgba(148,163,184,0.12);border-radius:3px;padding:1px 35px">&nbsp;</span><br><br>'
            'You have the right to request a free copy of your credit report from '
            '<span style="color:rgba(99,102,241,0.55)">Credit Bureau Singapore (CBS)</span>'
            ' within 30 days of receiving this notice.<br><br>'
            'We are committed to fair lending practices in accordance with the '
            '<span style="color:rgba(99,102,241,0.55)">Monetary Authority of Singapore&#39;s Notice on Fair Dealing</span>.'
            '<br><br>Sincerely,<br>Credit Risk Review Team<br>Credit Decision Intelligence'
            '</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        render_checklist(checks, ghost=True)

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

                st.divider()
                render_checklist(checks)


# ═════════════════════════════════════════════════════════════════════════════
# LIMITATIONS  —  below the fold
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<div style='text-align:center;padding:0.6rem 0 0;font-size:0.65rem;"
    "color:rgba(148,163,184,0.28);letter-spacing:0.08em;text-transform:uppercase'>"
    "\u2193 &nbsp; Limitations &nbsp;\u00b7&nbsp; Pipeline Demo below"
    "</div>",
    unsafe_allow_html=True,
)

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
    This is a proof-of-concept demonstrating architecture and regulatory thinking.
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
      · Feature set (20 columns) is from a 1990s German dataset — not reflective of a real institution's data universe
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
# Pre-defined profiles illustrating how the system evaluates different risk profiles.
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

import base64 as _b64
PIPELINE_HTML = _b64.b64decode("PCFET0NUWVBFIGh0bWw+CjxodG1sPjxoZWFkPjxtZXRhIGNoYXJzZXQ9InV0Zi04Ij48c3R5bGU+Cip7bWFyZ2luOjA7cGFkZGluZzowO2JveC1zaXppbmc6Ym9yZGVyLWJveH0KaHRtbCxib2R5e3dpZHRoOjEwMCU7YmFja2dyb3VuZDojMGQwZjFhO2NvbG9yOiNjYmQ1ZTE7Zm9udC1mYW1pbHk6LWFwcGxlLXN5c3RlbSxCbGlua01hY1N5c3RlbUZvbnQsJ1NlZ29lIFVJJyxzYW5zLXNlcmlmO2ZvbnQtc2l6ZToxMnB4fQojd3JhcHtiYWNrZ3JvdW5kOiMwZDBmMWE7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDgpO2JvcmRlci1yYWRpdXM6NnB4O292ZXJmbG93OmhpZGRlbn0KI2hkcntkaXNwbGF5OmZsZXg7anVzdGlmeS1jb250ZW50OnNwYWNlLWJldHdlZW47YWxpZ24taXRlbXM6Y2VudGVyO3BhZGRpbmc6OHB4IDE0cHg7YmFja2dyb3VuZDojMDYwNzEwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNyl9CiNodHtmb250LXNpemU6MTBweDtmb250LXdlaWdodDo3MDA7bGV0dGVyLXNwYWNpbmc6LjFlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7Y29sb3I6cmdiYSgxNDgsMTYzLDE4NCwuODUpO2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjdweH0KLmRwe2ZvbnQtc2l6ZTo5cHg7Zm9udC13ZWlnaHQ6NzAwO2JhY2tncm91bmQ6cmdiYSgyNDUsMTU4LDExLC4xNSk7Y29sb3I6I2Y1OWUwYjtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjQ1LDE1OCwxMSwuMjUpO2JvcmRlci1yYWRpdXM6M3B4O3BhZGRpbmc6MXB4IDVweH0KI2hze2Rpc3BsYXk6ZmxleDtnYXA6NXB4fQouc3B7cGFkZGluZzoycHggN3B4O2JvcmRlci1yYWRpdXM6M3B4O2ZvbnQtd2VpZ2h0OjYwMDtmb250LXNpemU6OS41cHh9Ci5zYXtiYWNrZ3JvdW5kOnJnYmEoMjIsMTYzLDc0LC4xNSk7Y29sb3I6IzRhZGU4MDtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjIsMTYzLDc0LC4yNSl9Ci5zZHtiYWNrZ3JvdW5kOnJnYmEoMjIwLDM4LDM4LC4xNSk7Y29sb3I6I2Y4NzE3MTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjIwLDM4LDM4LC4yNSl9Ci5zdHtiYWNrZ3JvdW5kOnJnYmEoOTksMTAyLDI0MSwuMTUpO2NvbG9yOiNhNWI0ZmM7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDk5LDEwMiwyNDEsLjI1KX0KI2xheXtkaXNwbGF5OmZsZXh9CiN0cHtmbGV4OjE7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtib3JkZXItcmlnaHQ6MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4wNyl9CiNjaHtkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjU1cHggMWZyIDM2cHggODBweCA1OHB4IDcycHggOTZweDtwYWRkaW5nOjAgMTBweDtoZWlnaHQ6MjZweDthbGlnbi1pdGVtczpjZW50ZXI7YmFja2dyb3VuZDojMDYwNzEwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsLjA1KTtmbGV4LXNocmluazowO2ZvbnQtc2l6ZTo5cHg7Zm9udC13ZWlnaHQ6NjAwO2xldHRlci1zcGFjaW5nOi4wOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTtjb2xvcjpyZ2JhKDEwMCwxMTYsMTM5LC43NSl9CiN0YntmbGV4OjF9Ci5yb3d7ZGlzcGxheTpncmlkO2dyaWQtdGVtcGxhdGUtY29sdW1uczo1NXB4IDFmciAzNnB4IDgwcHggNThweCA3MnB4IDk2cHg7cGFkZGluZzowIDEwcHg7aGVpZ2h0OjQycHg7YWxpZ24taXRlbXM6Y2VudGVyO2N1cnNvcjpwb2ludGVyO2JvcmRlci1ib3R0b206MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsLjA0KTtib3JkZXItbGVmdDoycHggc29saWQgdHJhbnNwYXJlbnR9Ci5yb3c6aG92ZXJ7YmFja2dyb3VuZDpyZ2JhKDk5LDEwMiwyNDEsLjA4KX0KLnJvdy5zZWx7YmFja2dyb3VuZDpyZ2JhKDk5LDEwMiwyNDEsLjEyKTtib3JkZXItbGVmdC1jb2xvcjojNjM2NmYxfQoucmVme2ZvbnQtZmFtaWx5Om1vbm9zcGFjZTtmb250LXNpemU6OXB4O2NvbG9yOnJnYmEoMTAwLDExNiwxMzksLjcpfQoubm17Zm9udC13ZWlnaHQ6NjAwO2NvbG9yOiNlMmU4ZjA7Zm9udC1zaXplOjExcHg7d2hpdGUtc3BhY2U6bm93cmFwO292ZXJmbG93OmhpZGRlbjt0ZXh0LW92ZXJmbG93OmVsbGlwc2lzfQouamJ7Zm9udC1zaXplOjlweDtjb2xvcjojNDc1NTY5O21hcmdpbi10b3A6MXB4O3doaXRlLXNwYWNlOm5vd3JhcDtvdmVyZmxvdzpoaWRkZW47dGV4dC1vdmVyZmxvdzplbGxpcHNpc30KLnJie2Rpc3BsYXk6aW5saW5lLWZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDozcHh9Ci5yZHt3aWR0aDo1cHg7aGVpZ2h0OjVweDtib3JkZXItcmFkaXVzOjUwJX0KLnBpbGx7ZGlzcGxheTppbmxpbmUtZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjRweDtwYWRkaW5nOjJweCA2cHg7Ym9yZGVyLXJhZGl1czozcHg7Zm9udC1zaXplOjlweDtmb250LXdlaWdodDo3MDB9Ci5va3tiYWNrZ3JvdW5kOnJnYmEoMjIsMTYzLDc0LC4xMik7Y29sb3I6IzRhZGU4MDtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjIsMTYzLDc0LC4yMil9Ci5ub3tiYWNrZ3JvdW5kOnJnYmEoMjIwLDM4LDM4LC4xMik7Y29sb3I6I2Y4NzE3MTtib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjIwLDM4LDM4LC4yMil9CiNkcHt3aWR0aDozMDBweDtmbGV4LXNocmluazowO3BhZGRpbmc6MTFweCAxM3B4O2JhY2tncm91bmQ6IzA0MDUwOH0KLnBoe2Rpc3BsYXk6ZmxleDtmbGV4LWRpcmVjdGlvbjpjb2x1bW47YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7bWluLWhlaWdodDozMDBweDtjb2xvcjojMzM0MTU1O2ZvbnQtc2l6ZToxMXB4O3RleHQtYWxpZ246Y2VudGVyO2dhcDo2cHh9Ci5kbntmb250LXNpemU6MTNweDtmb250LXdlaWdodDo3MDA7Y29sb3I6I2UyZThmMDttYXJnaW4tYm90dG9tOjFweH0KLmRte2ZvbnQtc2l6ZTo5LjVweDtjb2xvcjojNDc1NTY5O21hcmdpbi1ib3R0b206NnB4fQouZHNie2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjhweDtwYWRkaW5nOjZweCA5cHg7Ym9yZGVyLXJhZGl1czo0cHg7bWFyZ2luLWJvdHRvbTo1cHh9Ci5zbHtmb250LXNpemU6OC41cHg7Zm9udC13ZWlnaHQ6NzAwO2xldHRlci1zcGFjaW5nOi4xZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlO2NvbG9yOiM0NzU1Njk7bWFyZ2luOjdweCAwIDNweH0KLmZpe2ZvbnQtc2l6ZToxMHB4O2NvbG9yOiM2NDc0OGI7bGluZS1oZWlnaHQ6MS42NTtwYWRkaW5nOjFweCAwIDFweCAxMHB4O3Bvc2l0aW9uOnJlbGF0aXZlfQouZmk6OmJlZm9yZXtjb250ZW50OidcMjAzYSc7cG9zaXRpb246YWJzb2x1dGU7bGVmdDowO2NvbG9yOiM0ZjQ2ZTU7Zm9udC13ZWlnaHQ6NzAwfQoubmJ7YmFja2dyb3VuZDojMDQwNTA4O2JvcmRlcjoxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwuMDcpO2JvcmRlci1yYWRpdXM6NHB4O3BhZGRpbmc6OHB4IDEwcHg7Zm9udC1zaXplOjEwcHg7bGluZS1oZWlnaHQ6MS43O2NvbG9yOiM2NDc0OGI7bWFyZ2luLXRvcDozcHg7d2hpdGUtc3BhY2U6cHJlLXdyYXB9Ci5hYntiYWNrZ3JvdW5kOnJnYmEoMjIsMTYzLDc0LC4wNyk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIyLDE2Myw3NCwuMTUpO2JvcmRlci1yYWRpdXM6NHB4O3BhZGRpbmc6OHB4IDEwcHg7Zm9udC1zaXplOjEwcHg7bGluZS1oZWlnaHQ6MS42O2NvbG9yOiM0YWRlODA7bWFyZ2luLXRvcDozcHh9Ci5rdntkaXNwbGF5OmdyaWQ7Z3JpZC10ZW1wbGF0ZS1jb2x1bW5zOjFmciAxZnI7Z2FwOjJweCA2cHg7bWFyZ2luLWJvdHRvbTo1cHh9Ci5rdml7Zm9udC1zaXplOjkuNXB4O2NvbG9yOiMzMzQxNTV9Lmt2aSBzcGFue2NvbG9yOiM2NDc0OGJ9Ci5ybXtoZWlnaHQ6M3B4O2JhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwuMDgpO2JvcmRlci1yYWRpdXM6MnB4O292ZXJmbG93OmhpZGRlbjttYXJnaW46MnB4IDAgN3B4fQoucmZ7aGVpZ2h0OjEwMCU7Ym9yZGVyLXJhZGl1czoycHh9Cjwvc3R5bGU+PC9oZWFkPjxib2R5Pgo8ZGl2IGlkPSJ3cmFwIj4KPGRpdiBpZD0iaGRyIj48ZGl2IGlkPSJodCI+JiM5NjcyOyBDcmVkaXQgRGVjaXNpb24gSW50ZWxsaWdlbmNlICZtaWRkb3Q7IEFwcGxpY2FudCBQaXBlbGluZTxzcGFuIGNsYXNzPSJkcCI+REVNTzwvc3Bhbj48L2Rpdj48ZGl2IGlkPSJocyI+PHNwYW4gY2xhc3M9InNwIHNhIiBpZD0ic29rIj48L3NwYW4+PHNwYW4gY2xhc3M9InNwIHNkIiBpZD0ic25vIj48L3NwYW4+PHNwYW4gY2xhc3M9InNwIHN0IiBpZD0ic3RvdCI+PC9zcGFuPjwvZGl2PjwvZGl2Pgo8ZGl2IGlkPSJsYXkiPjxkaXYgaWQ9InRwIj48ZGl2IGlkPSJjaCI+PGRpdj5SZWYuPC9kaXY+PGRpdj5BcHBsaWNhbnQ8L2Rpdj48ZGl2PkFnZTwvZGl2PjxkaXY+QW1vdW50PC9kaXY+PGRpdj5UZXJtPC9kaXY+PGRpdj5SaXNrPC9kaXY+PGRpdj5EZWNpc2lvbjwvZGl2PjwvZGl2PjxkaXYgaWQ9InRiIj48L2Rpdj48L2Rpdj4KPGRpdiBpZD0iZHAiPjxkaXYgY2xhc3M9InBoIj48ZGl2IHN0eWxlPSJmb250LXNpemU6MjJweDtvcGFjaXR5Oi4yIj4mIzk2NzI7PC9kaXY+PGRpdj5Ib3ZlciBvciBjbGljayBhIHJvdzxicj50byB2aWV3IGRldGFpbHM8L2Rpdj48L2Rpdj48L2Rpdj48L2Rpdj48L2Rpdj4KPHNjcmlwdD4KdmFyIGxvY2tlZD1udWxsLE5MPSdcbic7CmZ1bmN0aW9uIHJjKHMpe3JldHVybiBzPDMwPycjNGFkZTgwJzpzPDYwPycjZmI5MjNjJzonI2Y4NzE3MSc7fQp2YXIgQ0JTPSdZb3UgaGF2ZSB0aGUgcmlnaHQgdG8gcmVxdWVzdCBhIGZyZWUgY29weSBvZiB5b3VyIGNyZWRpdCByZXBvcnQgZnJvbSBDcmVkaXQgQnVyZWF1IFNpbmdhcG9yZSAoQ0JTKSB3aXRoaW4gMzAgZGF5cy4nOwp2YXIgTUFTPSdXZSBhcmUgY29tbWl0dGVkIHRvIGZhaXIgbGVuZGluZyBwcmFjdGljZXMgaW4gYWNjb3JkYW5jZSB3aXRoIE1BUyBOb3RpY2Ugb24gRmFpciBEZWFsaW5nLic7CmZ1bmN0aW9uIG1rKGIpe3JldHVybiAnRGVhciBBcHBsaWNhbnQsJytOTCtOTCsnV2UgcmVncmV0IHRvIGluZm9ybSB5b3UgdGhhdCB5b3VyIGNyZWRpdCBhcHBsaWNhdGlvbiBoYXMgYmVlbiBkZWNsaW5lZC4nK05MK05MK2IrTkwrTkwrQ0JTK05MK05MK01BUytOTCtOTCsnU2luY2VyZWx5LCcrTkwrJ0NyZWRpdCBSaXNrIFJldmlldyBUZWFtJytOTCsnQ3JlZGl0IERlY2lzaW9uIEludGVsbGlnZW5jZSc7fQp2YXIgRD1bCntpOicwMScsbjonU2FyYWggVGFuJyxqOidTb2Z0d2FyZSBFbmdpbmVlcicsYTozMixtOidTR0QgMTIsMDAwJyxkOicyNCBtbycscjoxOC4zLGNrOic+PSAyMDAgRE0nLHN2Oic+PSAxLDAwMCBETScsZW06Jz49IDcgeXJzJyxoczonT3duJyxoaTonRXhpc3RpbmcgcGFpZCcsczonQVBQUk9WRUQnLGY6WydMb25nLXRlcm0gZW1wbG95bWVudCBjb25maXJtcyBpbmNvbWUgY29udGludWl0eScsJ1N0cm9uZyBzYXZpbmdzIGJ1ZmZlciBwcm92aWRlcyByZXNpbGllbmNlJywnUG9zaXRpdmUgYWNjb3VudCBoaXN0b3J5IGFjcm9zcyBhbGwgaW5kaWNhdG9ycyddfSwKe2k6JzAyJyxuOidNYXJjdXMgTGVlJyxqOidSZXRpcmVkIEV4ZWN1dGl2ZScsYTo2NyxtOidTR0QgMTUsMDAwJyxkOic2MCBtbycscjo4OS40LGNrOicwLTIwMCBETScsc3Y6JzwgMTAwIERNJyxlbTonPj0gNyB5cnMnLGhzOidPd24nLGhpOidBbGwgcGFpZCcsczonREVDTElORUQnLGY6WydMb2FuIG1hdHVyZXMgYXQgYWdlIDcyLCBzZXZlbiB5ZWFycyBwYXN0IHJldGlyZW1lbnQgdGhyZXNob2xkJywnTWluaW1hbCBzYXZpbmdzIGNhbm5vdCBzdXBwb3J0IGEgNS15ZWFyIG9ibGlnYXRpb24nLCdGaXhlZCBpbmNvbWUgbGltaXRzIHJlcGF5bWVudCBmbGV4aWJpbGl0eSddLG50Om1rKCctIExvYW4gbWF0dXJlcyBzZXZlbiB5ZWFycyBwYXN0IHRoZSByZXRpcmVtZW50IHRocmVzaG9sZC4nK05MKyctIFNhdmluZ3MgYmFsYW5jZSBpcyBpbnN1ZmZpY2llbnQgZm9yIHRoaXMgY29tbWl0bWVudC4nKX0sCntpOicwMycsbjonUHJpeWEgTmFpcicsajonU2Nob29sIFRlYWNoZXInLGE6MjcsbTonU0dEIDYsMDAwJyxkOicyNCBtbycscjozMS43LGNrOicwLTIwMCBETScsc3Y6JzEwMC01MDAgRE0nLGVtOicxLTQgeXJzJyxoczonRm9yIGZyZWUnLGhpOidFeGlzdGluZyBwYWlkJyxzOidBUFBST1ZFRCcsZjpbJ1B1YmxpYy1zZWN0b3IgZW1wbG95bWVudCBwcm92aWRlcyBzdGFibGUgaW5jb21lJywnTW9kZXN0IGFtb3VudCB3aXRoaW4gcmVwYXltZW50IGNhcGFjaXR5JywnQ29uc2lzdGVudCBvbi10aW1lIHJlcGF5bWVudCByZWNvcmQnXX0sCntpOicwNCcsbjonUnlhbiBUZW8nLGo6J01hcmtldGluZyBFeGVjJyxhOjIyLG06J1NHRCA4LDAwMCcsZDonMzYgbW8nLHI6ODQuOSxjazonTm8gYWNjb3VudCcsc3Y6J05vIHNhdmluZ3MnLGVtOic8IDEgeXInLGhzOidSZW50JyxoaTonTm8gY3JlZGl0cycsczonREVDTElORUQnLGY6WydFbXBsb3ltZW50IHVuZGVyIG9uZSB5ZWFyOyBpbmNvbWUgc3RhYmlsaXR5IG5vdCBlc3RhYmxpc2hlZCcsJ05vIGNoZWNraW5nIGFjY291bnQ7IGJlaGF2aW91ciBjYW5ub3QgYmUgYXNzZXNzZWQnLCdObyBzYXZpbmdzIGJ1ZmZlciBhZ2FpbnN0IGRpc3J1cHRpb24nXSxudDptaygnLSBFbXBsb3ltZW50IHVuZGVyIG9uZSB5ZWFyOyBpbmNvbWUgc3RhYmlsaXR5IG5vdCBlc3RhYmxpc2hlZC4nK05MKyctIE5vIGNoZWNraW5nIGFjY291bnQgb24gcmVjb3JkLicrTkwrJy0gU2F2aW5ncyBwcm92aWRlIG5vIGJ1ZmZlci4nKX0sCntpOicwNScsbjonTWljaGVsbGUgT25nJyxqOidIUiBNYW5hZ2VyJyxhOjQ1LG06J1NHRCAxMCwwMDAnLGQ6JzM2IG1vJyxyOjIyLjEsY2s6JzAtMjAwIERNJyxzdjonNTAwLTEsMDAwIERNJyxlbTonNC03IHlycycsaHM6J093bicsaGk6J0V4aXN0aW5nIHBhaWQnLHM6J0FQUFJPVkVEJyxmOlsnUHJvcGVydHkgb3duZXJzaGlwIHByb3ZpZGVzIGFzc2V0IGJhY2tpbmcnLCdNaWQtY2FyZWVyIHRlbnVyZSByZWR1Y2VzIGRpc3J1cHRpb24gcmlzaycsJ0FkZXF1YXRlIHNhdmluZ3MgYnVmZmVyJ119LAp7aTonMDYnLG46J0JlbmphbWluIEtvaCcsajonU2FsZXMgRGlyZWN0b3InLGE6MzgsbTonU0dEIDE4LDAwMCcsZDonNDggbW8nLHI6NzYuMyxjazonPCAwIERNJyxzdjonPCAxMDAgRE0nLGVtOic0LTcgeXJzJyxoczonUmVudCcsaGk6J0RlbGF5ZWQgcHJldmlvdXNseScsczonREVDTElORUQnLGY6WydDaGVja2luZyBhY2NvdW50IG92ZXJkcmF3biAtIGFjdGl2ZSBmaW5hbmNpYWwgZGlzdHJlc3MnLCdQcmV2aW91cyBwYXltZW50IGRlbGF5cyBhcmUgYSBtYXRlcmlhbCByaXNrIHNpZ25hbCcsJ0Ftb3VudCBoaWdoIHJlbGF0aXZlIHRvIGZpbmFuY2lhbCBzdGFuZGluZyddLG50Om1rKCctIENoZWNraW5nIGFjY291bnQgaXMgY3VycmVudGx5IG92ZXJkcmF3bi4nK05MKyctIENyZWRpdCBoaXN0b3J5IHNob3dzIHByZXZpb3VzIHBheW1lbnQgZGVsYXlzLicrTkwrJy0gUmVxdWVzdGVkIGFtb3VudCBpcyBoaWdoIHJlbGF0aXZlIHRvIGFzc2Vzc2VkIHByb2ZpbGUuJyl9LAp7aTonMDcnLG46J0RyLiBBbml0YSBTaW5naCcsajonTWVkaWNhbCBDb25zdWx0YW50JyxhOjUyLG06J1NHRCA1LDAwMCcsZDonMTIgbW8nLHI6MTAuOCxjazonPj0gMjAwIERNJyxzdjonPj0gMSwwMDAgRE0nLGVtOic+PSA3IHlycycsaHM6J093bicsaGk6J0FsbCBwYWlkJyxzOidBUFBST1ZFRCcsZjpbJ0FsbCBwcmlvciBjcmVkaXRzIGZ1bGx5IHNldHRsZWQnLCdTZW5pb3IgcHJvZmVzc2lvbmFsIGF0IGhpZ2hlc3Qgc3RhYmlsaXR5IHRpZXInLCdTaG9ydCAxMi1tb250aCB0ZXJtIG1pbmltaXNlcyBleHBvc3VyZSddfSwKe2k6JzA4JyxuOidLZXZpbiBMaW0nLGo6J0xvZ2lzdGljcyBDb29yZGluYXRvcicsYTozNSxtOidTR0QgOSwwMDAnLGQ6JzMwIG1vJyxyOjY4LjcsY2s6J05vIGFjY291bnQnLHN2Oic8IDEwMCBETScsZW06JzEtNCB5cnMnLGhzOidSZW50JyxoaTonTm8gY3JlZGl0cycsczonREVDTElORUQnLGY6WydObyBhY2NvdW50OyBmaW5hbmNpYWwgaGlzdG9yeSB1bmF2YWlsYWJsZScsJ1NhdmluZ3MgYmVsb3cgdGhyZXNob2xkIGZvciB0aGlzIGNvbW1pdG1lbnQnLCdSZW50YWwgY29zdHMgcmVkdWNlIGRpc3Bvc2FibGUgaW5jb21lJ10sbnQ6bWsoJy0gTm8gY2hlY2tpbmcgYWNjb3VudCBtZWFucyBiZWhhdmlvdXIgY2Fubm90IGJlIGFzc2Vzc2VkLicrTkwrJy0gU2F2aW5ncyBiZWxvdyByZXF1aXJlZCBsZXZlbC4nK05MKyctIFJlbnRhbCBjb3N0cyByZWR1Y2UgZGlzcG9zYWJsZSBpbmNvbWUuJyl9LAp7aTonMDknLG46J0phbWVzIFRhbicsajonQ2l2aWwgU2VydmFudCcsYTo0MCxtOidTR0QgNyw1MDAnLGQ6JzE4IG1vJyxyOjE0LjIsY2s6Jz49IDIwMCBETScsc3Y6JzUwMC0xLDAwMCBETScsZW06Jz49IDcgeXJzJyxoczonT3duJyxoaTonQWxsIHBhaWQnLHM6J0FQUFJPVkVEJyxmOlsnR292ZXJubWVudCBlbXBsb3ltZW50OiBtYXhpbXVtIHN0YWJpbGl0eScsJ1Nob3J0IDE4LW1vbnRoIHRlcm0gbWluaW1pc2VzIHJpc2snLCdTdHJvbmcgY2hlY2tpbmcgYW5kIHNhdmluZ3MgcHJvZmlsZSddfSwKe2k6JzEwJyxuOidNZWkgTGluIENodWEnLGo6J0ZyZWVsYW5jZSBEZXNpZ25lcicsYToyOSxtOidTR0QgMTEsMDAwJyxkOiczNiBtbycscjo3MS41LGNrOic8IDAgRE0nLHN2OicxMDAtNTAwIERNJyxlbTonPCAxIHlyJyxoczonUmVudCcsaGk6J05vIGNyZWRpdHMnLHM6J0RFQ0xJTkVEJyxmOlsnT3ZlcmRyYXduIGFjY291bnQgaW5kaWNhdGVzIGZpbmFuY2lhbCBzdHJlc3MnLCdGcmVlbGFuY2UgdGVudXJlIHVuZGVyIDEgeWVhcjsgaW5jb21lIGlycmVndWxhcicsJ05vIGNyZWRpdCBoaXN0b3J5IGxpbWl0cyByZXBheW1lbnQgYXNzZXNzbWVudCddLG50Om1rKCctIENoZWNraW5nIGFjY291bnQgaXMgb3ZlcmRyYXduLicrTkwrJy0gRnJlZWxhbmNlIHRlbnVyZSB1bmRlciBvbmUgeWVhcjsgaW5jb21lIGNhbm5vdCBiZSByZWxpYWJseSBwcm9qZWN0ZWQuJytOTCsnLSBObyBjcmVkaXQgaGlzdG9yeSBvbiByZWNvcmQuJyl9LAp7aTonMTEnLG46J0RhdmlkIE5nJyxqOidCYW5rIE1hbmFnZXInLGE6NDgsbTonU0dEIDIwLDAwMCcsZDonNDggbW8nLHI6MjkuNCxjazonPj0gMjAwIERNJyxzdjonPj0gMSwwMDAgRE0nLGVtOic+PSA3IHlycycsaHM6J093bicsaGk6J0V4aXN0aW5nIHBhaWQnLHM6J0FQUFJPVkVEJyxmOlsnU2VuaW9yIGZpbmFuY2lhbCBzZWN0b3Igcm9sZSB3aXRoIGxvbmcgdGVudXJlJywnU3Ryb25nIHNhdmluZ3MgcmVsYXRpdmUgdG8gbG9hbiBhbW91bnQnLCdDb25zaXN0ZW50IHJlcGF5bWVudCB0cmFjayByZWNvcmQnXX0sCntpOicxMicsbjonU2l0aSBSYWhpbWFoJyxqOidBZG1pbiBFeGVjdXRpdmUnLGE6MzMsbTonU0dEIDQsNTAwJyxkOicxMiBtbycscjoyNi44LGNrOicwLTIwMCBETScsc3Y6JzEwMC01MDAgRE0nLGVtOic0LTcgeXJzJyxoczonRm9yIGZyZWUnLGhpOidFeGlzdGluZyBwYWlkJyxzOidBUFBST1ZFRCcsZjpbJ1Nob3J0IDEyLW1vbnRoIHRlcm0gcmVkdWNlcyBleHBvc3VyZScsJ0VzdGFibGlzaGVkIGVtcGxveW1lbnQgY29uZmlybXMgc3RhYmlsaXR5JywnT24tdGltZSByZXBheW1lbnQgaGlzdG9yeSddfSwKe2k6JzEzJyxuOidBYXJvbiBZZW8nLGo6J1JpZGUtaGFpbGluZyBEcml2ZXInLGE6NDQsbTonU0dEIDEzLDAwMCcsZDonNDIgbW8nLHI6NzkuMSxjazonTm8gYWNjb3VudCcsc3Y6JzwgMTAwIERNJyxlbTonMS00IHlycycsaHM6J1JlbnQnLGhpOidEZWxheWVkIHByZXZpb3VzbHknLHM6J0RFQ0xJTkVEJyxmOlsnTm8gYWNjb3VudCBwcmV2ZW50cyB0cmFuc2FjdGlvbiBhc3Nlc3NtZW50JywnUHJldmlvdXMgbGF0ZSBwYXltZW50cyBhcmUgc2lnbmlmaWNhbnQnLCdHaWcgaW5jb21lIGlzIHZhcmlhYmxlIG92ZXIgNDIgbW9udGhzJ10sbnQ6bWsoJy0gTm8gY2hlY2tpbmcgYWNjb3VudCBvbiByZWNvcmQuJytOTCsnLSBDcmVkaXQgaGlzdG9yeSBzaG93cyBwcmV2aW91cyBwYXltZW50IGRlbGF5cy4nK05MKyctIEdpZy1lY29ub215IGluY29tZSBpcyBpbmhlcmVudGx5IHZhcmlhYmxlLicpfSwKe2k6JzE0JyxuOidHcmFjZSBXb25nJyxqOidQaGFybWFjaXN0JyxhOjM2LG06J1NHRCA4LDAwMCcsZDonMjQgbW8nLHI6MTYuOSxjazonPj0gMjAwIERNJyxzdjonPj0gMSwwMDAgRE0nLGVtOic0LTcgeXJzJyxoczonT3duJyxoaTonRXhpc3RpbmcgcGFpZCcsczonQVBQUk9WRUQnLGY6WydSZWd1bGF0ZWQgaGVhbHRoY2FyZSBpbmNvbWUgd2l0aCBzdHJvbmcgc3RhYmlsaXR5JywnUHJvcGVydHkgb3duZXJzaGlwIGFkZHMgYXNzZXQgYmFja2luZycsJ0NvbnNpc3RlbnQgcmVwYXltZW50IGhpc3RvcnknXX0sCntpOicxNScsbjonUmF2aSBLdW1hcicsajonQ29uc3RydWN0aW9uIFN1cGVydmlzb3InLGE6NTUsbTonU0dEIDE2LDAwMCcsZDonNjAgbW8nLHI6ODIuNixjazonMC0yMDAgRE0nLHN2Oic8IDEwMCBETScsZW06JzQtNyB5cnMnLGhzOidSZW50JyxoaTonTm8gY3JlZGl0cycsczonREVDTElORUQnLGY6WydMb2FuIG1hdHVyZXMgYXQgYWdlIDYwLCBhcHByb2FjaGluZyBpbmNvbWUgdHJhbnNpdGlvbicsJ01pbmltYWwgc2F2aW5ncyBmb3IgNS15ZWFyIGNvbW1pdG1lbnQnLCdObyBjcmVkaXQgaGlzdG9yeSBsaW1pdHMgYXNzZXNzbWVudCddLG50Om1rKCctIExvYW4gbWF0dXJlcyBhdCBhZ2UgNjAsIGEgcGVyaW9kIG9mIHBvdGVudGlhbCBpbmNvbWUgdHJhbnNpdGlvbi4nK05MKyctIFNhdmluZ3MgaW5zdWZmaWNpZW50IGZvciB0aGlzIGNvbW1pdG1lbnQuJytOTCsnLSBObyBwcmlvciBjcmVkaXQgaGlzdG9yeS4nKX0sCntpOicxNicsbjonTGluZGEgSG8nLGo6J0FjY291bnRhbnQnLGE6MzEsbTonU0dEIDksNTAwJyxkOiczMCBtbycscjoyNC4zLGNrOicwLTIwMCBETScsc3Y6JzUwMC0xLDAwMCBETScsZW06JzQtNyB5cnMnLGhzOidGb3IgZnJlZScsaGk6J0FsbCBwYWlkJyxzOidBUFBST1ZFRCcsZjpbJ0ZpbmFuY2UgcHJvZmVzc2lvbmFsIHdpdGggc3RhYmxlIHByb2ZpbGUnLCdBbGwgY3JlZGl0cyBmdWxseSBzZXR0bGVkJywnTG9hbiBwcm9wb3J0aW9uYXRlIHRvIGZpbmFuY2lhbCBzdGFuZGluZyddfSwKe2k6JzE3JyxuOidBaG1hZCBGYWR6bGknLGo6J0lUIFN5c3RlbXMgTWFuYWdlcicsYTo0MyxtOidTR0QgMTQsMDAwJyxkOiczNiBtbycscjoyMC41LGNrOic+PSAyMDAgRE0nLHN2Oic+PSAxLDAwMCBETScsZW06Jz49IDcgeXJzJyxoczonT3duJyxoaTonRXhpc3RpbmcgcGFpZCcsczonQVBQUk9WRUQnLGY6WydMb25nLXRlbnVyZWQgSVQgcm9sZSB3aXRoIHN0YWJsZSBpbmNvbWUnLCdTdHJvbmcgc2F2aW5ncyBhbmQgY2hlY2tpbmcgYmFsYW5jZScsJ09uLXRpbWUgcmVwYXltZW50IHdpdGggbm8gYWR2ZXJzZSBzaWduYWxzJ119LAp7aTonMTgnLG46J0phc21pbmUgTG9oJyxqOidQYXJ0LXRpbWUgUmV0YWlsJyxhOjI0LG06J1NHRCA2LDUwMCcsZDonMzAgbW8nLHI6NzcuNCxjazonTm8gYWNjb3VudCcsc3Y6JzwgMTAwIERNJyxlbTonPCAxIHlyJyxoczonUmVudCcsaGk6J05vIGNyZWRpdHMnLHM6J0RFQ0xJTkVEJyxmOlsnUGFydC10aW1lIGluY29tZSBpbnN1ZmZpY2llbnQgZm9yIHJlcGF5bWVudCcsJ05vIGFjY291bnQ7IGJlaGF2aW91ciB1bmtub3duJywnTm8gY3JlZGl0IGhpc3RvcnkgYW5kIG1pbmltYWwgc2F2aW5ncyddLG50Om1rKCctIFBhcnQtdGltZSBlbXBsb3ltZW50IGRvZXMgbm90IHByb3ZpZGUgcmVxdWlyZWQgaW5jb21lIHN0YWJpbGl0eS4nK05MKyctIE5vIGNoZWNraW5nIGFjY291bnQgb24gcmVjb3JkLicrTkwrJy0gU2F2aW5ncyBwcm92aWRlIG5vIGJ1ZmZlci4nKX0sCntpOicxOScsbjonUGV0ZXIgQ2hhbicsajonU2VuaW9yIExhd3llcicsYTo0MixtOidTR0QgMjUsMDAwJyxkOic0OCBtbycscjoxMy43LGNrOic+PSAyMDAgRE0nLHN2Oic+PSAxLDAwMCBETScsZW06Jz49IDcgeXJzJyxoczonT3duJyxoaTonQWxsIHBhaWQnLHM6J0FQUFJPVkVEJyxmOlsnTGVnYWwgcHJvZmVzc2lvbmFsIHdpdGggcHJlbWl1bSBzdGFibGUgaW5jb21lJywnRXhjZWxsZW50IHNhdmluZ3MgcmVsYXRpdmUgdG8gbG9hbiBhbW91bnQnLCdGbGF3bGVzcyByZXBheW1lbnQgYWNyb3NzIGFsbCBwcmlvciBjcmVkaXRzJ119LAp7aTonMjAnLG46J051cnVsIEFpc3lhaCcsajonVW5lbXBsb3llZCcsYTozMCxtOidTR0QgNywwMDAnLGQ6JzI0IG1vJyxyOjkzLjEsY2s6J05vIGFjY291bnQnLHN2OidObyBzYXZpbmdzJyxlbTonVW5lbXBsb3llZCcsaHM6J1JlbnQnLGhpOidEZWxheWVkIHByZXZpb3VzbHknLHM6J0RFQ0xJTkVEJyxmOlsnTm8gaW5jb21lOyByZXBheW1lbnQgY2FwYWNpdHkgbm90IGVzdGFibGlzaGVkJywnTm8gYWNjb3VudCBvciBzYXZpbmdzIG9uIHJlY29yZCcsJ1ByZXZpb3VzIGRlbGF5cyBjb21wb3VuZCB0aGUgcmlzayBwcm9maWxlJ10sbnQ6bWsoJy0gTm8gZW1wbG95bWVudCBtZWFucyByZXBheW1lbnQgY2FwYWNpdHkgY2Fubm90IGJlIGFzc2Vzc2VkLicrTkwrJy0gTm8gY2hlY2tpbmcgYWNjb3VudCBvciBzYXZpbmdzIG9uIHJlY29yZC4nK05MKyctIENyZWRpdCBoaXN0b3J5IHNob3dzIHByZXZpb3VzIHBheW1lbnQgZGVsYXlzLicpfSwKe2k6JzIxJyxuOidXZWkgSmllIFRhbicsajonUmVzZWFyY2ggU2NpZW50aXN0JyxhOjM0LG06J1NHRCA4LDUwMCcsZDonMjQgbW8nLHI6MTkuOCxjazonMC0yMDAgRE0nLHN2Oic1MDAtMSwwMDAgRE0nLGVtOic0LTcgeXJzJyxoczonRm9yIGZyZWUnLGhpOidFeGlzdGluZyBwYWlkJyxzOidBUFBST1ZFRCcsZjpbJ1Jlc2VhcmNoIGluc3RpdHV0aW9uIGVtcGxveW1lbnQgaXMgcmVsaWFibGUnLCdNb2Rlc3QgbG9hbiByZWxhdGl2ZSB0byBwcm9mZXNzaW9uYWwgcHJvZmlsZScsJ0NvbnNpc3RlbnQgcmVwYXltZW50IGJlaGF2aW91ciddfQpdOwpmdW5jdGlvbiBidWlsZCgpewp2YXIgb2s9MCxubz0wOwpmb3IodmFyIGk9MDtpPEQubGVuZ3RoO2krKyl7aWYoRFtpXS5zPT09J0FQUFJPVkVEJylvaysrO2Vsc2Ugbm8rKzt9CmRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzb2snKS50ZXh0Q29udGVudD0n4pyTICcrb2srJyBBcHByb3ZlZCc7CmRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzbm8nKS50ZXh0Q29udGVudD0n4pyVICcrbm8rJyBEZWNsaW5lZCc7CmRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzdG90JykudGV4dENvbnRlbnQ9RC5sZW5ndGgrJyBUb3RhbCc7CnZhciBoPScnOwpmb3IodmFyIGk9MDtpPEQubGVuZ3RoO2krKyl7CnZhciBhPURbaV0sYz1yYyhhLnIpLGRlYz0oYS5zPT09J0RFQ0xJTkVEJyk7CmgrPSc8ZGl2IGNsYXNzPSJyb3ciIGRhdGEtaWQ9IicrYS5pKyciIG9ubW91c2VvdmVyPSJob3YodGhpcykiIG9ubW91c2VvdXQ9Im91dCgpIiBvbmNsaWNrPSJwaWNrKHRoaXMpIj4nOwpoKz0nPGRpdiBjbGFzcz0icmVmIj4nK2EuaSsnPC9kaXY+JzsKaCs9JzxkaXYgc3R5bGU9Im92ZXJmbG93OmhpZGRlbiI+PGRpdiBjbGFzcz0ibm0iPicrYS5uKyc8L2Rpdj48ZGl2IGNsYXNzPSJqYiI+JythLmorJzwvZGl2PjwvZGl2Pic7CmgrPSc8ZGl2IHN0eWxlPSJmb250LXNpemU6MTFweDtjb2xvcjojOTRhM2I4Ij4nK2EuYSsnPC9kaXY+JzsKaCs9JzxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMXB4O2NvbG9yOiM5NGEzYjgiPicrYS5tKyc8L2Rpdj4nOwpoKz0nPGRpdiBzdHlsZT0iZm9udC1zaXplOjExcHg7Y29sb3I6Izk0YTNiOCI+JythLmQrJzwvZGl2Pic7CmgrPSc8ZGl2PjxkaXYgY2xhc3M9InJiIj48ZGl2IGNsYXNzPSJyZCIgc3R5bGU9ImJhY2tncm91bmQ6JytjKyciPjwvZGl2PjxzcGFuIHN0eWxlPSJjb2xvcjonK2MrJztmb250LXNpemU6MTFweDtmb250LXdlaWdodDo3MDAiPicrYS5yLnRvRml4ZWQoMSkrJyU8L3NwYW4+PC9kaXY+PC9kaXY+JzsKaCs9JzxkaXY+PHNwYW4gY2xhc3M9InBpbGwgJysoZGVjPydubyc6J29rJykrJyI+JysoZGVjPyfinJUnOifinJMnKSsnICcrYS5zKyc8L3NwYW4+PC9kaXY+JzsKaCs9JzwvZGl2Pic7fQpkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndGInKS5pbm5lckhUTUw9aDt9CmZ1bmN0aW9uIGRldChpZCl7CnZhciBhPW51bGw7Zm9yKHZhciBpPTA7aTxELmxlbmd0aDtpKyspe2lmKERbaV0uaT09PWlkKXthPURbaV07YnJlYWs7fX1pZighYSlyZXR1cm47CnZhciBkZWM9KGEucz09PSdERUNMSU5FRCcpLGM9cmMoYS5yKTsKdmFyIHNiZz1kZWM/J3JnYmEoMjIwLDM4LDM4LDAuMDgpJzoncmdiYSgyMiwxNjMsNzQsMC4wOCknOwp2YXIgc2JyPWRlYz8ncmdiYSgyMjAsMzgsMzgsMC4xOCknOidyZ2JhKDIyLDE2Myw3NCwwLjE4KSc7CnZhciBzYz1kZWM/JyNmODcxNzEnOicjNGFkZTgwJyxzeW09ZGVjPyfinJUnOifinJMnOwp2YXIgZmg9Jyc7Zm9yKHZhciBqPTA7ajxhLmYubGVuZ3RoO2orKyl7ZmgrPSc8ZGl2IGNsYXNzPSJmaSI+JythLmZbal0rJzwvZGl2Pic7fQp2YXIgaD0nPGRpdiBjbGFzcz0iZG4iPicrYS5uKyc8L2Rpdj4nCisnPGRpdiBjbGFzcz0iZG0iPicrYS5qKycgJm1pZGRvdDsgQWdlICcrYS5hKycgJm1pZGRvdDsgJythLmQrJzwvZGl2PicKKyc8ZGl2IGNsYXNzPSJkc2IiIHN0eWxlPSJiYWNrZ3JvdW5kOicrc2JnKyc7Ym9yZGVyOjFweCBzb2xpZCAnK3NicisnIj4nCisnPHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxNHB4Ij4nK3N5bSsnPC9zcGFuPicKKyc8ZGl2PjxkaXYgc3R5bGU9ImZvbnQtc2l6ZToxMC41cHg7Zm9udC13ZWlnaHQ6NzAwO2NvbG9yOicrc2MrJztsZXR0ZXItc3BhY2luZzouMDVlbSI+JythLnMrJzwvZGl2PicKKyc8ZGl2IHN0eWxlPSJmb250LXNpemU6OXB4O2NvbG9yOiM0NzU1NjkiPkRlZmF1bHQgcHJvYjogPHN0cm9uZyBzdHlsZT0iY29sb3I6JytjKyciPicrYS5yLnRvRml4ZWQoMSkrJyU8L3N0cm9uZz48L2Rpdj48L2Rpdj48L2Rpdj4nCisnPGRpdiBjbGFzcz0icm0iPjxkaXYgY2xhc3M9InJmIiBzdHlsZT0id2lkdGg6JytNYXRoLm1pbihhLnIsMTAwKSsnJTtiYWNrZ3JvdW5kOicrYysnIj48L2Rpdj48L2Rpdj4nCisnPGRpdiBjbGFzcz0ia3YiPicKKyc8ZGl2IGNsYXNzPSJrdmkiPkNoZWNraW5nIDxzcGFuPicrYS5jaysnPC9zcGFuPjwvZGl2PicKKyc8ZGl2IGNsYXNzPSJrdmkiPlNhdmluZ3MgPHNwYW4+JythLnN2Kyc8L3NwYW4+PC9kaXY+JworJzxkaXYgY2xhc3M9Imt2aSI+RW1wbG95bWVudCA8c3Bhbj4nK2EuZW0rJzwvc3Bhbj48L2Rpdj4nCisnPGRpdiBjbGFzcz0ia3ZpIj5Ib3VzaW5nIDxzcGFuPicrYS5ocysnPC9zcGFuPjwvZGl2PicKKyc8L2Rpdj4nCisnPGRpdiBjbGFzcz0ic2wiPicrKGRlYz8nRGVjbGluZSBGYWN0b3JzJzonQXBwcm92YWwgRmFjdG9ycycpKyc8L2Rpdj4nK2ZoOwppZihkZWMmJmEubnQpe2grPSc8ZGl2IGNsYXNzPSJzbCIgc3R5bGU9Im1hcmdpbi10b3A6N3B4Ij5DcmVkaXQgRGVjaXNpb24gTm90aWNlPC9kaXY+PGRpdiBjbGFzcz0ibmIiPicrYS5udCsnPC9kaXY+Jzt9CmlmKCFkZWMpe2grPSc8ZGl2IGNsYXNzPSJhYiIgc3R5bGU9Im1hcmdpbi10b3A6N3B4Ij48c3Ryb25nPuKckyBBcHBsaWNhdGlvbiBBcHByb3ZlZDwvc3Ryb25nPjxkaXYgc3R5bGU9ImZvbnQtc2l6ZTo5LjVweDtvcGFjaXR5Oi44O21hcmdpbi10b3A6MnB4Ij5ObyBhZHZlcnNlIGFjdGlvbiBub3RpY2UgcmVxdWlyZWQuPC9kaXY+PC9kaXY+Jzt9CmRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdkcCcpLmlubmVySFRNTD1oO30KZnVuY3Rpb24gaG92KGVsKXtkZXQoZWwuZ2V0QXR0cmlidXRlKCdkYXRhLWlkJykpO30KZnVuY3Rpb24gb3V0KCl7aWYobG9ja2VkKXtkZXQobG9ja2VkKTt9ZWxzZXtkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZHAnKS5pbm5lckhUTUw9JzxkaXYgY2xhc3M9InBoIj48ZGl2IHN0eWxlPSJmb250LXNpemU6MjJweDtvcGFjaXR5Oi4yIj4mIzk2NzI7PC9kaXY+PGRpdj5Ib3ZlciBvciBjbGljayBhIHJvdzxicj50byB2aWV3IGRldGFpbHM8L2Rpdj48L2Rpdj4nO319CmZ1bmN0aW9uIHBpY2soZWwpe2xvY2tlZD1lbC5nZXRBdHRyaWJ1dGUoJ2RhdGEtaWQnKTt2YXIgcm93cz1kb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcucm93Jyk7Zm9yKHZhciBpPTA7aTxyb3dzLmxlbmd0aDtpKyspe3Jvd3NbaV0uY2xhc3NMaXN0LnJlbW92ZSgnc2VsJyk7fWVsLmNsYXNzTGlzdC5hZGQoJ3NlbCcpO2RldChsb2NrZWQpO30KYnVpbGQoKTsKPC9zY3JpcHQ+PC9ib2R5PjwvaHRtbD4=").decode("utf-8")
components.html(PIPELINE_HTML, height=970, scrolling=False)
