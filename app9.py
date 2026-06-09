import streamlit as st
import os
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from modelo_termosolar import simular_planta, calculate_financial_metrics

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="CSP-CAES Simulator", layout="wide")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    h1 {color: #F49000;}
    h2, h3 {color: #6B91DD;}
    [data-testid="stMetricValue"] {color: #6B91DD;}
    </style>
    """, unsafe_allow_html=True)

# ─── Loading animation HTML ───────────────────────────────────────────────────
# Rendered as a fixed full-screen overlay (position:fixed, z-index:9999) so it
# is always centred in the viewport regardless of how far the user has scrolled.
# The semi-transparent white backdrop dims the page content behind it.
# The SVG is the updated animation from animacion1.txt, IDs suffixed with "3"
# to avoid any conflicts with other SVG elements on the page.
LOADING_ANIMATION = """
<div id="csp-overlay" style="
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    background: rgba(248,249,250,0.88);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 9999;">
  <div style="
      background: #ffffff;
      border-radius: 16px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.12);
      padding: 2rem 2.5rem;
      display: flex;
      flex-direction: column;
      align-items: center;
      max-width: 820px;
      width: 90%;">
    <style>
      .ray3  { stroke-dasharray: 8 12; animation: dash3 1.5s linear infinite; }
      @keyframes dash3 { to { stroke-dashoffset: -20; } }
      .thermal-pulse3 { animation: pulse3 2s infinite alternate ease-in-out; }
      @keyframes pulse3 {
        0%   { fill: #EF8E0B; stroke: #EF8E0B; filter: drop-shadow(0 0 2px #EF8E0B); }
        100% { fill: #FFA726; stroke: #FFA726; filter: drop-shadow(0 0 8px #FFA726); }
      }
      .air-pulse3 { animation: air3 2s infinite alternate ease-in-out; }
      .air-pulse3-async { animation: air3 2s infinite alternate ease-in-out; animation-delay: -2s; }
      @keyframes air3 {
        0%   { opacity: 0.4; filter: drop-shadow(0 0 2px #6B91DD); }
        100% { opacity: 0.8; filter: drop-shadow(0 0 6px #6B91DD); }
      }
      .flow3      { stroke-dasharray: 8 8; animation: flow3-anim 1.5s linear infinite; }
      @keyframes flow3-anim { to { stroke-dashoffset: -16; } }
      .flow3-down { stroke-dasharray: 8 8; animation: flow3-down 1s linear infinite; }
      @keyframes flow3-down { to { stroke-dashoffset: -16; } }
      .flow3-up   { stroke-dasharray: 8 8; animation: flow3-up 1s linear infinite; }
      @keyframes flow3-up   { to { stroke-dashoffset: 16; } }
      .spin3 { animation: spin3 2s linear infinite; transform-origin: 550px 240px; }
      @keyframes spin3 { 100% { transform: rotate(360deg); } }
      .elec-pulse3 { animation: elec-pulse3 1.5s infinite alternate ease-in-out; }
      @keyframes elec-pulse3 {
        0%   { opacity: 0.6; filter: drop-shadow(0 0 2px #ffca28); }
        100% { opacity: 1;   filter: drop-shadow(0 0 8px #ffca28); }
      }
      .elec-flow3 { stroke-dasharray: 8 16; animation: elec3-anim 2.5s ease-in-out infinite alternate; }
      @keyframes elec3-anim {
        0%   { stroke-dashoffset:  48; }
        100% { stroke-dashoffset: -48; }
      }
    </style>
    <svg width="100%" style="max-width:750px;" height="320" viewBox="0 0 750 360" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <filter id="shadow3" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="2" dy="5" stdDeviation="4" flood-color="#000000" flood-opacity="0.15"/>
        </filter>
        <linearGradient id="towerGrad3" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stop-color="#b0b5be" />
          <stop offset="50%"  stop-color="#e2e6eb" />
          <stop offset="100%" stop-color="#8d949e" />
        </linearGradient>
        <linearGradient id="tankGrad3" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stop-color="#9aa0ab" />
          <stop offset="50%"  stop-color="#ccd2db" />
          <stop offset="100%" stop-color="#7a818c" />
        </linearGradient>
        <radialGradient id="sunGrad3" cx="50%" cy="50%" r="50%">
          <stop offset="0%"   stop-color="#ffffff" />
          <stop offset="40%"  stop-color="#ffca28" />
          <stop offset="100%" stop-color="#EF8E0B" />
        </radialGradient>
      </defs>
      <!-- Ground -->
      <line x1="30" y1="270" x2="750" y2="270" stroke="#cbd0d8" stroke-width="4" stroke-linecap="round" />
      <!-- Sun -->
      <circle cx="50" cy="50" r="25" fill="url(#sunGrad3)" filter="url(#shadow3)" />
      <circle cx="50" cy="50" r="38" fill="none" stroke-width="0.5" class="thermal-pulse3" opacity="0.5"/>
      <!-- Thermal pipe: Tower -> TES -->
      <path d="M 245 90 L 245 140 L 490 140 L 490 190" fill="none" stroke="#EF8E0B" stroke-width="8" stroke-linecap="round" stroke-linejoin="round" />
      <path d="M 245 90 L 245 140 L 490 140 L 490 190" fill="none" stroke="#FFE0B2" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="flow3" />
      <!-- Tower -->
      <polygon points="230,270 270,270 260,90 240,90" fill="url(#towerGrad3)" filter="url(#shadow3)" />
      <line x1="250" y1="90" x2="250" y2="270" stroke="#717885" stroke-width="1.5" opacity="0.6" />
      <rect x="238" y="86" width="24" height="4" fill="#a0a5af" />
      <rect x="240" y="60" width="20" height="26" rx="2" class="thermal-pulse3" />
      <rect x="238" y="56" width="24" height="4" fill="#a0a5af" />
      <!-- Heliostat pedestals -->
      <rect x="93"  y="254" width="4" height="16" fill="#8d949e" />
      <rect x="173" y="264" width="4" height="6"  fill="#8d949e" />
      <rect x="323" y="264" width="4" height="6"  fill="#8d949e" />
      <rect x="403" y="254" width="4" height="16" fill="#8d949e" />
      <!-- Heliostats -->
      <rect x="70"  y="250" width="50" height="8" rx="1" transform="rotate(25 95 254)"   fill="#6B91DD" stroke="#4A6AA8" stroke-width="1" filter="url(#shadow3)" />
      <rect x="150" y="260" width="50" height="8" rx="1" transform="rotate(10 175 264)"  fill="#6B91DD" stroke="#4A6AA8" stroke-width="1" filter="url(#shadow3)" />
      <rect x="300" y="260" width="50" height="8" rx="1" transform="rotate(-10 325 264)" fill="#6B91DD" stroke="#4A6AA8" stroke-width="1" filter="url(#shadow3)" />
      <rect x="380" y="250" width="50" height="8" rx="1" transform="rotate(-25 405 254)" fill="#6B91DD" stroke="#4A6AA8" stroke-width="1" filter="url(#shadow3)" />
      <!-- Solar rays (incident) -->
      <line x1="50" y1="50" x2="95"  y2="254" stroke="#ffca28" stroke-width="2" opacity="0.6" class="ray3" />
      <line x1="50" y1="50" x2="175" y2="264" stroke="#ffca28" stroke-width="2" opacity="0.6" class="ray3" />
      <line x1="50" y1="50" x2="325" y2="264" stroke="#ffca28" stroke-width="2" opacity="0.6" class="ray3" />
      <line x1="50" y1="50" x2="405" y2="254" stroke="#ffca28" stroke-width="2" opacity="0.6" class="ray3" />
      <!-- Solar rays (reflected) -->
      <line x1="95"  y1="254" x2="250" y2="75" stroke="#ffca28" stroke-width="3" class="ray3" />
      <line x1="175" y1="264" x2="250" y2="75" stroke="#ffca28" stroke-width="3" class="ray3" />
      <line x1="325" y1="264" x2="250" y2="75" stroke="#ffca28" stroke-width="3" class="ray3" />
      <line x1="405" y1="254" x2="250" y2="75" stroke="#ffca28" stroke-width="3" class="ray3" />
      <!-- TES tank -->
      <rect x="465" y="190" width="50" height="80" rx="3" fill="url(#tankGrad3)" filter="url(#shadow3)" />
      <rect x="468" y="220" width="44" height="48" rx="1" class="thermal-pulse3" />
      <rect x="465" y="185" width="50" height="5"  rx="2" fill="#a0a5af" />
      <!-- Thermal pipe: TES -> Generator -->
      <path d="M 490 240 L 525 240" fill="none" stroke="#EF8E0B" stroke-width="8" stroke-linecap="round" />
      <path d="M 490 240 L 525 240" fill="none" stroke="#FFE0B2" stroke-width="2.5" stroke-linecap="round" class="flow3" />
      <!-- CAES cavern -->
      <path d="M 530 270 L 530 280 Q 500 285 490 310 Q 480 345 515 350 Q 550 355 575 345 Q 610 330 600 300 Q 590 285 570 280 L 570 270 Z"
            fill="#e2e6eb" stroke="#a0a5af" stroke-width="2" />
      <path d="M 535 285 Q 510 290 500 312 Q 492 340 520 343 Q 550 347 568 338 Q 598 325 590 302 Q 582 290 565 285 Z"
            fill="#6B91DD" class="air-pulse3" />
      <!-- CAES bidirectional pipes -->
      <path d="M 540 265 L 540 295" fill="none" stroke="#6B91DD" stroke-width="8" stroke-linecap="round" class="air-pulse3" />
      <path d="M 540 265 L 540 295" fill="none" stroke="#D0E0FF" stroke-width="2.5" stroke-linecap="round" class="flow3-down" />
      <path d="M 560 265 L 560 295" fill="none" stroke="#6B91DD" stroke-width="8" stroke-linecap="round" class="air-pulse3-async" />
      <path d="M 560 265 L 560 295" fill="none" stroke="#D0E0FF" stroke-width="2.5" stroke-linecap="round" class="flow3-up" />
      <!-- Generator / turbine building -->
      <rect x="525" y="210" width="50" height="60" rx="3" fill="url(#towerGrad3)" filter="url(#shadow3)" />
      <rect x="522" y="205" width="56" height="5"  rx="2" fill="#8d949e" />
      <!-- Spinning turbine -->
      <circle cx="550" cy="240" r="16" fill="#e2e6eb" />
      <circle cx="550" cy="240" r="13" fill="#a0a5af" />
      <g class="spin3">
        <rect x="548" y="228" width="4" height="24" rx="1" fill="#4b5563" />
        <rect x="538" y="238" width="24" height="4" rx="1" fill="#4b5563" />
        <circle cx="550" cy="240" r="3" fill="#1f2937" />
      </g>
      <!-- Transformer -->
      <rect x="585" y="240" width="16" height="30" rx="2" fill="#7a818c" filter="url(#shadow3)" />
      <line x1="585" y1="247" x2="601" y2="247" stroke="#4b5563" stroke-width="1" />
      <line x1="585" y1="254" x2="601" y2="254" stroke="#4b5563" stroke-width="1" />
      <line x1="585" y1="261" x2="601" y2="261" stroke="#4b5563" stroke-width="1" />
      <line x1="575" y1="265" x2="585" y2="265" stroke="#a0a5af" stroke-width="3" />
      <!-- Transmission tower (lattice) -->
      <polyline points="640,270 660,70 680,270" fill="none" stroke="#8d949e" stroke-width="2" filter="url(#shadow3)" />
      <line x1="644" y1="230" x2="676" y2="230" stroke="#8d949e" stroke-width="1.5" />
      <line x1="648" y1="190" x2="672" y2="190" stroke="#8d949e" stroke-width="1.5" />
      <line x1="652" y1="150" x2="668" y2="150" stroke="#8d949e" stroke-width="1.5" />
      <line x1="656" y1="110" x2="664" y2="110" stroke="#8d949e" stroke-width="1.5" />
      <line x1="640" y1="270" x2="676" y2="230" stroke="#8d949e" stroke-width="1" />
      <line x1="680" y1="270" x2="644" y2="230" stroke="#8d949e" stroke-width="1" />
      <line x1="644" y1="230" x2="672" y2="190" stroke="#8d949e" stroke-width="1" />
      <line x1="676" y1="230" x2="648" y2="190" stroke="#8d949e" stroke-width="1" />
      <line x1="648" y1="190" x2="668" y2="150" stroke="#8d949e" stroke-width="1" />
      <line x1="672" y1="190" x2="652" y2="150" stroke="#8d949e" stroke-width="1" />
      <line x1="652" y1="150" x2="664" y2="110" stroke="#8d949e" stroke-width="1" />
      <line x1="668" y1="150" x2="656" y2="110" stroke="#8d949e" stroke-width="1" />
      <line x1="620" y1="150" x2="700" y2="150" stroke="#8d949e" stroke-width="2" />
      <line x1="640" y1="110" x2="680" y2="110" stroke="#8d949e" stroke-width="2" />
      <line x1="620" y1="150" x2="620" y2="160" stroke="#e2e6eb" stroke-width="2" />
      <line x1="660" y1="110" x2="660" y2="120" stroke="#e2e6eb" stroke-width="2" />
      <line x1="700" y1="150" x2="700" y2="160" stroke="#e2e6eb" stroke-width="2" />
      <!-- Transmission cables (background) -->
      <path d="M 595 242 Q 605 200 620 160" fill="none" stroke="#a0a5af" stroke-width="2" />
      <path d="M 595 242 Q 625 180 660 120" fill="none" stroke="#a0a5af" stroke-width="2" />
      <path d="M 595 242 Q 645 200 700 160" fill="none" stroke="#a0a5af" stroke-width="2" />
      <path d="M 620 160 Q 685 190 750 160" fill="none" stroke="#a0a5af" stroke-width="2" />
      <path d="M 660 120 Q 705 150 750 120" fill="none" stroke="#a0a5af" stroke-width="2" />
      <path d="M 700 160 Q 725 180 750 170" fill="none" stroke="#a0a5af" stroke-width="2" />
      <!-- Electric flow animation -->
      <path d="M 595 242 Q 605 200 620 160" fill="none" stroke="#ffca28" stroke-width="3" class="elec-flow3" />
      <path d="M 595 242 Q 625 180 660 120" fill="none" stroke="#ffca28" stroke-width="3" class="elec-flow3" />
      <path d="M 595 242 Q 645 200 700 160" fill="none" stroke="#ffca28" stroke-width="3" class="elec-flow3" />
      <path d="M 620 160 Q 685 190 750 160" fill="none" stroke="#ffca28" stroke-width="3" class="elec-flow3" />
      <path d="M 660 120 Q 705 150 750 120" fill="none" stroke="#ffca28" stroke-width="3" class="elec-flow3" />
      <path d="M 700 160 Q 725 180 750 170" fill="none" stroke="#ffca28" stroke-width="3" class="elec-flow3" />
      <!-- Lightning bolt icon -->
      <path d="M 570 165 L 555 195 L 568 195 L 555 225 L 585 185 L 570 185 Z"
            fill="#ffca28" stroke="#EF8E0B" stroke-width="1.5" class="elec-pulse3" />
    </svg>
    <p style="color:#4b5563; font-size:1rem; margin-top:0.75rem; font-family:sans-serif; font-weight:500;">
      Running annual energy simulation — this may take about a minute...
    </p>
  </div>
</div>
"""

# ─── Logo ─────────────────────────────────────────────────────────────────────
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_container_width=True)
elif os.path.exists("logo.jpg"):
    st.sidebar.image("logo.jpg", use_container_width=True)

# ─── Title ────────────────────────────────────────────────────────────────────
st.title("Techno-Economic Analysis of ASTERIx-CAESar Plant")
st.markdown(
    "Adjust the design parameters of the concentrated solar power plant "
    "integrated with compressed air energy storage."
)

# ─── Session state initialisation ────────────────────────────────────────────
for key, default in [
    ("results",       None),
    ("sim_params",    None),
    ("scenarios",     []),
    ("sens_price",    100),
    ("sens_subsidy",  0),
    ("sens_disc",     6),
    ("sens_om",       1.5),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Sidebar: design parameters ──────────────────────────────────────────────
st.sidebar.header("Design Parameters")

# Display label → internal key used by simular_planta
LOCATION_OPTIONS = {
    "Australia":   "Australia",
    "California":  "California",
    "Chile":       "Chile",
    "Greece":      "Greece",
    "Iran":        "Iran",
    "Morocco":     "Morocco",
    "South Africa":"Southafrica",
    "Spain":       "Spain",
    "Texas":       "Texas",
}
selected_location_display = st.sidebar.selectbox("Select Location", list(LOCATION_OPTIONS.keys()))
selected_location = LOCATION_OPTIONS[selected_location_display]
power_rating      = st.sidebar.number_input("Electrical Power Rating (MW)",  min_value=10,     max_value=400,     value=100,    step=10)
tes_hours         = st.sidebar.slider(      "Thermal Energy Storage (Hours)", min_value=2,      max_value=24,      value=8)
reservoir_volume  = st.sidebar.number_input("Air Reservoir Volume (m³)",      min_value=50000,  max_value=600000,  value=177000, step=1000)
aperture_area     = st.sidebar.number_input("Solar Aperture Area (m²)",       min_value=100000, max_value=2000000, value=437000, step=5000)
charging_power    = st.sidebar.number_input("Compressor Charging Power (MW)", min_value=5,      max_value=800,     value=100,    step=5)

current_params = dict(
    location=selected_location,
    location_display=selected_location_display,
    power_rating=power_rating,
    tes_hours=tes_hours,
    reservoir_volume=reservoir_volume,
    aperture_area=aperture_area,
    charging_power=charging_power,
)

# ─── Cached model wrapper ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_model(pr, th, rv, aa, location, cp):
    return simular_planta(pr, th, rv, aa, location, cp)

# ─── Run button ───────────────────────────────────────────────────────────────
if st.sidebar.button("Run Simulation"):
    # Show the plant animation while the solver runs, then remove it cleanly.
    anim_slot = st.empty()
    anim_slot.markdown(LOADING_ANIMATION, unsafe_allow_html=True)

    results = run_model(
        power_rating, tes_hours, reservoir_volume, aperture_area, selected_location, charging_power
    )

    anim_slot.empty()   # remove animation as soon as results are ready

    status_val = results["status"]
    status_str = status_val[0] if isinstance(status_val, (list, tuple)) else str(status_val)

    if status_str != "ok":
        st.error(
            "The simulation could not find a valid solution. "
            f"Technical detail: {status_val}"
        )
        st.session_state.results = None
    else:
        st.session_state.results    = results
        st.session_state.sim_params = current_params.copy()
        # FIX 2 — reset sensitivity sliders to base values after every new
        # simulation. This guarantees that the sliders both LOOK and ARE at
        # their neutral position right after a run, avoiding any visual
        # mismatch between the displayed position and the stored session value.
        st.session_state.sens_price   = 100
        st.session_state.sens_subsidy = 0
        st.session_state.sens_disc    = 6
        st.session_state.sens_om      = 1.5

# ─── Param-change detection (evaluated after the button block) ────────────────
params_changed = (
    st.session_state.results is not None
    and st.session_state.sim_params is not None
    and current_params != st.session_state.sim_params
)
if params_changed:
    st.sidebar.warning(
        "Parameters have changed since the last simulation. "
        "Press **Run Simulation** to update the results."
    )

# ═════════════════════════════════════════════════════════════════════════════
# RESULTS SECTION
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.results is not None:
    results = st.session_state.results

    if not params_changed:
        st.success("Simulation completed successfully.")

    # ── 1. Key Performance Indicators ────────────────────────────────────────
    st.subheader("Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "CAPEX", f"€ {results['capex_meur']:.2f} M",
        help=(
            "Capital Expenditure — total upfront investment required to build the plant, "
            "including solar field, turbines, compressors, thermal storage and air reservoir."
        ),
    )
    col2.metric(
        "NPV", f"€ {results['npv_meur']:.2f} M",
        help=(
            "Net Present Value — sum of all future revenues and costs discounted to "
            "today's money (6 % discount rate, 30-year project life). "
            "A positive NPV means the project creates value."
        ),
    )
    col3.metric(
        "IRR", f"{results['irr_pct']:.2f} %",
        help=(
            "Internal Rate of Return — the annual return rate the investment generates. "
            "If higher than the discount rate (6 %), the project is financially attractive."
        ),
    )

    dpp_val     = results["dpp_years"]
    dpp_display = (
        f"{dpp_val:.1f} years"
        if (dpp_val and dpp_val > 0 and results["npv_meur"] > 0)
        else "Not reached"
    )
    col4.metric(
        "Payback (DPP)", dpp_display,
        help=(
            "Discounted Payback Period — years until cumulative discounted cash flows "
            "recover the initial investment. 'Not reached' means the project does not "
            "break even within its 30-year lifetime."
        ),
    )

    col5, col6, col7, _ = st.columns(4)
    col5.metric(
        "Operational Profit", f"€ {results['profit_eur']/1e6:.2f} M",
        help=(
            "Annual operating profit — revenues from electricity sales and industrial heat "
            "minus the cost of electricity consumed by the compressors."
        ),
    )
    col6.metric(
        "Capacity Factor", f"{results['cf_pct']:.2f} %",
        help=(
            "Capacity Factor — fraction of the year during which the plant generates "
            "electricity at its rated power. A higher value means more intensive use."
        ),
    )
    col7.metric(
        "eRTE", f"{results['erte_pct']:.1f} %",
        help=(
            "Electrical Round-Trip Efficiency — ratio of electricity discharged to the "
            "grid versus electricity consumed for compression."
        ),
    )

    # ── Save scenario (placed right after KPIs) ───────────────────────────────
    sp    = st.session_state.sim_params
    label = (
        f"{sp['location_display']} | {sp['power_rating']} MW | "
        f"{sp['tes_hours']}h TES | {sp['aperture_area']//1000}k m² | "
        f"{sp['reservoir_volume']//1000}k m³ | {sp['charging_power']} MW ch."
    )
    if st.button(f'Save scenario: "{label}"'):
        existing = [s["_label"] for s in st.session_state.scenarios]
        if label not in existing:
            st.session_state.scenarios.append({
                "_label":              label,
                "Location":            sp['location_display'],
                "Power Rating (MW)":   sp['power_rating'],
                "TES (h)":             sp['tes_hours'],
                "Aperture (k m²)":     sp['aperture_area'] // 1000,
                "Reservoir (k m³)":    sp['reservoir_volume'] // 1000,
                "Charging Power (MW)": sp['charging_power'],
                "CAPEX (M€)":          round(results["capex_meur"], 2),
                "NPV (M€)":            round(results["npv_meur"],   2),
                "IRR (%)":             round(results["irr_pct"],    2),
                "DPP (years)":         (
                    round(dpp_val, 1)
                    if (dpp_val and dpp_val > 0 and results["npv_meur"] > 0)
                    else "N/A"
                ),
                "Profit (M€)":         round(results["profit_eur"] / 1e6, 2),
                "CF (%)":              round(results["cf_pct"],     2),
                "eRTE (%)":            round(results["erte_pct"],   1),
            })
            st.success("Scenario saved. Scroll down to see the comparison table.")
        else:
            st.info("This scenario is already in the comparison table.")

    st.divider()

    # ── 2. Financial Sensitivity Analysis (What-If) ───────────────────────────
    st.subheader("Financial Sensitivity Analysis")
    st.markdown(
        "Adjust the economic assumptions below to see how the financial indicators "
        "respond — without re-running the energy simulation."
    )

    df         = results["df_hourly"]
    rev_base   = (df["P_Disch"] * df["Price"]).sum()
    cost_base  = (-df["P_Ch"]   * df["Price"]).sum()
    iph_base   = results["profit_eur"] - rev_base + cost_base
    capex_base = results["capex_meur"] * 1e6

    def reset_sensitivity():
        st.session_state.sens_price   = 100
        st.session_state.sens_subsidy = 0
        st.session_state.sens_disc    = 6
        st.session_state.sens_om      = 1.5

    st.button("Reset to base values", on_click=reset_sensitivity)

    wf1, wf2, wf3, wf4 = st.columns(4)

    # FIX 2 — sliders read their initial value from session_state via key=.
    # Because we reset those keys to base values both on simulation completion
    # and via the reset button, the visual position and the computed value are
    # always in sync. Each slider shows its base reference in a caption below.
    price_mult = wf1.slider(
        "Electricity Market Price (% of base)", 50, 200, step=5,
        key="sens_price",
        help="Scale all hourly market prices up or down uniformly.",
    ) / 100
    wf1.caption("Base: 100 %")

    subsidy_pct = wf2.slider(
        "Investment Grant / Subsidy (%)", 0, 60, step=5,
        key="sens_subsidy",
        help="Portion of the CAPEX covered by a public grant.",
    )
    wf2.caption("Base: 0 %")

    disc_rate = wf3.slider(
        "Discount Rate (%)", 3, 15, step=1,
        key="sens_disc",
        help="Annual rate used to bring future cash flows to present value.",
    ) / 100
    wf3.caption("Base: 6 %")

    om_pct = wf4.slider(
        "Annual O&M Cost (% of CAPEX)", 0.5, 3.0, step=0.5,
        key="sens_om",
        help="Yearly operations and maintenance expenditure as a share of total investment.",
    ) / 100
    wf4.caption("Base: 1.5 %")

    adj_profit = rev_base * price_mult - cost_base * price_mult + iph_base
    adj_capex  = capex_base * (1 - subsidy_pct / 100)

    adj_npv, adj_irr, adj_dpp = calculate_financial_metrics(
        adj_profit, adj_capex,
        operation_maintenance_factor=om_pct,
        real_discount_rate=disc_rate,
    )

    adj_dpp_display = (
        f"{adj_dpp:.1f} years"
        if (adj_dpp is not None and adj_dpp > 0 and adj_npv > 0)
        else "Not reached"
    )

    # Compute deltas; pass None (no coloured arrow) when the rounded value is 0.00
    # so Streamlit doesn't show a misleading green/red indicator for a zero change.
    npv_delta    = adj_npv/1e6 - results['npv_meur']
    profit_delta = (adj_profit - results['profit_eur']) / 1e6
    irr_delta    = (adj_irr * 100 - results['irr_pct']) if adj_irr is not None else None

    r1, r2, r3, r4 = st.columns(4)
    r1.metric(
        "Adjusted NPV",
        f"€ {adj_npv/1e6:.2f} M",
        delta=f"{npv_delta:+.2f} M vs. base" if abs(npv_delta) >= 0.005 else None,
        help="NPV recalculated with the assumptions above.",
    )
    r2.metric(
        "Adjusted IRR",
        f"{adj_irr*100:.2f} %" if adj_irr is not None else "N/A",
        delta=(
            f"{irr_delta:+.2f} pp vs. base"
            if (irr_delta is not None and abs(irr_delta) >= 0.005) else None
        ),
        help="IRR recalculated with the assumptions above.",
    )
    r3.metric(
        "Adjusted DPP",
        adj_dpp_display,
        help="Payback period recalculated with the assumptions above.",
    )
    r4.metric(
        "Adjusted Annual Profit",
        f"€ {adj_profit/1e6:.2f} M",
        delta=f"{profit_delta:+.2f} M vs. base" if abs(profit_delta) >= 0.005 else None,
        help="Operating profit with the adjusted electricity price.",
    )

    st.divider()

    # ── 3. State of Charge Dynamics ───────────────────────────────────────────
    st.subheader("State of Charge Dynamics")

    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    selected_month = st.selectbox("Select Month for Chart Visualization", months)
    month_idx = months.index(selected_month) + 1

    df_plot = results["df_hourly"][["SoC_Thermal_%", "SoC_Air_%"]].copy()
    df_plot = df_plot[df_plot.index.month == month_idx]

    fig_soc = go.Figure()
    fig_soc.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot["SoC_Thermal_%"],
        name="Thermal TES",
        line=dict(color="#F49000", width=2),
        hovertemplate="<b>Thermal TES:</b> %{y:.0f}%<extra></extra>",
    ))
    fig_soc.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot["SoC_Air_%"],
        name="Air Reservoir",
        line=dict(color="#6B91DD", width=2),
        hovertemplate="<b>Air Reservoir:</b> %{y:.0f}%<extra></extra>",
    ))

    fig_soc.update_layout(
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="State of Charge (%)",
        yaxis=dict(range=[0, 105]),
        # FIX: opaque legend background so text is always legible over the
        # chart lines; slightly larger font for readability.
        legend=dict(
            orientation="v",
            x=0.01, y=0.97,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.95)",  # white opaque background
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            font=dict(size=14, color="#1a1c21"),  # larger, dark text
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
    )
    fig_soc.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig_soc.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")

    st.plotly_chart(fig_soc, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO COMPARATOR
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.scenarios:
    st.divider()
    st.subheader("Scenario Comparison")
    st.markdown(
        "All saved scenarios are listed below. "
        "Simulate a new configuration and press **Save scenario** to add it to the table."
    )

    df_sc = pd.DataFrame(st.session_state.scenarios).drop(columns=["_label"])
    st.dataframe(df_sc, use_container_width=True)

    if st.button("Clear all saved scenarios"):
        st.session_state.scenarios = []
        st.rerun()
