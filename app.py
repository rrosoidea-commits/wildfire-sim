"""
app.py - High-Performance Vectorized 3D Wildfire Simulation & Evacuation System.
Academic Research Prototype.
"""

from typing import Optional, List, Dict, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import os
import base64
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import time
from PIL import Image

from core.terrain import TerrainGrid, FuelType, Structure
from core.weather import WeatherCondition
from core.risk_assessment import WildfireRiskAssessment, RiskAssessmentResult
from core.fire_ca import WildfireSimulation, FireState, SimulationStepStats
from core.firefighting import FirefightingManager
from core.wui_evacuation import WUIEvacuationAnalyzer, EvacuationAnalysisResult
from presets.scenarios import PRESET_SCENARIOS, ScenarioPreset
from visualization.plot_3d import Wildfire3DVisualizer, get_cardinal_dir
from visualization.plot_3d_webgl import Wildfire3DWebGLVisualizer

# Resolve and Load Logo
def load_logo() -> Tuple[Optional[Image.Image], Optional[str]]:
    candidates = [
        Path(__file__).parent / "logo.png",
        Path(__file__).parent / "logo_transparent.png",
        Path(__file__).parent / "visualization" / "logo.png",
        Path("logo.png"),
        Path("logo_transparent.png"),
        Path("visualization/logo.png"),
        Path("visualization/logo.png.png")
    ]
    for p in candidates:
        if p.exists():
            try:
                img = Image.open(p)
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                return img, f"data:image/png;base64,{b64}"
            except Exception:
                continue
    return None, None

logo_img, logo_b64 = load_logo()

# Page Configuration
st.set_page_config(
    page_title="Wildfire Simulation & WUI Evacuation",
    page_icon=logo_img if logo_img is not None else "🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

if logo_img is not None:
    try:
        st.logo(logo_img)
    except Exception:
        pass

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0B1120; }
    div[data-testid="metric-container"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 12px 16px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="metric-container"] label {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    .disclaimer-banner {
        background: linear-gradient(90deg, #7F1D1D 0%, #991B1B 100%);
        color: #FEE2E2;
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 0.88rem;
        margin-bottom: 16px;
        border-left: 5px solid #EF4444;
    }
    .section-title {
        color: #F8FAFC;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 6px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid #EF4444;
        color: #F87171;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .live-badge.running {
        background: rgba(16, 185, 129, 0.2);
        border-color: #10B981;
        color: #34D399;
    }
</style>
""", unsafe_allow_html=True)

# Simulation Initialization
def init_simulation(scenario_key: str = "santa_ana_canyon", custom_args: Optional[Dict[str, Any]] = None):
    scenario = PRESET_SCENARIOS.get(scenario_key, PRESET_SCENARIOS["santa_ana_canyon"])

    if scenario_key != "custom_procedural":
        terrain = TerrainGrid.create_synthetic(
            nx=scenario.grid_size,
            ny=scenario.grid_size,
            cell_size_m=scenario.cell_size_m,
            preset=scenario.terrain_preset,
            base_elevation=scenario.base_elevation,
            roughness=scenario.roughness,
            water_level=scenario.water_level,
            forest_density_scale=scenario.forest_density_scale
        )
        weather = scenario.weather
        sim = WildfireSimulation(
            terrain=terrain,
            weather=weather,
            spread_rate_factor=scenario.spread_rate_factor,
            enable_spotting=scenario.enable_spotting
        )
        sim.ignite_established_front(scenario.default_ignition[0], scenario.default_ignition[1], radius=1)
    else:
        args = custom_args or {}
        grid_size = args.get("grid_size", 60)
        terrain = TerrainGrid.create_synthetic(
            nx=grid_size,
            ny=grid_size,
            cell_size_m=30.0,
            preset=args.get("preset", "canyon"),
            seed=args.get("seed", 42),
            base_elevation=args.get("base_elev", 400.0),
            roughness=args.get("roughness", 1.0),
            water_level=args.get("water_level", 0.05),
            forest_density_scale=args.get("density_scale", 1.0)
        )
        weather = WeatherCondition(
            wind_speed_kmh=args.get("wind_speed", 35.0),
            wind_direction_deg=args.get("wind_dir", 225.0),
            temperature_c=args.get("temp", 34.0),
            relative_humidity_pct=args.get("humidity", 18.0)
        )
        sim = WildfireSimulation(
            terrain=terrain,
            weather=weather,
            spread_rate_factor=args.get("spread_factor", 1.0),
            enable_spotting=args.get("enable_spotting", True)
        )
        sim.ignite_established_front(grid_size // 2, grid_size // 2, radius=1)

    risk_res = WildfireRiskAssessment.calculate(terrain, weather)
    evac_res = WUIEvacuationAnalyzer.evaluate_threats_and_corridors(sim)

    st.session_state.simulation = sim
    st.session_state.terrain = terrain
    st.session_state.weather = weather
    st.session_state.risk_result = risk_res
    st.session_state.evac_result = evac_res
    st.session_state.is_playing = False
    st.session_state.active_scenario_key = scenario_key

if "simulation" not in st.session_state:
    init_simulation("santa_ana_canyon")

sim: WildfireSimulation = st.session_state.simulation
weather: WeatherCondition = sim.weather
terrain: TerrainGrid = sim.terrain

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    if logo_img is not None:
        col_sb1, col_sb2, col_sb3 = st.columns([1, 2, 1])
        with col_sb2:
            st.image(logo_img, width=120)

    st.markdown(
        """
        <h2 style="
            text-align: center;
            margin-top: 0;
            font-family: 'Trebuchet MS', sans-serif;
            font-size: 23px;
            font-weight: 800;
            letter-spacing: 0.5px;
        ">
            WILDFIRE SIMULATION
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align: center; color: #94A3B8; margin-bottom: 15px;'>Vectorized Cellular Automata & WUI Evacuation</p>",
        unsafe_allow_html=True
    )

    # 1. Fire Settings
    st.markdown('<div class="section-title">⚙️ Fire Settings</div>', unsafe_allow_html=True)
    scenario_options: Dict[str, str] = {k: v.title for k, v in PRESET_SCENARIOS.items()}
    scenario_options["custom_procedural"] = "Custom Procedural Generator"

    active_key = st.session_state.get("active_scenario_key", "santa_ana_canyon")
    current_idx = list(scenario_options.keys()).index(active_key) if active_key in scenario_options else 0

    selected_scenario = st.selectbox(
        "Landscape Preset",
        options=list(scenario_options.keys()),
        format_func=lambda x: scenario_options[x],
        index=current_idx,
        key="sb_selected_scenario"
    )

    custom_params: Dict[str, Any] = {}
    if selected_scenario == "custom_procedural":
        with st.expander("Procedural Terrain Parameters", expanded=True):
            custom_params["preset"] = st.selectbox("Terrain Geometry", ["canyon", "alpine_ridge", "rolling_hills", "plains_chaparral", "procedural"], key="sb_cust_preset")
            custom_params["grid_size"] = st.slider("Grid Size", 40, 80, 60, step=10, key="sb_cust_grid_size")
            custom_params["seed"] = st.number_input("Random Seed", value=42, step=1, key="sb_cust_seed")
            custom_params["base_elev"] = st.slider("Base Elevation (m)", 100.0, 1000.0, 400.0, step=50.0, key="sb_cust_base_elev")
            custom_params["roughness"] = st.slider("Roughness", 0.3, 2.0, 1.0, step=0.1, key="sb_cust_roughness")
            custom_params["water_level"] = st.slider("Water Level", 0.0, 0.15, 0.04, step=0.01, key="sb_cust_water_level")
            custom_params["density_scale"] = st.slider("Forest Density", 0.5, 1.5, 1.0, step=0.1, key="sb_cust_density_scale")
            custom_params["wind_speed"] = st.slider("Wind Speed (km/h)", 0.0, 80.0, 30.0, key="sb_cust_wind_speed")
            custom_params["wind_dir"] = st.slider("Wind Direction (°)", 0.0, 360.0, 225.0, key="sb_cust_wind_dir")
            custom_params["temp"] = st.slider("Temperature (°C)", 5.0, 48.0, 32.0, key="sb_cust_temp")
            custom_params["humidity"] = st.slider("Humidity (%)", 5.0, 95.0, 18.0, key="sb_cust_humidity")
            custom_params["spread_factor"] = st.slider("Spread Sensitivity", 0.3, 2.5, 1.0, key="sb_cust_spread_factor")
            custom_params["enable_spotting"] = st.checkbox("Spot Fires / Embers", value=True, key="sb_cust_enable_spotting")

        if st.button("Generate Landscape", use_container_width=True, key="sb_btn_gen_landscape"):
            init_simulation("custom_procedural", custom_params)
            st.rerun()
    elif selected_scenario != st.session_state.active_scenario_key:
        init_simulation(selected_scenario)
        st.rerun()

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        sim.spread_rate_factor = st.slider("Fire Spread Factor", 0.3, 2.5, float(sim.spread_rate_factor), step=0.1, key="sb_spread_rate_factor")
    with c_s2:
        sim.enable_spotting = st.checkbox("Spotting Embers", value=sim.enable_spotting, key="sb_enable_spotting")

    st.markdown("---")

    # 2. Weather Controls
    st.markdown('<div class="section-title">🌤️ Weather</div>', unsafe_allow_html=True)
    c_w1, c_w2 = st.columns(2)
    with c_w1:
        new_wind_speed = st.slider("Wind Speed (km/h)", 0.0, 90.0, float(weather.wind_speed_kmh), step=2.0, key="sb_wind_speed")
    with c_w2:
        new_wind_dir = st.slider("Wind Direction (°)", 0.0, 360.0, float(weather.wind_direction_deg), step=5.0, key="sb_wind_dir")

    cardinal = get_cardinal_dir(new_wind_dir)
    st.caption(f"💨 Prevailing Wind: **{new_wind_speed:.0f} km/h** from **{new_wind_dir:.0f}° ({cardinal})**")

    c_t1, c_t2 = st.columns(2)
    with c_t1:
        new_temp = st.slider("Temperature (°C)", 5.0, 48.0, float(weather.temperature_c), step=1.0, key="sb_temp")
    with c_t2:
        new_rh = st.slider("Humidity (%)", 5.0, 95.0, float(weather.relative_humidity_pct), step=1.0, key="sb_humidity")

    weather.wind_speed_kmh = new_wind_speed
    weather.wind_direction_deg = new_wind_dir
    weather.temperature_c = new_temp
    weather.relative_humidity_pct = new_rh

    st.markdown("---")

    # 3. Fire Location Controls
    st.markdown('<div class="section-title">📍 Fire Location</div>', unsafe_allow_html=True)
    col_ix, col_iy = st.columns(2)
    with col_ix:
        ignite_x = st.slider("Fire Location X", 0, sim.nx - 1, sim.nx // 2, key="sb_ignite_x")
    with col_iy:
        ignite_y = st.slider("Fire Location Y", 0, sim.ny - 1, sim.ny // 2, key="sb_ignite_y")

    col_ir, col_ii = st.columns(2)
    with col_ir:
        ignite_rad = st.slider("Ignition Radius", 0, 3, 1, key="sb_ignite_rad")
    with col_ii:
        ignite_int = st.slider("Fire Intensity", 0.4, 1.0, 0.85, step=0.05, key="sb_ignite_int")

    c_ig1, c_ig2 = st.columns(2)
    with c_ig1:
        if st.button("Set Fire Location", use_container_width=True, key="sb_btn_set_fire"):
            if ignite_rad == 0:
                sim.ignite_cell(ignite_x, ignite_y, initial_intensity=ignite_int)
            else:
                sim.ignite_radius(ignite_x, ignite_y, radius=ignite_rad, initial_intensity=ignite_int)
            st.session_state.evac_result = WUIEvacuationAnalyzer.evaluate_threats_and_corridors(sim)
            st.success(f"Ignited fire at ({ignite_x}, {ignite_y})")
            st.rerun()
    with c_ig2:
        if st.button("Random Spark", use_container_width=True, key="sb_btn_random_spark"):
            rand_x = int(np.random.randint(5, sim.nx - 5))
            rand_y = int(np.random.randint(5, sim.ny - 5))
            sim.ignite_cell(rand_x, rand_y, initial_intensity=ignite_int)
            st.session_state.evac_result = WUIEvacuationAnalyzer.evaluate_threats_and_corridors(sim)
            st.success(f"Ignited at ({rand_x}, {rand_y})")
            st.rerun()

    st.markdown("---")

    # 4. Interactive Firefighting & Containment
    st.markdown('<div class="section-title">🚒 Firefighting & Containment</div>', unsafe_allow_html=True)
    ff_tool = st.selectbox("Firefighting Action", ["Firelines (Barrier)", "Water Drop (Aerial)", "Backburn (Burnout)"], key="sb_ff_tool")

    if ff_tool == "Firelines (Barrier)":
        fl_c1, fl_c2 = st.columns(2)
        with fl_c1:
            fl_x1 = int(st.number_input("Start X", 0, sim.nx-1, 10, key="sb_fl_x1"))
            fl_y1 = int(st.number_input("Start Y", 0, sim.ny-1, int(sim.ny*0.4), key="sb_fl_y1"))
        with fl_c2:
            fl_x2 = int(st.number_input("End X", 0, sim.nx-1, sim.nx-10, key="sb_fl_x2"))
            fl_y2 = int(st.number_input("End Y", 0, sim.ny-1, int(sim.ny*0.4), key="sb_fl_y2"))
        if st.button("Build Fireline", use_container_width=True, key="sb_btn_build_fireline"):
            act = sim.firefighting_mgr.add_fireline(fl_x1, fl_y1, fl_x2, fl_y2)
            st.session_state.evac_result = WUIEvacuationAnalyzer.evaluate_threats_and_corridors(sim)
            st.success(f"Built Fireline #{act.id} ({act.length_m:.0f}m)")
            st.rerun()

    elif ff_tool == "Water Drop (Aerial)":
        wd_c1, wd_c2 = st.columns(2)
        with wd_c1:
            wd_x = int(st.number_input("Target X", 0, sim.nx-1, sim.nx//2, key="sb_wd_x"))
        with wd_c2:
            wd_y = int(st.number_input("Target Y", 0, sim.ny-1, sim.ny//2, key="sb_wd_y"))
        wd_r = int(st.slider("Drop Radius", 1, 5, 2, key="sb_wd_r"))
        if st.button("Execute Water Drop", use_container_width=True, key="sb_btn_water_drop"):
            act = sim.firefighting_mgr.apply_water_drop(sim, wd_x, wd_y, radius=wd_r)
            st.session_state.evac_result = WUIEvacuationAnalyzer.evaluate_threats_and_corridors(sim)
            st.success(f"Water Drop #{act.id}: {act.cells_extinguished} cells extinguished ({act.coverage_area_ha:.2f} ha)")
            st.rerun()

    else: # Backburn
        bb_c1, bb_c2 = st.columns(2)
        with bb_c1:
            bb_x = int(st.number_input("Burnout X", 0, sim.nx-1, sim.nx//2 + 5, key="sb_bb_x"))
        with bb_c2:
            bb_y = int(st.number_input("Burnout Y", 0, sim.ny-1, sim.ny//2 + 5, key="sb_bb_y"))
        bb_r = int(st.slider("Burnout Radius", 1, 4, 2, key="sb_bb_r"))
        if st.button("Ignite Backburn", use_container_width=True, key="sb_btn_backburn"):
            act = sim.firefighting_mgr.apply_backburn(sim, bb_x, bb_y, radius=bb_r)
            st.session_state.evac_result = WUIEvacuationAnalyzer.evaluate_threats_and_corridors(sim)
            st.success(f"Backburn #{act.id}: {act.cells_backburned} cells burned ({act.area_ha:.2f} ha)")
            st.rerun()

# ----------------- MAIN WORKSPACE -----------------

# Scenario Description
scenario_meta = PRESET_SCENARIOS.get(st.session_state.active_scenario_key)
if scenario_meta:
    st.info(f"**{scenario_meta.title}**: {scenario_meta.description}")

# ----------------- MASTER SIMULATION METRICS -----------------
active_burn_cells = int(np.count_nonzero(sim.state == FireState.BURNING))
is_playing = st.session_state.get("is_playing", False)

if is_playing:
    status_badge_html = f'<span class="live-badge running">🟢 SIMULATION RUNNING — Step {sim.current_step} ({sim.elapsed_minutes:.0f} min) | Active Front: {active_burn_cells} cells</span>'
elif active_burn_cells > 0:
    status_badge_html = f'<span class="live-badge">🟡 SIMULATION PAUSED — Step {sim.current_step} ({sim.elapsed_minutes:.0f} min) | Active Front: {active_burn_cells} cells</span>'
else:
    status_badge_html = f'<span class="live-badge" style="border-color:#64748B; color:#94A3B8;">⚪ SIMULATION READY — Step {sim.current_step} | Use 3D viewer controls to run</span>'

st.markdown(status_badge_html, unsafe_allow_html=True)

# Environmental Advisory Banner if weather severely impedes spread
if weather.relative_humidity_pct > 65.0 or sim.spread_rate_factor < 0.5:
    st.warning(f"⚠️ Atmospheric humidity is high ({weather.relative_humidity_pct:.0f}%) or spread sensitivity is low ({sim.spread_rate_factor:.1f}). Fire spread may be slow or self-extinguishing. Increase wind speed or spread factor for rapid spread.", icon="⚠️")

evac_res: EvacuationAnalysisResult = st.session_state.get("evac_result") or WUIEvacuationAnalyzer.evaluate_threats_and_corridors(sim)
risk_res: RiskAssessmentResult = st.session_state.get("risk_result") or WildfireRiskAssessment.calculate(sim.terrain, sim.weather)

# Real-Time Metric Banner
latest_stats: Optional[SimulationStepStats] = sim.history[-1] if len(sim.history) > 0 else None
m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    burned_ha = latest_stats.total_burned_area_ha if latest_stats else 0.0
    burned_pct = latest_stats.burned_area_pct if latest_stats else 0.0
    st.metric("Total Burned Area", f"{burned_ha:.1f} ha", f"{burned_pct:.1f}% landscape", delta_color="inverse")

with m2:
    active_ha = latest_stats.active_fire_area_ha if latest_stats else 0.0
    active_cells = latest_stats.burning_cells if latest_stats else np.count_nonzero(sim.state == FireState.BURNING)
    st.metric("Active Fire Area", f"{active_ha:.1f} ha", f"{active_cells} active cells")

with m3:
    threatened_s = evac_res.threatened_count
    total_s = evac_res.total_structures
    st.metric("Threatened Structures", f"{threatened_s} / {total_s}", f"{evac_res.burned_count} burned", delta_color="inverse" if threatened_s > 0 else "normal")

with m4:
    perimeter = latest_stats.fire_perimeter_km if latest_stats else 0.0
    st.metric("Fire Perimeter", f"{perimeter:.2f} km")

with m5:
    max_d = latest_stats.max_spread_distance_m if latest_stats else 0.0
    st.metric("Max Spread Distance", f"{max_d:.0f} m")

with m6:
    fwi = sim.weather.compute_fire_weather_index_proxy()
    fwi_label = "Extreme" if fwi > 75 else "High" if fwi > 50 else "Moderate" if fwi > 25 else "Low"
    st.metric("Fire Weather Index", f"{fwi:.0f} / 100", fwi_label, delta_color="inverse")

# ----------------- 3D INTERACTIVE MAP -----------------
st.markdown('<div class="section-title">🗺️ 3D Landscape & Wildfire Simulation Visualization</div>', unsafe_allow_html=True)

c_hdr1, c_hdr2, c_hdr3 = st.columns([1.8, 1.2, 1.0])
with c_hdr1:
    renderer_mode = st.radio(
        "3D View Engine",
        options=["webgl_engine", "plotly_mesh"],
        format_func=lambda x: "🎮 Interactive 3D WebGL (60 FPS, Orbit Controls, No Flickering — Recommended)" if x == "webgl_engine" else "📊 Plotly 3D (Static Mesh / Export)",
        horizontal=True,
        key="rad_renderer_mode"
    )
with c_hdr2:
    layer_mode = st.selectbox(
        "Initial Layer Mode",
        options=["fire_dynamic", "risk_map", "slope", "fuel_moisture", "historical_risk", "elevation"],
        format_func=lambda x: {
            "fire_dynamic": "🔥 Dynamic Flames, Ash & Vegetation",
            "risk_map": "📊 Wildfire Risk Index (0-100)",
            "slope": "⛰️ Slope Gradient (°)",
            "fuel_moisture": "💧 Fuel Moisture Content (%)",
            "historical_risk": "📈 Historical Risk (%)",
            "elevation": "🏔️ Digital Elevation Model"
        }[x],
        key="sel_3d_layer_mode"
    )
with c_hdr3:
    z_exagg = st.slider("3D Elevation Scale", 0.5, 2.5, 1.0, step=0.1, key="slider_3d_z_exagg")

if renderer_mode == "webgl_engine":
    # Ultra-smooth 60 FPS WebGL Interactive 3D Viewer with OrbitControls and Live Table Drawer
    html_3d = Wildfire3DWebGLVisualizer.generate_html(
        simulation=sim,
        risk_result=risk_res,
        evacuation_result=evac_res,
        layer_mode=layer_mode,
        z_exaggeration=z_exagg,
        height=730
    )
    components.html(html_3d, height=740, scrolling=False)
else:
    # Plotly 3D Fallback
    c_opts3, c_opts4, c_opts5 = st.columns(3)
    with c_opts3:
        show_structs = st.checkbox("Show WUI Structures", value=True, key="cb_3d_show_structs")
    with c_opts4:
        show_evac = st.checkbox("Show Safe Evac Routes", value=True, key="cb_3d_show_evac")
    with c_opts5:
        show_tactics = st.checkbox("Show Firefighting", value=True, key="cb_3d_show_tactics")

    fig_3d = Wildfire3DVisualizer.create_3d_figure(
        simulation=sim,
        risk_result=risk_res,
        evacuation_result=evac_res,
        layer_mode=layer_mode,
        show_wind_vectors=True,
        show_ignition_markers=True,
        show_structures=show_structs,
        show_evac_corridors=show_evac,
        show_firefighting=show_tactics,
        z_exaggeration=z_exagg
    )
    st.plotly_chart(fig_3d, use_container_width=True, key="wildfire_3d_map")

# ----------------- CONTINUOUS PLAYBACK ADVANCER -----------------
if st.session_state.get("is_playing", False):
    active_count = int(np.count_nonzero(sim.state == FireState.BURNING))
    if active_count > 0 and sim.current_step < 500:
        sim.step()
        st.session_state.evac_result = WUIEvacuationAnalyzer.evaluate_threats_and_corridors(sim)
        time.sleep(0.1)
        st.rerun()
    else:
        st.session_state.is_playing = False
        if active_count == 0:
            st.toast("🔥 Fire front extinguished or burned out of fuel!", icon="🔥")
        st.rerun()
