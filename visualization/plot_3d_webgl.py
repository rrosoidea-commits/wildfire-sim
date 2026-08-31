"""
plot_3d_webgl.py - High-Performance 60 FPS WebGL/Three.js 3D Wildfire Simulation Visualizer.
Features:
- Real-time 60 FPS hardware accelerated 3D landscape terrain rendering
- Uninterrupted OrbitControls (rotate, zoom, pan while fire is spreading)
- Dynamic 3D flame particles, glowing ash, and wind-driven smoke/embers
- 3D WUI structures with hover tooltips and live threat distance tracking
- 3D evacuation corridors and firefighting tactic overlays
- Live real-time Simulation Results Table drawer updating at every timestep
- Instant CSV export and step telemetry recording
"""

import json
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from core.terrain import TerrainGrid, FuelType, Structure
from core.weather import WeatherCondition
from core.fire_ca import WildfireSimulation, FireState
from core.risk_assessment import RiskAssessmentResult
from core.wui_evacuation import EvacuationAnalysisResult

class Wildfire3DWebGLVisualizer:
    @staticmethod
    def generate_html(
        simulation: WildfireSimulation,
        risk_result: Optional[RiskAssessmentResult] = None,
        evacuation_result: Optional[EvacuationAnalysisResult] = None,
        layer_mode: str = "fire_dynamic",
        z_exaggeration: float = 1.0,
        height: int = 740
    ) -> str:
        terrain = simulation.terrain
        weather = simulation.weather
        nx, ny = terrain.nx, terrain.ny
        cell_size = float(terrain.cell_size_m)

        # Prepare 2D arrays for JSON serialization
        elev_list = terrain.elevation.tolist()
        fuel_type_list = terrain.fuel_type.tolist()
        fuel_density_list = terrain.fuel_density.tolist()
        fuel_moist_list = terrain.fuel_moisture.tolist()
        hist_risk_list = terrain.historical_risk.tolist()
        slope_list = terrain.slope_deg.tolist()

        if risk_result is not None:
            composite_risk_list = risk_result.composite_risk_score.tolist()
        else:
            composite_risk_list = (terrain.historical_risk * 100.0).tolist()

        # Structures data
        structures_data = []
        for s in terrain.structures:
            structures_data.append({
                "id": s.id,
                "name": s.name,
                "type": s.structure_type,
                "x": int(s.x),
                "y": int(s.y),
                "status": s.status,
                "defensibility": float(s.defensibility_score),
                "distance_to_fire_m": float(s.distance_to_fire_m),
                "threat_level": s.threat_level
            })

        # Evacuation corridors data
        corridors_data = []
        if evacuation_result is not None:
            for c in evacuation_result.corridors:
                corridors_data.append({
                    "id": c.id,
                    "structure_id": c.structure_id,
                    "structure_name": c.structure_name,
                    "exit_name": c.exit_name,
                    "path_coords": [[int(pt[0]), int(pt[1])] for pt in c.path_coords],
                    "path_length_m": float(c.path_length_m),
                    "status": c.status,
                    "status_notes": c.status_notes
                })

        # Firefighting data
        firelines_data = []
        water_drops_data = []
        backburns_data = []
        if hasattr(simulation, 'firefighting_mgr') and simulation.firefighting_mgr is not None:
            fm = simulation.firefighting_mgr
            for fl in fm.firelines:
                firelines_data.append({
                    "id": fl.id,
                    "x1": int(fl.x1), "y1": int(fl.y1),
                    "x2": int(fl.x2), "y2": int(fl.y2),
                    "status": fl.status, "length_m": float(fl.length_m)
                })
            for wd in fm.water_drops:
                water_drops_data.append({
                    "id": wd.id, "x": int(wd.x), "y": int(wd.y),
                    "radius": int(wd.radius),
                    "cells_extinguished": int(wd.cells_extinguished),
                    "area_ha": float(wd.coverage_area_ha)
                })
            for bb in fm.backburns:
                backburns_data.append({
                    "id": bb.id, "x": int(bb.x), "y": int(bb.y),
                    "radius": int(bb.radius),
                    "cells_backburned": int(bb.cells_backburned),
                    "area_ha": float(bb.area_ha)
                })

        # Initial Simulation State
        state_list = simulation.state.tolist()
        intensity_list = simulation.fire_intensity.tolist()
        flame_height_list = simulation.flame_height_m.tolist()
        ignition_points_list = [[int(pt[0]), int(pt[1])] for pt in simulation.ignition_points]

        # Fuel Multipliers & Residence times
        spread_mult_dict = {int(k): float(v) for k, v in FuelType.SPREAD_MULTIPLIER.items()}
        residence_dict = {int(k): int(v) for k, v in FuelType.BURN_RESIDENCE_STEPS.items()}

        payload = {
            "nx": nx,
            "ny": ny,
            "cellSize": cell_size,
            "zExagg": float(z_exaggeration),
            "layerMode": layer_mode,
            "elevation": elev_list,
            "fuelType": fuel_type_list,
            "fuelDensity": fuel_density_list,
            "fuelMoisture": fuel_moist_list,
            "historicalRisk": hist_risk_list,
            "slopeDeg": slope_list,
            "compositeRisk": composite_risk_list,
            "structures": structures_data,
            "corridors": corridors_data,
            "firefighting": {
                "firelines": firelines_data,
                "waterDrops": water_drops_data,
                "backburns": backburns_data
            },
            "weather": {
                "windSpeed": float(weather.wind_speed_kmh),
                "windDir": float(weather.wind_direction_deg),
                "temp": float(weather.temperature_c),
                "humidity": float(weather.relative_humidity_pct),
                "fwi": float(weather.compute_fire_weather_index_proxy())
            },
            "simulation": {
                "currentStep": int(simulation.current_step),
                "elapsedMinutes": float(simulation.elapsed_minutes),
                "state": state_list,
                "intensity": intensity_list,
                "flameHeight": flame_height_list,
                "ignitionPoints": ignition_points_list,
                "baseSpreadProb": float(simulation.base_spread_probability),
                "spreadFactor": float(simulation.spread_rate_factor),
                "enableSpotting": bool(simulation.enable_spotting),
                "spottingProb": float(simulation.spotting_probability),
                "minutesPerStep": float(simulation.minutes_per_step),
                "spreadMultDict": spread_mult_dict,
                "residenceDict": residence_dict
            }
        }

        payload_json = json.dumps(payload)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>3D Wildfire Simulation Visualizer</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; }}
    body {{
        background: #0B1120;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        overflow: hidden;
        width: 100vw;
        height: 100vh;
    }}
    #webgl-canvas-container {{
        width: 100%;
        height: 100%;
        position: absolute;
        top: 0;
        left: 0;
        cursor: grab;
    }}
    #webgl-canvas-container:active {{
        cursor: grabbing;
    }}

    /* Glassmorphic UI Overlays */
    .overlay-panel {{
        position: absolute;
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
        z-index: 100;
        pointer-events: auto;
    }}

    /* Top Left: Simulation Metrics & HUD */
    #hud-panel {{
        top: 14px;
        left: 14px;
        min-width: 280px;
        max-width: 320px;
    }}
    .hud-title {{
        font-size: 0.95rem;
        font-weight: 700;
        color: #38BDF8;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 6px;
    }}
    .hud-stat-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        font-size: 0.78rem;
    }}
    .hud-stat-box {{
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(51, 65, 85, 0.8);
        border-radius: 8px;
        padding: 6px 10px;
    }}
    .hud-label {{
        color: #94A3B8;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .hud-val {{
        color: #F8FAFC;
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 2px;
    }}
    .hud-val.alert {{ color: #EF4444; }}
    .hud-val.warn {{ color: #F59E0B; }}
    .hud-val.safe {{ color: #10B981; }}

    /* Top Right: Layer & Display Controls */
    #layer-panel {{
        top: 14px;
        right: 14px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        min-width: 220px;
    }}
    .select-styled {{
        background: #1E293B;
        color: #F8FAFC;
        border: 1px solid #475569;
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 0.82rem;
        font-weight: 600;
        width: 100%;
        outline: none;
        cursor: pointer;
    }}
    .toggle-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.78rem;
        color: #CBD5E1;
        cursor: pointer;
        padding: 2px 0;
    }}
    .toggle-row input[type="checkbox"] {{
        cursor: pointer;
        accent-color: #EF4444;
        transform: scale(1.15);
    }}

    /* Bottom Center: Master Simulation Control Deck */
    #control-deck {{
        bottom: 16px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 18px;
        border-radius: 50px;
        background: rgba(15, 23, 42, 0.90);
    }}
    .ctrl-btn {{
        background: #1E293B;
        color: #F8FAFC;
        border: 1px solid #475569;
        border-radius: 30px;
        padding: 8px 16px;
        font-size: 0.85rem;
        font-weight: 700;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
        transition: all 0.15s ease;
    }}
    .ctrl-btn:hover {{
        background: #334155;
        border-color: #64748B;
        transform: translateY(-1px);
    }}
    .ctrl-btn:active {{
        transform: translateY(1px);
    }}
    .ctrl-btn.primary {{
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        border-color: #F87171;
        color: #FFFFFF;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
    }}
    .ctrl-btn.primary:hover {{
        background: linear-gradient(135deg, #F87171 0%, #EF4444 100%);
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.6);
    }}
    .ctrl-btn.active {{
        background: #0284C7;
        border-color: #38BDF8;
        color: #FFFFFF;
    }}

    .speed-group {{
        display: flex;
        align-items: center;
        background: #1E293B;
        border-radius: 20px;
        border: 1px solid #475569;
        padding: 2px 4px;
        gap: 2px;
    }}
    .speed-btn {{
        background: transparent;
        border: none;
        color: #94A3B8;
        font-size: 0.76rem;
        font-weight: 700;
        padding: 4px 8px;
        border-radius: 12px;
        cursor: pointer;
    }}
    .speed-btn.active {{
        background: #EF4444;
        color: #FFFFFF;
    }}

    /* Bottom Left: Camera View Presets */
    #camera-panel {{
        bottom: 16px;
        left: 14px;
        display: flex;
        gap: 6px;
        padding: 8px 12px;
        border-radius: 24px;
    }}
    .cam-btn {{
        background: #1E293B;
        color: #CBD5E1;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 5px 10px;
        font-size: 0.74rem;
        font-weight: 600;
        cursor: pointer;
    }}
    .cam-btn:hover {{
        background: #334155;
        color: #FFFFFF;
    }}

    /* Bottom Right: Interactive Navigation Help */
    #help-panel {{
        bottom: 16px;
        right: 14px;
        font-size: 0.72rem;
        color: #94A3B8;
        padding: 8px 12px;
        border-radius: 20px;
        line-height: 1.4;
    }}
    .key-badge {{
        background: #334155;
        color: #F8FAFC;
        padding: 2px 5px;
        border-radius: 4px;
        font-weight: 600;
    }}

    /* Interactive Tooltip on 3D Object Hover */
    #tooltip-3d {{
        position: absolute;
        display: none;
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid #38BDF8;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 0.78rem;
        color: #F8FAFC;
        pointer-events: none;
        z-index: 1000;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
        max-width: 250px;
    }}

    /* Live Simulation Results Table Drawer */
    #table-drawer {{
        position: absolute;
        bottom: 72px;
        left: 50%;
        transform: translateX(-50%);
        width: 94%;
        max-height: 280px;
        background: rgba(15, 23, 42, 0.96);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid #38BDF8;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.7);
        z-index: 500;
        display: none;
        flex-direction: column;
        gap: 8px;
        animation: slideUp 0.2s ease-out;
    }}
    @keyframes slideUp {{
        from {{ opacity: 0; transform: translate(-50%, 20px); }}
        to {{ opacity: 1; transform: translate(-50%, 0); }}
    }}
    .drawer-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #334155;
        padding-bottom: 6px;
    }}
    .drawer-title {{
        font-size: 0.88rem;
        font-weight: 700;
        color: #38BDF8;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .drawer-actions {{
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .drawer-btn {{
        background: #1E293B;
        color: #F8FAFC;
        border: 1px solid #475569;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.76rem;
        font-weight: 600;
        cursor: pointer;
    }}
    .drawer-btn:hover {{
        background: #334155;
        border-color: #38BDF8;
    }}
    .drawer-btn.primary {{
        background: #EF4444;
        border-color: #F87171;
    }}
    .table-container {{
        overflow-y: auto;
        max-height: 200px;
        border-radius: 6px;
        border: 1px solid #334155;
    }}
    .live-sim-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.75rem;
        text-align: right;
    }}
    .live-sim-table th {{
        background: #1E293B;
        color: #94A3B8;
        padding: 6px 10px;
        position: sticky;
        top: 0;
        font-weight: 600;
        border-bottom: 1px solid #475569;
    }}
    .live-sim-table td {{
        padding: 5px 10px;
        border-bottom: 1px solid #1E293B;
        color: #E2E8F0;
    }}
    .live-sim-table tr:hover td {{
        background: rgba(56, 189, 248, 0.1);
    }}
    .live-sim-table tr:last-child td {{
        background: rgba(239, 68, 68, 0.15);
        color: #FFFFFF;
        font-weight: 700;
    }}
</style>

<!-- Load Three.js and OrbitControls via reliable CDNs -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>

<div id="webgl-canvas-container"></div>
<div id="tooltip-3d"></div>

<!-- Top Left HUD: Real-Time Telemetry -->
<div class="overlay-panel" id="hud-panel">
    <div class="hud-title">
        <span>🔥</span>
        <span id="hud-status-badge">SIMULATION READY</span>
    </div>
    <div class="hud-stat-grid">
        <div class="hud-stat-box">
            <div class="hud-label">Step / Time</div>
            <div class="hud-val" id="hud-step-val">0 <span style="font-size:0.75rem; color:#94A3B8;">(0m)</span></div>
        </div>
        <div class="hud-stat-box">
            <div class="hud-label">Active Fire Front</div>
            <div class="hud-val warn" id="hud-active-val">0 <span style="font-size:0.75rem; color:#94A3B8;">cells</span></div>
        </div>
        <div class="hud-stat-box">
            <div class="hud-label">Total Burned</div>
            <div class="hud-val alert" id="hud-burned-val">0.0 <span style="font-size:0.75rem; color:#94A3B8;">ha</span></div>
        </div>
        <div class="hud-stat-box">
            <div class="hud-label">Threatened WUI</div>
            <div class="hud-val" id="hud-wui-val">0 / 0</div>
        </div>
        <div class="hud-stat-box">
            <div class="hud-label">Fire Weather (FWI)</div>
            <div class="hud-val alert" id="hud-fwi-val">--</div>
        </div>
        <div class="hud-stat-box">
            <div class="hud-label">Prevailing Wind</div>
            <div class="hud-val" id="hud-wind-val">-- km/h</div>
        </div>
    </div>
</div>

<!-- Top Right: Layer & Display Controls -->
<div class="overlay-panel" id="layer-panel">
    <select class="select-styled" id="sel-layer-mode">
        <option value="fire_dynamic">🔥 Dynamic Flames & Ash</option>
        <option value="risk_map">📊 Wildfire Risk Score (0-100)</option>
        <option value="slope">⛰️ Slope Gradient (°)</option>
        <option value="fuel_moisture">💧 Fuel Moisture (%)</option>
        <option value="historical_risk">📈 Historical Climatology Risk</option>
        <option value="elevation">🏔️ Digital Elevation Model</option>
    </select>

    <label class="toggle-row">
        <span>🏡 WUI Structures</span>
        <input type="checkbox" id="chk-structures" checked>
    </label>
    <label class="toggle-row">
        <span>🚶 Safe Evac Corridors</span>
        <input type="checkbox" id="chk-evac" checked>
    </label>
    <label class="toggle-row">
        <span>🚒 Firefighting Tactics</span>
        <input type="checkbox" id="chk-tactics" checked>
    </label>
    <label class="toggle-row">
        <span>💨 3D Wind Vector</span>
        <input type="checkbox" id="chk-wind" checked>
    </label>
    <label class="toggle-row">
        <span>✨ Flame Particle FX</span>
        <input type="checkbox" id="chk-particles" checked>
    </label>
</div>

<!-- Live Simulation Results Table Drawer -->
<div id="table-drawer">
    <div class="drawer-header">
        <div class="drawer-title">
            <span>📋</span>
            <span>Live Step-by-Step Simulation Recording Table (<span id="drawer-step-count">0</span> timesteps recorded)</span>
        </div>
        <div class="drawer-actions">
            <button class="drawer-btn primary" id="btn-export-csv">📥 Download CSV</button>
            <button class="drawer-btn" id="btn-close-table">✕ Close</button>
        </div>
    </div>
    <div class="table-container" id="table-scroll-wrap">
        <table class="live-sim-table">
            <thead>
                <tr>
                    <th style="text-align:left;">Step</th>
                    <th>Time (min)</th>
                    <th>Wind (km/h)</th>
                    <th>Temp (°C)</th>
                    <th>Humidity (%)</th>
                    <th>Active Cells</th>
                    <th>Burned Cells</th>
                    <th>Active Area (ha)</th>
                    <th>Total Burned (ha)</th>
                    <th>Burned (%)</th>
                    <th>Threatened WUI</th>
                    <th>Burned WUI</th>
                </tr>
            </thead>
            <tbody id="sim-records-tbody">
                <!-- Rows injected dynamically on every step -->
            </tbody>
        </table>
    </div>
</div>

<!-- Bottom Center: Master Simulation Control Deck -->
<div class="overlay-panel" id="control-deck">
    <button class="ctrl-btn primary" id="btn-play-pause">
        <span id="btn-play-icon">▶</span>
        <span id="btn-play-text">RUN SIMULATION</span>
    </button>
    <button class="ctrl-btn" id="btn-step" title="Advance 1 simulation step">
        <span>⏭</span> Step (+1)
    </button>
    <button class="ctrl-btn" id="btn-reset" title="Reset fire to initial state">
        <span>🔄</span> Reset
    </button>

    <div class="speed-group">
        <span style="font-size:0.72rem; color:#94A3B8; margin-left:4px;">⚡</span>
        <button class="speed-btn" data-speed="1">1x</button>
        <button class="speed-btn active" data-speed="3">3x</button>
        <button class="speed-btn" data-speed="6">6x</button>
        <button class="speed-btn" data-speed="12">12x</button>
    </div>

    <button class="ctrl-btn" id="btn-toggle-table" title="Toggle Live Simulation Results Recording Table">
        <span>📋</span> Results Table (<span id="btn-table-counter">0</span>)
    </button>

    <button class="ctrl-btn" id="btn-ignite-tool" title="Toggle Click-on-Terrain Ignition Tool">
        <span>📍</span> Click-to-Ignite
    </button>
</div>

<!-- Bottom Left: Camera View Presets -->
<div class="overlay-panel" id="camera-panel">
    <button class="cam-btn" id="cam-persp">Perspective</button>
    <button class="cam-btn" id="cam-top">Top-Down 2D</button>
    <button class="cam-btn" id="cam-canyon">Canyon Angle</button>
    <button class="cam-btn" id="cam-reset">Reset View</button>
</div>

<!-- Bottom Right: Interactive Navigation Help -->
<div class="overlay-panel" id="help-panel">
    🖱️ <span class="key-badge">Left Drag</span> Orbit | <span class="key-badge">Right Drag</span> Pan | <span class="key-badge">Wheel</span> Zoom
</div>

<script>
// ==================== INITIALIZATION & DATA INGESTION ====================
const simData = {payload_json};

const nx = simData.nx;
const ny = simData.ny;
const cellSize = simData.cellSize;
const cellAreaHa = (cellSize * cellSize) / 10000.0;
const totalGridAreaHa = (nx * ny) * cellAreaHa;
const zExagg = simData.zExagg || 1.0;
let currentLayer = simData.layerMode || "fire_dynamic";

const elevation = simData.elevation;
const fuelType = simData.fuelType;
const fuelDensity = simData.fuelDensity;
const fuelMoisture = simData.fuelMoisture;
const historicalRisk = simData.historicalRisk;
const slopeDeg = simData.slopeDeg;
const compositeRisk = simData.compositeRisk;

// Live Step Recording History
const simRecords = [];

// Simulation Cellular Automata State
const simState = {{
    nx: nx,
    ny: ny,
    cellSize: cellSize,
    state: simData.simulation.state.map(row => [...row]),
    intensity: simData.simulation.intensity.map(row => [...row]),
    flameHeight: simData.simulation.flameHeight.map(row => [...row]),
    stepsBurning: Array.from({{length: ny}}, () => new Int16Array(nx)),
    burnProgress: Array.from({{length: ny}}, () => new Float32Array(nx)),
    ignitionPoints: JSON.parse(JSON.stringify(simData.simulation.ignitionPoints)),
    currentStep: simData.simulation.currentStep || 0,
    elapsedMinutes: simData.simulation.elapsedMinutes || 0.0,
    isPlaying: false,
    simSpeed: 3, // Steps per second multiplier
    stepIntervalMs: 250,
    clickToIgnite: false,
    baseSpreadProb: simData.simulation.baseSpreadProb,
    spreadFactor: simData.simulation.spreadFactor,
    enableSpotting: simData.simulation.enableSpotting,
    spottingProb: simData.simulation.spottingProb,
    minutesPerStep: simData.simulation.minutesPerStep,
    spreadMultDict: simData.simulation.spreadMultDict,
    residenceDict: simData.simulation.residenceDict,
    weather: {{ ...simData.weather }},
    structures: JSON.parse(JSON.stringify(simData.structures)),
    corridors: JSON.parse(JSON.stringify(simData.corridors)),
    firefighting: JSON.parse(JSON.stringify(simData.firefighting))
}};

// Precompute initial burn residence steps
for (let y = 0; y < ny; y++) {{
    for (let x = 0; x < nx; x++) {{
        if (simState.state[y][x] === 1) {{
            simState.stepsBurning[y][x] = 1;
        }}
    }}
}}

// Record Step 0 in table history
recordCurrentTimestep();

// ==================== THREE.JS 3D SCENE SETUP ====================
const container = document.getElementById("webgl-canvas-container");
const tooltip = document.getElementById("tooltip-3d");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0B1120);
scene.fog = new THREE.FogExp2(0x0B1120, 0.00035);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 10, 50000);
const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false, powerPreference: "high-performance" }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2.0));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

// OrbitControls setup
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.maxPolarAngle = Math.PI / 2.05; // Prevent camera going beneath terrain
controls.minDistance = 50;
controls.maxDistance = 15000;

// World Dimensions
const worldWidth = nx * cellSize;
const worldHeight = ny * cellSize;
const centerX = worldWidth * 0.5;
const centerY = worldHeight * 0.5;

// Find min/max elevation for camera framing
let minZ = 999999, maxZ = -999999;
for (let y = 0; y < ny; y++) {{
    for (let x = 0; x < nx; x++) {{
        const z = elevation[y][x] * zExagg;
        if (z < minZ) minZ = z;
        if (z > maxZ) maxZ = z;
    }}
}}
const centerZ = (minZ + maxZ) * 0.5;

// Default Perspective Camera Framing
function setPerspectiveView() {{
    camera.position.set(centerX + worldWidth * 0.8, centerZ + (maxZ - minZ) * 2.2 + worldWidth * 0.6, centerY + worldHeight * 0.85);
    controls.target.set(centerX, centerZ, centerY);
    controls.update();
}}

// Restore camera from sessionStorage if available, else default view
const savedCam = sessionStorage.getItem("wildfire_cam_state");
if (savedCam) {{
    try {{
        const camObj = JSON.parse(savedCam);
        camera.position.set(camObj.px, camObj.py, camObj.pz);
        controls.target.set(camObj.tx, camObj.ty, camObj.tz);
        controls.update();
    }} catch (e) {{
        setPerspectiveView();
    }}
}} else {{
    setPerspectiveView();
}}

// Save camera position on change
controls.addEventListener("change", () => {{
    sessionStorage.setItem("wildfire_cam_state", JSON.stringify({{
        px: camera.position.x, py: camera.position.y, pz: camera.position.z,
        tx: controls.target.x, ty: controls.target.y, tz: controls.target.z
    }}));
}});

// ==================== LIGHTING ====================
const ambientLight = new THREE.AmbientLight(0xE2E8F0, 0.70);
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xFFF7ED, 1.15);
sunLight.position.set(centerX + worldWidth * 0.5, maxZ + 2500, centerY - worldHeight * 0.4);
sunLight.castShadow = true;
scene.add(sunLight);

const hemisphereLight = new THREE.HemisphereLight(0x38BDF8, 0x0F172A, 0.45);
scene.add(hemisphereLight);

// ==================== 3D TERRAIN MESH GENERATION ====================
// PlaneGeometry with segments matching grid
const terrainGeom = new THREE.PlaneGeometry(worldWidth, worldHeight, nx - 1, ny - 1);
terrainGeom.rotateX(-Math.PI / 2); // Rotate to lie in X-Z ground plane (Three.js Y is UP)

// Displace vertices with elevation data
const posAttr = terrainGeom.attributes.position;
for (let i = 0; i < posAttr.count; i++) {{
    const ix = i % nx;
    const iy = Math.floor(i / nx);
    const zVal = elevation[iy][ix] * zExagg;
    
    // In Three.js: X is East, Y is Elevation (Up), Z is North
    posAttr.setX(i, ix * cellSize);
    posAttr.setY(i, zVal);
    posAttr.setZ(i, iy * cellSize);
}}
terrainGeom.computeVertexNormals();

// 2D Dynamic Canvas Texture for 60 FPS pixel-perfect terrain coloring
const terrainCanvas = document.createElement("canvas");
terrainCanvas.width = nx;
terrainCanvas.height = ny;
const terrainCtx = terrainCanvas.getContext("2d");
const terrainImgData = terrainCtx.createImageData(nx, ny);

const terrainTexture = new THREE.CanvasTexture(terrainCanvas);
terrainTexture.minFilter = THREE.LinearFilter;
terrainTexture.magFilter = THREE.NearestFilter; // Sharp cell boundaries
terrainTexture.generateMipmaps = false;

// Fuel Type Base Colors (Hex -> RGB)
const FUEL_COLORS = {{
    0: [21, 101, 192],  // Water Deep Blue
    1: [158, 158, 158], // Bare Rock/Ground Grey
    2: [205, 220, 57],  // Grass Lime Green
    3: [251, 192, 45],  // Shrub Gold
    4: [85, 139, 47],   // Mixed Woodland Green
    5: [27, 94, 32],    // Dense Conifer Dark Green
    6: [141, 110, 99]   // Slash Earth Brown
}};

// Function to update the 2D canvas texture based on layer and fire state
function updateTerrainTexture() {{
    const data = terrainImgData.data;
    let ptr = 0;

    for (let y = 0; y < ny; y++) {{
        for (let x = 0; x < nx; x++) {{
            const ftype = fuelType[y][x];
            const state = simState.state[y][x];
            const intensity = simState.intensity[y][x];
            let r = 80, g = 80, b = 80;

            if (currentLayer === "fire_dynamic") {{
                if (state === 1) {{
                    // BURNING: Dynamic Fire Glow (White-Yellow-Red based on intensity)
                    if (intensity > 0.8) {{
                        r = 255; g = 255; b = 220; // White-hot
                    }} else if (intensity > 0.5) {{
                        r = 255; g = 200; b = 0;   // Bright Gold
                    }} else {{
                        r = 239; g = 68; b = 68;   // Deep Fire Orange-Red
                    }}
                }} else if (state === 2) {{
                    // BURNED OUT: Charcoal Ash Dark Grey/Black
                    r = 33; g = 33; b = 33;
                }} else {{
                    // Base Landscape Fuel Color
                    const baseCol = FUEL_COLORS[ftype] || [100, 100, 100];
                    r = baseCol[0]; g = baseCol[1]; b = baseCol[2];

                    // Overlay Firefighting tactics
                    if (simState.firefighting) {{
                        // Check firelines
                        for (let fl of simState.firefighting.firelines) {{
                            const minX = Math.min(fl.x1, fl.x2), maxX = Math.max(fl.x1, fl.x2);
                            const minY = Math.min(fl.y1, fl.y2), maxY = Math.max(fl.y1, fl.y2);
                            if (x >= minX && x <= maxX && y >= minY && y <= maxY) {{
                                r = 255; g = 145; b = 0; // Fireline Orange Earth
                            }}
                        }}
                    }}
                }}
            }} else if (currentLayer === "risk_map") {{
                // Composite Risk (0 - 100): Green -> Yellow -> Orange -> Red -> Purple
                const score = Math.max(0, Math.min(100, compositeRisk[y][x]));
                if (score < 25) {{ r = 46; g = 125; b = 50; }}
                else if (score < 50) {{ r = 253; g = 216; b = 53; }}
                else if (score < 75) {{ r = 251; g = 140; b = 0; }}
                else if (score < 90) {{ r = 229; g = 57; b = 53; }}
                else {{ r = 136; g = 14; b = 79; }}
            }} else if (currentLayer === "slope") {{
                // Slope (0 - 45 deg): Viridis gradient approximation
                const s = Math.min(45, slopeDeg[y][x]) / 45.0;
                r = Math.floor(68 + s * 187);
                g = Math.floor(1 + s * 220);
                b = Math.floor(84 + (1 - s) * 100);
            }} else if (currentLayer === "fuel_moisture") {{
                // Moisture (2 - 35%): Blue-Green gradient
                const m = Math.min(0.35, fuelMoisture[y][x]) / 0.35;
                r = Math.floor((1 - m) * 255);
                g = Math.floor(180 + m * 75);
                b = Math.floor(50 + m * 205);
            }} else if (currentLayer === "historical_risk") {{
                // Historical Climatology (0 - 1)
                const hr = historicalRisk[y][x];
                r = Math.floor(hr * 255);
                g = Math.floor((1 - hr) * 150);
                b = 30;
            }} else {{
                // Elevation DEM (minZ to maxZ)
                const elevNorm = (elevation[y][x] * zExagg - minZ) / Math.max(1, maxZ - minZ);
                r = Math.floor(elevNorm * 220 + 35);
                g = Math.floor(elevNorm * 200 + 40);
                b = Math.floor(elevNorm * 180 + 40);
            }}

            data[ptr++] = r;
            data[ptr++] = g;
            data[ptr++] = b;
            data[ptr++] = 255;
        }}
    }}

    terrainCtx.putImageData(terrainImgData, 0, 0);
    terrainTexture.needsUpdate = true;
}}

updateTerrainTexture();

const terrainMat = new THREE.MeshStandardMaterial({{
    map: terrainTexture,
    roughness: 0.75,
    metalness: 0.10,
    flatShading: false
}});

const terrainMesh = new THREE.Mesh(terrainGeom, terrainMat);
terrainMesh.receiveShadow = true;
scene.add(terrainMesh);

// ==================== 3D OBJECT GROUPS ====================
const flameParticlesGroup = new THREE.Group();
const structuresGroup = new THREE.Group();
const corridorsGroup = new THREE.Group();
const firefightingGroup = new THREE.Group();
const windArrowGroup = new THREE.Group();

scene.add(flameParticlesGroup);
scene.add(structuresGroup);
scene.add(corridorsGroup);
scene.add(firefightingGroup);
scene.add(windArrowGroup);

// ==================== 3D FLAME & SMOKE PARTICLE SYSTEM ====================
const flameGeom = new THREE.ConeGeometry(cellSize * 0.45, cellSize * 1.8, 5);
flameGeom.translate(0, cellSize * 0.9, 0);

const flameMat = new THREE.MeshBasicMaterial({{
    color: 0xFFEA00,
    wireframe: false,
    transparent: true,
    opacity: 0.85
}});

function updateFlameMeshes() {{
    // Clear old flames
    while (flameParticlesGroup.children.length > 0) {{
        flameParticlesGroup.remove(flameParticlesGroup.children[0]);
    }}

    if (!document.getElementById("chk-particles").checked) return;

    for (let y = 0; y < ny; y++) {{
        for (let x = 0; x < nx; x++) {{
            if (simState.state[y][x] === 1) {{
                const zVal = elevation[y][x] * zExagg;
                const mesh = new THREE.Mesh(flameGeom, flameMat);
                mesh.position.set(x * cellSize, zVal, y * cellSize);
                
                // Scale with flame height & intensity
                const hScale = Math.max(0.6, simState.flameHeight[y][x] / 2.5);
                mesh.scale.set(hScale, hScale * (0.8 + Math.random() * 0.4), hScale);
                flameParticlesGroup.add(mesh);
            }}
        }}
    }}
}}

// ==================== 3D WUI STRUCTURES ====================
const structureObjects = [];
const houseBaseGeom = new THREE.BoxGeometry(cellSize * 0.7, cellSize * 0.5, cellSize * 0.7);
houseBaseGeom.translate(0, cellSize * 0.25, 0);
const roofGeom = new THREE.ConeGeometry(cellSize * 0.55, cellSize * 0.4, 4);
roofGeom.translate(0, cellSize * 0.7, 0);
roofGeom.rotateY(Math.PI / 4);

const STATUS_COLORS = {{
    "INTACT": 0x00E676,     // Emerald Green
    "THREATENED": 0xFF1744, // Crimson Red Alert
    "DEFENDED": 0x2979FF,   // Protected Blue
    "BURNED": 0x212121      // Burned Charcoal
}};

function createStructures() {{
    while (structuresGroup.children.length > 0) {{
        structuresGroup.remove(structuresGroup.children[0]);
    }}
    structureObjects.length = 0;

    if (!document.getElementById("chk-structures").checked) return;

    simState.structures.forEach(s => {{
        const zVal = elevation[s.y][s.x] * zExagg;
        const colorHex = STATUS_COLORS[s.status] || 0x00E676;

        const sGroup = new THREE.Group();
        sGroup.position.set(s.x * cellSize, zVal, s.y * cellSize);

        // Building walls
        const wallMat = new THREE.MeshStandardMaterial({{ color: 0xF8FAFC, roughness: 0.6 }});
        const wallMesh = new THREE.Mesh(houseBaseGeom, wallMat);
        sGroup.add(wallMesh);

        // Roof with status color
        const roofMat = new THREE.MeshStandardMaterial({{ color: colorHex, roughness: 0.5 }});
        const roofMesh = new THREE.Mesh(roofGeom, roofMat);
        sGroup.add(roofMesh);

        // Floating Threat Beacon
        const beaconGeom = new THREE.SphereGeometry(cellSize * 0.2, 8, 8);
        const beaconMat = new THREE.MeshBasicMaterial({{ color: colorHex }});
        const beaconMesh = new THREE.Mesh(beaconGeom, beaconMat);
        beaconMesh.position.set(0, cellSize * 1.3, 0);
        sGroup.add(beaconMesh);

        sGroup.userData = {{ structure: s }};
        structuresGroup.add(sGroup);
        structureObjects.push(sGroup);
    }});
}}

// ==================== 3D EVACUATION CORRIDORS ====================
function createCorridors() {{
    while (corridorsGroup.children.length > 0) {{
        corridorsGroup.remove(corridorsGroup.children[0]);
    }}

    if (!document.getElementById("chk-evac").checked) return;

    simState.corridors.forEach(c => {{
        if (!c.path_coords || c.path_coords.length < 2) return;

        const points = c.path_coords.map(pt => {{
            const z = elevation[pt[1]][pt[0]] * zExagg + 4.0;
            return new THREE.Vector3(pt[0] * cellSize, z, pt[1] * cellSize);
        }});

        const curve = new THREE.CatmullRomCurve3(points);
        const tubeGeom = new THREE.TubeGeometry(curve, points.length * 2, cellSize * 0.12, 6, false);
        
        let col = 0x00E676;
        if (c.status === "CAUTION") col = 0xFFD600;
        if (c.status === "BLOCKED") col = 0xFF1744;

        const tubeMat = new THREE.MeshBasicMaterial({{ color: col, transparent: true, opacity: 0.85 }});
        const tubeMesh = new THREE.Mesh(tubeGeom, tubeMat);
        corridorsGroup.add(tubeMesh);
    }});
}}

// ==================== 3D FIREFIGHTING TACTICS ====================
function createFirefighting() {{
    while (firefightingGroup.children.length > 0) {{
        firefightingGroup.remove(firefightingGroup.children[0]);
    }}

    if (!document.getElementById("chk-tactics").checked || !simState.firefighting) return;

    // Water Drops
    simState.firefighting.waterDrops.forEach(wd => {{
        const z = elevation[wd.y][wd.x] * zExagg + 10.0;
        const geom = new THREE.RingGeometry(cellSize * 0.5, cellSize * wd.radius, 16);
        geom.rotateX(-Math.PI / 2);
        const mat = new THREE.MeshBasicMaterial({{ color: 0x00E5FF, transparent: true, opacity: 0.7, side: THREE.DoubleSide }});
        const ring = new THREE.Mesh(geom, mat);
        ring.position.set(wd.x * cellSize, z, wd.y * cellSize);
        firefightingGroup.add(ring);
    }});

    // Backburns
    simState.firefighting.backburns.forEach(bb => {{
        const z = elevation[bb.y][bb.x] * zExagg + 6.0;
        const geom = new THREE.RingGeometry(cellSize * 0.3, cellSize * bb.radius, 16);
        geom.rotateX(-Math.PI / 2);
        const mat = new THREE.MeshBasicMaterial({{ color: 0xFF3D00, transparent: true, opacity: 0.7, side: THREE.DoubleSide }});
        const ring = new THREE.Mesh(geom, mat);
        ring.position.set(bb.x * cellSize, z, bb.y * cellSize);
        firefightingGroup.add(ring);
    }});
}}

// ==================== 3D WIND VECTOR COMPASS ====================
function updateWindArrow() {{
    while (windArrowGroup.children.length > 0) {{
        windArrowGroup.remove(windArrowGroup.children[0]);
    }}

    if (!document.getElementById("chk-wind").checked) return;

    const windDeg = simState.weather.windDir;
    const blowDeg = (windDeg + 180) % 360;
    const blowRad = blowDeg * Math.PI / 180;

    // Unit vector: East (X) is sin, North (Z) is cos
    const u = Math.sin(blowRad);
    const v = Math.cos(blowRad);

    const arrowLen = worldWidth * 0.22;
    const arrowHeight = maxZ + 120.0;

    const startPt = new THREE.Vector3(centerX, arrowHeight, centerY);
    const endPt = new THREE.Vector3(centerX + u * arrowLen, arrowHeight, centerY + v * arrowLen);

    const dir = new THREE.Vector3().subVectors(endPt, startPt).normalize();
    const arrowHelper = new THREE.ArrowHelper(dir, startPt, arrowLen, 0x00E5FF, cellSize * 2.0, cellSize * 1.5);
    windArrowGroup.add(arrowHelper);
}}

// Initial Setup of 3D objects
createStructures();
createCorridors();
createFirefighting();
updateWindArrow();
updateFlameMeshes();

// ==================== STEP RECORDING & LIVE TABLE LOGIC ====================
function recordCurrentTimestep() {{
    let burningCells = 0, burnedCells = 0;
    for (let y = 0; y < ny; y++) {{
        for (let x = 0; x < nx; x++) {{
            if (simState.state[y][x] === 1) burningCells++;
            if (simState.state[y][x] === 2) burnedCells++;
        }}
    }}

    const activeAreaHa = burningCells * cellAreaHa;
    const totalBurnedHa = (burningCells + burnedCells) * cellAreaHa;
    const burnedPct = (totalBurnedHa / totalGridAreaHa) * 100.0;
    const threatenedCount = simState.structures.filter(s => s.status === "THREATENED").length;
    const burnedStructures = simState.structures.filter(s => s.status === "BURNED").length;

    const record = {{
        step: simState.currentStep,
        elapsedMin: simState.elapsedMinutes,
        windSpeed: simState.weather.windSpeed,
        windDir: simState.weather.windDir,
        temp: simState.weather.temp,
        humidity: simState.weather.humidity,
        burningCells: burningCells,
        burnedCells: burnedCells,
        activeAreaHa: activeAreaHa,
        totalBurnedHa: totalBurnedHa,
        burnedPct: burnedPct,
        threatenedCount: threatenedCount,
        burnedStructures: burnedStructures
    }};

    simRecords.push(record);

    // Update Drawer Table UI
    appendRecordToTable(record);

    // Update Counter badge
    document.getElementById("btn-table-counter").innerText = simRecords.length;
    document.getElementById("drawer-step-count").innerText = simRecords.length;
}}

function appendRecordToTable(rec) {{
    const tbody = document.getElementById("sim-records-tbody");
    if (!tbody) return;

    const row = document.createElement("tr");
    row.innerHTML = `
        <td style="text-align:left; font-weight:700; color:#38BDF8;">#${{rec.step}}</td>
        <td>${{rec.elapsedMin.toFixed(1)}}</td>
        <td>${{rec.windSpeed.toFixed(0)}}</td>
        <td>${{rec.temp.toFixed(1)}}</td>
        <td>${{rec.humidity.toFixed(0)}}%</td>
        <td style="color:#F59E0B;">${{rec.burningCells}}</td>
        <td style="color:#94A3B8;">${{rec.burnedCells}}</td>
        <td>${{rec.activeAreaHa.toFixed(1)}}</td>
        <td style="color:#EF4444; font-weight:700;">${{rec.totalBurnedHa.toFixed(1)}}</td>
        <td>${{rec.burnedPct.toFixed(1)}}%</td>
        <td style="color:${{rec.threatenedCount > 0 ? '#EF4444' : '#10B981'}}">${{rec.threatenedCount}}</td>
        <td style="color:${{rec.burnedStructures > 0 ? '#94A3B8' : '#10B981'}}">${{rec.burnedStructures}}</td>
    `;
    tbody.appendChild(row);

    // Auto-scroll table to bottom
    const container = document.getElementById("table-scroll-wrap");
    if (container) {{
        container.scrollTop = container.scrollHeight;
    }}
}}

// ==================== CELLULAR AUTOMATA PROPAGATION ENGINE ====================
function computeSimulationStep() {{
    const activeMask = [];
    for (let y = 0; y < ny; y++) {{
        for (let x = 0; x < nx; x++) {{
            if (simState.state[y][x] === 1) activeMask.push([y, x]);
        }}
    }}

    if (activeMask.length === 0) {{
        simState.isPlaying = false;
        updatePlayButtonUI();
        return;
    }}

    const newlyIgnited = Array.from({{length: ny}}, () => new Uint8Array(nx));
    const newIntensities = Array.from({{length: ny}}, () => new Float32Array(nx));

    // Wind blowing angle unit vector
    const blowDeg = (simState.weather.windDir + 180) % 360;
    const blowRad = blowDeg * Math.PI / 180;
    const u_base = Math.sin(blowRad);
    const v_base = Math.cos(blowRad);
    const windSpeed = simState.weather.windSpeed;

    const f_temp = 1.0 + 0.025 * (simState.weather.temp - 20.0);
    const f_humid = Math.exp(-0.018 * simState.weather.humidity);

    const neighbors = [
        [1, 0, 1.0], [-1, 0, 1.0], [0, -1, 1.0], [0, 1, 1.0],
        [1, -1, 1.414], [1, 1, 1.414], [-1, -1, 1.414], [-1, 1, 1.414]
    ];

    // Evaluate spread to unburned neighbors
    for (let i = 0; i < activeMask.length; i++) {{
        const [sy, sx] = activeMask[i];
        const srcInt = simState.intensity[sy][sx];
        const srcElev = elevation[sy][sx];

        for (let j = 0; j < neighbors.length; j++) {{
            const [dy, dx, dist] = neighbors[j];
            const ty = sy + dy, tx = sx + dx;

            if (tx < 0 || tx >= nx || ty < 0 || ty >= ny) continue;
            if (simState.state[ty][tx] !== 0) continue; // Must be UNBURNED

            const dstFuel = fuelType[ty][tx];
            const dstFuelMult = simState.spreadMultDict[dstFuel] || 0.0;
            if (dstFuelMult <= 0.0) continue;

            const dstElev = elevation[ty][tx];
            const effSlope = (dstElev - srcElev) / (dist * cellSize);
            const f_slope = effSlope > 0 ? Math.exp(2.8 * Math.min(1.2, effSlope)) : Math.exp(-1.4 * Math.min(0.8, Math.abs(effSlope)));

            // Wind alignment
            const r_u = dx / dist, r_v = dy / dist;
            const cosPhi = (r_u * u_base + r_v * v_base);
            const f_wind = cosPhi > 0 ? Math.exp(0.042 * windSpeed * cosPhi) : Math.exp(-0.025 * windSpeed * Math.abs(cosPhi));

            const dstDens = fuelDensity[ty][tx];
            const dstMoist = fuelMoisture[ty][tx];
            const dstRisk = historicalRisk[ty][tx];

            const f_fuel = dstFuelMult * Math.pow(dstDens, 0.8);
            const f_moist = Math.pow(Math.max(0.02, Math.min(1.0, 1.0 - (dstMoist / 0.32))), 1.4);
            const f_risk = 1.0 + 0.45 * dstRisk;

            const spreadRate = simState.baseSpreadProb * simState.spreadFactor * srcInt * f_fuel * f_moist * f_slope * f_wind * f_temp * f_humid * f_risk / dist;
            const prob = Math.max(0.0, Math.min(0.98, 1.0 - Math.exp(-0.35 * spreadRate)));

            if (Math.random() < prob) {{
                newlyIgnited[ty][tx] = 1;
                newIntensities[ty][tx] = Math.max(newIntensities[ty][tx], Math.min(1.0, Math.max(0.3, 0.6 * srcInt + 0.4 * (f_fuel / 2.0))));
            }}
        }}
    }}

    // Spot fires / ember lofting downwind
    if (simState.enableSpotting && windSpeed > 25.0) {{
        const spotProb = simState.spottingProb * (windSpeed / 40.0);
        for (let i = 0; i < activeMask.length; i++) {{
            const [sy, sx] = activeMask[i];
            if (simState.intensity[sy][sx] > 0.65 && Math.random() < spotProb) {{
                const distCells = Math.floor(Math.random() * 4) + 2;
                const spX = Math.round(sx + u_base * distCells + (Math.random() - 0.5) * 1.5);
                const spY = Math.round(sy + v_base * distCells + (Math.random() - 0.5) * 1.5);
                if (spX >= 0 && spX < nx && spY >= 0 && spY < ny && simState.state[spY][spX] === 0) {{
                    if ((simState.spreadMultDict[fuelType[spY][spX]] || 0) > 0) {{
                        newlyIgnited[spY][spX] = 1;
                        newIntensities[spY][spX] = 0.75;
                    }}
                }}
            }}
        }}
    }}

    // Burn progress & burnout decay
    for (let i = 0; i < activeMask.length; i++) {{
        const [sy, sx] = activeMask[i];
        simState.stepsBurning[sy][sx] += 1;
        const maxRes = simState.residenceDict[fuelType[sy][sx]] || 8;
        const steps = simState.stepsBurning[sy][sx];

        if (steps >= maxRes) {{
            simState.state[sy][sx] = 2; // BURNED OUT
            simState.intensity[sy][sx] = 0.0;
            simState.flameHeight[sy][sx] = 0.0;
        }} else {{
            const normT = steps / maxRes;
            if (normT < 0.25) {{
                simState.intensity[sy][sx] = Math.min(1.0, simState.intensity[sy][sx] * 1.15);
            }} else {{
                simState.intensity[sy][sx] = Math.max(0.1, simState.intensity[sy][sx] * (1.0 - (normT - 0.25) / 0.75));
            }}
            simState.flameHeight[sy][sx] = Math.max(0.5, simState.intensity[sy][sx] * 3.5);
        }}
    }}

    // Apply newly ignited cells
    for (let y = 0; y < ny; y++) {{
        for (let x = 0; x < nx; x++) {{
            if (newlyIgnited[y][x] === 1) {{
                simState.state[y][x] = 1;
                simState.intensity[y][x] = newIntensities[y][x];
                simState.stepsBurning[y][x] = 1;
                simState.flameHeight[y][x] = 2.5;
            }}
        }}
    }}

    simState.currentStep += 1;
    simState.elapsedMinutes += simState.minutesPerStep;

    // Update Structure Threat Status
    updateStructureThreats();

    // Record Step in Live Table History
    recordCurrentTimestep();

    // Re-render visual layers
    updateTerrainTexture();
    updateFlameMeshes();
    updateHUD();
}}

// Update distances & threat status of WUI structures
function updateStructureThreats() {{
    const activeCoords = [];
    for (let y = 0; y < ny; y++) {{
        for (let x = 0; x < nx; x++) {{
            if (simState.state[y][x] === 1) activeCoords.push([y, x]);
        }}
    }}

    simState.structures.forEach(s => {{
        if (simState.state[s.y][s.x] === 2) {{
            s.status = "BURNED";
            s.threat_level = "CRITICAL";
            s.distance_to_fire_m = 0;
        }} else if (simState.state[s.y][s.x] === 1) {{
            s.status = "THREATENED";
            s.threat_level = "CRITICAL";
            s.distance_to_fire_m = 0;
        }} else if (activeCoords.length > 0) {{
            let minDist = 999999;
            for (let i = 0; i < activeCoords.length; i++) {{
                const [ay, ax] = activeCoords[i];
                const d = Math.sqrt((ax - s.x)**2 + (ay - s.y)**2) * cellSize;
                if (d < minDist) minDist = d;
            }}
            s.distance_to_fire_m = minDist;
            if (minDist < 300) {{
                s.status = "THREATENED";
                s.threat_level = "HIGH";
            }} else if (minDist < 800) {{
                s.status = "THREATENED";
                s.threat_level = "MODERATE";
            }} else {{
                s.status = "INTACT";
                s.threat_level = "LOW";
            }}
        }}
    }});

    createStructures();
}}

// ==================== HUD TELEMETRY UPDATE ====================
function updateHUD() {{
    let burningCells = 0, burnedCells = 0;
    for (let y = 0; y < ny; y++) {{
        for (let x = 0; x < nx; x++) {{
            if (simState.state[y][x] === 1) burningCells++;
            if (simState.state[y][x] === 2) burnedCells++;
        }}
    }}

    const totalBurnedHa = (burningCells + burnedCells) * cellAreaHa;
    const threatenedCount = simState.structures.filter(s => s.status === "THREATENED").length;
    const totalStructures = simState.structures.length;

    document.getElementById("hud-step-val").innerHTML = `${{simState.currentStep}} <span style="font-size:0.75rem; color:#94A3B8;">(${{Math.round(simState.elapsedMinutes)}}m)</span>`;
    document.getElementById("hud-active-val").innerHTML = `${{burningCells}} <span style="font-size:0.75rem; color:#94A3B8;">cells</span>`;
    document.getElementById("hud-burned-val").innerHTML = `${{totalBurnedHa.toFixed(1)}} <span style="font-size:0.75rem; color:#94A3B8;">ha</span>`;
    
    const wuiEl = document.getElementById("hud-wui-val");
    wuiEl.innerText = `${{threatenedCount}} / ${{totalStructures}}`;
    wuiEl.className = threatenedCount > 0 ? "hud-val alert" : "hud-val safe";

    document.getElementById("hud-fwi-val").innerText = `${{Math.round(simState.weather.fwi)}} / 100`;
    document.getElementById("hud-wind-val").innerText = `${{Math.round(simState.weather.windSpeed)}} km/h`;

    const badge = document.getElementById("hud-status-badge");
    if (simState.isPlaying) {{
        badge.innerText = "SIMULATION RUNNING";
        badge.style.color = "#10B981";
    }} else if (burningCells > 0) {{
        badge.innerText = "SIMULATION PAUSED";
        badge.style.color = "#F59E0B";
    }} else {{
        badge.innerText = "SIMULATION IDLE";
        badge.style.color = "#94A3B8";
    }}
}}

// ==================== UI CONTROLS & INTERACTIVITY ====================
const btnPlayPause = document.getElementById("btn-play-pause");
const btnPlayIcon = document.getElementById("btn-play-icon");
const btnPlayText = document.getElementById("btn-play-text");

function updatePlayButtonUI() {{
    if (simState.isPlaying) {{
        btnPlayIcon.innerText = "⏸";
        btnPlayText.innerText = "PAUSE";
        btnPlayPause.classList.add("primary");
    }} else {{
        btnPlayIcon.innerText = "▶";
        btnPlayText.innerText = simState.currentStep > 0 ? "RESUME" : "RUN SIMULATION";
        btnPlayPause.classList.remove("primary");
    }}
    updateHUD();
}}

btnPlayPause.addEventListener("click", () => {{
    let burning = 0;
    for (let y = 0; y < ny; y++) {{
        for (let x = 0; x < nx; x++) {{
            if (simState.state[y][x] === 1) burning++;
        }}
    }}
    if (burning === 0 && !simState.isPlaying) {{
        // Auto ignite at center or preset ignition
        const igPt = simState.ignitionPoints[0] || [Math.floor(nx / 2), Math.floor(ny / 2)];
        ignitePoint(igPt[0], igPt[1], 1);
    }}
    simState.isPlaying = !simState.isPlaying;
    updatePlayButtonUI();
}});

document.getElementById("btn-step").addEventListener("click", () => {{
    simState.isPlaying = false;
    computeSimulationStep();
    updatePlayButtonUI();
}});

document.getElementById("btn-reset").addEventListener("click", () => {{
    simState.isPlaying = false;
    simState.currentStep = 0;
    simState.elapsedMinutes = 0;
    for (let y = 0; y < ny; y++) {{
        for (let x = 0; x < nx; x++) {{
            simState.state[y][x] = (fuelType[y][x] === 0) ? 3 : 0;
            simState.intensity[y][x] = 0;
            simState.flameHeight[y][x] = 0;
            simState.stepsBurning[y][x] = 0;
        }}
    }}
    simRecords.length = 0;
    document.getElementById("sim-records-tbody").innerHTML = "";
    const igPt = simState.ignitionPoints[0] || [Math.floor(nx / 2), Math.floor(ny / 2)];
    ignitePoint(igPt[0], igPt[1], 1);
    updatePlayButtonUI();
    updateTerrainTexture();
    updateFlameMeshes();
}});

// Speed Buttons
document.querySelectorAll(".speed-btn").forEach(btn => {{
    btn.addEventListener("click", (e) => {{
        document.querySelectorAll(".speed-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        simState.simSpeed = parseInt(btn.dataset.speed);
        simState.stepIntervalMs = Math.round(1000 / (simState.simSpeed * 1.5));
    }});
}});

// Toggle Live Results Table Drawer
const tableDrawer = document.getElementById("table-drawer");
const btnToggleTable = document.getElementById("btn-toggle-table");
btnToggleTable.addEventListener("click", () => {{
    const isShown = (tableDrawer.style.display === "flex");
    tableDrawer.style.display = isShown ? "none" : "flex";
    btnToggleTable.classList.toggle("active", !isShown);
}});
document.getElementById("btn-close-table").addEventListener("click", () => {{
    tableDrawer.style.display = "none";
    btnToggleTable.classList.remove("active");
}});

// Export CSV Functionality
document.getElementById("btn-export-csv").addEventListener("click", () => {{
    if (simRecords.length === 0) return;
    const headers = ["Step", "Simulation Time (min)", "Wind Speed (km/h)", "Wind Direction (°)", "Temperature (°C)", "Humidity (%)", "Burning Cells", "Burned Cells", "Active Fire Area (ha)", "Total Burned Area (ha)", "Burned Area (%)", "Threatened Structures", "Burned Structures"];
    const rows = simRecords.map(r => [
        r.step,
        r.elapsedMin.toFixed(1),
        r.windSpeed.toFixed(1),
        r.windDir.toFixed(0),
        r.temp.toFixed(1),
        r.humidity.toFixed(0),
        r.burningCells,
        r.burnedCells,
        r.activeAreaHa.toFixed(2),
        r.totalBurnedHa.toFixed(2),
        r.burnedPct.toFixed(1),
        r.threatenedCount,
        r.burnedStructures
    ]);
    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "wildfire_simulation_records.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}});

// Click to Ignite Toggle
const btnIgniteTool = document.getElementById("btn-ignite-tool");
btnIgniteTool.addEventListener("click", () => {{
    simState.clickToIgnite = !simState.clickToIgnite;
    btnIgniteTool.classList.toggle("active", simState.clickToIgnite);
    container.style.cursor = simState.clickToIgnite ? "crosshair" : "grab";
}});

function ignitePoint(gx, gy, radius = 1) {{
    for (let dy = -radius; dy <= radius; dy++) {{
        for (let dx = -radius; dx <= radius; dx++) {{
            const x = gx + dx, y = gy + dy;
            if (x >= 0 && x < nx && y >= 0 && y < ny) {{
                if (fuelType[y][x] !== 0) {{
                    simState.state[y][x] = 1;
                    simState.intensity[y][x] = 0.95;
                    simState.stepsBurning[y][x] = 1;
                    simState.flameHeight[y][x] = 2.5;
                }}
            }}
        }}
    }}
    recordCurrentTimestep();
    updateTerrainTexture();
    updateFlameMeshes();
    updateHUD();
}}

// Layer Selector
document.getElementById("sel-layer-mode").addEventListener("change", (e) => {{
    currentLayer = e.target.value;
    updateTerrainTexture();
}});

// Checkbox Toggles
document.getElementById("chk-structures").addEventListener("change", createStructures);
document.getElementById("chk-evac").addEventListener("change", createCorridors);
document.getElementById("chk-tactics").addEventListener("change", createFirefighting);
document.getElementById("chk-wind").addEventListener("change", updateWindArrow);
document.getElementById("chk-particles").addEventListener("change", updateFlameMeshes);

// Camera Preset Buttons
document.getElementById("cam-persp").addEventListener("click", setPerspectiveView);
document.getElementById("cam-top").addEventListener("click", () => {{
    camera.position.set(centerX, maxZ + worldWidth * 1.3, centerY + 0.1);
    controls.target.set(centerX, centerZ, centerY);
    controls.update();
}});
document.getElementById("cam-canyon").addEventListener("click", () => {{
    camera.position.set(centerX - worldWidth * 0.45, centerZ + (maxZ - minZ) * 0.8, centerY - worldHeight * 0.45);
    controls.target.set(centerX, centerZ + 50, centerY);
    controls.update();
}});
document.getElementById("cam-reset").addEventListener("click", setPerspectiveView);

// ==================== RAYCASTING & 3D MOUSE INTERACTIONS ====================
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

container.addEventListener("click", (event) => {{
    if (!simState.clickToIgnite) return;

    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);

    const intersects = raycaster.intersectObject(terrainMesh);
    if (intersects.length > 0) {{
        const pt = intersects[0].point;
        const gx = Math.round(pt.x / cellSize);
        const gy = Math.round(pt.z / cellSize);
        if (gx >= 0 && gx < nx && gy >= 0 && gy < ny) {{
            ignitePoint(gx, gy, 1);
        }}
    }}
}});

// Hover Tooltip on WUI Structures
window.addEventListener("mousemove", (event) => {{
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);

    const intersects = raycaster.intersectObjects(structureObjects, true);
    if (intersects.length > 0) {{
        let obj = intersects[0].object;
        while (obj && !obj.userData.structure && obj.parent) {{
            obj = obj.parent;
        }}
        if (obj && obj.userData.structure) {{
            const s = obj.userData.structure;
            tooltip.style.display = "block";
            tooltip.style.left = (event.clientX + 14) + "px";
            tooltip.style.top = (event.clientY + 14) + "px";
            tooltip.innerHTML = `
                <div style="font-weight:700; color:#38BDF8; font-size:0.85rem; margin-bottom:3px;">🏡 ${{s.name}}</div>
                <div style="color:#94A3B8; font-size:0.75rem;">Type: ${{s.type}}</div>
                <div style="margin-top:4px;">Status: <b style="color:${{s.status === 'THREATENED' ? '#EF4444' : s.status === 'BURNED' ? '#94A3B8' : '#10B981'}}">${{s.status}}</b></div>
                <div>Threat Level: <b>${{s.threat_level}}</b></div>
                <div>Dist to Fire: <b>${{s.distance_to_fire_m < 9000 ? s.distance_to_fire_m.toFixed(0) + 'm' : 'Safe (>5km)'}}</b></div>
                <div>Defensibility: <b>${{Math.round(s.defensibility * 100)}}%</b></div>
            `;
            return;
        }}
    }}
    tooltip.style.display = "none";
}});

// Window Resize Handling
window.addEventListener("resize", () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}});

// ==================== MAIN 60 FPS ANIMATION LOOP ====================
let lastStepTime = 0;

function animate(now) {{
    requestAnimationFrame(animate);

    // Simulation Timer Step
    if (simState.isPlaying && (now - lastStepTime > simState.stepIntervalMs)) {{
        computeSimulationStep();
        lastStepTime = now;
    }}

    // Subtle Flame Animation Flicker
    if (flameParticlesGroup.children.length > 0) {{
        flameParticlesGroup.children.forEach(mesh => {{
            mesh.scale.y = mesh.scale.x * (0.85 + Math.sin(now * 0.015 + mesh.position.x) * 0.25);
        }});
    }}

    controls.update();
    renderer.render(scene, camera);
}}

updateHUD();
requestAnimationFrame(animate);

</script>
</body>
</html>
"""
        return html_content
