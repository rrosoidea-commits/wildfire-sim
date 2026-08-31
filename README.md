# 🔥 Wildfire Simulation & WUI Evacuation System

An interactive, physics-driven Python wildfire simulation featuring a **NumPy-vectorized cellular automata combustion engine**, **interactive firefighting tactics** (firelines, water drops, backburns), **Wildland-Urban Interface (WUI) structure threat analysis**, and **safe evacuation corridor routing** with full 3D interactive terrain.

---

> **⚠️ ACADEMIC & RESEARCH PROTOTYPE DISCLAIMER:**  
> This software is an educational simulation and scientific prototype built for demonstrating cellular automata modeling and spatial analysis. It is **not** certified or intended for operational emergency management, wildfire command decisions, or real-time evacuation planning.

---

## 🌟 Key Features

1. **NumPy-Vectorized Combustion Engine (`NumPy`, `SciPy`)**:
   - High-performance vectorized 8-neighbor combustion transitions executing under 6 ms per step.
   - Comprehensive physics multipliers: Topographic slope & aspect, localized wind vectors with ridge acceleration, fuel types, canopy density, fuel moisture, temperature, humidity, and historical risk climatology.
   - Long-range spotting / ember lofting under extreme winds.

2. **Interactive Firefighting & Containment**:
   - **Firelines (Containment Lines)**: Construct bulldozer / handline containment barriers to stop or redirect fire spread.
   - **Aerial Water Drops**: Deploy helicopter/airtanker retardant drops that extinguish active burning cells and increase local fuel moisture.
   - **Tactical Backburns**: Prescribed burnout operations that consume available fuel ahead of the wildfire front.
   - Live tracking of constructed fireline distance, water drop counts, and backburned hectares.

3. **Wildland-Urban Interface (WUI) & Safe Evacuation Corridor Analysis**:
   - Synthetic WUI structures (homes, community facilities, commercial lodges, emergency shelters) distributed logically along valleys and roads.
   - Real-time threat classification: *Intact (Safe)*, *Threatened (Critical/High Proximity)*, *Defended (Protected by tactics)*, and *Burned*.
   - Dijkstra-based least-cost evacuation corridor solver identifying safe egress routes away from active fire fronts and steep terrain.

4. **Automatic Timestep Recording & CSV Export**:
   - Automatically records 14+ variables at every step: *Simulation time, wind speed, wind direction, temperature, humidity, fuel moisture, max & mean fire intensity, burning cells, burned cells, active fire area, total burned area, fire perimeter, max spread distance, and threatened structures*.
   - Interactive data table and one-click **CSV Download**.

5. **Simplified & Intuitive UI**:
   - Clear and easy-to-understand controls organized logically:
     **Fire Settings → Weather → Fire Location → Simulation Controls → 3D Map → Simulation Results**
   - Essential simulation controls: **RUN SIMULATION**, **PAUSE**, **RESUME**, **STOP FIRE**, **RESET**, **NEW FIRE LOCATION**.
   - Fully interactive 3D map allowing zoom, pan, rotate, and inspection of dynamic flames, structures, and safe evacuation corridors.

---

## 🛠️ Project Architecture

```
wildfire_sim/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Dependency specifications
├── README.md                   # Project documentation & reference
├── core/
│   ├── __init__.py
│   ├── terrain.py              # TerrainGrid, FuelType, Structure dataclasses & road network
│   ├── weather.py              # WeatherCondition, wind vector field & FWI calculation
│   ├── risk_assessment.py      # Multi-criteria wildfire hazard & risk assessment
│   ├── fire_ca.py              # Vectorized CA simulation engine & combustion physics
│   ├── firefighting.py         # FirefightingManager, firelines, water drops, backburns
│   ├── wui_evacuation.py       # Structure threat assessment & Dijkstra evacuation solver
│   └── gis_io.py               # ESRI ASCII Grid & GeoJSON perimeter export/import
├── visualization/
│   ├── __init__.py
│   ├── plot_3d.py              # Plotly 3D terrain surface, WUI buildings & evacuation corridors
│   ├── plot_2d.py              # 2D heatmaps (fire spread, risk maps, slope gradient)
│   └── dashboard.py           # Metric charts, time-series plots & structure safety charts
└── presets/
    ├── __init__.py
    └── scenarios.py            # Predefined realistic disaster scenarios & terrain presets
```

---

## 🚀 Installation & How to Run

### 1. Requirements
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application
Navigate to the `wildfire_sim` directory and start Streamlit:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
