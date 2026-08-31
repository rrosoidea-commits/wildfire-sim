"""
plot_3d.py - Interactive 3D Terrain, Wildfire Front, Firefighting, WUI Structures, and Evacuation Visualizer.
Renders elevation mesh, realistic dynamic fire front, burned ash zones,
firelines, water drops, WUI buildings with threat status, and safe evacuation corridors.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, List, Tuple, Dict, Any, Union
from core.terrain import TerrainGrid, FuelType, Structure
from core.weather import WeatherCondition
from core.fire_ca import WildfireSimulation, FireState
from core.risk_assessment import RiskAssessmentResult
from core.wui_evacuation import EvacuationAnalysisResult, EvacuationCorridor

class Wildfire3DVisualizer:
    @staticmethod
    def create_3d_figure(
        simulation: WildfireSimulation,
        risk_result: Optional[RiskAssessmentResult] = None,
        evacuation_result: Optional[EvacuationAnalysisResult] = None,
        layer_mode: str = "fire_dynamic",
        show_wind_vectors: bool = True,
        show_ignition_markers: bool = True,
        show_structures: bool = True,
        show_evac_corridors: bool = True,
        show_firefighting: bool = True,
        z_exaggeration: float = 1.0
    ) -> go.Figure:
        terrain = simulation.terrain
        nx, ny = terrain.nx, terrain.ny
        cell_size = terrain.cell_size_m

        xx, yy = terrain.get_mesh_grid()
        zz = terrain.elevation * z_exaggeration

        # Build surface color matrix based on layer_mode
        if layer_mode == "fire_dynamic":
            surface_color = terrain.get_base_surface_color().copy()

            # Overlay Firefighting Tactics on surface
            if hasattr(simulation, 'firefighting_mgr') and simulation.firefighting_mgr is not None:
                fm = simulation.firefighting_mgr
                if np.any(fm.tactic_mask == 1):
                    surface_color[fm.tactic_mask == 1] = 0.82 # Fireline Mineral Earth
                if np.any(fm.tactic_mask == 2):
                    surface_color[fm.tactic_mask == 2] = 0.12 # Water Drenched Retardant

            # Overlay Burned Out / Ash cells
            burned_mask = (simulation.state == FireState.BURNED_OUT)
            if np.any(burned_mask):
                surface_color[burned_mask] = 0.91

            # Overlay Active Burning cells with intensity
            active_mask = (simulation.state == FireState.BURNING)
            if np.any(active_mask):
                intensities = simulation.fire_intensity[active_mask]
                surface_color[active_mask] = 0.94 + 0.06 * np.clip(intensities, 0.0, 1.0)

            custom_colorscale = [
                [0.00, "#1565C0"], # Water Deep Blue
                [0.15, "#42A5F5"], # Water Light Blue
                [0.16, "#9E9E9E"], # Rock Light Grey
                [0.28, "#424242"], # Road / Bare Soil
                [0.29, "#CDDC39"], # Grass Pale Lime
                [0.40, "#9E9D24"], # Grass Olive
                [0.41, "#FBC02D"], # Shrub Amber Gold
                [0.52, "#F57F17"], # Shrub Warm Gold
                [0.53, "#558B2F"], # Woodland Olive Green
                [0.65, "#33691E"], # Woodland Deep Olive
                [0.66, "#1B5E20"], # Dense Conifer Dark Forest
                [0.78, "#004D40"], # Conifer Deep Teal/Pine
                [0.79, "#8D6E63"], # Slash Earth Brown
                [0.88, "#5D4037"], # Slash Dark Brown
                [0.89, "#212121"], # Ash Charcoal
                [0.93, "#37474F"], # Ash Grey-Black
                [0.94, "#D50000"], # Fire Red
                [0.96, "#FF6D00"], # Fire Orange
                [0.98, "#FFD600"], # Fire Bright Gold
                [1.00, "#FFFFFF"]  # Fire White Hot
            ]
            cmin, cmax = 0.0, 1.0
            colorbar_title = "Landscape & Fire State"

        elif layer_mode == "risk_map" and risk_result is not None:
            surface_color = risk_result.composite_risk_score
            custom_colorscale = [
                [0.00, "#2E7D32"], # Low Risk (Green)
                [0.25, "#8BC34A"], # Low-Moderate (Light Green)
                [0.45, "#FDD835"], # Moderate (Yellow)
                [0.65, "#FB8C00"], # High (Orange)
                [0.80, "#E53935"], # Very High (Red)
                [1.00, "#880E4F"]  # Extreme (Purple/Deep Crimson)
            ]
            cmin, cmax = 0.0, 100.0
            colorbar_title = "Wildfire Risk (0-100)"

        elif layer_mode == "slope":
            surface_color = terrain.slope_deg
            custom_colorscale = "Viridis"
            cmin, cmax = 0.0, 45.0
            colorbar_title = "Slope Angle (°)"

        elif layer_mode == "fuel_moisture":
            surface_color = terrain.fuel_moisture * 100.0
            custom_colorscale = "YlGnBu_r"
            cmin, cmax = 2.0, 35.0
            colorbar_title = "Fuel Moisture (%)"

        elif layer_mode == "historical_risk":
            surface_color = terrain.historical_risk * 100.0
            custom_colorscale = "Hot_r"
            cmin, cmax = 0.0, 100.0
            colorbar_title = "Historical Risk (%)"

        else: # Elevation
            surface_color = zz
            custom_colorscale = "Earth"
            cmin, cmax = float(zz.min()), float(zz.max())
            colorbar_title = "Elevation (m)"

        fig = go.Figure()

        # 1. Main 3D Terrain Surface
        fig.add_trace(go.Surface(
            x=xx,
            y=yy,
            z=zz,
            surfacecolor=surface_color,
            cmin=cmin,
            cmax=cmax,
            colorscale=custom_colorscale,
            showscale=True,
            colorbar=dict(
                title=dict(text=colorbar_title, side="right", font=dict(color="#ECEFF1", size=12)),
                tickfont=dict(color="#ECEFF1", size=10),
                len=0.70,
                thickness=16,
                x=1.02
            ),
            lighting=dict(ambient=0.65, diffuse=0.85, specular=0.25, roughness=0.6, fresnel=0.3),
            contours=dict(z=dict(show=True, usecolormap=False, project_z=False, color="rgba(255,255,255,0.06)", size=50)),
            name="Terrain Surface",
            hoverinfo="x+y+z+text"
        ))

        # 2. Firefighting Tactics Markers in 3D
        if show_firefighting and hasattr(simulation, 'firefighting_mgr') and simulation.firefighting_mgr is not None:
            fm = simulation.firefighting_mgr

            # Firelines (Containment Lines)
            if len(fm.firelines) > 0:
                fl_x, fl_y, fl_z = [], [], []
                for fl in fm.firelines:
                    num_pts = max(abs(fl.x2 - fl.x1), abs(fl.y2 - fl.y1), 1) * 2
                    lx = np.linspace(fl.x1, fl.x2, num_pts)
                    ly = np.linspace(fl.y1, fl.y2, num_pts)
                    for px, py in zip(lx, ly):
                        ix, iy = int(round(px)), int(round(py))
                        if 0 <= ix < nx and 0 <= iy < ny:
                            fl_x.append(ix * cell_size)
                            fl_y.append(iy * cell_size)
                            fl_z.append(zz[iy, ix] + 4.0)

                if fl_x:
                    fig.add_trace(go.Scatter3d(
                        x=fl_x, y=fl_y, z=fl_z,
                        mode="markers",
                        marker=dict(size=4, color="#FF9100", symbol="square"),
                        name=f"🛡️ Firelines Built ({len(fm.firelines)} lines, {fm.total_fireline_km:.1f} km)",
                        hoverinfo="text",
                        text=[f"Fireline Barrier" for _ in fl_x]
                    ))

            # Water Drops (Aerial Suppression)
            if len(fm.water_drops) > 0:
                wd_x = [d.x * cell_size for d in fm.water_drops]
                wd_y = [d.y * cell_size for d in fm.water_drops]
                wd_z = [zz[d.y, d.x] + 15.0 for d in fm.water_drops]

                fig.add_trace(go.Scatter3d(
                    x=wd_x, y=wd_y, z=wd_z,
                    mode="markers+text",
                    marker=dict(size=9, color="#00E5FF", symbol="diamond", line=dict(color="#FFFFFF", width=2)),
                    text=[f"💧 Drop #{d.id}" for d in fm.water_drops],
                    textposition="top center",
                    textfont=dict(color="#00E5FF", size=10),
                    name=f"💧 Water Drops ({len(fm.water_drops)})"
                ))

            # Backburns (Tactical Burnout)
            if len(fm.backburns) > 0:
                bb_x = [b.x * cell_size for b in fm.backburns]
                bb_y = [b.y * cell_size for b in fm.backburns]
                bb_z = [zz[b.y, b.x] + 8.0 for b in fm.backburns]

                fig.add_trace(go.Scatter3d(
                    x=bb_x, y=bb_y, z=bb_z,
                    mode="markers+text",
                    marker=dict(size=7, color="#FF3D00", symbol="circle-open", line=dict(color="#FFD600", width=2)),
                    text=[f"🔥 Backburn #{b.id}" for b in fm.backburns],
                    textposition="top center",
                    textfont=dict(color="#FFD600", size=10),
                    name=f"🔥 Tactical Backburns ({len(fm.backburns)})"
                ))

        # 3. WUI Structures in 3D
        if show_structures and len(terrain.structures) > 0:
            status_colors = {
                "INTACT": "#00E676",      # Emerald Green
                "THREATENED": "#FF1744",  # Crimson / Orange Alert
                "BURNED": "#212121",      # Charcoal / Black
                "DEFENDED": "#2979FF"     # Protected Blue
            }
            symbols = {
                "INTACT": "square",
                "THREATENED": "diamond",
                "BURNED": "cross",
                "DEFENDED": "circle"
            }

            for status_key, group_name in [
                ("THREATENED", "⚠️ Threatened Structures"),
                ("DEFENDED", "🛡️ Defended Structures"),
                ("INTACT", "🏡 Intact WUI Buildings"),
                ("BURNED", "⬛ Burned Structures")
            ]:
                group_structs = [s for s in terrain.structures if s.status == status_key]
                if group_structs:
                    st_x = [s.x * cell_size for s in group_structs]
                    st_y = [s.y * cell_size for s in group_structs]
                    st_z = [zz[s.y, s.x] + 12.0 for s in group_structs]

                    hover_texts = [
                        f"<b>{s.name}</b><br>Type: {s.structure_type}<br>Status: <b>{s.status}</b> ({s.threat_level} Threat)<br>Dist to Fire: {s.distance_to_fire_m:.0f}m<br>Defensibility: {s.defensibility_score*100:.0f}%"
                        for s in group_structs
                    ]

                    fig.add_trace(go.Scatter3d(
                        x=st_x, y=st_y, z=st_z,
                        mode="markers+text",
                        marker=dict(
                            size=8 if status_key != "BURNED" else 6,
                            color=status_colors[status_key],
                            symbol=symbols[status_key],
                            line=dict(color="#FFFFFF", width=1.5)
                        ),
                        text=[s.id for s in group_structs],
                        textposition="top center",
                        textfont=dict(color=status_colors[status_key], size=9),
                        name=f"{group_name} ({len(group_structs)})",
                        hoverinfo="text",
                        texttemplate="%{text}",
                        customdata=hover_texts,
                        hovertemplate="%{customdata}<extra></extra>"
                    ))

        # 4. Evacuation Corridors in 3D
        if show_evac_corridors and evacuation_result is not None and len(evacuation_result.corridors) > 0:
            for idx, corr in enumerate(evacuation_result.corridors):
                if len(corr.path_coords) < 2:
                    continue

                px = [pt[0] * cell_size for pt in corr.path_coords]
                py = [pt[1] * cell_size for pt in corr.path_coords]
                pz = [zz[pt[1], pt[0]] + 6.0 for pt in corr.path_coords]

                if corr.status == "SAFE":
                    line_color = "#00E676" # Green
                elif corr.status == "CAUTION":
                    line_color = "#FFD600" # Yellow
                else:
                    line_color = "#FF1744" # Red Blocked

                show_leg = (idx == 0) # Only show first in legend to avoid clutter
                fig.add_trace(go.Scatter3d(
                    x=px, y=py, z=pz,
                    mode="lines",
                    line=dict(color=line_color, width=5),
                    name="🚶 Evacuation Corridors" if show_leg else None,
                    showlegend=show_leg,
                    hoverinfo="text",
                    text=[f"Evacuation Route: {corr.structure_name} → {corr.exit_name}<br>Status: <b>{corr.status}</b><br>{corr.status_notes}" for _ in px]
                ))

        # 5. Active Fire Flame Front in 3D
        active_coords = np.argwhere(simulation.state == FireState.BURNING)
        if len(active_coords) > 0:
            flame_x = active_coords[:, 1] * cell_size
            flame_y = active_coords[:, 0] * cell_size
            flame_z = zz[active_coords[:, 0], active_coords[:, 1]] + simulation.flame_height_m[active_coords[:, 0], active_coords[:, 1]] * 2.8

            fig.add_trace(go.Scatter3d(
                x=flame_x,
                y=flame_y,
                z=flame_z,
                mode="markers",
                marker=dict(
                    size=4,
                    color="#FFEA00",
                    symbol="circle",
                    opacity=0.9,
                    line=dict(color="#FF1744", width=1.5)
                ),
                name=f"🔥 Active Flame Front ({len(active_coords)} cells)",
                hoverinfo="text",
                text=[f"Active Flame<br>Grid: ({int(x/cell_size)}, {int(y/cell_size)})<br>Flame: {h:.1f}m" for x, y, h in zip(flame_x, flame_y, simulation.flame_height_m[active_coords[:, 0], active_coords[:, 1]])]
            ))

        # 6. Ignition Origins
        if show_ignition_markers and len(simulation.ignition_points) > 0:
            ig_x = [pt[0] * cell_size for pt in simulation.ignition_points]
            ig_y = [pt[1] * cell_size for pt in simulation.ignition_points]
            ig_z = [zz[pt[1], pt[0]] + 25.0 for pt in simulation.ignition_points]

            fig.add_trace(go.Scatter3d(
                x=ig_x, y=ig_y, z=ig_z,
                mode="markers+text",
                marker=dict(size=8, color="#D500F9", symbol="diamond", line=dict(color="#FFFFFF", width=2)),
                text=[f"Ignition #{i+1}" for i in range(len(ig_x))],
                textposition="top center",
                textfont=dict(color="#FFFFFF", size=11),
                name="🎯 Ignition Origins"
            ))

        # 7. Spot Fires
        if len(simulation.spot_fire_events) > 0:
            spot_y = [ev[1] for ev in simulation.spot_fire_events]
            spot_x = [ev[2] for ev in simulation.spot_fire_events]
            sp_x = [x * cell_size for x in spot_x]
            sp_y = [y * cell_size for y in spot_y]
            sp_z = [zz[y, x] + 15.0 for y, x in zip(spot_y, spot_x)]

            fig.add_trace(go.Scatter3d(
                x=sp_x, y=sp_y, z=sp_z,
                mode="markers",
                marker=dict(size=6, color="#FF3D00", symbol="cross", line=dict(color="#FFD600", width=1.5)),
                name=f"⚡ Spot Fires ({len(simulation.spot_fire_events)})"
            ))

        # 8. Prevailing Wind Vector Vector in 3D
        if show_wind_vectors:
            u_dir, v_dir = simulation.weather.wind_vector
            center_x = (nx * cell_size) * 0.5
            center_y = (ny * cell_size) * 0.5
            max_z = float(np.max(zz)) + 120.0

            arrow_len = (nx * cell_size) * 0.22
            tip_x = center_x + u_dir * arrow_len
            tip_y = center_y + v_dir * arrow_len

            fig.add_trace(go.Scatter3d(
                x=[center_x, tip_x],
                y=[center_y, tip_y],
                z=[max_z, max_z],
                mode="lines+markers+text",
                line=dict(color="#00E5FF", width=6),
                marker=dict(size=[3, 8], color=["#00E5FF", "#FFFFFF"], symbol=["circle", "diamond"]),
                text=["", f"💨 Wind: {simulation.weather.wind_speed_kmh:.0f} km/h from {simulation.get_cardinal_dir(simulation.weather.wind_direction_deg)} ({simulation.weather.wind_direction_deg:.0f}°)\nBlowing Towards {simulation.get_cardinal_dir((simulation.weather.wind_direction_deg + 180)%360)}"],
                textposition="top center",
                textfont=dict(color="#00E5FF", size=12),
                name="💨 Wind Vector"
            ))

        # Layout styling with uirevision to keep user camera, rotation, and zoom persistent
        fig.update_layout(
            uirevision="wildfire_3d_map",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            scene=dict(
                uirevision="wildfire_3d_map",
                xaxis=dict(title=dict(text="West → East (m)", font=dict(color="#94A3B8")), tickfont=dict(color="#94A3B8"), gridcolor="#1E293B", backgroundcolor="#0F172A", showbackground=True),
                yaxis=dict(title=dict(text="South → North (m)", font=dict(color="#94A3B8")), tickfont=dict(color="#94A3B8"), gridcolor="#1E293B", backgroundcolor="#0F172A", showbackground=True),
                zaxis=dict(title=dict(text="Elevation (m)", font=dict(color="#94A3B8")), tickfont=dict(color="#94A3B8"), gridcolor="#1E293B", backgroundcolor="#0F172A", showbackground=True),
                aspectratio=dict(x=1.0, y=1.0, z=0.38),
                camera=dict(eye=dict(x=1.45, y=-1.45, z=0.95), center=dict(x=0, y=0, z=-0.1))
            ),
            legend=dict(
                font=dict(color="#E2E8F0", size=11),
                bgcolor="rgba(15, 23, 42, 0.85)",
                bordercolor="#334155",
                borderwidth=1,
                x=0.01,
                y=0.99
            ),
            margin=dict(l=0, r=0, t=20, b=0),
            height=680
        )

        return fig

def get_cardinal_dir(deg: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int(round(deg / 22.5)) % 16
    return dirs[idx]
