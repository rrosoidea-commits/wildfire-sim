"""
plot_2d.py - 2D Spatial Analysis, Risk Heatmaps, and Fire Front Visualizations.
Provides interactive 2D maps for real-time fire monitoring, risk assessment layers,
topographic slope analysis, WUI structure locations, and evacuation corridors.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, List, Tuple, Dict, Any, Union
from core.terrain import TerrainGrid, FuelType, Structure
from core.fire_ca import WildfireSimulation, FireState
from core.risk_assessment import RiskAssessmentResult
from core.wui_evacuation import EvacuationAnalysisResult

class Wildfire2DVisualizer:
    @staticmethod
    def create_fire_map_2d(
        simulation: WildfireSimulation,
        evacuation_result: Optional[EvacuationAnalysisResult] = None
    ) -> go.Figure:
        terrain = simulation.terrain
        nx, ny = terrain.nx, terrain.ny

        # Discrete 2D visualization matrix
        # 0: Water, 1: Rock, 2: Grass, 3: Shrub, 4: Woodland, 5: Conifer, 6: Slash, 7: Ash, 8: Fire
        vis_grid = np.zeros((ny, nx), dtype=int)

        for ftype in range(7):
            vis_grid[terrain.fuel_type == ftype] = ftype

        # Fireline and water drops
        if hasattr(simulation, 'firefighting_mgr') and simulation.firefighting_mgr is not None:
            fm = simulation.firefighting_mgr
            vis_grid[fm.tactic_mask == 1] = 1 # Rock / Bare
            vis_grid[fm.tactic_mask == 2] = 0 # Water

        vis_grid[simulation.state == FireState.BURNED_OUT] = 7
        vis_grid[simulation.state == FireState.BURNING] = 8

        colorscale = [
            [0/8, "#1E88E5"], # 0: Water
            [1/8, "#B0BEC5"], # 1: Bare Ground
            [2/8, "#D4E157"], # 2: Grassland
            [3/8, "#FDD835"], # 3: Shrubland
            [4/8, "#7CB342"], # 4: Woodland
            [5/8, "#2E7D32"], # 5: Dense Forest
            [6/8, "#8D6E63"], # 6: Slash
            [7/8, "#263238"], # 7: Burned Ash
            [8/8, "#FF1744"]  # 8: Active Fire
        ]

        fig = go.Figure()

        # Main 2D Heatmap
        fig.add_trace(go.Heatmap(
            z=vis_grid,
            x=np.arange(nx),
            y=np.arange(ny),
            zmin=0,
            zmax=8,
            colorscale=colorscale,
            showscale=False,
            hoverinfo="x+y+text",
            text=[[f"({x}, {y})<br>Elev: {terrain.elevation[y, x]:.0f}m<br>Slope: {terrain.slope_deg[y, x]:.1f}°<br>Fuel: {FuelType.NAMES.get(terrain.fuel_type[y, x], 'Unknown')}" for x in range(nx)] for y in range(ny)]
        ))

        # Evacuation Corridors in 2D
        if evacuation_result is not None:
            for idx, corr in enumerate(evacuation_result.corridors):
                if len(corr.path_coords) > 1:
                    cx = [p[0] for p in corr.path_coords]
                    cy = [p[1] for p in corr.path_coords]
                    color = "#00E676" if corr.status == "SAFE" else "#FFD600" if corr.status == "CAUTION" else "#FF1744"
                    fig.add_trace(go.Scatter(
                        x=cx, y=cy,
                        mode="lines",
                        line=dict(color=color, width=3),
                        name="Evacuation Route" if idx == 0 else None,
                        showlegend=(idx == 0),
                        hoverinfo="skip"
                    ))

        # Active Fire Outline / Points
        active_coords = np.argwhere(simulation.state == FireState.BURNING)
        if len(active_coords) > 0:
            fig.add_trace(go.Scatter(
                x=active_coords[:, 1],
                y=active_coords[:, 0],
                mode="markers",
                marker=dict(size=7, color="#FFEA00", symbol="circle", line=dict(color="#D50000", width=1.5)),
                name=f"🔥 Active Fire Front ({len(active_coords)} cells)",
                hoverinfo="skip"
            ))

        # WUI Structures in 2D
        if len(terrain.structures) > 0:
            for status_key, s_col, s_sym in [
                ("THREATENED", "#FF1744", "diamond"),
                ("DEFENDED", "#2979FF", "circle"),
                ("INTACT", "#00E676", "square"),
                ("BURNED", "#212121", "cross")
            ]:
                g_structs = [s for s in terrain.structures if s.status == status_key]
                if g_structs:
                    fig.add_trace(go.Scatter(
                        x=[s.x for s in g_structs],
                        y=[s.y for s in g_structs],
                        mode="markers+text",
                        marker=dict(size=10, color=s_col, symbol=s_sym, line=dict(color="#FFFFFF", width=1.5)),
                        text=[s.id for s in g_structs],
                        textposition="top center",
                        textfont=dict(color=s_col, size=10),
                        name=f"Structure: {status_key.title()} ({len(g_structs)})",
                        hoverinfo="text",
                        hovertext=[f"{s.name} ({s.status}) - {s.distance_to_fire_m:.0f}m from fire" for s in g_structs]
                    ))

        # Ignitions
        if len(simulation.ignition_points) > 0:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in simulation.ignition_points],
                y=[p[1] for p in simulation.ignition_points],
                mode="markers+text",
                marker=dict(size=12, color="#E040FB", symbol="diamond", line=dict(color="#FFFFFF", width=2)),
                text=[f"P{i+1}" for i in range(len(simulation.ignition_points))],
                textposition="top center",
                textfont=dict(color="#FFFFFF", size=12),
                name="🎯 Ignition Points"
            ))

        fig.update_layout(
            title=dict(text=f"🗺️ Wildfire Spread & WUI Map (Step {simulation.current_step} - T+{simulation.elapsed_minutes:.0f}m)", font=dict(color="#E2E8F0", size=16)),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            xaxis=dict(title="Grid X Coordinate", color="#94A3B8", gridcolor="#1E293B", dtick=10),
            yaxis=dict(title="Grid Y Coordinate", color="#94A3B8", gridcolor="#1E293B", dtick=10, scaleanchor="x", scaleratio=1),
            legend=dict(font=dict(color="#E2E8F0", size=11), bgcolor="rgba(15, 23, 42, 0.8)"),
            margin=dict(l=40, r=40, t=50, b=40),
            height=580
        )
        return fig

    @staticmethod
    def create_risk_map_2d(risk_result: RiskAssessmentResult, terrain: TerrainGrid) -> go.Figure:
        nx, ny = terrain.nx, terrain.ny
        fig = go.Figure()

        custom_colorscale = [
            [0.00, "#2E7D32"], # Low Risk (Green)
            [0.25, "#8BC34A"], # Low-Moderate (Light Green)
            [0.45, "#FDD835"], # Moderate (Yellow)
            [0.65, "#FB8C00"], # High (Orange)
            [0.80, "#E53935"], # Very High (Red)
            [1.00, "#880E4F"]  # Extreme (Deep Crimson)
        ]

        fig.add_trace(go.Heatmap(
            z=risk_result.composite_risk_score,
            x=np.arange(nx),
            y=np.arange(ny),
            zmin=0,
            zmax=100,
            colorscale=custom_colorscale,
            colorbar=dict(
                title=dict(text="Risk Index (0-100)", font=dict(color="#E2E8F0", size=12)),
                tickfont=dict(color="#E2E8F0", size=10),
                thickness=15
            ),
            hoverinfo="x+y+z+text",
            text=[[f"Grid: ({x}, {y})<br>Composite Risk: {risk_result.composite_risk_score[y, x]:.1f}<br>Fuel Hazard: {risk_result.fuel_hazard_score[y, x]:.1f}<br>Topo Hazard: {risk_result.topographic_hazard_score[y, x]:.1f}" for x in range(nx)] for y in range(ny)]
        ))

        # Add elevation contour lines
        fig.add_trace(go.Contour(
            z=terrain.elevation,
            x=np.arange(nx),
            y=np.arange(ny),
            showscale=False,
            contours=dict(coloring='none', showlabels=True, labelfont=dict(size=9, color="#FFFFFF")),
            line=dict(color="rgba(255, 255, 255, 0.25)", width=1),
            hoverinfo="skip"
        ))

        fig.update_layout(
            title=dict(text="📊 Multi-Criteria Wildfire Hazard & Risk Index (WHRI)", font=dict(color="#E2E8F0", size=16)),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            xaxis=dict(title="Grid X Coordinate", color="#94A3B8", gridcolor="#1E293B"),
            yaxis=dict(title="Grid Y Coordinate", color="#94A3B8", gridcolor="#1E293B", scaleanchor="x", scaleratio=1),
            margin=dict(l=40, r=40, t=50, b=40),
            height=580
        )
        return fig

    @staticmethod
    def create_slope_aspect_2d(terrain: TerrainGrid) -> go.Figure:
        nx, ny = terrain.nx, terrain.ny
        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=terrain.slope_deg,
            x=np.arange(nx),
            y=np.arange(ny),
            zmin=0,
            zmax=45,
            colorscale="Plasma",
            colorbar=dict(title="Slope (°)", tickfont=dict(color="#E2E8F0"), thickness=15),
            hoverinfo="x+y+z+text",
            text=[[f"Grid: ({x}, {y})<br>Slope: {terrain.slope_deg[y, x]:.1f}°<br>Aspect: {terrain.aspect_deg[y, x]:.0f}°" for x in range(nx)] for y in range(ny)]
        ))

        fig.update_layout(
            title=dict(text="⛰️ Topographic Slope & Elevation Gradient Map", font=dict(color="#E2E8F0", size=16)),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            xaxis=dict(title="Grid X Coordinate", color="#94A3B8", gridcolor="#1E293B"),
            yaxis=dict(title="Grid Y Coordinate", color="#94A3B8", gridcolor="#1E293B", scaleanchor="x", scaleratio=1),
            margin=dict(l=40, r=40, t=50, b=40),
            height=580
        )
        return fig
