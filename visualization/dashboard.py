"""
dashboard.py - Real-time Analytics Dashboard & Time-Series Visualizations.
Generates metrics cards, fire behavior progression graphs, rate of spread charts,
WUI structure safety breakdowns, and risk distribution analytics.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple, Union
from core.fire_ca import WildfireSimulation, SimulationStepStats
from core.risk_assessment import RiskAssessmentResult
from core.wui_evacuation import EvacuationAnalysisResult

class WildfireDashboard:
    @staticmethod
    def create_history_charts(simulation: WildfireSimulation) -> Tuple[go.Figure, go.Figure]:
        if len(simulation.history) == 0:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                paper_bgcolor="#0F172A",
                plot_bgcolor="#0F172A",
                annotations=[dict(text="Simulation not started yet. Click RUN SIMULATION to view real-time charts.", showarrow=False, font=dict(color="#94A3B8", size=14))],
                height=300
            )
            return empty_fig, empty_fig

        df = pd.DataFrame([
            {
                "Step": s.step,
                "Time (min)": s.elapsed_minutes,
                "Burned Area (ha)": s.total_burned_area_ha,
                "Burned Area (%)": s.burned_area_pct,
                "Active Fire Area (ha)": s.active_fire_area_ha,
                "Active Cells": s.burning_cells,
                "Perimeter (km)": s.fire_perimeter_km,
                "Rate of Spread (ha/step)": s.rate_of_spread_ha_step,
                "Fuel Consumed (tons)": s.fuel_consumed_tons,
                "Max Intensity": s.fire_intensity_max,
                "Threatened Structures": s.threatened_structures,
                "Spot Fires": s.spot_fires_count
            }
            for s in simulation.history
        ])

        # Chart 1: Burned Area vs Active Fire Area
        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(
            x=df["Time (min)"],
            y=df["Burned Area (ha)"],
            mode="lines+markers",
            name="Total Burned Area (ha)",
            line=dict(color="#FF5722", width=3),
            fill="tozeroy",
            fillcolor="rgba(255, 87, 34, 0.15)"
        ))
        fig_area.add_trace(go.Scatter(
            x=df["Time (min)"],
            y=df["Active Fire Area (ha)"],
            mode="lines+markers",
            name="Active Front Area (ha)",
            line=dict(color="#FFEB3B", width=2.5, dash="dash")
        ))
        fig_area.update_layout(
            title=dict(text="🔥 Cumulative Burned Area & Active Fire Front", font=dict(color="#E2E8F0", size=14)),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            xaxis=dict(title="Elapsed Time (minutes)", color="#94A3B8", gridcolor="#1E293B"),
            yaxis=dict(title="Area (hectares)", color="#94A3B8", gridcolor="#1E293B"),
            legend=dict(font=dict(color="#E2E8F0"), bgcolor="rgba(15,23,42,0.7)"),
            margin=dict(l=40, r=40, t=40, b=40),
            height=320
        )

        # Chart 2: Rate of Spread & Fire Perimeter & Threatened Structures
        fig_ros = go.Figure()
        fig_ros.add_trace(go.Scatter(
            x=df["Time (min)"],
            y=df["Perimeter (km)"],
            mode="lines+markers",
            name="Fire Perimeter (km)",
            line=dict(color="#00E5FF", width=2.5),
            yaxis="y1"
        ))
        fig_ros.add_trace(go.Bar(
            x=df["Time (min)"],
            y=df["Rate of Spread (ha/step)"],
            name="Rate of Spread (ha/step)",
            marker_color="rgba(233, 30, 99, 0.75)",
            yaxis="y2"
        ))
        fig_ros.update_layout(
            title=dict(text="📈 Fire Perimeter & Rate of Spread Progression", font=dict(color="#E2E8F0", size=14)),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            xaxis=dict(title="Elapsed Time (minutes)", color="#94A3B8", gridcolor="#1E293B"),
            yaxis=dict(title="Perimeter (km)", color="#00E5FF", gridcolor="#1E293B"),
            yaxis2=dict(title="Rate of Spread (ha/step)", color="#E91E63", overlaying="y", side="right"),
            legend=dict(font=dict(color="#E2E8F0"), bgcolor="rgba(15,23,42,0.7)"),
            margin=dict(l=40, r=40, t=40, b=40),
            height=320
        )

        return fig_area, fig_ros

    @staticmethod
    def create_wui_status_chart(evac_result: EvacuationAnalysisResult) -> go.Figure:
        categories = ["Intact / Safe", "Defended (Tactics)", "Threatened (Active)", "Burned"]
        counts = [evac_result.intact_count, evac_result.defended_count, evac_result.threatened_count, evac_result.burned_count]
        colors = ["#00E676", "#2979FF", "#FF1744", "#424242"]

        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=counts,
                marker_color=colors,
                text=counts,
                textposition="auto",
                textfont=dict(color="#FFFFFF", size=13)
            )
        ])
        fig.update_layout(
            title=dict(text="🏡 WUI Structure Protection & Threat Breakdown", font=dict(color="#E2E8F0", size=14)),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            xaxis=dict(color="#94A3B8", gridcolor="#1E293B"),
            yaxis=dict(title="Number of Structures", color="#94A3B8", gridcolor="#1E293B", dtick=1),
            margin=dict(l=40, r=40, t=40, b=40),
            height=300
        )
        return fig

    @staticmethod
    def create_risk_distribution_chart(risk_result: RiskAssessmentResult) -> go.Figure:
        categories = list(risk_result.category_percentages.keys())
        percentages = list(risk_result.category_percentages.values())
        colors = ["#2E7D32", "#FDD835", "#FB8C00", "#E53935", "#880E4F"]

        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=percentages,
                marker_color=colors,
                text=[f"{p:.1f}%" for p in percentages],
                textposition="auto",
                textfont=dict(color="#FFFFFF", size=12)
            )
        ])

        fig.update_layout(
            title=dict(text="📊 Wildfire Risk Class Distribution (% Landscape)", font=dict(color="#E2E8F0", size=14)),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            xaxis=dict(color="#94A3B8", gridcolor="#1E293B"),
            yaxis=dict(title="Percentage of Burnable Land (%)", color="#94A3B8", gridcolor="#1E293B", range=[0, max(60, max(percentages) + 10)]),
            margin=dict(l=40, r=40, t=40, b=40),
            height=300
        )
        return fig
