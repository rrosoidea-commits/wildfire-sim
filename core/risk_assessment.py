"""
risk_assessment.py - Multi-Criteria Wildfire Risk & Hazard Assessment Engine.
Generates baseline risk maps combining fuel load, topography/slope, weather conditions,
and historical fire climatology prior to or during simulation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any, Union
from .terrain import TerrainGrid, FuelType
from .weather import WeatherCondition

@dataclass
class RiskAssessmentResult:
    composite_risk_score: np.ndarray      # 2D array [0.0 to 100.0]
    fuel_hazard_score: np.ndarray         # 2D array [0.0 to 100.0]
    topographic_hazard_score: np.ndarray  # 2D array [0.0 to 100.0]
    weather_hazard_score: float           # Scalar [0.0 to 100.0]
    historical_hazard_score: np.ndarray   # 2D array [0.0 to 100.0]
    risk_category_map: np.ndarray         # 2D string/int category map
    category_percentages: Dict[str, float]
    mean_risk_score: float
    max_risk_score: float
    high_risk_area_ha: float

class WildfireRiskAssessment:
    @staticmethod
    def calculate(
        terrain: TerrainGrid,
        weather: WeatherCondition,
        w_fuel: float = 0.35,
        w_topo: float = 0.25,
        w_weather: float = 0.25,
        w_hist: float = 0.15
    ) -> RiskAssessmentResult:
        # Normalize weights
        total_w = w_fuel + w_topo + w_weather + w_hist
        w_fuel /= total_w
        w_topo /= total_w
        w_weather /= total_w
        w_hist /= total_w

        # 1. Fuel Hazard Score [0 - 100]
        # Based on fuel type spread multiplier, density, and dryness (1 - moisture)
        spread_mult_grid = np.zeros_like(terrain.fuel_density)
        for ftype, mult in FuelType.SPREAD_MULTIPLIER.items():
            spread_mult_grid[terrain.fuel_type == ftype] = mult

        norm_mult = np.clip(spread_mult_grid / 2.0, 0.0, 1.0)
        dryness = np.clip(1.0 - (terrain.fuel_moisture / 0.35), 0.0, 1.0)
        fuel_hazard = 100.0 * (0.45 * norm_mult + 0.35 * terrain.fuel_density + 0.20 * dryness)
        # Water and bare rock have zero fuel hazard
        fuel_hazard[terrain.fuel_type == FuelType.WATER] = 0.0
        fuel_hazard[terrain.fuel_type == FuelType.BARE_GROUND] = np.clip(fuel_hazard[terrain.fuel_type == FuelType.BARE_GROUND] * 0.05, 0.0, 5.0)

        # 2. Topographic Hazard Score [0 - 100]
        # Steep slopes (> 25 deg) dramatically accelerate fire spread
        # South/Southwest facing slopes (aspect 180 - 225) are sun-exposed
        slope_term = np.clip(terrain.slope_deg / 40.0, 0.0, 1.0)
        aspect_rad = np.deg2rad(terrain.aspect_deg)
        solar_term = np.clip((np.sin(aspect_rad) * 0.3 + np.cos(aspect_rad - np.pi) * 0.7 + 1.0) / 2.0, 0.0, 1.0)
        topo_hazard = 100.0 * (0.70 * (slope_term ** 1.3) + 0.30 * solar_term)
        topo_hazard[terrain.fuel_type == FuelType.WATER] = 0.0

        # 3. Weather Hazard Score [0 - 100]
        weather_hazard = weather.compute_fire_weather_index_proxy()

        # 4. Historical Risk Score [0 - 100]
        hist_hazard = 100.0 * terrain.historical_risk
        hist_hazard[terrain.fuel_type == FuelType.WATER] = 0.0

        # Composite Score [0 - 100]
        composite = (
            w_fuel * fuel_hazard +
            w_topo * topo_hazard +
            w_weather * weather_hazard +
            w_hist * hist_hazard
        )
        composite[terrain.fuel_type == FuelType.WATER] = 0.0
        composite = np.clip(composite, 0.0, 100.0)

        # Category mapping:
        # 0: Low (<25), 1: Moderate (25-45), 2: High (45-65), 3: Very High (65-80), 4: Extreme (>=80)
        cat_map = np.zeros_like(composite, dtype=int)
        cat_map[composite >= 25.0] = 1
        cat_map[composite >= 45.0] = 2
        cat_map[composite >= 65.0] = 3
        cat_map[composite >= 80.0] = 4

        total_cells = terrain.nx * terrain.ny
        non_water_cells = np.count_nonzero(terrain.fuel_type != FuelType.WATER)
        denom = max(1, non_water_cells)

        pcts = {
            "Low (< 25)": float(np.count_nonzero((cat_map == 0) & (terrain.fuel_type != FuelType.WATER)) / denom * 100.0),
            "Moderate (25-45)": float(np.count_nonzero(cat_map == 1) / denom * 100.0),
            "High (45-65)": float(np.count_nonzero(cat_map == 2) / denom * 100.0),
            "Very High (65-80)": float(np.count_nonzero(cat_map == 3) / denom * 100.0),
            "Extreme (>= 80)": float(np.count_nonzero(cat_map == 4) / denom * 100.0),
        }

        cell_area_ha = (terrain.cell_size_m ** 2) / 10000.0
        high_risk_cells = np.count_nonzero(cat_map >= 2)
        high_risk_ha = float(high_risk_cells * cell_area_ha)

        return RiskAssessmentResult(
            composite_risk_score=composite,
            fuel_hazard_score=fuel_hazard,
            topographic_hazard_score=topo_hazard,
            weather_hazard_score=weather_hazard,
            historical_hazard_score=hist_hazard,
            risk_category_map=cat_map,
            category_percentages=pcts,
            mean_risk_score=float(np.mean(composite[terrain.fuel_type != FuelType.WATER])),
            max_risk_score=float(np.max(composite)),
            high_risk_area_ha=high_risk_ha
        )