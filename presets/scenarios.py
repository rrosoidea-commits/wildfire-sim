"""
scenarios.py - Predefined realistic wildfire simulation scenarios and terrain presets.
Provides ready-to-simulate case studies showcasing extreme weather, topographic funneling,
river firebreaks, and prescribed burn behaviors.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List, Union
from core.terrain import TerrainGrid
from core.weather import WeatherCondition

@dataclass
class ScenarioPreset:
    id: str
    title: str
    description: str
    terrain_preset: str
    grid_size: int
    cell_size_m: float
    base_elevation: float
    roughness: float
    water_level: float
    forest_density_scale: float
    weather: WeatherCondition
    default_ignition: Tuple[int, int]
    spread_rate_factor: float
    enable_spotting: bool

PRESET_SCENARIOS: Dict[str, ScenarioPreset] = {
    "santa_ana_canyon": ScenarioPreset(
        id="santa_ana_canyon",
        title="Santa Ana Diablo Windstorm (Canyon Channelling)",
        description="Fierce, dry offshore gale-force winds funnel down a steep canyon corridor. Extremely low relative humidity (8%) and high temperatures produce explosive uphill spread and long-range ember spotting.",
        terrain_preset="canyon",
        grid_size=60,
        cell_size_m=30.0,
        base_elevation=350.0,
        roughness=1.1,
        water_level=0.03,
        forest_density_scale=1.1,
        weather=WeatherCondition(
            wind_speed_kmh=62.0,
            wind_direction_deg=45.0, # From NE (Santa Ana direction)
            temperature_c=39.0,
            relative_humidity_pct=8.0,
            wind_gust_kmh=80.0,
            atmospheric_instability=0.85
        ),
        default_ignition=(20, 20),
        spread_rate_factor=1.35,
        enable_spotting=True
    ),

    "alpine_river_break": ScenarioPreset(
        id="alpine_river_break",
        title="Alpine Mountain Peak with River Firebreak",
        description="A lightning strike in the lower valley begins a massive uphill crown fire in dense conifer timber. The winding alpine river serves as a natural hydraulic firebreak for the western flank.",
        terrain_preset="alpine_ridge",
        grid_size=60,
        cell_size_m=30.0,
        base_elevation=500.0,
        roughness=1.2,
        water_level=0.06,
        forest_density_scale=1.2,
        weather=WeatherCondition(
            wind_speed_kmh=26.0,
            wind_direction_deg=220.0, # From SW
            temperature_c=29.0,
            relative_humidity_pct=22.0,
            wind_gust_kmh=38.0,
            atmospheric_instability=0.55
        ),
        default_ignition=(15, 40),
        spread_rate_factor=1.0,
        enable_spotting=True
    ),

    "rolling_hills_drought": ScenarioPreset(
        id="rolling_hills_drought",
        title="Rolling Oak & Grassland Summer Drought",
        description="Prolonged mid-summer heatwave with dried fine fuels and savanna grasses across undulating terrain. Fires propagate rapidly through surface grass before torching mixed woodland copses.",
        terrain_preset="rolling_hills",
        grid_size=60,
        cell_size_m=30.0,
        base_elevation=280.0,
        roughness=0.9,
        water_level=0.04,
        forest_density_scale=0.85,
        weather=WeatherCondition(
            wind_speed_kmh=34.0,
            wind_direction_deg=270.0, # From West
            temperature_c=36.0,
            relative_humidity_pct=14.0,
            wind_gust_kmh=45.0,
            atmospheric_instability=0.65
        ),
        default_ignition=(12, 30),
        spread_rate_factor=1.15,
        enable_spotting=True
    ),

    "chaparral_plains": ScenarioPreset(
        id="chaparral_plains",
        title="Chaparral Scrub Flash Fire",
        description="High volatile oil content in dense dry chaparral combined with sustained southerly winds drives a broad, fast-moving continuous flame front with minimal topographical obstruction.",
        terrain_preset="plains_chaparral",
        grid_size=60,
        cell_size_m=30.0,
        base_elevation=200.0,
        roughness=0.6,
        water_level=0.0,
        forest_density_scale=0.95,
        weather=WeatherCondition(
            wind_speed_kmh=44.0,
            wind_direction_deg=180.0, # From South
            temperature_c=41.0,
            relative_humidity_pct=10.0,
            wind_gust_kmh=55.0,
            atmospheric_instability=0.75
        ),
        default_ignition=(30, 10),
        spread_rate_factor=1.25,
        enable_spotting=True
    ),

    "prescribed_burn": ScenarioPreset(
        id="prescribed_burn",
        title="Controlled Prescribed Fuel Reduction Burn",
        description="Ideal prescription window with calm winds, moderate temperatures, and high fuel moisture. Demonstrates controlled, low-intensity backing fire for hazardous fuel reduction and community defense.",
        terrain_preset="rolling_hills",
        grid_size=50,
        cell_size_m=30.0,
        base_elevation=300.0,
        roughness=0.7,
        water_level=0.05,
        forest_density_scale=0.75,
        weather=WeatherCondition(
            wind_speed_kmh=10.0,
            wind_direction_deg=0.0, # Gentle North wind
            temperature_c=18.0,
            relative_humidity_pct=58.0,
            wind_gust_kmh=14.0,
            atmospheric_instability=0.20
        ),
        default_ignition=(25, 40),
        spread_rate_factor=0.60,
        enable_spotting=False
    )
}
