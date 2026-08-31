"""
weather.py - Meteorological and Atmospheric State Engine for Wildfire Simulation.
Models wind velocity vector fields, relative humidity, temperature, wind gusts,
and their combined atmospheric drying effect on fuel beds.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any, Union

@dataclass
class WeatherCondition:
    wind_speed_kmh: float = 25.0       # Wind speed in km/h (0 to 100)
    wind_direction_deg: float = 225.0   # Direction wind is coming FROM in degrees (0=N, 90=E, 180=S, 270=W, 225=SW)
    temperature_c: float = 32.0        # Ambient air temp in Celsius
    relative_humidity_pct: float = 18.0# Relative humidity (0 to 100%)
    wind_gust_kmh: float = 35.0        # Peak gust speed
    atmospheric_instability: float = 0.5 # Haines Index / stability factor [0=stable, 1=unstable]

    @property
    def wind_speed_ms(self) -> float:
        return self.wind_speed_kmh / 3.6

    @property
    def wind_vector(self) -> Tuple[float, float]:
        """
        Returns (u, v) unit vector in simulation grid coordinates.
        Note: If wind is COMING FROM angle theta, it BLOWS TOWARDS theta + 180 deg.
        In grid: +X is East, +Y is North.
        """
        # Blowing angle
        blow_deg = (self.wind_direction_deg + 180.0) % 360.0
        blow_rad = np.deg2rad(blow_deg)
        u = np.sin(blow_rad) # East component
        v = np.cos(blow_rad) # North component
        return (float(u), float(v))

    def get_effective_wind_field(self, nx: int, ny: int, elevation: np.ndarray, cell_size_m: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes localized wind field (speed_grid, u_grid, v_grid) accounting for terrain channeling and ridge acceleration.
        Ridgetops experience wind speedup (venturi effect), while leeward slopes experience sheltering.
        """
        u_base, v_base = self.wind_vector
        speed_base = self.wind_speed_kmh

        # Compute elevation gradient
        dz_dy, dz_dx = np.gradient(elevation, cell_size_m)
        
        # Terrain slope in direction of wind:
        # Wind blowing direction unit vector (u_base, v_base)
        slope_in_wind_dir = dz_dx * u_base + dz_dy * v_base

        # Ridge acceleration: ridgetops and windward crests speed up wind by up to 35%
        # Leeward valleys slow wind down
        speed_mod = 1.0 + 0.35 * np.tanh(slope_in_wind_dir * 1.5)
        speed_grid = np.clip(speed_base * speed_mod, 1.0, 150.0)

        u_grid = u_base * (speed_grid / (speed_base + 1e-6))
        v_grid = v_base * (speed_grid / (speed_base + 1e-6))

        return speed_grid, u_grid, v_grid

    def compute_fire_weather_index_proxy(self) -> float:
        """
        Computes a normalized Fire Weather Index (FWI) proxy [0 to 100].
        High temp, high wind, low humidity yield extreme index.
        """
        temp_term = np.clip((self.temperature_c - 10.0) / 35.0, 0.0, 1.0)
        humid_term = np.clip((100.0 - self.relative_humidity_pct) / 90.0, 0.0, 1.0)
        wind_term = np.clip(self.wind_speed_kmh / 70.0, 0.0, 1.0)

        fwi = 100.0 * (0.35 * temp_term + 0.40 * humid_term + 0.25 * wind_term)
        return float(np.clip(fwi, 0.0, 100.0))