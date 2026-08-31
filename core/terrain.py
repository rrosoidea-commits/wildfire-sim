"""
terrain.py - 3D Terrain & Landscape Generation Module for Wildfire Simulation.
Generates synthetic procedural elevations, slopes, aspects, fuel types, fuel densities,
fuel moisture, historical fire risk layers, road networks, and synthetic WUI structures.
Supports future GIS raster / DEM ingest.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

class FuelType:
    WATER = 0
    BARE_GROUND = 1
    SHORT_GRASS = 2
    DRY_SHRUBLAND = 3
    MIXED_WOODLAND = 4
    DENSE_FOREST = 5
    DRY_SLASH = 6

    NAMES = {
        WATER: "Water Body / River",
        BARE_GROUND: "Bare Ground / Rock",
        SHORT_GRASS: "Grassland / Savanna",
        DRY_SHRUBLAND: "Dry Shrubland / Chaparral",
        MIXED_WOODLAND: "Mixed Woodland",
        DENSE_FOREST: "Dense Conifer Forest",
        DRY_SLASH: "Dry Slash / Heavy Fuel"
    }

    SPREAD_MULTIPLIER = {
        WATER: 0.0,
        BARE_GROUND: 0.05,
        SHORT_GRASS: 1.45,
        DRY_SHRUBLAND: 1.75,
        MIXED_WOODLAND: 1.20,
        DENSE_FOREST: 1.45,
        DRY_SLASH: 1.95
    }

    FUEL_LOAD_T_HA = {
        WATER: 0.0,
        BARE_GROUND: 0.2,
        SHORT_GRASS: 4.5,
        DRY_SHRUBLAND: 14.0,
        MIXED_WOODLAND: 32.0,
        DENSE_FOREST: 55.0,
        DRY_SLASH: 70.0
    }

    BURN_RESIDENCE_STEPS = {
        WATER: 0,
        BARE_GROUND: 2,
        SHORT_GRASS: 4,
        DRY_SHRUBLAND: 7,
        MIXED_WOODLAND: 11,
        DENSE_FOREST: 15,
        DRY_SLASH: 18
    }

    COLORS = {
        WATER: "#1E88E5",
        BARE_GROUND: "#B0BEC5",
        SHORT_GRASS: "#D4E157",
        DRY_SHRUBLAND: "#FDD835",
        MIXED_WOODLAND: "#7CB342",
        DENSE_FOREST: "#2E7D32",
        DRY_SLASH: "#8D6E63"
    }

@dataclass
class Structure:
    id: str
    name: str
    x: int
    y: int
    structure_type: str = "Residential Home" # 'Residential Home', 'Community Facility', 'Commercial Lodge', 'Emergency Shelter'
    defensibility_score: float = 0.70 # 0.0 to 1.0
    status: str = "INTACT" # 'INTACT', 'THREATENED', 'BURNED', 'DEFENDED'
    distance_to_fire_m: float = 9999.0
    threat_level: str = "LOW" # 'LOW', 'MODERATE', 'HIGH', 'CRITICAL'
    arrival_time_min: Optional[float] = None

@dataclass
class TerrainGrid:
    nx: int
    ny: int
    cell_size_m: float
    elevation: np.ndarray
    slope_rad: np.ndarray
    slope_deg: np.ndarray
    aspect_deg: np.ndarray
    fuel_type: np.ndarray
    fuel_density: np.ndarray
    fuel_moisture: np.ndarray
    historical_risk: np.ndarray
    preset_name: str = "custom"
    roads_mask: Optional[np.ndarray] = None
    structures: List[Structure] = field(default_factory=list)
    _cached_base_surface_color: Optional[np.ndarray] = None
    _cached_xx: Optional[np.ndarray] = None
    _cached_yy: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.roads_mask is None:
            self.roads_mask = np.zeros((self.ny, self.nx), dtype=bool)

    def get_mesh_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._cached_xx is None or self._cached_yy is None:
            x_coords = np.arange(self.nx) * self.cell_size_m
            y_coords = np.arange(self.ny) * self.cell_size_m
            self._cached_xx, self._cached_yy = np.meshgrid(x_coords, y_coords)
        return self._cached_xx, self._cached_yy

    def get_base_surface_color(self) -> np.ndarray:
        if self._cached_base_surface_color is None:
            fuel_values = np.array([
                0.08, # FuelType.WATER = 0
                0.22, # FuelType.BARE_GROUND = 1
                0.35, # FuelType.SHORT_GRASS = 2
                0.47, # FuelType.DRY_SHRUBLAND = 3
                0.59, # FuelType.MIXED_WOODLAND = 4
                0.72, # FuelType.DENSE_FOREST = 5
                0.84  # FuelType.DRY_SLASH = 6
            ], dtype=float)
            fuel_clipped = np.clip(self.fuel_type, 0, 6)
            base = fuel_values[fuel_clipped] + (self.slope_deg / 50.0) * 0.04
            if self.roads_mask is not None:
                base[self.roads_mask] = 0.20
            self._cached_base_surface_color = base
        return self._cached_base_surface_color

    @property
    def total_area_ha(self) -> float:
        return (self.nx * self.ny * (self.cell_size_m ** 2)) / 10000.0

    @property
    def total_area_km2(self) -> float:
        return self.total_area_ha / 100.0

    @classmethod
    def create_synthetic(
        cls,
        nx: int = 60,
        ny: int = 60,
        cell_size_m: float = 30.0,
        preset: str = "canyon",
        seed: Optional[int] = 42,
        base_elevation: float = 400.0,
        roughness: float = 1.0,
        water_level: float = 0.05,
        forest_density_scale: float = 1.0,
        custom_params: Optional[dict] = None
    ) -> 'TerrainGrid':
        if seed is not None:
            np.random.seed(seed)

        x = np.linspace(0, nx - 1, nx)
        y = np.linspace(0, ny - 1, ny)
        xx, yy = np.meshgrid(x, y)

        if preset == "canyon":
            canyon_axis = (xx - yy) / np.sqrt(2.0)
            valley = 320.0 * np.exp(-((canyon_axis) ** 2) / (2.0 * (nx * 0.18) ** 2))
            flanking_ridges = 240.0 * np.cos(canyon_axis / (nx * 0.16))
            noise = gaussian_filter(np.random.randn(ny, nx) * 50.0, sigma=3.0)
            elevation = base_elevation + flanking_ridges - valley + noise * roughness
            elevation = np.clip(elevation, 50.0, 2000.0)

            is_river = (np.abs(canyon_axis) < (nx * 0.04)) & (elevation < (base_elevation + 80.0))

        elif preset == "alpine_ridge":
            dist_to_peak = np.sqrt((xx - nx * 0.75) ** 2 + (yy - ny * 0.75) ** 2)
            peak = 650.0 * np.exp(-((dist_to_peak) ** 2) / (2.0 * (nx * 0.38) ** 2))
            sub_ridge = 180.0 * np.cos(xx / 12.0) * np.sin(yy / 12.0)
            noise = gaussian_filter(np.random.randn(ny, nx) * 40.0, sigma=3.0)
            elevation = base_elevation + peak + sub_ridge + noise * roughness
            elevation = np.clip(elevation, 50.0, 2500.0)

            river_path = ny * 0.30 + 6.0 * np.sin(xx / 8.0)
            is_river = np.abs(yy - river_path) < 1.6

        elif preset == "rolling_hills":
            hills = 150.0 * (np.sin(xx / 11.0) + np.cos(yy / 11.0)) + 60.0 * np.sin((xx + yy) / 14.0)
            noise = gaussian_filter(np.random.randn(ny, nx) * 25.0, sigma=3.5)
            elevation = base_elevation + hills + noise * roughness
            elevation = np.clip(elevation, 50.0, 1500.0)

            lake_mask = (elevation < np.percentile(elevation, 5.0))
            is_river = lake_mask

        elif preset == "plains_chaparral":
            slight_tilt = (xx * 1.5 + yy * 0.8)
            noise = gaussian_filter(np.random.randn(ny, nx) * 15.0, sigma=4.0)
            elevation = base_elevation + slight_tilt + noise * roughness
            elevation = np.clip(elevation, 50.0, 1000.0)
            is_river = np.zeros((ny, nx), dtype=bool)

        else: # Procedural
            noise_coarse = gaussian_filter(np.random.randn(ny, nx) * 200.0, sigma=7.0)
            noise_med = gaussian_filter(np.random.randn(ny, nx) * 70.0, sigma=3.5)
            noise_fine = gaussian_filter(np.random.randn(ny, nx) * 20.0, sigma=1.5)
            elevation = base_elevation + (noise_coarse + noise_med + noise_fine) * roughness
            elevation = np.clip(elevation, 50.0, 2500.0)
            is_river = (elevation < np.percentile(elevation, max(1.0, water_level * 100)))

        dz_dy, dz_dx = np.gradient(elevation, cell_size_m)
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.rad2deg(slope_rad)

        aspect_rad = np.arctan2(-dz_dx, dz_dy)
        aspect_deg = (np.rad2deg(aspect_rad) + 360.0) % 360.0

        fuel_type = np.full((ny, nx), FuelType.MIXED_WOODLAND, dtype=int)
        fuel_density = np.ones((ny, nx), dtype=float) * 0.8
        fuel_moisture = np.ones((ny, nx), dtype=float) * 0.12

        norm_elev = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-6)

        fuel_type[norm_elev < 0.25] = FuelType.SHORT_GRASS
        fuel_type[(norm_elev >= 0.25) & (norm_elev < 0.55)] = FuelType.DRY_SHRUBLAND
        fuel_type[(norm_elev >= 0.55) & (norm_elev < 0.80)] = FuelType.MIXED_WOODLAND
        fuel_type[norm_elev >= 0.80] = FuelType.DENSE_FOREST

        fuel_type[slope_deg > 48.0] = FuelType.BARE_GROUND

        slash_noise = gaussian_filter(np.random.randn(ny, nx), sigma=2.5)
        fuel_type[(slash_noise > 1.35) & (fuel_type != FuelType.BARE_GROUND)] = FuelType.DRY_SLASH

        fuel_type[is_river] = FuelType.WATER
        fuel_density[is_river] = 0.0
        fuel_moisture[is_river] = 1.0

        density_noise = gaussian_filter(np.random.rand(ny, nx), sigma=2.5)
        density_noise = (density_noise - density_noise.min()) / (density_noise.max() - density_noise.min() + 1e-6)
        fuel_density = np.clip(density_noise * forest_density_scale, 0.2, 1.0)
        fuel_density[is_river] = 0.0
        fuel_density[fuel_type == FuelType.BARE_GROUND] = 0.05

        solar_insolation = np.cos(np.deg2rad(aspect_deg - 180.0)) * np.sin(slope_rad)
        solar_insolation = np.clip(solar_insolation, -1.0, 1.0)

        dist_from_water = gaussian_filter((~is_river).astype(float), sigma=3.0)
        base_moist = 0.14 - 0.05 * solar_insolation - 0.03 * (norm_elev - 0.5)
        fuel_moisture = np.clip(base_moist * (0.8 + 0.4 * dist_from_water), 0.03, 0.32)
        fuel_moisture[is_river] = 1.0

        risk_noise = gaussian_filter(np.random.randn(ny, nx), sigma=4.0)
        risk_noise = (risk_noise - risk_noise.min()) / (risk_noise.max() - risk_noise.min() + 1e-6)
        historical_risk = np.clip(
            0.4 * risk_noise +
            0.3 * (1.0 - fuel_moisture) +
            0.3 * (slope_deg / 40.0),
            0.0, 1.0
        )
        historical_risk[is_river] = 0.0

        # Generate Road Network
        roads_mask = np.zeros((ny, nx), dtype=bool)
        main_road_y = int(ny * 0.45)
        roads_mask[max(0, main_road_y - 1):min(ny, main_road_y + 1), :] = True
        main_road_x = int(nx * 0.60)
        roads_mask[:, max(0, main_road_x - 1):min(nx, main_road_x + 1)] = True
        diag_idx = np.arange(10, min(nx, ny) - 10)
        roads_mask[diag_idx, diag_idx] = True

        roads_mask[is_river & (roads_mask == False)] = False

        # Generate Synthetic WUI Structures
        structures = cls._generate_synthetic_structures(nx, ny, elevation, fuel_type, is_river, roads_mask, preset)

        return cls(
            nx=nx,
            ny=ny,
            cell_size_m=cell_size_m,
            elevation=elevation,
            slope_rad=slope_rad,
            slope_deg=slope_deg,
            aspect_deg=aspect_deg,
            fuel_type=fuel_type,
            fuel_density=fuel_density,
            fuel_moisture=fuel_moisture,
            historical_risk=historical_risk,
            preset_name=preset,
            roads_mask=roads_mask,
            structures=structures
        )

    @classmethod
    def _generate_synthetic_structures(
        cls,
        nx: int,
        ny: int,
        elevation: np.ndarray,
        fuel_type: np.ndarray,
        is_river: np.ndarray,
        roads_mask: np.ndarray,
        preset: str
    ) -> List[Structure]:
        """
        Generates realistic Wildland-Urban Interface (WUI) structures clustered near roads and scenic valleys.
        """
        structures: List[Structure] = []

        if preset == "canyon":
            clusters = [
                ("Canyon Valley Homes", int(nx * 0.65), int(ny * 0.45), 5, "Residential Home"),
                ("Highland Ridge Estates", int(nx * 0.30), int(ny * 0.75), 4, "Residential Home"),
                ("Canyon Ranger Station", int(nx * 0.58), int(ny * 0.42), 1, "Community Facility"),
                ("Riverbend Shelter", int(nx * 0.80), int(ny * 0.25), 2, "Emergency Shelter")
            ]
        elif preset == "alpine_ridge":
            clusters = [
                ("Alpine Village", int(nx * 0.60), int(ny * 0.45), 6, "Residential Home"),
                ("Summit Mountain Lodge", int(nx * 0.75), int(ny * 0.70), 2, "Commercial Lodge"),
                ("Valley Community School", int(nx * 0.55), int(ny * 0.40), 1, "Emergency Shelter"),
                ("Timberline Cabins", int(nx * 0.25), int(ny * 0.35), 3, "Residential Home")
            ]
        elif preset == "rolling_hills":
            clusters = [
                ("Oak Grove Suburb", int(nx * 0.62), int(ny * 0.48), 6, "Residential Home"),
                ("Meadow Ranch Cabins", int(nx * 0.35), int(ny * 0.65), 4, "Residential Home"),
                ("Hilltop Civic Center", int(nx * 0.60), int(ny * 0.42), 1, "Community Facility"),
                ("Valley Evac Center", int(nx * 0.85), int(ny * 0.20), 2, "Emergency Shelter")
            ]
        else: # plains / procedural
            clusters = [
                ("Prairie Edge Town", int(nx * 0.60), int(ny * 0.45), 6, "Residential Home"),
                ("Chaparral Ranch Estates", int(nx * 0.30), int(ny * 0.30), 4, "Residential Home"),
                ("County Emergency Depot", int(nx * 0.65), int(ny * 0.50), 2, "Emergency Shelter")
            ]

        struct_idx = 1
        for cluster_name, cx, cy, count, default_type in clusters:
            for i in range(count):
                if count == 1:
                    sx, sy = cx, cy
                else:
                    angle = (2 * np.pi * i) / count + np.random.uniform(-0.3, 0.3)
                    radius = np.random.uniform(1.5, 4.5)
                    sx = int(round(cx + radius * np.cos(angle)))
                    sy = int(round(cy + radius * np.sin(angle)))

                sx = int(np.clip(sx, 2, nx - 3))
                sy = int(np.clip(sy, 2, ny - 3))

                if fuel_type[sy, sx] == FuelType.WATER or is_river[sy, sx]:
                    sx = int(np.clip(sx + 2, 2, nx - 3))

                fuel_type[sy, sx] = FuelType.SHORT_GRASS

                s_id = f"WUI-{struct_idx:02d}"
                s_name = f"{cluster_name} #{i+1}" if count > 1 else cluster_name
                stype = default_type
                defens_score = float(np.random.uniform(0.60, 0.90))

                structures.append(Structure(
                    id=s_id,
                    name=s_name,
                    x=int(sx),
                    y=int(sy),
                    structure_type=stype,
                    defensibility_score=defens_score,
                    status="INTACT",
                    distance_to_fire_m=9999.0,
                    threat_level="LOW"
                ))
                struct_idx += 1

        return structures