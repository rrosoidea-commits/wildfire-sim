"""
fire_ca.py - Vectorized Cellular Automata Wildfire Spread Engine.
Implements high-performance NumPy-vectorized fire spread calculations incorporating:
- Topographic slope & aspect (gravity and convective preheating)
- Localized wind field & ridge acceleration
- Fuel moisture, fuel types, fuel load, and canopy density
- Atmospheric temperature, humidity, and Fire Weather Index
- Historical wildfire climatology risk
- Long-range spotting / ember lofting
- Interactive firefighting containment (firelines, water drops, backburns)
- WUI structure threat tracking
- Comprehensive timestep simulation recording for CSV export.
"""

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Union
from core.terrain import TerrainGrid, FuelType, Structure
from core.weather import WeatherCondition
from core.firefighting import FirefightingManager

class FireState:
    UNBURNED = 0
    BURNING = 1
    BURNED_OUT = 2
    UNBURNABLE = 3

@dataclass
class SimulationStepStats:
    step: int
    elapsed_minutes: float
    wind_speed_kmh: float
    wind_direction_deg: float
    wind_direction_cardinal: str
    temperature_c: float
    humidity_pct: float
    fuel_moisture_pct: float
    fire_intensity_max: float
    fire_intensity_mean: float
    burning_cells: int
    burned_cells: int
    active_fire_area_ha: float
    total_burned_area_ha: float
    burned_area_pct: float
    fire_perimeter_km: float
    max_spread_distance_m: float
    threatened_structures: int
    burned_structures: int
    rate_of_spread_ha_step: float
    fuel_consumed_tons: float
    spot_fires_count: int

class WildfireSimulation:
    def __init__(
        self,
        terrain: TerrainGrid,
        weather: WeatherCondition,
        base_spread_probability: float = 0.60,
        spread_rate_factor: float = 1.0,
        enable_spotting: bool = True,
        spotting_probability: float = 0.008,
        minutes_per_step: float = 10.0
    ):
        self.terrain = terrain
        self.weather = weather
        self.base_spread_probability = base_spread_probability
        self.spread_rate_factor = spread_rate_factor
        self.enable_spotting = enable_spotting
        self.spotting_probability = spotting_probability
        self.minutes_per_step = minutes_per_step

        self.nx = terrain.nx
        self.ny = terrain.ny
        self.cell_size_m = terrain.cell_size_m
        self.cell_area_ha = (self.cell_size_m ** 2) / 10000.0

        # State grids
        self.state = np.full((self.ny, self.nx), FireState.UNBURNED, dtype=np.int8)
        self.state[terrain.fuel_type == FuelType.WATER] = FireState.UNBURNABLE

        self.fire_intensity = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.steps_burning = np.zeros((self.ny, self.nx), dtype=np.int16)
        self.burn_progress = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.ignition_step = np.full((self.ny, self.nx), -1, dtype=np.int32)
        self.flame_height_m = np.zeros((self.ny, self.nx), dtype=np.float32)

        # Precomputed lookup arrays for vectorization
        self.spread_mult_grid = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.max_residence_grid = np.zeros((self.ny, self.nx), dtype=np.int16)
        self._update_fuel_lookup_grids()

        # Firefighting Manager
        self.firefighting_mgr = FirefightingManager(self.nx, self.ny, self.cell_size_m)

        # Tracking
        self.current_step = 0
        self.elapsed_minutes = 0.0
        self.ignition_points: List[Tuple[int, int]] = []
        self.spot_fire_events: List[Tuple[int, int, int]] = [] # (step, y, x)
        self.history: List[SimulationStepStats] = []
        self.is_running = False
        self.is_extinguished = False

    def _update_fuel_lookup_grids(self):
        """Precomputes 2D arrays of spread multipliers and burn residence times for instant vector indexing."""
        for ftype, mult in FuelType.SPREAD_MULTIPLIER.items():
            mask = (self.terrain.fuel_type == ftype)
            self.spread_mult_grid[mask] = mult
            self.max_residence_grid[mask] = FuelType.BURN_RESIDENCE_STEPS.get(ftype, 8)

    def record_initial_state(self):
        """Records initial Step 0 in history if empty."""
        if len(self.history) == 0:
            stats = self._compute_stats(spot_count=0)
            self.history.append(stats)

    def ignite_cell(self, x: int, y: int, initial_intensity: float = 0.85, auto_record: bool = True) -> bool:
        """
        Ignites a specific coordinate (x, y). If water, searches nearest land cell.
        """
        if not (0 <= x < self.nx and 0 <= y < self.ny):
            return False

        if self.terrain.fuel_type[y, x] == FuelType.WATER:
            # Find nearest burnable neighbor
            best_dist = 999.0
            best_coord = None
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    ny_idx, nx_idx = y + dy, x + dx
                    if 0 <= ny_idx < self.ny and 0 <= nx_idx < self.nx:
                        if self.terrain.fuel_type[ny_idx, nx_idx] != FuelType.WATER:
                            d = np.sqrt(dx**2 + dy**2)
                            if d < best_dist:
                                best_dist = d
                                best_coord = (nx_idx, ny_idx)
            if best_coord is not None:
                x, y = best_coord
            else:
                return False

        if self.terrain.fuel_type[y, x] == FuelType.BARE_GROUND:
            self.terrain.fuel_type[y, x] = FuelType.DRY_SHRUBLAND
            self.terrain.fuel_density[y, x] = 0.6
            self.terrain.fuel_moisture[y, x] = 0.10
            self._update_fuel_lookup_grids()

        self.state[y, x] = FireState.BURNING
        self.fire_intensity[y, x] = float(np.clip(initial_intensity, 0.2, 1.0))
        self.steps_burning[y, x] = 0
        self.ignition_step[y, x] = self.current_step
        self.flame_height_m[y, x] = 2.5
        self.is_extinguished = False
        if (x, y) not in self.ignition_points:
            self.ignition_points.append((x, y))

        if auto_record and len(self.history) == 0 and self.current_step == 0:
            self.record_initial_state()

        return True

    def ignite_radius(self, x: int, y: int, radius: int = 1, initial_intensity: float = 0.85):
        """Ignites all cells within circular radius."""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    self.ignite_cell(x + dx, y + dy, initial_intensity, auto_record=False)
        self.is_extinguished = False
        if len(self.history) == 0 and self.current_step == 0:
            self.record_initial_state()

    def ignite_established_front(self, x: int, y: int, radius: int = 1, initial_intensity: float = 0.95):
        """
        Ignites a robust initial core (e.g. 3x3 kernel) to ensure the flame front
        is well-established and does not suffer from single-cell stochastic extinction.
        """
        self.ignite_radius(x, y, radius=max(1, radius), initial_intensity=initial_intensity)
        self.is_extinguished = False

    def step(self) -> SimulationStepStats:
        """
        Advances the simulation by 1 timestep using high-performance NumPy vectorization.
        """
        active_mask = (self.state == FireState.BURNING)
        active_count = int(np.count_nonzero(active_mask))

        if active_count == 0:
            self.is_extinguished = True
            self.current_step += 1
            self.elapsed_minutes += self.minutes_per_step
            stats = self._compute_stats(spot_count=0)
            self.history.append(stats)
            return stats

        # Vectorized wind field computation with topographic channeling
        wind_speed_grid, u_grid, v_grid = self.weather.get_effective_wind_field(
            self.nx, self.ny, self.terrain.elevation, self.cell_size_m
        )

        # 8-Directional Vectorized Spread
        neighbors = [
            (1, 0, 1.0),          # North (+Y)
            (-1, 0, 1.0),         # South (-Y)
            (0, -1, 1.0),         # West (-X)
            (0, 1, 1.0),          # East (+X)
            (1, -1, 1.41421356),  # NW
            (1, 1, 1.41421356),   # NE
            (-1, -1, 1.41421356), # SW
            (-1, 1, 1.41421356)   # SE
        ]

        newly_ignited = np.zeros((self.ny, self.nx), dtype=bool)
        new_intensities = np.zeros((self.ny, self.nx), dtype=np.float32)

        f_temp = 1.0 + 0.025 * (self.weather.temperature_c - 20.0)
        f_humid = np.exp(-0.018 * self.weather.relative_humidity_pct)

        for dy, dx, dist in neighbors:
            src_y_min, src_y_max = max(0, -dy), min(self.ny, self.ny - dy)
            src_x_min, src_x_max = max(0, -dx), min(self.nx, self.nx - dx)
            dst_y_min, dst_y_max = max(0, dy), min(self.ny, self.ny + dy)
            dst_x_min, dst_x_max = max(0, dx), min(self.nx, self.nx + dx)

            src_slice = (slice(src_y_min, src_y_max), slice(src_x_min, src_x_max))
            dst_slice = (slice(dst_y_min, dst_y_max), slice(dst_x_min, dst_x_max))

            src_active = (self.state[src_slice] == FireState.BURNING)
            dst_unburned = (self.state[dst_slice] == FireState.UNBURNED)
            valid_pair = src_active & dst_unburned

            if not np.any(valid_pair):
                continue

            # Vectorized Slope Factor
            src_elev = self.terrain.elevation[src_slice]
            dst_elev = self.terrain.elevation[dst_slice]
            delta_z = dst_elev - src_elev
            eff_slope = delta_z / (dist * self.cell_size_m)
            f_slope = np.where(
                eff_slope > 0,
                np.exp(2.8 * np.clip(eff_slope, 0.0, 1.2)),
                np.exp(-1.4 * np.clip(np.abs(eff_slope), 0.0, 0.8))
            )

            # Vectorized Wind Factor
            src_u = u_grid[src_slice]
            src_v = v_grid[src_slice]
            src_w_speed = wind_speed_grid[src_slice]
            r_u = dx / dist
            r_v = dy / dist
            cos_phi = (r_u * src_u + r_v * src_v) / (np.sqrt(src_u**2 + src_v**2) + 1e-6)
            f_wind = np.where(
                cos_phi > 0,
                np.exp(0.042 * src_w_speed * cos_phi),
                np.exp(-0.025 * src_w_speed * np.abs(cos_phi))
            )

            # Destination Fuel & Environmental Multipliers
            dst_fuel_mult = self.spread_mult_grid[dst_slice]
            dst_density = self.terrain.fuel_density[dst_slice]
            dst_moist = self.terrain.fuel_moisture[dst_slice]
            dst_risk = self.terrain.historical_risk[dst_slice]
            dst_containment = self.firefighting_mgr.containment_grid[dst_slice]

            f_fuel = dst_fuel_mult * (dst_density ** 0.8)
            f_moist = np.clip((1.0 - (dst_moist / 0.32)), 0.02, 1.0) ** 1.4
            f_risk = 1.0 + 0.45 * dst_risk

            src_intensity = self.fire_intensity[src_slice]

            # Spread rate equation
            spread_rate = (
                self.base_spread_probability *
                self.spread_rate_factor *
                src_intensity *
                f_fuel *
                f_moist *
                f_slope *
                f_wind *
                f_temp *
                f_humid *
                f_risk *
                dst_containment /
                dist
            )

            prob = np.where(
                (dst_fuel_mult > 0.0) & (dst_containment > 0.0),
                np.clip(1.0 - np.exp(-0.35 * spread_rate), 0.0, 0.98),
                0.0
            )

            # Vectorized Stochastic Transition
            rand_draw = np.random.rand(*prob.shape)
            ignited = valid_pair & (rand_draw < prob)

            target_int = np.clip(0.6 * src_intensity + 0.4 * (f_fuel / 2.0), 0.3, 1.0)

            # Apply to output buffers
            newly_ignited[dst_slice] |= ignited
            current_dst_int = new_intensities[dst_slice]
            new_intensities[dst_slice] = np.maximum(current_dst_int, np.where(ignited, target_int, 0.0))

        # Vectorized Spotting / Lofted Firebrands
        spot_count = 0
        if self.enable_spotting and self.weather.wind_speed_kmh > 25.0:
            active_coords = np.argwhere(active_mask)
            if len(active_coords) > 0:
                high_int_mask = self.fire_intensity[active_mask] > 0.65
                spot_candidates = active_coords[high_int_mask]
                if len(spot_candidates) > 0:
                    u_dir, v_dir = self.weather.wind_vector
                    spot_prob = self.spotting_probability * (self.weather.wind_speed_kmh / 40.0)
                    for y, x in spot_candidates:
                        if np.random.rand() < spot_prob:
                            dist_cells = np.random.randint(2, 6)
                            spot_x = int(round(x + u_dir * dist_cells + np.random.randn() * 0.8))
                            spot_y = int(round(y + v_dir * dist_cells + np.random.randn() * 0.8))
                            if 0 <= spot_x < self.nx and 0 <= spot_y < self.ny:
                                if self.state[spot_y, spot_x] == FireState.UNBURNED:
                                    if self.spread_mult_grid[spot_y, spot_x] > 0 and self.firefighting_mgr.containment_grid[spot_y, spot_x] > 0:
                                        newly_ignited[spot_y, spot_x] = True
                                        new_intensities[spot_y, spot_x] = 0.75
                                        self.spot_fire_events.append((self.current_step, spot_y, spot_x))
                                        spot_count += 1

        # Vectorized Combustion Decay & Burn Residence
        self.steps_burning[active_mask] += 1
        active_residence = self.max_residence_grid[active_mask]
        active_steps = self.steps_burning[active_mask]

        # Calculate progress
        self.burn_progress[active_mask] = np.minimum(1.0, active_steps / np.maximum(1, active_residence))

        # Cells that burn out
        burned_out_mask = (active_steps >= active_residence)
        active_indices = np.argwhere(active_mask)
        if len(active_indices) > 0:
            for idx, (y, x) in enumerate(active_indices):
                if burned_out_mask[idx]:
                    self.state[y, x] = FireState.BURNED_OUT
                    self.fire_intensity[y, x] = 0.0
                    self.flame_height_m[y, x] = 0.0
                else:
                    norm_t = active_steps[idx] / max(1, active_residence[idx])
                    if norm_t < 0.25:
                        self.fire_intensity[y, x] = float(np.clip(self.fire_intensity[y, x] * 1.15, 0.2, 1.0))
                    else:
                        decay = (1.0 - norm_t) / 0.75
                        self.fire_intensity[y, x] = float(np.clip(self.fire_intensity[y, x] * (0.85 + 0.15 * decay), 0.05, 1.0))
                    self.flame_height_m[y, x] = float(1.0 + 4.5 * (self.fire_intensity[y, x] ** 0.8))

        # Apply newly ignited cells
        ignited_coords = np.argwhere(newly_ignited)
        for y, x in ignited_coords:
            self.state[y, x] = FireState.BURNING
            self.fire_intensity[y, x] = float(new_intensities[y, x])
            self.steps_burning[y, x] = 0
            self.ignition_step[y, x] = self.current_step
            self.flame_height_m[y, x] = 1.5

        # Compute and record metrics
        self.current_step += 1
        self.elapsed_minutes += self.minutes_per_step
        stats = self._compute_stats(spot_count=spot_count)
        self.history.append(stats)
        return stats

    def _compute_stats(self, spot_count: int = 0) -> SimulationStepStats:
        burned_mask = (self.state == FireState.BURNED_OUT)
        active_mask = (self.state == FireState.BURNING)
        total_affected = burned_mask | active_mask

        burning_cells = int(np.count_nonzero(active_mask))
        burned_cells = int(np.count_nonzero(burned_mask))
        total_affected_cells = int(np.count_nonzero(total_affected))

        burned_area_ha = float(total_affected_cells * self.cell_area_ha)
        burned_area_pct = float((total_affected_cells / (self.nx * self.ny)) * 100.0)
        active_fire_area_ha = float(burning_cells * self.cell_area_ha)

        if len(self.history) > 0:
            prev_ha = self.history[-1].total_burned_area_ha
            ros_ha = max(0.0, burned_area_ha - prev_ha)
        else:
            ros_ha = burned_area_ha

        # Perimeter calculation using binary morphological dilation
        struct = generate_binary_structure(2, 2)
        dilated = binary_dilation(total_affected, structure=struct)
        boundary_cells = np.count_nonzero(dilated ^ total_affected)
        perimeter_km = float((boundary_cells * self.cell_size_m) / 1000.0)

        # Maximum spread distance from origin points
        max_dist_m = 0.0
        if len(self.ignition_points) > 0 and total_affected_cells > 0:
            affected_coords = np.argwhere(total_affected) # (y, x)
            for ox, oy in self.ignition_points:
                dists = np.sqrt((affected_coords[:, 1] - ox)**2 + (affected_coords[:, 0] - oy)**2) * self.cell_size_m
                if len(dists) > 0:
                    max_dist_m = max(max_dist_m, float(np.max(dists)))

        # Fuel consumed in tons
        fuel_consumed = 0.0
        for ftype, load in FuelType.FUEL_LOAD_T_HA.items():
            type_mask = (self.terrain.fuel_type == ftype)
            if np.any(type_mask):
                consumed_in_type = np.sum(self.burn_progress[type_mask])
                density_mean = float(np.mean(self.terrain.fuel_density[type_mask]))
                fuel_consumed += float(consumed_in_type * self.cell_area_ha * load * density_mean)

        # Intensity metrics
        if burning_cells > 0:
            max_int = float(np.max(self.fire_intensity[active_mask]))
            mean_int = float(np.mean(self.fire_intensity[active_mask]))
        else:
            max_int = 0.0
            mean_int = 0.0

        # WUI Threatened & Burned Structures count
        threatened_count = 0
        burned_struct_count = 0
        active_coords = np.argwhere(active_mask)

        for s in self.terrain.structures:
            sx, sy = s.x, s.y
            if self.state[sy, sx] == FireState.BURNED_OUT or self.state[sy, sx] == FireState.BURNING:
                burned_struct_count += 1
            elif len(active_coords) > 0:
                min_d = np.min(np.sqrt((active_coords[:, 1] - sx)**2 + (active_coords[:, 0] - sy)**2)) * self.cell_size_m
                if min_d < 600.0:
                    threatened_count += 1

        # Mean fuel moisture across landscape
        avg_moist_pct = float(np.mean(self.terrain.fuel_moisture[self.terrain.fuel_type != FuelType.WATER]) * 100.0)

        # Cardinal direction
        cardinal_dir = self.get_cardinal_dir(self.weather.wind_direction_deg)

        return SimulationStepStats(
            step=self.current_step,
            elapsed_minutes=self.elapsed_minutes,
            wind_speed_kmh=float(self.weather.wind_speed_kmh),
            wind_direction_deg=float(self.weather.wind_direction_deg),
            wind_direction_cardinal=cardinal_dir,
            temperature_c=float(self.weather.temperature_c),
            humidity_pct=float(self.weather.relative_humidity_pct),
            fuel_moisture_pct=avg_moist_pct,
            fire_intensity_max=max_int,
            fire_intensity_mean=mean_int,
            burning_cells=burning_cells,
            burned_cells=burned_cells,
            active_fire_area_ha=active_fire_area_ha,
            total_burned_area_ha=burned_area_ha,
            burned_area_pct=burned_area_pct,
            fire_perimeter_km=perimeter_km,
            max_spread_distance_m=max_dist_m,
            threatened_structures=threatened_count,
            burned_structures=burned_struct_count,
            rate_of_spread_ha_step=ros_ha,
            fuel_consumed_tons=fuel_consumed,
            spot_fires_count=spot_count
        )

    def reset(self):
        self.state.fill(FireState.UNBURNED)
        self.state[self.terrain.fuel_type == FuelType.WATER] = FireState.UNBURNABLE
        self.fire_intensity.fill(0.0)
        self.steps_burning.fill(0)
        self.burn_progress.fill(0.0)
        self.ignition_step.fill(-1)
        self.flame_height_m.fill(0.0)
        self.current_step = 0
        self.elapsed_minutes = 0.0
        self.ignition_points.clear()
        self.spot_fire_events.clear()
        self.history.clear()
        self.is_running = False
        self.is_extinguished = False
        self.firefighting_mgr.reset()
        for s in self.terrain.structures:
            s.status = "INTACT"
            s.threat_level = "LOW"
            s.distance_to_fire_m = 9999.0

    @staticmethod
    def get_cardinal_dir(deg: float) -> str:
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        idx = int(round(deg / 22.5)) % 16
        return dirs[idx]
