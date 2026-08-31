"""
firefighting.py - Interactive Firefighting and Containment Mechanism.
Implements:
1. Firelines (Containment Lines / Bulldozer Lines) - prevents or reduces fire spread.
2. Aerial Water Drops / Retardant - reduces or extinguishes active fire cells and raises fuel moisture.
3. Backburns (Prescribed Tactical Burnouts) - consumes available fuel ahead of the wildfire.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Union

@dataclass
class FirelineAction:
    id: int
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    length_m: float
    status: str = "ACTIVE"

@dataclass
class WaterDropAction:
    id: int
    x: int
    y: int
    radius: int
    step_applied: int
    cells_extinguished: int
    coverage_area_ha: float

@dataclass
class BackburnAction:
    id: int
    x: int
    y: int
    radius: int
    step_applied: int
    cells_backburned: int
    area_ha: float

class FirefightingManager:
    def __init__(self, nx: int, ny: int, cell_size_m: float = 30.0):
        self.nx = nx
        self.ny = ny
        self.cell_size_m = cell_size_m
        self.cell_area_ha = (cell_size_m ** 2) / 10000.0

        # Containment spread multiplier grid [0.0 = total barrier, 1.0 = normal]
        self.containment_grid = np.ones((self.ny, self.nx), dtype=np.float32)
        # Visual markers mask: 0=none, 1=fireline, 2=water drop, 3=backburn
        self.tactic_mask = np.zeros((self.ny, self.nx), dtype=np.int8)

        self.firelines: List[FirelineAction] = []
        self.water_drops: List[WaterDropAction] = []
        self.backburns: List[BackburnAction] = []

        self.total_fireline_km = 0.0
        self.total_water_drops_count = 0
        self.total_backburned_ha = 0.0

    def add_fireline(self, x1: int, y1: int, x2: int, y2: int, width: int = 1) -> FirelineAction:
        """
        Builds a continuous fireline / trench between (x1, y1) and (x2, y2).
        Sets containment spread multiplier to 0.0 along the line.
        """
        x1 = int(np.clip(x1, 0, self.nx - 1))
        y1 = int(np.clip(y1, 0, self.ny - 1))
        x2 = int(np.clip(x2, 0, self.nx - 1))
        y2 = int(np.clip(y2, 0, self.ny - 1))

        # Bresenham-style line interpolation
        num_points = max(abs(x2 - x1), abs(y2 - y1), 1) * 2
        xs = np.linspace(x1, x2, num_points)
        ys = np.linspace(y1, y2, num_points)

        for px, py in zip(xs, ys):
            ix, iy = int(round(px)), int(round(py))
            for dy in range(-width + 1, width):
                for dx in range(-width + 1, width):
                    nx_i, ny_i = ix + dx, iy + dy
                    if 0 <= nx_i < self.nx and 0 <= ny_i < self.ny:
                        self.containment_grid[ny_i, nx_i] = 0.0
                        self.tactic_mask[ny_i, nx_i] = 1

        dist_m = float(np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * self.cell_size_m)
        action = FirelineAction(
            id=len(self.firelines) + 1,
            x1=x1, y1=y1, x2=x2, y2=y2,
            width=width,
            length_m=dist_m
        )
        self.firelines.append(action)
        self.total_fireline_km += (dist_m / 1000.0)
        return action

    def apply_water_drop(
        self,
        simulation,
        x: int,
        y: int,
        radius: int = 2,
        retardant_power: float = 0.90
    ) -> WaterDropAction:
        """
        Executes an aerial water / retardant drop at (x, y) with radius.
        Directly reduces/extinguishes active flames and dramatically increases fuel moisture.
        """
        x = int(np.clip(x, 0, self.nx - 1))
        y = int(np.clip(y, 0, self.ny - 1))

        extinguished = 0
        total_cells = 0

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    nx_i, ny_i = x + dx, y + dy
                    if 0 <= nx_i < self.nx and 0 <= ny_i < self.ny:
                        total_cells += 1
                        # If cell is actively burning, extinguish or suppress it
                        if simulation.state[ny_i, nx_i] == 1: # BURNING
                            if np.random.rand() < retardant_power:
                                simulation.state[ny_i, nx_i] = 2 # BURNED_OUT / Extinguished
                                simulation.fire_intensity[ny_i, nx_i] = 0.0
                                simulation.flame_height_m[ny_i, nx_i] = 0.0
                                extinguished += 1
                            else:
                                simulation.fire_intensity[ny_i, nx_i] *= 0.25

                        # Drench fuel: increase fuel moisture significantly
                        simulation.terrain.fuel_moisture[ny_i, nx_i] = min(0.45, simulation.terrain.fuel_moisture[ny_i, nx_i] + 0.25)
                        # Set partial containment retardant factor
                        self.containment_grid[ny_i, nx_i] = min(self.containment_grid[ny_i, nx_i], 0.15)
                        self.tactic_mask[ny_i, nx_i] = 2

        action = WaterDropAction(
            id=len(self.water_drops) + 1,
            x=x, y=y,
            radius=radius,
            step_applied=simulation.current_step,
            cells_extinguished=extinguished,
            coverage_area_ha=float(total_cells * self.cell_area_ha)
        )
        self.water_drops.append(action)
        self.total_water_drops_count += 1
        return action

    def apply_backburn(
        self,
        simulation,
        x: int,
        y: int,
        radius: int = 2
    ) -> BackburnAction:
        """
        Executes an intentional tactical backburn to consume fuel ahead of the wildfire.
        Transitions fuel in the radius to BURNED_OUT immediately or ignites low-intensity burnout.
        """
        x = int(np.clip(x, 0, self.nx - 1))
        y = int(np.clip(y, 0, self.ny - 1))

        backburned_count = 0

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    nx_i, ny_i = x + dx, y + dy
                    if 0 <= nx_i < self.nx and 0 <= ny_i < self.ny:
                        if simulation.state[ny_i, nx_i] == 0: # UNBURNED
                            simulation.state[ny_i, nx_i] = 2 # Immediately consumed as controlled ash
                            simulation.burn_progress[ny_i, nx_i] = 1.0
                            simulation.fire_intensity[ny_i, nx_i] = 0.0
                            simulation.flame_height_m[ny_i, nx_i] = 0.0
                            simulation.terrain.fuel_density[ny_i, nx_i] = 0.0
                            self.tactic_mask[ny_i, nx_i] = 3
                            backburned_count += 1

        area_ha = float(backburned_count * self.cell_area_ha)
        action = BackburnAction(
            id=len(self.backburns) + 1,
            x=x, y=y,
            radius=radius,
            step_applied=simulation.current_step,
            cells_backburned=backburned_count,
            area_ha=area_ha
        )
        self.backburns.append(action)
        self.total_backburned_ha += area_ha
        return action

    def reset(self):
        self.containment_grid.fill(1.0)
        self.tactic_mask.fill(0)
        self.firelines.clear()
        self.water_drops.clear()
        self.backburns.clear()
        self.total_fireline_km = 0.0
        self.total_water_drops_count = 0
        self.total_backburned_ha = 0.0
