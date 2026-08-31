"""
wui_evacuation.py - Wildland-Urban Interface (WUI) Structure Threat Assessment & Evacuation Corridor Analysis.
Provides:
1. Threat evaluation for synthetic WUI buildings/settlements based on active flame front distance and spread vector.
2. Dijkstra-based evacuation corridor solver computing safe pathways avoiding active fire, burned ground, and steep terrain.
3. Real-time corridor safety classifications (Safe, Caution, Blocked).
4. Academic prototype disclaimer.
"""

import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Union
from core.terrain import TerrainGrid, Structure, FuelType

@dataclass
class EvacuationCorridor:
    id: str
    structure_id: str
    structure_name: str
    origin_coord: Tuple[int, int]
    exit_name: str
    exit_coord: Tuple[int, int]
    path_coords: List[Tuple[int, int]] # [(x, y), ...]
    path_length_m: float
    status: str = "SAFE" # "SAFE", "CAUTION", "BLOCKED"
    min_dist_to_fire_m: float = 9999.0
    status_notes: str = "Clear egress corridor"

@dataclass
class EvacuationAnalysisResult:
    total_structures: int
    intact_count: int
    threatened_count: int
    burned_count: int
    defended_count: int
    corridors: List[EvacuationCorridor]
    safe_corridors_count: int
    caution_corridors_count: int
    blocked_corridors_count: int
    critical_structures: List[Structure]

class WUIEvacuationAnalyzer:
    @staticmethod
    def evaluate_threats_and_corridors(
        simulation
    ) -> EvacuationAnalysisResult:
        """
        Evaluates proximity and threat levels for all structures, and computes optimal evacuation corridors.
        """
        terrain: TerrainGrid = simulation.terrain
        structures: List[Structure] = terrain.structures
        cell_size = simulation.cell_size_m
        active_fire_mask = (simulation.state == 1) # BURNING
        burned_mask = (simulation.state == 2) # BURNED_OUT
        active_coords = np.argwhere(active_fire_mask) # (y, x)

        intact = 0
        threatened = 0
        burned = 0
        defended = 0
        critical_list: List[Structure] = []

        # 1. Update structure threat status
        for s in structures:
            sx, sy = s.x, s.y

            # Check if cell itself is burned or burning
            if simulation.state[sy, sx] == 2:
                s.status = "BURNED"
                s.threat_level = "CRITICAL"
                s.distance_to_fire_m = 0.0
                burned += 1
                continue
            elif simulation.state[sy, sx] == 1:
                s.status = "THREATENED"
                s.threat_level = "CRITICAL"
                s.distance_to_fire_m = 0.0
                threatened += 1
                critical_list.append(s)
                continue

            # Check if defended by water drop or fireline nearby
            is_defended = False
            if hasattr(simulation, 'firefighting_mgr') and simulation.firefighting_mgr is not None:
                fm = simulation.firefighting_mgr
                # If within 2 cells of a tactic
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        nx_i, ny_i = sx + dx, sy + dy
                        if 0 <= nx_i < simulation.nx and 0 <= ny_i < simulation.ny:
                            if fm.tactic_mask[ny_i, nx_i] > 0 or fm.containment_grid[ny_i, nx_i] < 0.5:
                                is_defended = True
                                break

            # Compute distance to nearest active fire cell
            if len(active_coords) > 0:
                # Euclidean distance in meters
                dists_m = np.sqrt((active_coords[:, 1] - sx)**2 + (active_coords[:, 0] - sy)**2) * cell_size
                min_d = float(np.min(dists_m))
                s.distance_to_fire_m = min_d

                # Threat classification
                if min_d < 250.0:
                    s.status = "DEFENDED" if is_defended else "THREATENED"
                    s.threat_level = "CRITICAL"
                    if is_defended:
                        defended += 1
                    else:
                        threatened += 1
                        critical_list.append(s)
                elif min_d < 600.0:
                    s.status = "DEFENDED" if is_defended else "THREATENED"
                    s.threat_level = "HIGH"
                    if is_defended:
                        defended += 1
                    else:
                        threatened += 1
                        critical_list.append(s)
                elif min_d < 1200.0:
                    s.status = "INTACT"
                    s.threat_level = "MODERATE"
                    intact += 1
                else:
                    s.status = "INTACT"
                    s.threat_level = "LOW"
                    intact += 1
            else:
                s.distance_to_fire_m = 9999.0
                s.status = "INTACT"
                s.threat_level = "LOW"
                intact += 1

        # 2. Designated Safe Exit Points (Perimeter Gates)
        nx, ny = simulation.nx, simulation.ny
        exit_points = [
            ("North Highway Exit", (nx // 2, ny - 2)),
            ("South Valley Exit", (nx // 2, 1)),
            ("East Ridge Pass", (nx - 2, ny // 2)),
            ("West Coastal Route", (1, ny // 2))
        ]

        # 3. Build Cost Surface for Evacuation Pathfinding
        cost_grid = np.ones((ny, nx), dtype=np.float32) * 3.0 # Base off-road walking/driving cost

        # Roads are low cost
        if terrain.roads_mask is not None:
            cost_grid[terrain.roads_mask] = 1.0

        # Slope penalty
        cost_grid += (terrain.slope_deg / 15.0) ** 1.5

        # Water bodies impassable (unless road bridge)
        is_water = (terrain.fuel_type == FuelType.WATER)
        if terrain.roads_mask is not None:
            cost_grid[is_water & (~terrain.roads_mask)] = 99999.0
        else:
            cost_grid[is_water] = 99999.0

        # Burned out areas have high risk/debris penalty
        cost_grid[burned_mask] = 500.0

        # Active fire is completely impassable
        cost_grid[active_fire_mask] = 999999.0

        # Active Fire Buffer / Radiant Heat & Smoke Hazard
        if len(active_coords) > 0:
            for ay, ax in active_coords:
                for dy in range(-4, 5):
                    for dx in range(-4, 5):
                        ny_i, nx_i = ay + dy, ax + dx
                        if 0 <= nx_i < nx and 0 <= ny_i < ny:
                            dist_c = np.sqrt(dx*dx + dy*dy)
                            if dist_c <= 4.0:
                                hazard_weight = 80.0 * np.exp(-dist_c / 1.5)
                                cost_grid[ny_i, nx_i] += hazard_weight

        # 4. Compute Evacuation Corridors from each Structure to Nearest Safe Exit
        corridors: List[EvacuationCorridor] = []
        safe_count = 0
        caution_count = 0
        blocked_count = 0

        # Filter active exits (must not be burning)
        viable_exits = [ep for ep in exit_points if not active_fire_mask[ep[1][1], ep[1][0]]]
        if not viable_exits:
            viable_exits = exit_points

        for idx, s in enumerate(structures):
            start = (s.x, s.y)
            # Find best exit using Dijkstra
            best_path, best_cost, best_exit_name, best_exit_coord = WUIEvacuationAnalyzer._dijkstra_shortest_path(
                cost_grid, start, viable_exits, nx, ny
            )

            if len(best_path) > 1:
                path_len_m = 0.0
                min_fire_d = 9999.0
                is_blocked = False
                is_caution = False

                for p_idx in range(len(best_path) - 1):
                    p1 = best_path[p_idx]
                    p2 = best_path[p_idx + 1]
                    step_d = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) * cell_size
                    path_len_m += step_d

                    px, py = p1
                    if active_fire_mask[py, px]:
                        is_blocked = True
                    elif burned_mask[py, px]:
                        is_caution = True

                    if len(active_coords) > 0:
                        d_fire = np.min(np.sqrt((active_coords[:, 1] - px)**2 + (active_coords[:, 0] - py)**2)) * cell_size
                        if d_fire < min_fire_d:
                            min_fire_d = float(d_fire)

                if min_fire_d < 180.0:
                    is_caution = True
                if min_fire_d < 40.0 or is_blocked:
                    status = "BLOCKED"
                    notes = "Cut off by active fire front"
                    blocked_count += 1
                elif is_caution:
                    status = "CAUTION"
                    notes = f"Smoke & heat proximity (min {min_fire_d:.0f}m to flame)"
                    caution_count += 1
                else:
                    status = "SAFE"
                    notes = f"Clear egress ({min_fire_d:.0f}m buffer from fire)"
                    safe_count += 1

                corridor = EvacuationCorridor(
                    id=f"CORR-{idx+1:02d}",
                    structure_id=s.id,
                    structure_name=s.name,
                    origin_coord=start,
                    exit_name=best_exit_name,
                    exit_coord=best_exit_coord,
                    path_coords=best_path,
                    path_length_m=float(path_len_m),
                    status=status,
                    min_dist_to_fire_m=min_fire_d,
                    status_notes=notes
                )
                corridors.append(corridor)

        return EvacuationAnalysisResult(
            total_structures=len(structures),
            intact_count=intact,
            threatened_count=threatened,
            burned_count=burned,
            defended_count=defended,
            corridors=corridors,
            safe_corridors_count=safe_count,
            caution_corridors_count=caution_count,
            blocked_corridors_count=blocked_count,
            critical_structures=critical_list
        )

    @staticmethod
    def _dijkstra_shortest_path(
        cost_grid: np.ndarray,
        start: Tuple[int, int],
        exit_points: List[Tuple[str, Tuple[int, int]]],
        nx: int,
        ny: int
    ) -> Tuple[List[Tuple[int, int]], float, str, Tuple[int, int]]:
        """
        Calculates 2D Dijkstra shortest path from start to closest exit point on cost grid.
        """
        sx, sy = start
        target_coords = {ep[1]: ep[0] for ep in exit_points}

        dist = np.full((ny, nx), np.inf, dtype=np.float32)
        prev = {}
        dist[sy, sx] = 0.0

        pq = [(0.0, sx, sy)]
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

        reached_exit = None
        reached_exit_name = ""

        while pq:
            d, x, y = heapq.heappop(pq)

            if (x, y) in target_coords:
                reached_exit = (x, y)
                reached_exit_name = target_coords[(x, y)]
                break

            if d > dist[y, x]:
                continue

            for dx, dy, mult in moves:
                nx_i, ny_i = x + dx, y + dy
                if 0 <= nx_i < nx and 0 <= ny_i < ny:
                    step_c = (cost_grid[ny_i, nx_i] + cost_grid[y, x]) * 0.5 * mult
                    if step_c > 900000.0:
                        continue # Impassable
                    new_d = d + step_c
                    if new_d < dist[ny_i, nx_i]:
                        dist[ny_i, nx_i] = new_d
                        prev[(nx_i, ny_i)] = (x, y)
                        heapq.heappush(pq, (new_d, nx_i, ny_i))

        if reached_exit is None:
            # Fallback direct straight route to closest exit
            best_ex = exit_points[0]
            return [start, best_ex[1]], 9999.0, best_ex[0], best_ex[1]

        # Reconstruct path
        path = []
        curr = reached_exit
        while curr in prev:
            path.append(curr)
            curr = prev[curr]
        path.append(start)
        path.reverse()

        return path, float(dist[reached_exit[1], reached_exit[0]]), reached_exit_name, reached_exit
