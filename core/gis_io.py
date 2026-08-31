"""
gis_io.py - Modular Geographic Information System (GIS) Data Ingestion and Export.
Enables importing real-world Digital Elevation Models (DEMs), landcover/fuel rasters,
and exporting simulation burn perimeters & severity maps to GeoJSON / ESRI ASCII grid formats.
"""

import json
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union

class GISDataLoader:
    """
    Interface for ingesting raster/DEM data and translating them into simulation grids.
    """
    @staticmethod
    def from_ascii_grid(asc_content: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Parses ESRI ASCII Grid format (.asc) into a 2D numpy array and header dict.
        """
        lines = asc_content.strip().splitlines()
        header = {}
        data_lines = []
        for line in lines:
            tokens = line.strip().split()
            if len(tokens) == 2 and tokens[0].lower() in ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'nodata_value']:
                header[tokens[0].lower()] = float(tokens[1]) if '.' in tokens[1] else int(tokens[1])
            else:
                data_lines.append(line)

        raw_data = " ".join(data_lines)
        grid = np.fromstring(raw_data, sep=' ', dtype=float)
        ncols = int(header.get('ncols', 0))
        nrows = int(header.get('nrows', 0))
        if ncols > 0 and nrows > 0 and len(grid) == ncols * nrows:
            elevation = grid.reshape((nrows, ncols))
            return elevation, header
        else:
            raise ValueError("Invalid ASCII grid dimensions or data content.")

    @staticmethod
    def to_ascii_grid(grid: np.ndarray, cell_size_m: float = 30.0, nodata: float = -9999.0) -> str:
        """
        Exports a 2D array to ESRI ASCII Grid string.
        """
        nrows, ncols = grid.shape
        header_lines = [
            f"ncols         {ncols}",
            f"nrows         {nrows}",
            f"xllcorner     500000.0",
            f"yllcorner     4100000.0",
            f"cellsize      {cell_size_m}",
            f"NODATA_value  {nodata}"
        ]
        header = "\n".join(header_lines) + "\n"
        body = []
        for row in grid:
            body.append(" ".join(f"{val:.2f}" for val in row))
        return header + "\n".join(body)

    @staticmethod
    def export_fire_perimeter_geojson(
        fire_state: np.ndarray,
        cell_size_m: float = 30.0,
        origin_lon: float = -120.5,
        origin_lat: float = 38.5
    ) -> str:
        """
        Exports active & burned boundary cells as a standard GeoJSON FeatureCollection.
        """
        ny, nx = fire_state.shape
        m_to_deg = 1.0 / 111320.0
        
        features = []
        burned_indices = np.argwhere(fire_state >= 1) # Burning or Burned Out

        for y, x in burned_indices:
            lon0 = origin_lon + (x * cell_size_m) * m_to_deg
            lat0 = origin_lat + (y * cell_size_m) * m_to_deg
            lon1 = origin_lon + ((x + 1) * cell_size_m) * m_to_deg
            lat1 = origin_lat + ((y + 1) * cell_size_m) * m_to_deg

            state_name = "Active Burning" if fire_state[y, x] == 1 else "Burned Out (Ash)"

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon0, lat0],
                        [lon1, lat0],
                        [lon1, lat1],
                        [lon0, lat1],
                        [lon0, lat0]
                    ]]
                },
                "properties": {
                    "grid_x": int(x),
                    "grid_y": int(y),
                    "status": state_name,
                    "fire_state": int(fire_state[y, x])
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "features": features
        }
        return json.dumps(geojson, indent=2)