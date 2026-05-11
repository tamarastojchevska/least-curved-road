import json
import math
from typing import Dict, List, Optional

import networkx as nx
import osmnx as ox
import requests

ox.settings.use_cache = False
ox.settings.log_console = False

WALKING_SPEED_M_PER_S = 1.4
DRIVING_FALLBACK_SPEED_M_PER_S = 13.9
OSRM_MATCH_URL = "{base_url}/match/v1/{profile}/{coordinates}"
OSRM_DEFAULT_BASE_URL = "https://router.project-osrm.org"
OSRM_MAX_COORDS = 100


def _validate_coordinates(origin_lat, origin_lon, dest_lat, dest_lon):
    vals = [origin_lat, origin_lon, dest_lat, dest_lon]
    if any(v is None for v in vals):
        raise ValueError("invalid_coordinates")

    origin_lat = float(origin_lat)
    origin_lon = float(origin_lon)
    dest_lat = float(dest_lat)
    dest_lon = float(dest_lon)

    if not (-90 <= origin_lat <= 90 and -90 <= dest_lat <= 90):
        raise ValueError("invalid_coordinates")
    if not (-180 <= origin_lon <= 180 and -180 <= dest_lon <= 180):
        raise ValueError("invalid_coordinates")

    return origin_lat, origin_lon, dest_lat, dest_lon


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _get_graph(origin_lat, origin_lon, dest_lat, dest_lon, mode="driving"):
    center_lat = (origin_lat + dest_lat) / 2
    center_lon = (origin_lon + dest_lon) / 2

    straight_distance = _haversine_m(origin_lat, origin_lon, dest_lat, dest_lon)
    dist = max(1500, int(straight_distance * 1.8))

    network_type = "walk" if mode == "walking" else "drive"

    g = ox.graph_from_point(
        (center_lat, center_lon),
        dist=dist,
        dist_type="bbox",
        network_type=network_type,
        simplify=True,
    )

    return g

def _prepare_graph_weights(g, mode="driving", use_curvature=False, curvature_weight=0.3):
    if use_curvature:
        return _prepare_graph_weights_with_curvature(g, mode, curvature_weight)

    for _, _, _, data in g.edges(keys=True, data=True):
        if "length" not in data or data["length"] is None:
            data["length"] = 0.0

    if mode == "driving":
        g = ox.routing.add_edge_speeds(g)
        g = ox.routing.add_edge_travel_times(g)
    else:
        for _, _, _, data in g.edges(keys=True, data=True):
            length_m = float(data.get("length", 0.0))
            data["travel_time"] = length_m / WALKING_SPEED_M_PER_S if length_m > 0 else 0.0

    return g


def _nearest_nodes(g, origin_lat, origin_lon, dest_lat, dest_lon):
    origin_node = ox.distance.nearest_nodes(g, X=origin_lon, Y=origin_lat)
    dest_node = ox.distance.nearest_nodes(g, X=dest_lon, Y=dest_lat)
    return origin_node, dest_node


def _edge_data_for_pair(g, u, v, weight_key="length"):
    edge_dict = g.get_edge_data(u, v)
    if not edge_dict:
        return None

    best_key = min(
        edge_dict,
        key=lambda k: float(edge_dict[k].get(weight_key, float("inf"))),
    )
    return edge_dict[best_key]

def _calculate_bearing(lat1, lon1, lat2, lon2):
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)

    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360

def _calculate_turn_angle(bearing1, bearing2):
    angle_diff = abs(bearing2 - bearing1)
    # Take the shortest angle (0-180 degrees)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    return angle_diff

def _compute_edge_curvature(g, u, v):
    node_u = g.nodes[u]
    node_v = g.nodes[v]

    lat_u, lon_u = node_u["y"], node_u["x"]
    lat_v, lon_v = node_v["y"], node_v["x"]

    predecessors = list(g.predecessors(u))
    successors = list(g.successors(v))

    total_turn_angle = 0.0
    turn_count = 0

    if predecessors:
        for pred in predecessors:
            lat_p = g.nodes[pred]["y"]
            lon_p = g.nodes[pred]["x"]
            bearing_in = _calculate_bearing(lat_p, lon_p, lat_u, lon_u)
            bearing_out = _calculate_bearing(lat_u, lon_u, lat_v, lon_v)
            turn_angle = _calculate_turn_angle(bearing_in, bearing_out)
            total_turn_angle += turn_angle
            turn_count += 1

    if successors:
        for succ in successors:
            lat_s = g.nodes[succ]["y"]
            lon_s = g.nodes[succ]["x"]
            bearing_in = _calculate_bearing(lat_u, lon_u, lat_v, lon_v)
            bearing_out = _calculate_bearing(lat_v, lon_v, lat_s, lon_s)
            turn_angle = _calculate_turn_angle(bearing_in, bearing_out)
            total_turn_angle += turn_angle
            turn_count += 1

    avg_turn = total_turn_angle / turn_count if turn_count > 0 else 0.0
    curvature_cost = avg_turn / 180.0

    return curvature_cost

def _prepare_graph_weights_with_curvature(g, mode="driving", curvature_weight=0.3):
    """
    curvature_weight: float between 0 and 1
        0.0 = ignore curvature
        1.0 = prioritize curvature equally with distance/time
        0.3 = default: 30% curvature, 70% distance/time
    """

    for u, v, key, data in g.edges(keys=True, data=True):
        curvature = _compute_edge_curvature(g, u, v)
        data["curvature"] = curvature

    for u, v, key, data in g.edges(keys=True, data=True):
        base_weight = float(data.get("travel_time", data.get("length", 0.0)))
        curvature_penalty = data.get("curvature", 0.0) * base_weight
        data["hybrid_weight"] = (
                (1.0 - curvature_weight) * base_weight +
                curvature_weight * curvature_penalty
        )

    return g


def _route_to_polyline(g, route_nodes: List[int]) -> List[Dict[str, float]]:
    polyline = []
    for node_id in route_nodes:
        node = g.nodes[node_id]
        polyline.append({"lat": float(node["y"]), "lng": float(node["x"])})
    return polyline


def _compute_route_metrics(g, route_nodes: List[int], mode="driving"):
    total_distance = 0.0
    total_seconds = 0.0

    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        edge = _edge_data_for_pair(g, u, v, "length")
        if edge is None:
            continue

        length_m = float(edge.get("length", 0.0))
        total_distance += length_m

        if "travel_time" in edge and edge["travel_time"] is not None:
            total_seconds += float(edge["travel_time"])
        else:
            speed = WALKING_SPEED_M_PER_S if mode == "walking" else DRIVING_FALLBACK_SPEED_M_PER_S
            total_seconds += length_m / speed if speed > 0 else 0.0

    return round(total_distance, 2), int(round(total_seconds))


def _compute_bounds(polyline: List[Dict[str, float]]):
    lats = [p["lat"] for p in polyline]
    lngs = [p["lng"] for p in polyline]

    return {
        "north": max(lats),
        "south": min(lats),
        "east": max(lngs),
        "west": min(lngs),
    }


def _chunk_polyline(
    polyline: List[Dict[str, float]], chunk_size: int
) -> List[List[Dict[str, float]]]:
    if len(polyline) <= chunk_size:
        return [polyline]

    chunks = []
    step = chunk_size - 1
    for i in range(0, len(polyline), step):
        chunk = polyline[i : i + chunk_size]
        if len(chunk) < 2:
            break
        chunks.append(chunk)
    return chunks


def _map_match_chunk(
    chunk: List[Dict[str, float]],
    profile: str,
    base_url: str,
) -> Optional[List[Dict[str, float]]]:
    coord_str = ";".join(f"{p['lng']},{p['lat']}" for p in chunk)
    url = OSRM_MATCH_URL.format(base_url=base_url, profile=profile, coordinates=coord_str)

    params = {
        "geometries": "geojson",
        "overview": "full",
        "annotations": "false",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException:
        return None

    if data.get("code") != "Ok":
        return None

    matchings = data.get("matchings")
    if not matchings:
        return None

    geometry_coords = matchings[0].get("geometry", {}).get("coordinates", [])

    if not geometry_coords:
        return None

    return [{"lat": c[1], "lng": c[0]} for c in geometry_coords]


def _post_process_map_matching(
    polyline: List[Dict[str, float]],
    mode: str,
    osrm_base_url: str,
) -> List[Dict[str, float]]:
    profile = "driving" if mode == "driving" else "foot"
    chunks = _chunk_polyline(polyline, OSRM_MAX_COORDS)

    snapped: List[Dict[str, float]] = []

    for i, chunk in enumerate(chunks):
        matched = _map_match_chunk(chunk, profile, osrm_base_url)
        if matched is None:
            return polyline
        if i == 0:
            snapped.extend(matched)
        else:
            snapped.extend(matched[1:])

    return snapped if snapped else polyline


def calculate_route(
    origin_lat,
    origin_lon,
    dest_lat,
    dest_lon,
    mode="driving",
    osrm_base_url: str = OSRM_DEFAULT_BASE_URL,
    map_matching: bool = True,
    use_curvature: bool = True,
    curvature_weight: float = 0.3,
):
    try:
        origin_lat, origin_lon, dest_lat, dest_lon = _validate_coordinates(
            origin_lat, origin_lon, dest_lat, dest_lon
        )

        mode = "walking" if str(mode).lower() == "walking" else "driving"

        g = _get_graph(origin_lat, origin_lon, dest_lat, dest_lon, mode)
        g = _prepare_graph_weights(g, mode, use_curvature=use_curvature, curvature_weight=curvature_weight)

        origin_node, dest_node = _nearest_nodes(g, origin_lat, origin_lon, dest_lat, dest_lon)

        weight = "hybrid_weight" if use_curvature else ("travel_time" if mode == "driving" else "length")
        route_nodes = nx.shortest_path(g, origin_node, dest_node, weight=weight)

        if not route_nodes or len(route_nodes) < 2:
            return json.dumps({
                "status": "error",
                "route": None,
                "bounds": None,
                "error": "no_path_found",
            })

        raw_polyline = _route_to_polyline(g, route_nodes)
        total_distance_meters, estimated_duration_seconds = _compute_route_metrics(
            g, route_nodes, mode
        )

        if map_matching:
            polyline = _post_process_map_matching(raw_polyline, mode, osrm_base_url)
        else:
            polyline = raw_polyline

        bounds = _compute_bounds(polyline)

        return json.dumps({
            "status": "success",
            "route": {
                "polyline": polyline,
                "total_distance_meters": total_distance_meters,
                "estimated_duration_seconds": estimated_duration_seconds,
                "mode": mode,
                "map_matched": map_matching and polyline is not raw_polyline,
            },
            "bounds": bounds,
            "error": None,
        })

    except nx.NetworkXNoPath:
        return json.dumps({
            "status": "error",
            "route": None,
            "bounds": None,
            "error": "no_path_found",
        })
    except ValueError as e:
        return json.dumps({
            "status": "error",
            "route": None,
            "bounds": None,
            "error": str(e),
        })
    except Exception:
        return json.dumps({
            "status": "error",
            "route": None,
            "bounds": None,
            "error": "graph_build_failed",
        })
