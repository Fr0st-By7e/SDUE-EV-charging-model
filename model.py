from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# =========================
# Global configuration
# =========================

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

MINUTES_PER_DAY = 24 * 60


# =========================
# Data classes
# =========================

@dataclass
class Link:
    u: int
    v: int
    length_km: float
    free_time_min: float
    capacity: float = 2000.0  # simplified traffic capacity per time bin


@dataclass
class Trip:
    origin: int
    destination: int
    departure_min: int
    destination_type: str  # "H", "W", "O"


@dataclass
class EV:
    ev_id: int
    home_zone: int
    has_home_charger: bool
    trip_chain: List[Trip]
    battery_kwh: float
    soc_init_kwh: float
    soc_current_kwh: float = 0.0

    def reset_soc(self) -> None:
        self.soc_current_kwh = self.soc_init_kwh


@dataclass
class FastChargeEvent:
    ev_id: int
    trip_index: int
    fcs_zone: int
    arrival_min: float
    departure_min: float
    charged_kwh: float
    cost: float


@dataclass
class SlowChargeEvent:
    ev_id: int
    trip_index: int
    location_zone: int
    location_type: str
    arrival_min: float
    departure_min: float
    charged_kwh: float
    deferrable_kwh: float
    max_delay_min: float


@dataclass
class SimulationOutputs:
    avg_utilities: List[float] = field(default_factory=list)
    fast_events: List[FastChargeEvent] = field(default_factory=list)
    slow_events: List[SlowChargeEvent] = field(default_factory=list)
    final_link_times: Dict[Tuple[int, int, int], float] = field(default_factory=dict)
    final_prices: Dict[Tuple[int, int], float] = field(default_factory=dict)
    final_fast_load_kw: Dict[Tuple[int, int], float] = field(default_factory=dict)
    case_name: str = "case"


# =========================
# Model parameters
# =========================

@dataclass
class Params:
    # Battery and charging
    E_B: float = 40.0          # battery capacity kWh
    E_min: float = 0.2 * 40.0  # minimum SOC threshold
    E_anx: float = 0.4 * 40.0  # anxiety SOC threshold
    P_f_max: float = 30.0      # fast charger kW
    P_s_home: float = 7.2      # home slow charger kW
    P_s_work_other: float = 11.0

    # Utility
    alpha_tra: float = 0.96    # utils/hour
    alpha_mon: float = 1.0   # utils/$
    beta_u: float = 15.0       # logit scale parameter

    # Pricing
    cp_nor: float = 0.4        # $/kWh
    theta: float = 2.0        # price tuning
    dt_bin_min: int = 15

    # Infrastructure coverage
    LC_pri: float = 0.5        # home slow charging coverage
    LC_pub: float = 0.5        # public slow charging coverage

    # Iterations
    max_iters: int = 60
    eps: float = 1e-4
    min_iters_before_stop: int = 20

    # Simplified traffic model
    traffic_sensitivity: float = 0.15
    energy_per_km_base: float = 0.2  # kWh/km baseline

    # Demand scaling for demo
    num_demo_evs: int = 2000

    # Fast load target mode
    fixed_fast_load_limit_kw: Optional[float] = None


# =========================
# Utility helpers
# =========================

def minute_to_bin(t_min: float, dt_bin_min: int) -> int:
    t_min = max(0, min(MINUTES_PER_DAY - 1e-9, t_min))
    return int(t_min // dt_bin_min)


def overlap_minutes(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =========================
# Demo data generation
# =========================

def build_demo_network() -> Tuple[nx.DiGraph, Dict[int, Tuple[float, float]]]:
    G = nx.DiGraph()
    coords: Dict[int, Tuple[float, float]] = {}

    idx = 1
    for r in range(4):
        for c in range(6):
            coords[idx] = (c, 3 - r)
            idx += 1

    def add_bidir(a: int, b: int, length: float, speed_kmh: float = 40.0) -> None:
        free_time = 60.0 * length / speed_kmh
        G.add_edge(a, b, link=Link(a, b, length, free_time))
        G.add_edge(b, a, link=Link(b, a, length, free_time))

    for r in range(4):
        for c in range(5):
            n1 = r * 6 + c + 1
            n2 = r * 6 + c + 2
            add_bidir(n1, n2, length=4.0 + 0.2 * ((n1 + n2) % 3))

    for r in range(3):
        for c in range(6):
            n1 = r * 6 + c + 1
            n2 = (r + 1) * 6 + c + 1
            add_bidir(n1, n2, length=2.5 + 0.2 * ((n1 + n2) % 4))

    extra = [
        (1, 8, 4.0), (2, 9, 4.2), (3, 10, 4.0), (4, 11, 4.1), (5, 12, 4.3),
        (7, 14, 4.0), (8, 15, 4.1), (9, 16, 4.0), (10, 17, 4.1), (11, 18, 4.0),
        (13, 20, 4.2), (14, 21, 4.0), (15, 22, 4.1), (16, 23, 4.0), (17, 24, 4.2),
        (1, 7, 2.8), (7, 13, 2.8), (13, 19, 2.8), (6, 12, 2.8), (12, 18, 2.8), (18, 24, 2.8),
    ]
    for a, b, l in extra:
        add_bidir(a, b, l, speed_kmh=50.0)

    return G, coords


def zone_weights_demo() -> Dict[int, float]:
    weights = {}
    central = {8, 9, 10, 11, 14, 15, 16, 17}
    for z in range(1, 25):
        weights[z] = 3.0 if z in central else 1.0 + 0.2 * ((z * 7) % 5)
    return weights


def sample_zone(weights: Dict[int, float]) -> int:
    zones = list(weights.keys())
    w = np.array([weights[z] for z in zones], dtype=float)
    w = w / w.sum()
    return int(np.random.choice(zones, p=w))


def generate_demo_trip_chain(home_zone: int) -> List[Trip]:
    kind = "HWH" if random.random() < 0.73 else "HOH"

    if kind == "HWH":
        work_zone = random.choice([z for z in range(1, 25) if z != home_zone])
        dep_home = int(np.random.normal(7.5 * 60, 15))
        dep_work = int(np.random.normal(17.5 * 60, 15))
        dep_home = int(clamp(dep_home, 5 * 60, 10 * 60))
        dep_work = int(clamp(dep_work, 15 * 60, 21 * 60))
        return [
            Trip(origin=home_zone, destination=work_zone, departure_min=dep_home, destination_type="W"),
            Trip(origin=work_zone, destination=home_zone, departure_min=dep_work, destination_type="H"),
        ]
    else:
        other_zone = random.choice([z for z in range(1, 25) if z != home_zone])
        dep_home = random.randint(7 * 60, 20 * 60)
        stay = random.randint(120, 300)
        dep_other = min(dep_home + stay, 22 * 60)
        return [
            Trip(origin=home_zone, destination=other_zone, departure_min=dep_home, destination_type="O"),
            Trip(origin=other_zone, destination=home_zone, departure_min=dep_other, destination_type="H"),
        ]


def generate_demo_population(params: Params) -> List[EV]:
    weights = zone_weights_demo()
    evs: List[EV] = []
    for i in range(params.num_demo_evs):
        home = sample_zone(weights)
        has_home_charger = random.random() < params.LC_pri
        trip_chain = generate_demo_trip_chain(home)
        if has_home_charger:
         soc0 = np.random.uniform(0.5, 0.9) * params.E_B
        else:
            soc0 = np.random.uniform(0.2, 0.6) * params.E_B
        evs.append(
            EV(
                ev_id=i,
                home_zone=home,
                has_home_charger=has_home_charger,
                trip_chain=trip_chain,
                battery_kwh=params.E_B,
                soc_init_kwh=soc0,
            )
        )
    return evs


# =========================
# Optional CSV loading
# =========================

def load_network_from_csv(nodes_csv: str, links_csv: str) -> Tuple[nx.DiGraph, Dict[int, Tuple[float, float]]]:
    if not (os.path.exists(nodes_csv) and os.path.exists(links_csv)):
        raise FileNotFoundError("Network CSV files not found.")

    coords: Dict[int, Tuple[float, float]] = {}
    with open(nodes_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = int(row["node_id"])
            coords[node] = (float(row["x"]), float(row["y"]))

    G = nx.DiGraph()
    with open(links_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = int(row["u"])
            v = int(row["v"])
            length = float(row["length_km"])
            free_time = float(row["free_time_min"])
            cap = float(row["capacity"]) if "capacity" in row and row["capacity"] else 2000.0
            G.add_edge(u, v, link=Link(u, v, length, free_time, cap))

    return G, coords


# =========================
# Time-dependent network methods
# =========================


class TimeDependentNetwork:
    def __init__(self, G: nx.DiGraph, params: Params):
        self.G = G
        self.params = params
        self.num_bins = MINUTES_PER_DAY // params.dt_bin_min
        self.expected_link_times: Dict[Tuple[int, int, int], float] = {}
        self._init_free_flow_times()

    def _init_free_flow_times(self) -> None:
        for u, v, data in self.G.edges(data=True):
            free_time = data["link"].free_time_min
            for b in range(self.num_bins):
                self.expected_link_times[(u, v, b)] = free_time

    def shortest_path_and_time(self, origin: int, destination: int, dep_min: float) -> Tuple[List[int], float, float]:
        b = minute_to_bin(dep_min, self.params.dt_bin_min)
        H = nx.DiGraph()
        for u, v, data in self.G.edges(data=True):
            H.add_edge(u, v, weight=self.expected_link_times[(u, v, b)])

        path = nx.shortest_path(H, origin, destination, weight="weight")
        travel_time = 0.0
        distance = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            travel_time += self.expected_link_times[(u, v, b)]
            distance += self.G[u][v]["link"].length_km
        return path, travel_time, distance

    def path_distance(self, path: List[int]) -> float:
        d = 0.0
        for i in range(len(path) - 1):
            d += self.G[path[i]][path[i + 1]]["link"].length_km
        return d


# =========================
# Pricing model
# =========================

class PricingModel:
    def __init__(self, zone_ids: List[int], params: Params):
        self.zone_ids = zone_ids
        self.params = params
        self.num_bins = MINUTES_PER_DAY // params.dt_bin_min
        self.expected_prices: Dict[Tuple[int, int], float] = {}
        self.reset()

    def reset(self) -> None:
        for z in self.zone_ids:
            for b in range(self.num_bins):
                self.expected_prices[(z, b)] = self.params.cp_nor


# =========================
# Traffic state accumulation
# =========================

@dataclass
class IterationState:
    link_bin_flows: Dict[Tuple[int, int, int], int] = field(default_factory=dict)
    fast_load_kw: Dict[Tuple[int, int], float] = field(default_factory=dict)  # (zone, bin) -> avg kW
    fast_events: List[FastChargeEvent] = field(default_factory=list)
    slow_events: List[SlowChargeEvent] = field(default_factory=list)
    avg_utility: float = 0.0


# =========================
# Energy model
# =========================

def energy_per_km(speed_kmh: float, params: Params) -> float:
    speed_kmh = max(5.0, speed_kmh)
    base = params.energy_per_km_base
    penalty = 0.0009 * abs(speed_kmh - 35.0) ** 1.15 / 10.0
    return base + penalty


def expected_speed_kmh(length_km: float, travel_time_min: float) -> float:
    if travel_time_min <= 1e-9:
        return 35.0
    return 60.0 * length_km / travel_time_min


# =========================
# Charging rules
# =========================

def public_slow_charger_available(params: Params) -> bool:
    return random.random() < params.LC_pub


def slow_charge_power(location_type: str, params: Params) -> float:
    return params.P_s_home if location_type == "H" else params.P_s_work_other


def should_fast_charge_intermediate(soc_dep: float, energy_needed_trip: float, params: Params) -> bool:
    return (soc_dep - energy_needed_trip) < params.E_min


def should_fast_charge_last_trip(soc_dep: float, energy_needed_trip: float, has_home_charger: bool, params: Params) -> bool:
    if has_home_charger:
        return (soc_dep - energy_needed_trip) < params.E_min
    return ((soc_dep - energy_needed_trip) < params.E_min) or (soc_dep < params.E_anx)


def should_slow_charge_intermediate(
    soc_arr: float,
    next_trip_energy_expected: float,
    parking_min: float,
    location_type: str,
    params: Params,
) -> bool:
    p_s = slow_charge_power(location_type, params)
    crit1 = (soc_arr - next_trip_energy_expected) < params.E_min
    crit2 = (soc_arr < params.E_anx) and (soc_arr + (parking_min / 60.0) * p_s >= 0.8 * params.E_B)
    return crit1 or crit2


def should_slow_charge_last_destination(has_home_charger: bool) -> bool:
    return has_home_charger


def slow_charge_metrics(arrival_min: float, departure_min: float, charged_kwh: float, p_s_kw: float) -> Tuple[float, float]:
    parking_h = max(0.0, (departure_min - arrival_min) / 60.0)
    if parking_h <= 0 or p_s_kw <= 0 or charged_kwh <= 0:
        return 0.0, 0.0

    full_possible = parking_h * p_s_kw
    if full_possible <= charged_kwh + 1e-9:
        return 0.0, 0.0

    deferrable = min(full_possible - charged_kwh, charged_kwh)
    t_late = departure_min - 60.0 * charged_kwh / p_s_kw
    max_delay = max(0.0, t_late - arrival_min)
    return deferrable, max_delay


# =========================
# Choice model
# =========================

def compute_trip_utility(travel_time_min: float, fast_charge_cost: float, params: Params) -> float:
    return -(params.alpha_tra * (travel_time_min / 60.0) + params.alpha_mon * fast_charge_cost)


def logit_probs(utilities: List[float], beta_u: float) -> np.ndarray:
    if len(utilities) == 1:
        return np.array([1.0])
    arr = np.array(utilities, dtype=float)
    arr = beta_u * (arr - arr.max())
    ex = np.exp(arr)
    return ex / ex.sum()


# =========================
# Simulation engine
# =========================

class EVChargingSDUEModel:
    def __init__(self, G: nx.DiGraph, coords: Dict[int, Tuple[float, float]], evs: List[EV], params: Params):
        self.G = G
        self.coords = coords
        self.evs = evs
        self.params = params
        self.network = TimeDependentNetwork(G, params)
        self.pricing = PricingModel(sorted(coords.keys()), params)
        self.num_bins = MINUTES_PER_DAY // params.dt_bin_min

    def reachable_fcs_candidates(
        self,
        origin: int,
        destination: int,
        dep_time_min: float,
        soc_dep_kwh: float,
    ) -> List[Dict[str, Any]]:
        candidates = []
        for k in self.coords.keys():
            try:
                path1, tt1, dist1 = self.network.shortest_path_and_time(origin, k, dep_time_min)
                speed1 = expected_speed_kmh(dist1, tt1)
                e1 = dist1 * energy_per_km(speed1, self.params)

                soc_arr = soc_dep_kwh - e1
                if soc_arr < self.params.E_min:
                    continue

                tp_h = max(0.0, (0.6 * self.params.E_B - soc_arr) / self.params.P_f_max)
                ta = dep_time_min + tt1
                td = ta + 60.0 * tp_h

                path2, tt2, dist2 = self.network.shortest_path_and_time(k, destination, td)
                speed2 = expected_speed_kmh(dist2, tt2)
                e2 = dist2 * energy_per_km(speed2, self.params)

                if (0.8 * self.params.E_B - e2) < self.params.E_min:
                    continue

                b = minute_to_bin(ta, self.params.dt_bin_min)
                price = self.pricing.expected_prices[(k, b)]
                charged_kwh = max(0.0, 0.8 * self.params.E_B - soc_arr)
                cost = price * charged_kwh
                total_tt = tt1 + 60.0 * tp_h + tt2
                util = compute_trip_utility(total_tt, cost, self.params)

                candidates.append(
                    {
                        "fcs": k,
                        "path1": path1,
                        "path2": path2,
                        "tt1": tt1,
                        "tt2": tt2,
                        "dist1": dist1,
                        "dist2": dist2,
                        "e1": e1,
                        "e2": e2,
                        "soc_arr": soc_arr,
                        "ta": ta,
                        "td": td,
                        "charged_kwh": charged_kwh,
                        "price": price,
                        "cost": cost,
                        "utility": util,
                    }
                )
            except nx.NetworkXNoPath:
                continue

        return candidates

    def simulate_trip_no_fast(
        self,
        ev: EV,
        trip: Trip,
        state: IterationState,
    ) -> Tuple[float, float, List[int]]:
        path, tt, dist = self.network.shortest_path_and_time(trip.origin, trip.destination, trip.departure_min)
        speed = expected_speed_kmh(dist, tt)
        e_cons = dist * energy_per_km(speed, self.params)

        current_time = trip.departure_min
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            b = minute_to_bin(current_time, self.params.dt_bin_min)
            state.link_bin_flows[(u, v, b)] = state.link_bin_flows.get((u, v, b), 0) + 1
            current_time += self.network.expected_link_times[(u, v, b)]

        return tt, e_cons, path

    def add_fast_load_event(self, zone: int, arrival_min: float, departure_min: float, power_kw: float, state: IterationState) -> None:
        for b in range(self.num_bins):
            bin_start = b * self.params.dt_bin_min
            bin_end = (b + 1) * self.params.dt_bin_min
            ov = overlap_minutes(arrival_min, departure_min, bin_start, bin_end)
            if ov > 0:
                avg_kw = power_kw * (ov / self.params.dt_bin_min)
                state.fast_load_kw[(zone, b)] = state.fast_load_kw.get((zone, b), 0.0) + avg_kw

    def simulate_iteration(self, price_incentives: bool) -> IterationState:
        state = IterationState()
        utility_sum = 0.0

        for ev in self.evs:
            ev.reset_soc()

        for ev in self.evs:
            trip_chain_utility = 0.0

            for m, trip in enumerate(ev.trip_chain):
                direct_path, direct_tt, direct_dist = self.network.shortest_path_and_time(trip.origin, trip.destination, trip.departure_min)
                speed_direct = expected_speed_kmh(direct_dist, direct_tt)
                energy_direct = direct_dist * energy_per_km(speed_direct, self.params)

                is_last_trip = (m == len(ev.trip_chain) - 1)
                if is_last_trip:
                    need_fast = should_fast_charge_last_trip(
                        ev.soc_current_kwh, energy_direct, ev.has_home_charger, self.params
                    )
                else:
                    need_fast = should_fast_charge_intermediate(
                        ev.soc_current_kwh, energy_direct, self.params
                    )

                if need_fast:
                    candidates = self.reachable_fcs_candidates(
                        trip.origin, trip.destination, trip.departure_min, ev.soc_current_kwh
                    )
                    if not candidates:
                        tt, e_cons, _ = self.simulate_trip_no_fast(ev, trip, state)
                        ev.soc_current_kwh = max(0.0, ev.soc_current_kwh - e_cons)
                        trip_chain_utility += compute_trip_utility(tt, 0.0, self.params)
                        arr_time = trip.departure_min + tt
                    else:
                        
                        utils = [c["utility"] for c in candidates]
                        probs = logit_probs(utils, self.params.beta_u)
                        idx = int(np.random.choice(len(candidates), p=probs))
                        c = candidates[idx]

                        current_time = trip.departure_min
                        for i in range(len(c["path1"]) - 1):
                            u, v = c["path1"][i], c["path1"][i + 1]
                            b = minute_to_bin(current_time, self.params.dt_bin_min)
                            state.link_bin_flows[(u, v, b)] = state.link_bin_flows.get((u, v, b), 0) + 1
                            current_time += self.network.expected_link_times[(u, v, b)]

                        charged_kwh = c["charged_kwh"]
                        cost = c["cost"]
                        ta = c["ta"]
                        td = c["td"]
                        fcs_zone = c["fcs"]

                        state.fast_events.append(
                            FastChargeEvent(
                                ev_id=ev.ev_id,
                                trip_index=m,
                                fcs_zone=fcs_zone,
                                arrival_min=ta,
                                departure_min=td,
                                charged_kwh=charged_kwh,
                                cost=cost,
                            )
                        )
                        self.add_fast_load_event(fcs_zone, ta, td, self.params.P_f_max, state)

                        current_time = td
                        for i in range(len(c["path2"]) - 1):
                            u, v = c["path2"][i], c["path2"][i + 1]
                            b = minute_to_bin(current_time, self.params.dt_bin_min)
                            state.link_bin_flows[(u, v, b)] = state.link_bin_flows.get((u, v, b), 0) + 1
                            current_time += self.network.expected_link_times[(u, v, b)]

                        ev.soc_current_kwh = max(0.0, 0.8 * self.params.E_B - c["e2"])
                        total_tt = c["tt1"] + (td - ta) + c["tt2"]
                        trip_chain_utility += compute_trip_utility(total_tt, cost, self.params)
                        arr_time = trip.departure_min + total_tt
                else:
                    tt, e_cons, _ = self.simulate_trip_no_fast(ev, trip, state)
                    ev.soc_current_kwh = max(0.0, ev.soc_current_kwh - e_cons)
                    trip_chain_utility += compute_trip_utility(tt, 0.0, self.params)
                    arr_time = trip.departure_min + tt

                if is_last_trip:
                    parking_departure = MINUTES_PER_DAY
                    need_slow = should_slow_charge_last_destination(ev.has_home_charger)
                else:
                    next_trip = ev.trip_chain[m + 1]
                    parking_departure = next_trip.departure_min

                    _, next_tt, next_dist = self.network.shortest_path_and_time(
                        next_trip.origin, next_trip.destination, next_trip.departure_min
                    )
                    next_speed = expected_speed_kmh(next_dist, next_tt)
                    next_energy = next_dist * energy_per_km(next_speed, self.params)

                    parking_min = max(0.0, parking_departure - arr_time)
                    need_slow = should_slow_charge_intermediate(
                        ev.soc_current_kwh, next_energy, parking_min, trip.destination_type, self.params
                    )

                    if trip.destination_type in ("W", "O"):
                        need_slow = need_slow and public_slow_charger_available(self.params)

                if need_slow:
                    p_s = slow_charge_power(trip.destination_type, self.params)
                    e_req = max(0.0, 0.8 * self.params.E_B - ev.soc_current_kwh)
                    parking_h = max(0.0, (parking_departure - arr_time) / 60.0)
                    charged_kwh = min(parking_h * p_s, e_req)

                    if charged_kwh > 1e-9:
                        deferrable, max_delay = slow_charge_metrics(arr_time, parking_departure, charged_kwh, p_s)
                        state.slow_events.append(
                            SlowChargeEvent(
                                ev_id=ev.ev_id,
                                trip_index=m,
                                location_zone=trip.destination,
                                location_type=trip.destination_type,
                                arrival_min=arr_time,
                                departure_min=parking_departure,
                                charged_kwh=charged_kwh,
                                deferrable_kwh=deferrable,
                                max_delay_min=max_delay,
                            )
                        )
                        ev.soc_current_kwh = min(self.params.E_B, ev.soc_current_kwh + charged_kwh)

            utility_sum += trip_chain_utility

        state.avg_utility = utility_sum / max(1, len(self.evs))
        return state

    def update_link_times_msa(self, state: IterationState, iteration: int) -> None:
        for u, v, data in self.G.edges(data=True):
            link: Link = data["link"]
            for b in range(self.num_bins):
                flow = state.link_bin_flows.get((u, v, b), 0)
                realized = link.free_time_min * (1.0 + self.params.traffic_sensitivity * (flow / max(1.0, link.capacity)))
                prev = self.network.expected_link_times[(u, v, b)]
                updated = realized / (iteration + 1) + prev * iteration / (iteration + 1)
                self.network.expected_link_times[(u, v, b)] = updated

    def compute_fast_load_limit(self, baseline_fast_load: Dict[Tuple[int, int], float]) -> float:
        if self.params.fixed_fast_load_limit_kw is not None:
            return self.params.fixed_fast_load_limit_kw

        per_bin_mean = []
        zones = list(self.coords.keys())
        for b in range(self.num_bins):
            vals = [baseline_fast_load.get((z, b), 0.0) for z in zones]
            per_bin_mean.append(float(np.mean(vals)))
        return max(per_bin_mean) if per_bin_mean else self.params.P_f_max

    def update_prices_msa(
        self,
        state: IterationState,
        iteration: int,
        price_incentives: bool,
        load_limit_kw: Optional[float] = None,
        
    ) -> None:
        for z in self.coords.keys():
            for b in range(self.num_bins):
                realized_load = state.fast_load_kw.get((z, b), 0.0)
                if not price_incentives:
                    assigned = self.params.cp_nor
                else:
                    limit = load_limit_kw if load_limit_kw is not None else self.params.P_f_max
                    if realized_load <= limit + 1e-9:
                        assigned = self.params.cp_nor
                    else:
                        ratio = clamp(limit / max(realized_load, 1e-6), 1e-9, 1.0)
                        assigned = self.params.cp_nor * (1 + 3* self.params.theta * (realized_load / limit))

                prev = self.pricing.expected_prices[(z, b)]
                updated = assigned / (iteration + 1) + prev * iteration / (iteration + 1)
                self.pricing.expected_prices[(z, b)] = updated

    def has_converged(self, avg_utils: List[float]) -> bool:
        n = len(avg_utils)
        if n < max(self.params.min_iters_before_stop, 3):
            return False
        prev_mean = np.mean(avg_utils[:-1])
        curr_mean = np.mean(avg_utils)
        return abs(prev_mean - curr_mean) <= self.params.eps

    def run_case(
        self,
        case_name: str,
        price_incentives: bool,
        baseline_fast_load_for_limit: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> SimulationOutputs:
        self.network._init_free_flow_times()
        self.pricing.reset()

        outputs = SimulationOutputs(case_name=case_name)
        load_limit_kw = None
        if price_incentives and baseline_fast_load_for_limit is not None:
            load_limit_kw = self.compute_fast_load_limit(baseline_fast_load_for_limit)

        last_state: Optional[IterationState] = None
        rolling_utils: List[float] = []

        for it in range(self.params.max_iters):
            state = self.simulate_iteration(price_incentives=price_incentives)
            outputs.avg_utilities.append(state.avg_utility)

            rolling_utils.append(state.avg_utility)
            if len(rolling_utils) > 10:
                rolling_utils.pop(0)

            self.update_link_times_msa(state, it)
            self.update_prices_msa(state, it, price_incentives, load_limit_kw)

            last_state = state

            if self.has_converged(rolling_utils):
                print(f"[{case_name}] converged at iteration {it + 1}")
                break

        if last_state is None:
            raise RuntimeError("Simulation did not produce any iteration state.")

        outputs.fast_events = last_state.fast_events
        outputs.slow_events = last_state.slow_events
        outputs.final_fast_load_kw = dict(last_state.fast_load_kw)
        outputs.final_link_times = dict(self.network.expected_link_times)
        outputs.final_prices = dict(self.pricing.expected_prices)
        return outputs


# =========================
# Analysis and plotting
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def summarize_slow_events(slow_events: List[SlowChargeEvent]) -> Dict[str, Dict[str, float]]:
    summary = {"H": {}, "W": {}, "O": {}}
    for loc in summary.keys():
        events = [e for e in slow_events if e.location_type == loc]
        summary[loc]["num_events"] = len(events)
        summary[loc]["charged_kwh"] = float(sum(e.charged_kwh for e in events))
        summary[loc]["deferrable_kwh"] = float(sum(e.deferrable_kwh for e in events))
        summary[loc]["avg_max_delay_h"] = float(np.mean([e.max_delay_min / 60.0 for e in events])) if events else 0.0
    return summary


def fast_load_timeseries_by_zone(fast_load_kw: Dict[Tuple[int, int], float], zones: List[int], num_bins: int) -> Dict[int, np.ndarray]:
    ts = {z: np.zeros(num_bins) for z in zones}
    for (z, b), kw in fast_load_kw.items():
        if z in ts:
            ts[z][b] = kw
    return ts


def top_k_zones_by_peak(fast_load_kw: Dict[Tuple[int, int], float], k: int = 6) -> List[int]:
    peak_by_zone: Dict[int, float] = {}
    for (z, _b), kw in fast_load_kw.items():
        peak_by_zone[z] = max(peak_by_zone.get(z, 0.0), kw)
    zones = sorted(peak_by_zone.keys(), key=lambda z: peak_by_zone[z], reverse=True)
    return zones[:k]


def plot_fast_load_profiles(outputs: SimulationOutputs, outdir: str) -> None:
    ensure_dir(outdir)
    zones = top_k_zones_by_peak(outputs.final_fast_load_kw, k=8)
    num_bins = MINUTES_PER_DAY // 15
    ts = fast_load_timeseries_by_zone(outputs.final_fast_load_kw, zones, num_bins)
    x = np.arange(num_bins) * 15 / 60.0

    plt.figure(figsize=(12, 6))
    for z in zones:
        plt.plot(x, ts[z], label=f"FCS {z}")
    plt.xlabel("Time of day (hour)")
    plt.ylabel("Average fast charging load (kW)")
    plt.title(f"Fast charging load profiles - {outputs.case_name}")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{outputs.case_name}_fast_load_profiles.png"), dpi=200)
    plt.close()


def plot_prices(outputs: SimulationOutputs, outdir: str) -> None:
    ensure_dir(outdir)
    num_bins = MINUTES_PER_DAY // 15
    zone_ids = sorted({z for (z, _b) in outputs.final_prices.keys()})
    mean_price = {
        z: np.mean([outputs.final_prices[(z, b)] for b in range(num_bins)])
        for z in zone_ids
    }
    top = sorted(zone_ids, key=lambda z: mean_price[z], reverse=True)[:8]
    x = np.arange(num_bins) * 15 / 60.0

    plt.figure(figsize=(12, 6))
    for z in top:
        y = [outputs.final_prices[(z, b)] for b in range(num_bins)]
        plt.plot(x, y, label=f"FCS {z}")
    plt.xlabel("Time of day (hour)")
    plt.ylabel("Charging price ($/kWh)")
    plt.title(f"FCS charging prices - {outputs.case_name}")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{outputs.case_name}_prices.png"), dpi=200)
    plt.close()


def plot_avg_utility(outputs: SimulationOutputs, outdir: str) -> None:
    ensure_dir(outdir)
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(outputs.avg_utilities) + 1), outputs.avg_utilities, marker="o", ms=3)
    plt.xlabel("Iteration")
    plt.ylabel("Average trip-chain utility")
    plt.title(f"Convergence - {outputs.case_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{outputs.case_name}_convergence.png"), dpi=200)
    plt.close()


def plot_slow_summary(outputs: SimulationOutputs, outdir: str) -> None:
    ensure_dir(outdir)
    summary = summarize_slow_events(outputs.slow_events)
    locs = ["H", "W", "O"]
    charged = [summary[l]["charged_kwh"] for l in locs]
    deferrable = [summary[l]["deferrable_kwh"] for l in locs]

    x = np.arange(len(locs))
    width = 0.36

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, charged, width=width, label="Charged energy (kWh)")
    plt.bar(x + width / 2, deferrable, width=width, label="Deferrable energy (kWh)")
    plt.xticks(x, ["Home", "Work", "Other"])
    plt.ylabel("Energy (kWh)")
    plt.title(f"Slow charging flexibility summary - {outputs.case_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{outputs.case_name}_slow_summary.png"), dpi=200)
    plt.close()


def plot_case_comparison(
    case1: SimulationOutputs,
    case2: SimulationOutputs,
    outdir: str,
    dt_bin_min: int,
) -> None:
    ensure_dir(outdir)

    num_bins = MINUTES_PER_DAY // dt_bin_min
    zone_ids = sorted({z for (z, _b) in set(case1.final_fast_load_kw.keys()) | set(case2.final_fast_load_kw.keys())})
    x = np.arange(num_bins) * dt_bin_min / 60.0

    total1 = np.zeros(num_bins)
    total2 = np.zeros(num_bins)
    for z in zone_ids:
        for b in range(num_bins):
            total1[b] += case1.final_fast_load_kw.get((z, b), 0.0)
            total2[b] += case2.final_fast_load_kw.get((z, b), 0.0)

    plt.figure(figsize=(10, 5))
    plt.plot(x, total1, label=case1.case_name)
    plt.plot(x, total2, label=case2.case_name)
    plt.xlabel("Time of day (hour)")
    plt.ylabel("Total fast charging load (kW)")
    plt.title("Case comparison: total fast charging load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "case_comparison_total_fast_load.png"), dpi=200)
    plt.close()


def export_fast_events(events: List[FastChargeEvent], filepath: str) -> None:
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ev_id", "trip_index", "fcs_zone", "arrival_min", "departure_min", "charged_kwh", "cost"])
        for e in events:
            writer.writerow([e.ev_id, e.trip_index, e.fcs_zone, e.arrival_min, e.departure_min, e.charged_kwh, e.cost])


def export_slow_events(events: List[SlowChargeEvent], filepath: str) -> None:
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ev_id", "trip_index", "location_zone", "location_type", "arrival_min",
            "departure_min", "charged_kwh", "deferrable_kwh", "max_delay_min"
        ])
        for e in events:
            writer.writerow([
                e.ev_id, e.trip_index, e.location_zone, e.location_type, e.arrival_min,
                e.departure_min, e.charged_kwh, e.deferrable_kwh, e.max_delay_min
            ])


def export_summary(case: SimulationOutputs, filepath: str) -> None:
    slow_summary = summarize_slow_events(case.slow_events)

    total_fast_energy_kwh = sum(e.charged_kwh for e in case.fast_events)
    total_fast_cost = sum(e.cost for e in case.fast_events)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["case_name", case.case_name])
        writer.writerow(["num_fast_events", len(case.fast_events)])
        writer.writerow(["total_fast_energy_kwh", total_fast_energy_kwh])
        writer.writerow(["total_fast_cost_$", total_fast_cost])
        writer.writerow(["num_slow_events", len(case.slow_events)])
        for loc in ["H", "W", "O"]:
            writer.writerow([f"{loc}_charged_kwh", slow_summary[loc]["charged_kwh"]])
            writer.writerow([f"{loc}_deferrable_kwh", slow_summary[loc]["deferrable_kwh"]])
            writer.writerow([f"{loc}_avg_max_delay_h", slow_summary[loc]["avg_max_delay_h"]])


# =========================
# Main driver
# =========================

def main() -> None:
    params = Params(
        E_B=24.0,
        E_min=0.2 * 24.0,
        E_anx=0.5 * 24.0,
        P_f_max=60.0,
        P_s_home=7.2,
        P_s_work_other=11.0,
        alpha_tra=0.5,
        alpha_mon=0.1,
        beta_u=40.0,
        cp_nor=0.4,
        theta=1.0,
        dt_bin_min=15,
        LC_pri=0.5,
        LC_pub=0.5,
        max_iters=50,
        eps=1e-4,
        min_iters_before_stop=15,
        traffic_sensitivity=0.22,
        num_demo_evs=1600,
    )
    params.fixed_fast_load_limit_kw = 60

    outdir = "outputs"
    ensure_dir(outdir)

    nodes_csv = "nodes.csv"
    links_csv = "links.csv"

    try:
        G, coords = load_network_from_csv(nodes_csv, links_csv)
        print("Loaded network from CSV.")
    except FileNotFoundError:
        G, coords = build_demo_network()
        print("Using built-in demo network.")

    evs = generate_demo_population(params)
    print(f"EV population: {len(evs)}")

    model = EVChargingSDUEModel(G, coords, evs, params)

    case1 = model.run_case(case_name="case1_no_price_incentives", price_incentives=False)
    export_fast_events(case1.fast_events, os.path.join(outdir, "case1_fast_events.csv"))
    export_slow_events(case1.slow_events, os.path.join(outdir, "case1_slow_events.csv"))
    export_summary(case1, os.path.join(outdir, "case1_summary.csv"))
    plot_fast_load_profiles(case1, outdir)
    plot_avg_utility(case1, outdir)
    plot_slow_summary(case1, outdir)

    case2 = model.run_case(
        case_name="case2_price_incentives",
        price_incentives=True,
        baseline_fast_load_for_limit=case1.final_fast_load_kw,
    )
    export_fast_events(case2.fast_events, os.path.join(outdir, "case2_fast_events.csv"))
    export_slow_events(case2.slow_events, os.path.join(outdir, "case2_slow_events.csv"))
    export_summary(case2, os.path.join(outdir, "case2_summary.csv"))
    plot_fast_load_profiles(case2, outdir)
    plot_prices(case2, outdir)
    plot_avg_utility(case2, outdir)
    plot_slow_summary(case2, outdir)

    plot_case_comparison(case1, case2, outdir, params.dt_bin_min)

    print("Done. Outputs saved in:", outdir)
    print("Generated files:")
    for fn in sorted(os.listdir(outdir)):
        print(" -", fn)


if __name__ == "__main__":
    main()