
from __future__ import annotations
import time, json, os
from typing import Dict, Any, List
import networkx as nx

GRAPH_PATH = os.path.join(os.path.dirname(__file__), "..", "storage", "graph.json")
os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)

def _now() -> float:
    return time.time()

def _ensure_graph():
    if os.path.exists(GRAPH_PATH):
        try:
            with open(GRAPH_PATH, "r") as f:
                data = json.load(f)
            return nx.node_link_graph(data)
        except Exception:
            pass
    G = nx.MultiDiGraph()
    G.graph["created_at"] = _now()
    return G

def save_graph(G: nx.MultiDiGraph):
    data = nx.node_link_data(G)
    with open(GRAPH_PATH, "w") as f:
        json.dump(data, f, indent=2)

def add_user_if_missing(G: nx.MultiDiGraph, user_id: str):
    if not G.has_node(("user", user_id)):
        G.add_node(("user", user_id), type="user", created_at=_now())

def upsert_thread(G: nx.MultiDiGraph, user_id: str, theme: str) -> str:
    node_id = ("thread", f"{user_id}:{theme}")
    if not G.has_node(node_id):
        G.add_node(node_id, type="thread", theme=theme, user=user_id, created_at=_now())
        G.add_edge(("user", user_id), node_id, type="has_thread", ts=_now())
    return node_id[1]

def add_message(G: nx.MultiDiGraph, user_id: str, thread_id: str, role: str, text: str, anchors: List[Dict[str, Any]]):
    mid = ("msg", f"{time.time_ns()}")
    G.add_node(mid, type="message", role=role, text=text, ts=_now())
    G.add_edge(("user", user_id), mid, type="authored")
    G.add_edge(("thread", thread_id), mid, type="in_thread")
    for a in anchors:
        aid = ("anchor", f"{a['key']}:{int(_now())}")
        if not G.has_node(aid):
            G.add_node(aid, type="anchor", key=a["key"], value=a.get("value"))
        G.add_edge(mid, aid, type="has_anchor", weight=a.get("weight", 1.0))

def add_symptom_solution(G: nx.MultiDiGraph, symptom: str, solution: str, user_id: str, thread_id: str):
    sid = ("symptom", symptom.lower().strip())
    if not G.has_node(sid):
        G.add_node(sid, type="symptom", text=symptom)
    rid = ("solution", solution.lower().strip())
    if not G.has_node(rid):
        G.add_node(rid, type="solution", text=solution)
    G.add_edge(sid, rid, type="treated_by", ts=_now())
    G.add_edge(("thread", thread_id), sid, type="mentions", ts=_now())
    G.add_edge(("thread", thread_id), rid, type="recommends", ts=_now())

def get_graph() -> nx.MultiDiGraph:
    return _ensure_graph()
