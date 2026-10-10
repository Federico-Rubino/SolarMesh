import re

import networkx as nx


def build_graph(nodes, edges):
    graph = nx.Graph()
    for node, voltage in nodes.items():
        graph.add_node(node, voltage=voltage, known=True)
    for a, b, wired in edges:
        for n in (a, b):
            if n not in graph:
                graph.add_node(n, voltage=None, known=False)
        graph.add_edge(a, b, wired=wired)
    return graph


def wiring(graph):
    wired = [(a, b) for a, b, w in graph.edges(data="wired") if w]
    sub = nx.Graph()
    sub.add_nodes_from(graph.nodes)
    sub.add_edges_from(wired)
    return sub


def chain_order(part):
    if not nx.is_tree(part) or any(d > 2 for _, d in part.degree):
        return None
    if part.number_of_nodes() == 1:
        return list(part.nodes)
    start = min(n for n, d in part.degree if d == 1)
    return list(nx.dfs_preorder_nodes(part, start))


def string_parts(graph):
    sub = wiring(graph)
    parts = [sub.subgraph(c) for c in nx.connected_components(sub)]
    if graph.number_of_nodes() > 1:
        parts = [p for p in parts if p.number_of_nodes() > 1]
    return sorted(parts, key=lambda p: natural(min(p.nodes, key=natural)))


def natural(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(name))]


def strings(graph):
    return [order for order in (chain_order(p) for p in string_parts(graph)) if order]


def neighbours(graph, node):
    return sorted(graph.neighbors(node))


def short(nodes, limit=8):
    nodes = sorted(nodes)
    extra = f" … (+{len(nodes) - limit})" if len(nodes) > limit else ""
    return ", ".join(nodes[:limit]) + extra


def validate(graph):
    issues = []
    unknown = [n for n, k in graph.nodes(data="known") if not k]
    if unknown:
        issues.append(f"nodes in edges but not reported: {short(unknown)}")

    isolated = list(nx.isolates(graph))
    if isolated:
        issues.append(f"modules with no edges: {short(isolated)}")

    sub = wiring(graph)
    unwired = [n for n, d in sub.degree if d == 0]
    if unwired and graph.number_of_nodes() > 1:
        issues.append(f"modules not electrically wired: {short(unwired)}")

    for part in string_parts(graph):
        if chain_order(part) is not None:
            continue
        branches = [n for n, d in part.degree if d > 2]
        if nx.cycle_basis(part):
            issues.append(f"string {min(part.nodes)} has a loop")
        if branches:
            issues.append(f"string {min(part.nodes)} branches at: {short(branches)}")
    return issues


def to_dict(graph):
    return {
        "nodes": [{"id": n, "known": k} for n, k in graph.nodes(data="known")],
        "edges": [{"source": a, "target": b, "wired": w} for a, b, w in graph.edges(data="wired")],
        "strings": strings(graph),
    }
