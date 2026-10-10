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


def series_order(graph):
    sub = wiring(graph)
    if sub.number_of_nodes() == 0:
        return []
    if not nx.is_connected(sub) or not nx.is_tree(sub):
        return None
    if any(d > 2 for _, d in sub.degree):
        return None
    if sub.number_of_nodes() == 1:
        return list(sub.nodes)
    start = min(n for n, d in sub.degree if d == 1)
    return list(nx.dfs_preorder_nodes(sub, start))


def neighbours(graph, node):
    return sorted(graph.neighbors(node))


def validate(graph):
    issues = []
    unknown = sorted(n for n, k in graph.nodes(data="known") if not k)
    if unknown:
        issues.append(f"nodes in edges but not reported: {unknown}")

    isolated = sorted(nx.isolates(graph))
    if isolated:
        issues.append(f"modules with no edges: {isolated}")

    sub = wiring(graph)
    unwired = sorted(n for n, d in sub.degree if d == 0)
    if unwired and graph.number_of_nodes() > 1:
        issues.append(f"modules not electrically wired: {unwired}")

    if sub.number_of_nodes() > 0 and not nx.is_connected(sub):
        parts = [sorted(c) for c in nx.connected_components(sub)]
        issues.append(f"wiring split into {len(parts)} parts: {parts}")

    if series_order(graph) is None:
        cycles = nx.cycle_basis(sub)
        branches = sorted(n for n, d in sub.degree if d > 2)
        if cycles:
            issues.append(f"wiring has loops: {cycles}")
        if branches:
            issues.append(f"wiring branches at: {branches}")
    return issues


def to_dict(graph):
    return {
        "nodes": [{"id": n, **attrs} for n, attrs in graph.nodes(data=True)],
        "edges": [{"source": a, "target": b, "wired": w} for a, b, w in graph.edges(data="wired")],
        "series": series_order(graph),
    }
