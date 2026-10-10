import time

import link
import topology
from optimizer import optimize

PERIOD = 1.0


def active_nodes(nodes, states):
    return {n for n in nodes if states.get(n, 1)}


def percent(value):
    return "n/a" if value is None else f"{value:+.0%}"


def report(result):
    on = [n for n, s in result["states"].items() if s]
    print(
        f"ON={on} "
        f"P_with={result['power_with']:.2f} "
        f"P_without={result['power_without']:.2f} "
        f"P_diode={result['power_diode']:.2f} "
        f"P_ideal={result['power_ideal']:.2f} "
        f"gain={percent(result['gain'])} "
        f"gain_vs_diode={percent(result['gain_vs_diode'])} "
        f"eff={result['efficiency']:.0%} "
        f"mse={result['mse']:.3f}",
        flush=True,
    )


def check_topology(mesh, last_issues):
    graph = topology.build_graph(mesh["nodes"], mesh["edges"])
    issues = topology.validate(graph)
    if issues != last_issues:
        for issue in issues:
            print(f"topology: {issue}", flush=True)
        if not issues:
            print(f"topology: ok, series={topology.series_order(graph)}", flush=True)
    return graph, issues


def step(states, issues):
    mesh = link.get_mesh()
    graph, issues = check_topology(mesh, issues)
    result = optimize(
        mesh["nodes"], active_nodes(mesh["nodes"], states), mesh["current"]
    )
    if result["states"] != states:
        link.set_nodes(result["states"])
    report(result)
    return result["states"], issues


def main():
    states, issues = {}, None
    while True:
        try:
            states, issues = step(states, issues)
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            print(f"error: {e}", flush=True)
        time.sleep(PERIOD)


if __name__ == "__main__":
    main()
