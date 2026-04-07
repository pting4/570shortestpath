#Runs the actual experiment
#Writes to a png and csv with all results
#What graph is used and what update method is used can be changed

#Requires matplotlib, as well as the graphs.py and algorithms.py files

import time, csv, random
import matplotlib.pyplot as plt
from graphs import generate_random_graph, generate_grid_graph, generate_road_graph
from algorithms import Dijkstra, DynamicDijkstra, LPAStar, DStarLite


def measure_algorithms(G, source, target, update_func):
    dijkstra = Dijkstra(G)
    dyn_dij = DynamicDijkstra(G, source)
    lpa = LPAStar(G, source, target, heuristic=lambda u, v: 0)
    dstar = DStarLite(G, source, target, heuristic=lambda u, v: 0)

    dijkstra.shortest_path(source, target)
    dyn_dij.shortest_path(target)
    lpa.shortest_path()
    dstar.shortest_path()

    start = time.perf_counter()
    update_func(G)
    dist_static, _ = dijkstra.shortest_path(source, target)
    t_static = time.perf_counter() - start

    start = time.perf_counter()
    update_func(dyn_dij.graph)
    dist_dyn, _ = dyn_dij.shortest_path(target)
    t_dyn = time.perf_counter() - start

    start = time.perf_counter()
    update_func(lpa.graph)
    dist_lpa, _ = lpa.shortest_path()
    t_lpa = time.perf_counter() - start

    start = time.perf_counter()
    update_func(dstar.graph)
    dist_dstar, _ = dstar.shortest_path()
    t_dstar = time.perf_counter() - start

    return {
        "Static": (t_static, dijkstra.nodes_expanded, dist_static),
        "Dynamic": (t_dyn, dyn_dij.nodes_expanded, dist_dyn),
        "LPA*": (t_lpa, lpa.nodes_expanded, dist_lpa),
        "D* Lite": (t_dstar, dstar.nodes_expanded, dist_dstar)
    }


def single_edge_increase(G):
    u, v = list(G.edges())[0]
    G[u][v]['weight'] *= 2


def single_edge_decrease(G):
    u, v = list(G.edges())[0]
    G[u][v]['weight'] *= 0.5


def localized_cluster_update(G, cluster_size=5):
    edges = list(G.edges())[:cluster_size]

    for u, v in edges:
        G[u][v]['weight'] *= random.uniform(0.5, 2.0)


def repeated_streaming_updates(G, num_updates=10):
    edges = list(G.edges())

    for _ in range(num_updates):
        u, v = random.choice(edges)
        G[u][v]['weight'] *= random.uniform(0.5, 2.0)


def run_experiment():
    #create a test graph and define source/target
    #Below in comments are two other cases for getting graphs

    G = generate_random_graph(50, 0.3, directed=True)
    #G = generate_grid_graph(50, 20, directed=True)
    #G = generate_road_graph(50, 30, directed=True)

    nodes = list(G.nodes())
    source, target = nodes[0], nodes[-1]

    experiments = {
        "Single Edge Increase": single_edge_increase,
        "Single Edge Decrease": single_edge_decrease,
        "Localized Cluster Update": localized_cluster_update,
        "Repeated Streaming Update": repeated_streaming_updates
    }

    all_results = []

    for exp_name, update_func in experiments.items():
        print(f"\nRunning: {exp_name}")

        results = measure_algorithms(G.copy(), source, target, update_func)

        labels = list(results.keys())
        times = [results[label][0] for label in labels]

        #WRITE TO PLT
        plt.figure(figsize=(6, 4))
        plt.bar(labels, times)
        plt.ylabel("Time (seconds)")
        plt.title(exp_name)
        plt.savefig(f"{exp_name.replace(' ', '_').lower()}.png")
        plt.show()

        for algo, vals in results.items():
            all_results.append([exp_name, algo, vals[0], vals[1], vals[2]])

    #WRITE TO CSV
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Experiment",
            "Algorithm",
            "Time",
            "NodesExpanded",
            "PathLength"
        ])
        writer.writerows(all_results)


if __name__ == "__main__":
    run_experiment()
