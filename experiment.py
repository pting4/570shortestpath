#Runs the actual experiment
#Writes to a png and csv with all restuls
#What graph is used and what update method is used can be changed

#Requires matplotlib, as well as the graphs.py and algorithms.py files

import time, csv
import matplotlib.pyplot as plt
from graphs import generate_random_graph
from algorithms import Dijkstra, DynamicDijkstra, LPAStar, DStarLite

def run_experiment():
    #create a test graph and define source/target
    G = generate_random_graph(50, 0.1, directed=True)
    nodes = list(G.nodes())
    source, target = nodes[0], nodes[-1]

    #initialize algorithms
    dijkstra    = Dijkstra(G)
    dyn_dij     = DynamicDijkstra(G, source)
    lpa         = LPAStar(G, source, target, heuristic=lambda u,v: 0)
    dstar       = DStarLite(G, source, target, heuristic=lambda u,v: 0)

    #baseline shortest paths static
    start = time.perf_counter()
    dist0, path0 = dijkstra.shortest_path(source, target)
    t0 = time.perf_counter() - start

    start = time.perf_counter()
    dist0b, path0b = dyn_dij.shortest_path(target)
    t0b = time.perf_counter() - start

    start = time.perf_counter()
    dist0c, path0c = lpa.shortest_path()
    t0c = time.perf_counter() - start

    start = time.perf_counter()
    dist0d, path0d = dstar.shortest_path()
    t0d = time.perf_counter() - start

    print("Initial path lengths:", dist0, dist0b, dist0c, dist0d)

    #Single-edge increase: increase one edge weight
    u,v = list(G.edges())[0]
    old_w = G[u][v]['weight']
    new_w = old_w * 2

    #Static (recompute)
    start = time.perf_counter()
    dist_static, path_static = dijkstra.shortest_path(source, target)
    t_static = time.perf_counter() - start

    #Dynamic Dijkstra update
    start = time.perf_counter()
    dyn_dij.update_edge(u, v, new_w)
    dist_dyn, path_dyn = dyn_dij.shortest_path(target)
    t_dyn = time.perf_counter() - start

    #LPA* update (mark node for update then replan)
    start = time.perf_counter()
    lpa.update_vertex(v)
    dist_lpa, path_lpa = lpa.shortest_path()
    t_lpa = time.perf_counter() - start

    #D* Lite update
    start = time.perf_counter()
    dstar.update_edge(u, v, new_w)
    dist_dstar, path_dstar = dstar.shortest_path()
    t_dstar = time.perf_counter() - start

    print("After weight increase (new distances):",
          dist_dyn, dist_lpa, dist_dstar)
    print(f"Time (sec): static={t_static:.4f}, dyn={t_dyn:.4f}, lpa={t_lpa:.4f}, dstar={t_dstar:.4f}")
    print("Nodes expanded (in update):",
          dijkstra.nodes_expanded, dyn_dij.nodes_expanded,
          lpa.nodes_expanded, dstar.nodes_expanded)

    # WRITE TO PLT
    labels = ['Static', 'Dynamic', 'LPA*', 'D* Lite']
    values = [dijkstra.nodes_expanded, dyn_dij.nodes_expanded, lpa.nodes_expanded, dstar.nodes_expanded]
    plt.figure(figsize=(6,4))
    plt.bar(labels, values, color=['gray','blue','green','orange'])
    plt.ylabel("Nodes expanded on update")
    plt.title("Comparison after single-edge weight increase")
    plt.savefig("update_comparison.png")
    plt.show()

    # WRITE TO CSV
    with open('results.csv','w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm','Time','NodesExpanded','PathLength'])
        writer.writerow(['Static Dijkstra',   t_static, dijkstra.nodes_expanded, dist_static])
        writer.writerow(['Dynamic Dijkstra',  t_dyn,    dyn_dij.nodes_expanded,  dist_dyn])
        writer.writerow(['LPA*',              t_lpa,    lpa.nodes_expanded,      dist_lpa])
        writer.writerow(['D* Lite',          t_dstar,  dstar.nodes_expanded,    dist_dstar])

if __name__ == "__main__":
    run_experiment()
