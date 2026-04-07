#Graph generation algorithm
#Parker Tingle 12343853

#Networkx python package needed.

import networkx as nx, random

def generate_grid_graph(rows: int, cols: int, directed: bool=True):
    """2D grid with random weights."""
    G = nx.grid_2d_graph(rows, cols)
    if directed:
        DG = nx.DiGraph()
        for u,v in G.edges():
            w = random.uniform(1,10)
            DG.add_edge(u,v, weight=w)
            DG.add_edge(v,u, weight=w)
        return DG
    else:
        for u,v in G.edges():
            w = random.uniform(1,10)
            G[u][v]['weight'] = w
            G[v][u]['weight'] = w
        return G

def generate_random_graph(n: int, p: float, directed: bool=True):
    """Random graph with weighted edges."""
    G = nx.fast_gnp_random_graph(n,p, directed=False)
    if directed:
        DG = nx.DiGraph()
        for u,v in G.edges():
            w = random.uniform(1,10)
            DG.add_edge(u,v, weight=w); DG.add_edge(v,u, weight=w)
        return DG
    else:
        for u,v in G.edges():
            w = random.uniform(1,10)
            G[u][v]['weight'] = w; G[v][u]['weight'] = w
        return G

def generate_road_graph(n: int, radius: float, directed: bool=True):
    """Random geometric graph as a road-like network."""
    G = nx.random_geometric_graph(n, radius)
    pos = nx.get_node_attributes(G, 'pos')
    if directed:
        DG = nx.DiGraph()
        for u,v in G.edges():
            dx,dy = pos[u][0]-pos[v][0], pos[u][1]-pos[v][1]
            w = (dx*dx+dy*dy)**0.5
            DG.add_edge(u,v, weight=w); DG.add_edge(v,u, weight=w)
        return DG
    else:
        for u,v in G.edges():
            dx,dy = pos[u][0]-pos[v][0], pos[u][1]-pos[v][1]
            w = (dx*dx+dy*dy)**0.5
            G[u][v]['weight'] = w; G[v][u]['weight'] = w
        return G
