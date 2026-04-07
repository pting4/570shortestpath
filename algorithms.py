#Contains algorithms for each of the 4 test methods
#Dijkstra's, dynamic Dijkstra's, LPA*, D Lite*
#Parker Tingle 12343853

import heapq
from typing import Tuple, List

class Dijkstra:
    """Static Dijkstra's algorithm."""
    def __init__(self, graph, weight='weight'):
        self.graph = graph; self.weight = weight; self.nodes_expanded = 0

    def shortest_path(self, source, target) -> Tuple[float, List]:
        self.nodes_expanded = 0
        dist = {n: float('inf') for n in self.graph.nodes()}
        prev = {n: None for n in self.graph.nodes()}
        dist[source] = 0
        pq = [(0, source)]
        visited = set()
        while pq:
            d,u = heapq.heappop(pq)
            if u == target: break
            if u in visited: continue
            visited.add(u); self.nodes_expanded += 1
            for v in self.graph.successors(u):
                w = self.graph[u][v][self.weight]
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd; prev[v] = u
                    heapq.heappush(pq, (nd, v))
        #reconstruct path
        path = []
        if dist[target] < float('inf'):
            cur = target
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
        return dist[target], path

class DynamicDijkstra:
    """Dynamic SSSP: updates Dijkstra's tree when an edge weight changes."""
    def __init__(self, graph, source, weight='weight'):
        self.graph = graph; self.source = source; self.weight = weight
        self.dist = {n: float('inf') for n in graph.nodes()}
        self.prev = {n: None         for n in graph.nodes()}
        self.nodes_expanded = 0
        self.dist[source] = 0
        #standard Dijkstra
        self._compute_initial()

    def _compute_initial(self):
        pq = [(0, self.source)]
        visited = set()
        while pq:
            d,u = heapq.heappop(pq)
            if u in visited: continue
            visited.add(u); self.nodes_expanded += 1
            for v in self.graph.successors(u):
                w = self.graph[u][v][self.weight]
                nd = d + w
                if nd < self.dist[v]:
                    self.dist[v] = nd; self.prev[v] = u
                    heapq.heappush(pq, (nd, v))

    def update_edge(self, u, v, new_weight):
        """Update weight of edge (u,v) and adjust the SPT."""
        self.nodes_expanded = 0
        old_w = self.graph[u][v][self.weight]
        self.graph[u][v][self.weight] = new_weight
        if new_weight < old_w:
            #edge cost decreased: try to relax path to v
            if self.dist[u] + new_weight < self.dist[v]:
                self.dist[v] = self.dist[u] + new_weight
                self.prev[v] = u
                #propagate improvements from v
                pq = [(self.dist[v], v)]
                visited = set()
                while pq:
                    d,x = heapq.heappop(pq)
                    if x in visited: continue
                    visited.add(x); self.nodes_expanded += 1
                    for y in self.graph.successors(x):
                        w = self.graph[x][y][self.weight]
                        nd = d + w
                        if nd < self.dist[y]:
                            self.dist[y] = nd; self.prev[y] = x
                            heapq.heappush(pq, (nd, y))
        elif new_weight > old_w:
            #edge cost increased
            if self.prev[v] == u:
                self.dist = {n: float('inf') for n in self.graph.nodes()}
                self.prev = {n: None         for n in self.graph.nodes()}
                self.dist[self.source] = 0
                self._compute_initial()

    def shortest_path(self, target) -> Tuple[float, List]:
        """Return (distance, path) from source to target."""
        d = self.dist.get(target, float('inf'))
        if d == float('inf'):
            return float('inf'), []
        path = []
        cur = target
        while cur is not None:
            path.append(cur)
            cur = self.prev[cur]
        path.reverse()
        return d, path

class LPAStar:
    """
    Lifelong Planning A* (incremental A*) from start to goal.
    Maintains two values per node (g and rhs) and a priority queue keyed by (min(g,rhs)+h, min(g,rhs)).
    Adapted from Koenig & Likhachev (2001).
    """
    def __init__(self, graph, start, goal, heuristic=None, weight='weight'):
        self.graph = graph; self.start = start; self.goal = goal; self.weight = weight
        self.heuristic = heuristic or (lambda u,v: 0)
        self.g   = {n: float('inf') for n in graph.nodes()}
        self.rhs = {n: float('inf') for n in graph.nodes()}
        self.g[start] = float('inf'); self.rhs[start] = 0
        self.queue = []  # heap of (k1, k2, node)
        self.entry_finder = {}
        self.nodes_expanded = 0
        self._push(start)

    def _calculate_key(self, u):
        return (min(self.g[u], self.rhs[u]) + self.heuristic(u, self.goal),
                min(self.g[u], self.rhs[u]))

    def _push(self, u):
        key = self._calculate_key(u)
        self.entry_finder[u] = key
        heapq.heappush(self.queue, (key[0], key[1], u))

    def compute_shortest_path(self):
        #continue until goal is locally consistent
        while self.queue and ((self.queue[0][0],self.queue[0][1]) < self._calculate_key(self.goal)
                              or self.rhs[self.goal] != self.g[self.goal]):
            k_old1, k_old2, u = heapq.heappop(self.queue)
            if u not in self.entry_finder: 
                continue
            del self.entry_finder[u]
            key_u = self._calculate_key(u)
            if (k_old1,k_old2) < key_u:
                self._push(u)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                self.nodes_expanded += 1
                for v in self.graph.successors(u):
                    self.update_vertex(v)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for v in self.graph.successors(u):
                    self.update_vertex(v)

    def update_vertex(self, u):
        if u != self.start:
            preds = list(self.graph.predecessors(u))
            self.rhs[u] = min(self.g[p] + self.graph[p][u][self.weight] for p in preds)
        if u in self.entry_finder:
            del self.entry_finder[u]
        if self.g[u] != self.rhs[u]:
            self._push(u)

    def shortest_path(self) -> Tuple[float, List]:
        """Return (distance, path) from start to goal."""
        self.compute_shortest_path()
        if self.g[self.goal] == float('inf'):
            return float('inf'), []
        path = []
        curr = self.goal
        while curr != self.start:
            path.append(curr)
            #select predecessor on optimal path
            curr = min(self.graph.predecessors(curr),
                       key=lambda p: self.g[p] + self.graph[p][curr][self.weight])
        path.append(self.start)
        path.reverse()
        return self.g[self.goal], path

class DStarLite:
    """
    D* Lite (Koenig & Likhachev, 2002) - plans backwards from goal to moving start.
    fix the start (agent) and goal, but still use its incremental update strategy.
    """
    def __init__(self, graph, start, goal, heuristic=None, weight='weight'):
        self.graph = graph; self.start = start; self.goal = goal; self.weight = weight
        self.heuristic = heuristic or (lambda u,v: 0)
        self.g   = {n: float('inf') for n in graph.nodes()}
        self.rhs = {n: float('inf') for n in graph.nodes()}
        self.rhs[goal] = 0  # goal is root (rhs=0)
        self.km = 0
        self.queue = []
        self.entry_finder = {}
        self.nodes_expanded = 0
        self._push(goal)

    def _calculate_key(self, u):
        return (min(self.g[u], self.rhs[u]) + self.heuristic(self.start, u) + self.km,
                min(self.g[u], self.rhs[u]))

    def _push(self, u):
        key = self._calculate_key(u)
        self.entry_finder[u] = key
        heapq.heappush(self.queue, (key[0], key[1], u))

    def compute_shortest_path(self):
        while self.queue and ((self.queue[0][0],self.queue[0][1]) < self._calculate_key(self.start)
                              or self.rhs[self.start] != self.g[self.start]):
            k_old1, k_old2, u = heapq.heappop(self.queue)
            if u not in self.entry_finder or self.entry_finder[u] != (k_old1,k_old2):
                continue
            del self.entry_finder[u]
            key_u = self._calculate_key(u)
            if (k_old1, k_old2) < key_u:
                self._push(u)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                self.nodes_expanded += 1
                for p in self.graph.predecessors(u):
                    self.update_vertex(p)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for p in self.graph.predecessors(u):
                    self.update_vertex(p)

    def update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min(self.graph[u][v][self.weight] + self.g[v] 
                              for v in self.graph.successors(u))
        if u in self.entry_finder:
            del self.entry_finder[u]
        if self.g[u] != self.rhs[u]:
            self._push(u)

    def update_edge(self, u, v, new_weight):
        """Handle an edge weight change and replan."""
        old_w = self.graph[u][v][self.weight]
        self.graph[u][v][self.weight] = new_weight
        self.update_vertex(u)
        self.compute_shortest_path()

    def shortest_path(self) -> Tuple[float, List]:
        """Return (distance, path) from start to goal."""
        self.compute_shortest_path()
        if self.g[self.start] == float('inf'):
            return float('inf'), []
        path = [self.start]
        curr = self.start
        while curr != self.goal:
            #choose successor on optimal path
            curr = min(self.graph.successors(curr),
                       key=lambda v: self.graph[curr][v][self.weight] + self.g[v])
            path.append(curr)
        return self.g[self.start], path