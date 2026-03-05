import heapq
import time


def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def rebuild_path(prev, node):
    path = []
    while node in prev:
        path.append(node)
        node = prev[node]
    path.append(node)
    path.reverse()
    return path


def graph_search(env, search_type="A*", weight=1.0):
    start = env.start
    goal  = env.goal
    t0    = time.time()

    heap = []
    heapq.heappush(heap, (0, 0, start))

    prev   = {}
    g      = {start: 0}
    closed = set()
    nodes  = 0
    peak   = 1

    while heap:
        peak = max(peak, len(heap))
        f, cost, cur = heapq.heappop(heap)

        if cur in closed:
            continue
        closed.add(cur)
        nodes += 1

        if cur == goal:
            return {
                "path": rebuild_path(prev, cur),
                "cost": g[cur],
                "nodes_developed": nodes,
                "time": time.time() - t0,
                "max_open_size": peak,
            }

        for nb in env.neighbors(cur):
            if nb in closed:
                continue
            ng = g[cur] + env.move_cost(cur, nb)
            if nb not in g or ng < g[nb]:
                prev[nb] = cur
                g[nb]    = ng
                h = heuristic(nb, goal)
                if search_type == "A*":
                    f_val = ng + weight * h
                elif search_type == "UCS":
                    f_val = ng
                else:  # Greedy
                    f_val = h
                heapq.heappush(heap, (f_val, ng, nb))

    return {"path": None, "cost": float("inf"),
            "nodes_developed": nodes, "time": time.time()-t0,
            "max_open_size": peak}
