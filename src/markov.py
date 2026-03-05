import numpy as np


def make_policy(path, env):
    pol  = {}
    goal = env.goal

    # follow the A* path step by step
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        pol[a] = (b[0]-a[0], b[1]-a[1])

    pol[goal] = (0, 0)  # stay at goal

    # for cells not on the path, move greedily toward goal
    for x in range(env.width):
        for y in range(env.height):
            s = (x, y)
            if env.passable(s) and s not in pol:
                best_h, best_a = float("inf"), (0,0)
                for nb in env.neighbors(s):
                    d = abs(nb[0]-goal[0]) + abs(nb[1]-goal[1])
                    if d < best_h:
                        best_h = d
                        best_a = (nb[0]-s[0], nb[1]-s[1])
                pol[s] = best_a
    return pol


def build_transition_matrix(env, policy, eps=0.1):
    states = [(x, y)
              for y in range(env.height)
              for x in range(env.width)
              if env.passable((x, y))]

    n   = len(states)
    sid = {s: i for i, s in enumerate(states)}
    P   = np.zeros((n, n))

    for s in states:
        i = sid[s]
        if s == env.goal:
            P[i, i] = 1.0
            continue

        dx, dy = policy.get(s, (0,0))
        if (dx, dy) == (0, 0):
            P[i, i] = 1.0
            continue

        fwd   = (s[0]+dx,  s[1]+dy)
        left  = (s[0]-dy,  s[1]+dx)
        right = (s[0]+dy,  s[1]-dx)

        for dest, prob in [(fwd, 1-eps), (left, eps/2), (right, eps/2)]:
            if env.in_bounds(dest) and env.passable(dest):
                P[i, sid[dest]] += prob
            else:
                P[i, i] += prob  # bounce back if blocked

    assert np.allclose(P.sum(axis=1), 1.0)
    return P, states, sid


def comm_classes(P, states):
    n       = len(states)
    seen    = [False]*n
    order   = []

    def dfs1(v):
        stk = [(v, iter(range(n)))]
        seen[v] = True
        while stk:
            u, it = stk[-1]
            try:
                w = next(it)
                if P[u,w] > 0 and not seen[w]:
                    seen[w] = True
                    stk.append((w, iter(range(n))))
            except StopIteration:
                order.append(u)
                stk.pop()

    for v in range(n):
        if not seen[v]:
            dfs1(v)

    seen2 = [False]*n
    result = []

    def dfs2(v):
        stk, comp = [v], []
        seen2[v] = True
        while stk:
            u = stk.pop()
            comp.append(u)
            for w in range(n):
                if P[w,u] > 0 and not seen2[w]:
                    seen2[w] = True
                    stk.append(w)
        return comp

    for v in reversed(order):
        if not seen2[v]:
            ci = dfs2(v)
            is_rec = all(P[i,j]==0 or j in ci
                         for i in ci for j in range(n))
            result.append({
                "states":  [states[i] for i in ci],
                "indices": ci,
                "type":    "recurrent" if is_rec else "transient"
            })
    return result


def expected_steps(P, states, goal):
    gi  = states.index(goal)
    tr  = [i for i in range(len(states)) if i != gi]
    Q   = P[np.ix_(tr, tr)]
    N   = np.linalg.inv(np.eye(len(tr)) - Q)
    t   = N @ np.ones(len(tr))
    return {states[tr[i]]: t[i] for i in range(len(tr))}


def absorb_prob(P, states, goal):
    gi  = states.index(goal)
    tr  = [i for i in range(len(states)) if i != gi]
    Q   = P[np.ix_(tr, tr)]
    R   = P[np.ix_(tr, [gi])]
    N   = np.linalg.inv(np.eye(len(tr)) - Q)
    B   = N @ R
    return {states[tr[i]]: float(B[i,0]) for i in range(len(tr))}


def monte_carlo(P, states, start, goal, N=10000, max_steps=300):
    si  = states.index(start)
    gi  = states.index(goal)
    rng = np.random.default_rng(42)
    times, hits = [], 0

    for _ in range(N):
        cur, k = si, 0
        while cur != gi and k < max_steps:
            cur = rng.choice(len(states), p=P[cur])
            k  += 1
        if cur == gi:
            hits += 1
            times.append(k)

    avg = float(np.mean(times)) if times else float("inf")
    return hits/N, avg, times
