import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings; warnings.filterwarnings("ignore")

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from grid   import make_easy_grid, make_medium_grid, make_hard_grid
from astar  import graph_search
from markov import (make_policy, build_transition_matrix, comm_classes, expected_steps, absorb_prob, monte_carlo)

ROOT = os.path.join(SRC, "..", "results")
DIRS = {k: os.path.join(ROOT, v) for k, v in {
    "E1": "E1_algorithm_comparison",
    "E2": "E2_epsilon_impact",
    "E3": "E3_heuristic_comparison",
    "E4": "E4_weighted_astar",
    "P3": "Phase3_markov_chain",
    "P4": "Phase4_absorption",
    "P5": "Phase5_monte_carlo",
}.items()}

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)


# ── light colour theme ────────────────────────────────────────
C = {
    "bg":     "#F7F3EE",   # warm off-white background
    "panel":  "#FFFFFF",   # pure white axes
    "border": "#CCBFA8",   # warm grey borders
    "grid":   "#E8DDD0",   # very light grid lines
    "text":   "#2C2C2C",   # near-black text
    "c1":     "#C0392B",   # brick red
    "c2":     "#2E86AB",   # teal blue
    "c3":     "#E8A838",   # amber
    "c4":     "#5B8C5A",   # forest green
    "c5":     "#7B4F9E",   # muted purple
    "wall":   "#4A4A5A",   # dark slate for walls
    "free":   "#F0EAE0",   # cream for open cells
}

ALGO_COLORS = [C["c1"], C["c3"], C["c2"]]   # UCS, Greedy, A*
EPS_COLORS  = [C["c4"], C["c2"], C["c3"], C["c1"]]


def style_ax(ax):
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["text"], labelsize=8.5)
    ax.xaxis.label.set_color(C["text"])
    ax.yaxis.label.set_color(C["text"])
    ax.title.set_color(C["text"])
    for sp in ax.spines.values():
        sp.set_color(C["border"])
    ax.grid(color=C["grid"], linewidth=0.8, linestyle="-")


def savefig(fig, key, name):
    p = os.path.join(DIRS[key], name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  saved:", os.path.relpath(p, ROOT))


def savejson(data, key, name):
    p = os.path.join(DIRS[key], name)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    print("  saved:", os.path.relpath(p, ROOT))


def draw_grid(ax, env, path, title, path2=None):
    W, H = env.width, env.height
    img = np.ones((H, W, 3))

    def hex2rgb(h):
        return tuple(int(h.lstrip("#")[i:i+2], 16)/255 for i in (0,2,4))

    wall = hex2rgb(C["wall"])
    free = hex2rgb(C["free"])

    for y in range(H):
        for x in range(W):
            img[y,x] = wall if (x,y) in env.obstacles else free

    ax.imshow(img, origin="upper")

    def pline(p, col, lbl, ls="-"):
        if p:
            ax.plot([s[0] for s in p], [s[1] for s in p],
                    color=col, lw=2.2, ls=ls, marker=".", ms=3.5,
                    label=lbl, zorder=3)

    pline(path, C["c2"], "A*")
    if path2:
        pline(path2, C["c1"], "UCS", ls="--")

    ax.plot(env.start[0], env.start[1], "o", color=C["c4"], ms=8, zorder=5, label="Start")
    ax.plot(env.goal[0],  env.goal[1],  "*", color=C["c3"], ms=11, zorder=5, label="Goal")
    ax.set_title(title, fontsize=8.5, color=C["text"], fontweight="bold", pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(C["border"])
    ax.legend(fontsize=6.5, loc="upper right",
              facecolor=C["panel"], edgecolor=C["border"])


# ═══════════════════════════════════════════════════════════
# E1
# ═══════════════════════════════════════════════════════════

def experiment_E1():
    print("\n=== E1 — UCS / Greedy / A* ===")
    envs  = [("Easy 10x10",   make_easy_grid()),
             ("Medium 15x15", make_medium_grid()),
             ("Hard 20x20",   make_hard_grid())]
    modes = ["UCS", "Greedy", "A*"]
    data  = []

    # grid path figure (3 rows x 3 cols)
    fig, axs = plt.subplots(3, 3, figsize=(14, 12))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("E1 — Paths found by each algorithm on 3 grids",
                 color=C["text"], fontsize=12, fontweight="bold", y=1.01)

    for gi, (name, env) in enumerate(envs):
        for mi, mode in enumerate(modes):
            r = graph_search(env, search_type=mode)
            print("  {:15} {:7} cost={:.0f}  nodes={}".format(
                name, mode, r["cost"], r["nodes_developed"]))
            data.append({"grid": name, "algo": mode,
                         "cost": r["cost"],
                         "nodes": r["nodes_developed"],
                         "open_max": r["max_open_size"],
                         "ms": round(r["time"]*1000, 3)})
            t = "{} | cost={:.0f} nodes={}".format(
                mode, r["cost"], r["nodes_developed"])
            draw_grid(axs[gi][mi], env, r["path"] or [], t)
        print()

    fig.tight_layout(pad=1.4)
    savefig(fig, "E1", "E1_grid_paths.png")

    # nodes bar chart
    fig2, axs2 = plt.subplots(1, 3, figsize=(13, 5))
    fig2.patch.set_facecolor(C["bg"])
    fig2.suptitle("E1 — Nodes Developed per Algorithm",
                  color=C["text"], fontsize=12, fontweight="bold")

    for gi, (name, env) in enumerate(envs):
        ax = axs2[gi]
        style_ax(ax)
        counts = [graph_search(env, search_type=m)["nodes_developed"] for m in modes]
        bars = ax.bar(modes, counts, color=ALGO_COLORS,
                      edgecolor="white", linewidth=1.2, width=0.48, zorder=3)
        for b, v in zip(bars, counts):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(counts)*0.015,
                    str(v), ha="center", va="bottom",
                    color=C["text"], fontweight="bold", fontsize=10)
        ax.set_title(name, color=C["text"], fontweight="bold")
        ax.set_ylabel("Nodes developed")
        ax.set_ylim(0, max(counts)*1.22)

    fig2.tight_layout()
    savefig(fig2, "E1", "E1_nodes_barchart.png")
    savejson({"experiment": "E1", "results": data}, "E1", "E1_results.json")


# ═══════════════════════════════════════════════════════════
# E2
# ═══════════════════════════════════════════════════════════

def experiment_E2():
    print("\n=== E2 — epsilon impact ===")
    env    = make_medium_grid()
    r      = graph_search(env, search_type="A*")
    pol    = make_policy(r["path"], env)
    opt    = r["cost"]
    eps_list = [0.0, 0.1, 0.2, 0.3]
    steps  = range(0, opt*4)
    data   = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("E2 — Effect of epsilon on goal-reaching probability",
                 color=C["text"], fontsize=12, fontweight="bold")
    style_ax(ax1); style_ax(ax2)

    pm_vals, mc_vals = [], []

    for eps, col in zip(eps_list, EPS_COLORS):
        P, states, sid = build_transition_matrix(env, pol, eps=eps)
        pi0 = np.zeros(len(states)); pi0[sid[env.start]] = 1.0
        gi  = sid[env.goal]

        probs = [float((pi0 @ np.linalg.matrix_power(P, n))[gi]) for n in steps]
        p_at  = probs[opt]

        rate, avg, _ = monte_carlo(P, states, env.start, env.goal,
                                   N=8000, max_steps=opt*6)
        pm_vals.append(p_at)
        mc_vals.append(rate)
        print("  eps={:.1f}  P_matrix={:.4f}  P_mc={:.4f}  avg={:.1f}".format(
            eps, p_at, rate, avg))
        data.append({"eps": eps, "p_matrix": round(p_at,6),
                     "p_mc": round(rate,6), "avg_steps": round(avg,2)})

        ax1.plot(list(steps), probs, color=col, lw=2,
                 marker=".", ms=2.5, label="eps="+str(eps))

    ax1.axvline(opt, color=C["c1"], lw=1.5, ls="--",
                label="A* cost ({})".format(opt))
    ax1.set_xlabel("Step n"); ax1.set_ylabel("P(goal at step n)")
    ax1.set_title("Matrix: pi(n) evolution")
    ax1.legend(fontsize=8, facecolor=C["panel"], edgecolor=C["border"])

    x = np.arange(len(eps_list)); w = 0.32
    b1 = ax2.bar(x-w/2, pm_vals, w, label="Matrix (n=opt)",
                 color=C["c2"], edgecolor="white", lw=1, zorder=3)
    b2 = ax2.bar(x+w/2, mc_vals, w, label="Monte Carlo",
                 color=C["c1"], edgecolor="white", lw=1, zorder=3)
    for b in list(b1)+list(b2):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                 "{:.3f}".format(b.get_height()),
                 ha="center", va="bottom", fontsize=8, color=C["text"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(["eps="+str(e) for e in eps_list])
    ax2.set_ylabel("P(reach goal)"); ax2.set_ylim(0, 1.2)
    ax2.set_title("Matrix vs Monte Carlo")
    ax2.legend(fontsize=8, facecolor=C["panel"], edgecolor=C["border"])

    fig.tight_layout()
    savefig(fig, "E2", "E2_epsilon_impact.png")
    savejson({"experiment": "E2", "opt_cost": opt, "results": data},
             "E2", "E2_results.json")


# ═══════════════════════════════════════════════════════════
# E3
# ═══════════════════════════════════════════════════════════

def experiment_E3():
    print("\n=== E3 — h=0 vs Manhattan ===")
    envs = [("Easy 10x10",   make_easy_grid()),
            ("Medium 15x15", make_medium_grid()),
            ("Hard 20x20",   make_hard_grid())]
    data = []
    names, n_ucs, n_ast = [], [], []

    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("E3 — A* (blue) vs UCS (red) paths on 3 grids",
                 color=C["text"], fontsize=12, fontweight="bold")

    for gi, (name, env) in enumerate(envs):
        ru = graph_search(env, search_type="UCS")
        ra = graph_search(env, search_type="A*")
        red = (1 - ra["nodes_developed"]/max(ru["nodes_developed"],1))*100
        print("  {}  UCS={} nodes  A*={} nodes  -{:.1f}%".format(
            name, ru["nodes_developed"], ra["nodes_developed"], red))
        data.append({"grid": name,
                     "ucs_nodes": ru["nodes_developed"], "ucs_cost": ru["cost"],
                     "astar_nodes": ra["nodes_developed"], "astar_cost": ra["cost"],
                     "reduction": round(red,2)})
        names.append(name.split()[0])
        n_ucs.append(ru["nodes_developed"])
        n_ast.append(ra["nodes_developed"])
        t = "{}\nA*={} / UCS={} | -{:.0f}%".format(
            name, ra["nodes_developed"], ru["nodes_developed"], red)
        draw_grid(axs[gi], env, ra["path"], t, path2=ru["path"])

    fig.tight_layout()
    savefig(fig, "E3", "E3_grid_paths.png")

    fig2, ax = plt.subplots(figsize=(9, 5))
    fig2.patch.set_facecolor(C["bg"])
    style_ax(ax)
    x = np.arange(len(names)); w = 0.32
    b1 = ax.bar(x-w/2, n_ucs, w, label="UCS (h=0)",
                color=C["c1"], edgecolor="white", lw=1, zorder=3)
    b2 = ax.bar(x+w/2, n_ast, w, label="A* (Manhattan)",
                color=C["c2"], edgecolor="white", lw=1, zorder=3)
    for b in list(b1)+list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                str(int(b.get_height())), ha="center", va="bottom",
                color=C["text"], fontweight="bold", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Nodes developed")
    ax.set_title("E3 — Heuristic impact on search exploration",
                 color=C["text"], fontweight="bold")
    ax.legend(facecolor=C["panel"], edgecolor=C["border"])
    fig2.tight_layout()
    savefig(fig2, "E3", "E3_heuristics_barchart.png")
    savejson({"experiment": "E3", "results": data}, "E3", "E3_results.json")


# ═══════════════════════════════════════════════════════════
# E4
# ═══════════════════════════════════════════════════════════

def experiment_E4():
    print("\n=== E4 — Weighted A* ===")
    env  = make_hard_grid()
    ws   = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
    opt  = graph_search(env, search_type="A*")["cost"]
    costs, nodes, data = [], [], []

    for w in ws:
        r   = graph_search(env, search_type="A*", weight=w)
        sub = (r["cost"]-opt)/max(opt,1)*100
        print("  W={:.1f}  cost={:.0f}  -{:.1f}%  nodes={}".format(
            w, r["cost"], sub, r["nodes_developed"]))
        costs.append(r["cost"]); nodes.append(r["nodes_developed"])
        data.append({"W": w, "cost": r["cost"],
                     "subopt_pct": round(sub,2),
                     "nodes": r["nodes_developed"]})

    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(C["bg"])
    style_ax(ax1)

    ax1.plot(ws, nodes, color=C["c2"], marker="o", lw=2.2,
             ms=7, label="Nodes", zorder=3)
    ax1.set_xlabel("Weight W"); ax1.set_ylabel("Nodes developed",
                                               color=C["c2"])
    ax1.tick_params(axis="y", labelcolor=C["c2"])

    ax2 = ax1.twinx()
    ax2.set_facecolor(C["panel"])
    ax2.plot(ws, costs, color=C["c1"], marker="s", lw=2.2,
             ms=7, ls="--", label="Path cost", zorder=3)
    ax2.axhline(opt, color=C["c3"], lw=1.4, ls=":",
                alpha=0.9, label="Optimal ({})".format(opt))
    ax2.set_ylabel("Path cost", color=C["c1"])
    ax2.tick_params(axis="y", labelcolor=C["c1"])
    for sp in ax2.spines.values():
        sp.set_color(C["border"])

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, loc="center right",
               facecolor=C["panel"], edgecolor=C["border"], fontsize=9)
    ax1.set_title("E4 — Weighted A*: speed vs optimality",
                  color=C["text"], fontweight="bold")
    fig.tight_layout()
    savefig(fig, "E4", "E4_weighted_astar.png")
    savejson({"experiment": "E4", "optimal": opt, "results": data},
             "E4", "E4_results.json")


# ═══════════════════════════════════════════════════════════
# Phase 3
# ═══════════════════════════════════════════════════════════

def phase3():
    print("\n=== Phase 3 — Markov chain P ===")
    env = make_medium_grid()
    r   = graph_search(env, search_type="A*")
    pol = make_policy(r["path"], env)
    eps = 0.1
    opt = r["cost"]

    P, states, sid = build_transition_matrix(env, pol, eps=eps)
    print("  P shape:", P.shape, " stochastic:", np.allclose(P.sum(1),1))

    pi0 = np.zeros(len(states)); pi0[sid[env.start]] = 1.0
    gi  = sid[env.goal]
    ns  = range(0, opt*5)
    probs = [float((pi0 @ np.linalg.matrix_power(P,n))[gi]) for n in ns]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(C["bg"])
    style_ax(ax)
    ax.plot(list(ns), probs, color=C["c2"], lw=2.2)
    ax.fill_between(list(ns), probs, color=C["c2"], alpha=0.10)
    ax.axvline(opt, color=C["c1"], lw=1.5, ls="--",
               label="A* cost ({})".format(opt))
    ax.set_xlabel("Step n"); ax.set_ylabel("P(agent at goal at step n)")
    ax.set_title("Phase 3 — pi(n) evolution  |  eps={}".format(eps),
                 color=C["text"], fontweight="bold")
    ax.legend(facecolor=C["panel"], edgecolor=C["border"])
    fig.tight_layout()
    savefig(fig, "P3", "Phase3_pi_evolution.png")
    savejson({"phase":3, "eps":eps, "opt_cost":opt,
              "P_shape": list(P.shape)}, "P3", "Phase3_summary.json")


# ═══════════════════════════════════════════════════════════
# Phase 4
# ═══════════════════════════════════════════════════════════

def phase4():
    print("\n=== Phase 4 — Classes & absorption ===")
    env = make_medium_grid()
    r   = graph_search(env, search_type="A*")
    pol = make_policy(r["path"], env)
    summary = []

    for eps in [0.0, 0.1, 0.2]:
        P, states, _ = build_transition_matrix(env, pol, eps=eps)
        cls = comm_classes(P, states)
        nr  = sum(1 for c in cls if c["type"]=="recurrent")
        nt  = sum(1 for c in cls if c["type"]=="transient")
        print("  eps={:.1f}  classes={}  rec={}  trans={}".format(
            eps, len(cls), nr, nt))
        row = {"eps": eps, "n_classes": len(cls),
               "recurrent": nr, "transient": nt,
               "p_goal": None, "mean_steps": None}
        if eps > 0:
            try:
                ap = absorb_prob(P, states, env.goal)
                et = expected_steps(P, states, env.goal)
                b  = ap.get(env.start, float("nan"))
                t  = et.get(env.start, float("nan"))
                print("    P(goal|start)={:.4f}  E[steps]={:.1f}".format(b,t))
                row["p_goal"] = round(b,6)
                row["mean_steps"] = round(t,4)
            except np.linalg.LinAlgError:
                print("    (I-Q) singular")
        summary.append(row)

    eps_range = [0.05,0.1,0.15,0.2,0.25,0.3]
    ap_vals, et_vals = [], []
    for eps in eps_range:
        P2, st2, _ = build_transition_matrix(env, pol, eps=eps)
        try:
            ap_vals.append(absorb_prob(P2,st2,env.goal).get(env.start,0))
            et_vals.append(expected_steps(P2,st2,env.goal).get(env.start,0))
        except:
            ap_vals.append(float("nan")); et_vals.append(float("nan"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(C["bg"])
    style_ax(ax1); style_ax(ax2)
    fig.suptitle("Phase 4 — Absorption analysis vs epsilon",
                 color=C["text"], fontsize=12, fontweight="bold")

    ax1.plot(eps_range, ap_vals, color=C["c4"], marker="o", lw=2.2, ms=7)
    ax1.fill_between(eps_range, ap_vals, color=C["c4"], alpha=0.10)
    ax1.set_xlabel("epsilon"); ax1.set_ylabel("P(reach goal | start)")
    ax1.set_title("Absorption probability"); ax1.set_ylim(0, 1.05)

    ax2.plot(eps_range, et_vals, color=C["c3"], marker="s", lw=2.2, ms=7)
    ax2.set_xlabel("epsilon"); ax2.set_ylabel("Steps (log scale)")
    ax2.set_title("Mean absorption time"); ax2.set_yscale("log")

    fig.tight_layout()
    savefig(fig, "P4", "Phase4_absorption.png")
    savejson({"phase":4, "summary": summary}, "P4", "Phase4_summary.json")


# ═══════════════════════════════════════════════════════════
# Phase 5
# ═══════════════════════════════════════════════════════════

def phase5():
    print("\n=== Phase 5 — Monte Carlo ===")
    env  = make_medium_grid()
    r    = graph_search(env, search_type="A*")
    pol  = make_policy(r["path"], env)
    opt  = r["cost"]
    eps_list = [0.0, 0.1, 0.2, 0.3]
    data, sims = [], []

    for eps in eps_list:
        P, states, _ = build_transition_matrix(env, pol, eps=eps)
        rate, avg, times = monte_carlo(P, states, env.start, env.goal,
                                       N=10000, max_steps=opt*8)
        med = float(np.median(times)) if times else float("nan")
        print("  eps={:.1f}  success={:.4f}  avg={:.1f}  med={:.0f}".format(
            eps, rate, avg, med))
        data.append({"eps":eps, "p_goal":round(rate,6),
                     "avg":round(avg,4), "median":round(med,1)})
        sims.append((eps, times, avg))

    # 2x2 histograms
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Phase 5 — Steps to reach goal (10 000 simulations)",
                 color=C["text"], fontsize=12, fontweight="bold")

    for ax, (eps, times, avg), col in zip(axs.flat, sims, EPS_COLORS):
        style_ax(ax)
        if times:
            ax.hist(times, bins=38, color=col, edgecolor="white",
                    alpha=0.85, zorder=3)
            ax.axvline(avg, color=C["c1"], lw=1.6, ls="--",
                       label="mean={:.0f}".format(avg))
            ax.axvline(opt, color=C["text"], lw=1.2, ls=":",
                       label="A* cost={}".format(opt))
            ax.set_title("eps={}  P(success)={:.3f}".format(
                eps, len(times)/10000), color=C["text"])
            ax.set_xlabel("Steps"); ax.set_ylabel("Count")
            ax.legend(fontsize=8, facecolor=C["panel"], edgecolor=C["border"])
        else:
            ax.text(0.5,0.5,"no successes", ha="center",
                    va="center", transform=ax.transAxes,
                    color=C["text"], fontsize=12)

    fig.tight_layout()
    savefig(fig, "P5", "Phase5_histograms.png")

    # success rate bar
    fig2, ax = plt.subplots(figsize=(7, 5))
    fig2.patch.set_facecolor(C["bg"])
    style_ax(ax)
    rates = [d["p_goal"] for d in data]
    bars  = ax.bar([str(e) for e in eps_list], rates,
                   color=EPS_COLORS, edgecolor="white", lw=1.2, width=0.46, zorder=3)
    for b, v in zip(bars, rates):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                "{:.3f}".format(v), ha="center", va="bottom",
                color=C["text"], fontweight="bold")
    ax.set_xlabel("epsilon"); ax.set_ylabel("P(reach goal)")
    ax.set_title("Phase 5 — Success rate vs epsilon",
                 color=C["text"], fontweight="bold")
    ax.set_ylim(0, 1.15)
    fig2.tight_layout()
    savefig(fig2, "P5", "Phase5_success_rate.png")
    savejson({"phase":5, "opt_cost":opt, "results":data},
             "P5", "Phase5_results.json")


# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    experiment_E1()
    experiment_E2()
    experiment_E3()
    experiment_E4()
    phase3()
    phase4()
    phase5()
    print("\nAll done — results saved in results/")
