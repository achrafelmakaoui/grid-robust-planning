# Mini-Projet — Planification Robuste sur Grille
## A* + Chaînes de Markov à Temps Discret

> **Filière :** Bases de l'Intelligence Artificielle  
> **Date :** Mars 2026

---

## Objectif

Planifier un chemin optimal sur une grille 2D avec obstacles **(A\*)**, puis évaluer la robustesse de ce plan face à une dynamique stochastique **(Chaînes de Markov)** via calcul matriciel et simulation Monte Carlo.

---

## Structure du projet

```
Mini_Projet_Bases_IA/
├── src/
│   ├── grid.py           # GridEnvironment + 3 grilles (Easy/Medium/Hard)
│   ├── astar.py          # A*, UCS, Greedy, Weighted A*
│   ├── markov.py         # Matrice P, classes, absorption, Monte Carlo
│   └── experiments.py    # Toutes les expériences (E1–E4, Phases 3–5)
│
├── notebooks/
│   └── exploration.ipynb    # Notebook interactif reproductible
│
├── results/
│   ├── E1_algorithm_comparison/     # Figures + CSV + JSON
│   ├── E2_epsilon_impact/
│   ├── E3_heuristic_comparison/
│   ├── E4_weighted_astar/
│   ├── Phase3_markov_chain/
│   ├── Phase4_absorption/
│   └── Phase5_monte_carlo/
│
├── docs/
│   └── rapport.pdf
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Cloner le projet
git clone <url>
cd Mini_Projet_Bases_IA

# 2. Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## Utilisation

### Lancer toutes les expériences (génère les résultats dans `results/`)
```bash
cd src
python experiments.py
```

### Lancer le notebook interactif
```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## Expériences

| ID | Description | Fichiers générés |
|----|-------------|-----------------|
| **E1** | UCS vs Greedy vs A* sur 3 grilles | `E1_grid_paths.png`, `E1_nodes_barchart.png`, `E1_results.csv` |
| **E2** | Impact de ε sur P(GOAL) | `E2_epsilon_impact.png`, `E2_results.csv` |
| **E3** | h=0 (UCS) vs Manhattan (A*) | `E3_grid_paths.png`, `E3_heuristics_barchart.png`, `E3_results.csv` |
| **E4** | Weighted A* vitesse vs optimalité | `E4_weighted_astar.png`, `E4_results.csv` |
| **Phase 3** | Matrice P + évolution π⁽ⁿ⁾ | `Phase3_pi_evolution.png`, `Phase3_pi_n.csv` |
| **Phase 4** | Classes de communication + absorption | `Phase4_absorption.png`, `Phase4_classes_summary.csv` |
| **Phase 5** | Simulation Monte Carlo (10 000 trajec.) | `Phase5_histograms.png`, `Phase5_success_rate.png`, `Phase5_results.csv` |

---

## Résultats clés

| Aspect | Résultat |
|--------|---------|
| **A\* vs UCS** | A\* réduit les nœuds explorés de **15–25%** grâce à h(Manhattan) |
| **A\* vs Greedy** | Greedy est plus rapide mais **sous-optimal** sur grilles complexes (+9–14%) |
| **Weighted A\*** | W≥2 réduit les nœuds mais introduit **~9% de suboptimalité** |
| **Impact ε=0.1** | P(GOAL en exactement n=opt étapes) ≈ **3%** seulement |
| **Absorption** | P(GOAL finalement) = **1.0** pour tout ε > 0 |
| **Distribution** | Temps d'atteinte **heavy-tailed** : médiane << moyenne |

---

## Modules Python

### `grid.py`
```python
from grid import make_easy_grid, make_medium_grid, make_hard_grid
env = make_medium_grid()   # GridEnvironment 15x15
```

### `astar.py`
```python
from astar import graph_search
result = graph_search(env, search_type="A*")   # aussi "UCS", "Greedy"
result = graph_search(env, search_type="A*", weight=2.0)  # Weighted A*
```

### `markov.py`
```python
from markov import generate_policy, build_transition_matrix, simulate_monte_carlo
policy = generate_policy(result["path"], env)
P, states, s2i = build_transition_matrix(env, policy, epsilon=0.1)
rate, avg, times = simulate_monte_carlo(P, states, env.start, env.goal)
```

---

## Références

- Synthèse *Chaînes de Markov à temps discret* : matrice P, Chapman–Kolmogorov, classes, absorption, simulation.
- Synthèse *Recherche heuristique — du Best-First à A\** : OPEN/CLOSED, g, h, f, admissibilité, cohérence, variantes.
