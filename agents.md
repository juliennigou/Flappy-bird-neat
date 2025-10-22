# agents.md – Règles pour l’agent LLM développeur

## 1) Philosophie
- **Sécurité & reproductibilité d’abord** : tests, seeds, CI locale avant ajout de features.
- **Petites itérations** : un objectif par PR/commit. Code simple > clever.
- **Contrats stables** : l’API de `neatlab/` ne dépend jamais d’un jeu Pygame spécifique.

## 2) Boucle d’itération standard
1. **Lire la spec** (docs du repo + issues). Résumer en 3–5 bullet points l’objectif courant.
2. **Écrire/mettre à jour les tests** (TDD léger) : cas nominaux + bords + erreurs.
3. **Coder minimalement** pour faire passer les tests (pas d’optim prématurée).
4. **Exécuter** `ruff` (lint), `mypy` (types), `pytest -q --maxfail=1` (rapide), puis tests lents/integ.
5. **Auto-doc** : docstrings (Google style), commentaires “pourquoi”, pas “quoi”.
6. **Mesurer** : si la tâche touche perf, ajouter un micro-benchmark reproductible.
7. **Refactor** sécurisé par tests.
8. **Commit** avec message clair (Conventional Commits). Ouvrir PR si applicable.

## 3) Standards de code
- **Python ≥ 3.10**, typage strict : `from __future__ import annotations`.
- **Lint** : `ruff` (PEP8 + règles import/order). **Types** : `mypy --strict` sur `neatlab/`.
- **Style** : `black` pour formatage.
- **Docstrings** : Google style + exemples d’usage (doctest si pertinent).
- **Erreurs** : lever exceptions claires (`ValueError`, `RuntimeError`, `TimeoutError`).
- **Logs** : `logging` niveau configurable; pas de `print()` hors CLI/reporters/tests.

## 4) Tests – exigences minimales
- Couverture **≥ 85%** sur `neatlab/` (branches principales).
- **Unitaires** : gènes/innovations/génome/network/species/repro/pop/evaluator/persistence.
- **Intégration** : Flappy `env_core`, `adapter`, CLI `train`/`play`.
- **E2E** : run 5–10 générations (population réduite) headless → fichiers `runs/` créés.
- **Perf** : bench headless (steps/s) et assert seuils minimaux configurables.
- **Determinisme** : tests seedés (mêmes résultats → mêmes checkpoints/totaux ± tolérance).
- **Property-based (option)** : `hypothesis` pour graphes acycliques, mutations sans cycles, etc.

## 5) Conventions d’API (réutilisabilité)
- `neatlab` **n’importe pas** Pygame. Interfaces : `Env.reset/step`, `obs_transform`, `action_transform`.
- Les modules NEAT **ne connaissent pas** les features Flappy.
- **Aucune** constante de jeu en dur dans `neatlab/`.

## 6) Parallélisation & OS
- **multiprocessing spawn** only (safe Windows). Protéger `if __name__ == "__main__":`.
- Pas de ressources Pygame/SDL dans les workers headless.
- **Seeding** : `seed_base + worker_id * 10_000 + local_counter`.

## 7) Persistance & compatibilité
- Checkpoints via `persistence.py` : versions sérialisées + schéma.
- **Backward compatible** sur mineures : ajouter champs avec défauts, migrations simples si besoin.

## 8) Performance
- Priorité au **headless** : pas d’horloge, pas d’allocs lourdes dans la boucle `step`.
- Minimiser copies d’objets (utiliser `array`/`numpy` si utile, sans surcomplexifier).
- Mesurer, ne pas deviner. Ajouter métriques `steps_per_sec`.

## 9) Qualité PR
- Checklist PR :
  - [ ] Tests ajoutés/maj et verts.
  - [ ] Lint + types OK.
  - [ ] Pas d’API publique cassée (ou changelog + migration).
  - [ ] Docs/docstrings mises à jour.
  - [ ] Bench (si perf-impact) joint.

## 10) Définition de Done (DoD)
- Fonctionnalité couverte par tests, docstrings, et examples.
- CI locale (lint/types/tests) **verte**.
- Pas de TODO bloquants; tickets créés pour dettes résiduelles.

## 11) Outils & commandes
- `make setup` (installer deps)
- `make lint` → ruff
- `make type` → mypy
- `make test` → pytest (rapide)
- `make test-all` → + intégration/E2E
- `make bench` → benchmarks headless

## 12) Anti-patterns (à éviter)
- Coupler `neatlab` à Pygame ou aux assets.
- Introduire du hasard non seedé.
- Surcharger les PR (> 400 lignes modifiées non testées).
- Optimisations micro avant d’avoir des métriques.

---
Ce fichier est l’autorité pour les bonnes pratiques. Toute dérogation doit être justifiée dans la PR et assortie de tests dédiés.

