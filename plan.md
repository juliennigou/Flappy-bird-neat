# plan.md – Découpage du projet par étapes

> Objectif : livrer une lib NEAT réutilisable + intégration Flappy, avec tests exhaustifs et CLI. Les étapes sont ordonnées. Chaque sous-étape produit des livrables et des critères d’acceptation.

## Étape 0 — Bootstrapping du repo
- **Livrables** : `pyproject.toml`, `README.md`, `CONTRIBUTING.md`, `agents.md`, `plan.md`, gabarits `makefile`.
- **Tâches** :
  - Initialiser tooling : ruff, black, mypy, pytest, coverage, pre-commit.
  - Structure dossiers (neatlab/, games/flappy/, tests/).
- **Acceptation** : `make lint`, `make type`, `make test` passent à vide.

## Étape 1 — Innovations & Gènes
1.1 **InnovationTracker**
- Livrables : `innovations.py`, tests unicité & persistance.
- Acceptation : mapping `(in,out)` → `innovation` stable, monotone.

1.2 **NodeGene / ConnGene**
- Livrables : `genes.py` + tests de validation des champs.
- Acceptation : création/égalité/copie, contraintes type de nœuds.

## Étape 2 — Genome Ops
2.1 **Mutation poids**
- Livrables : `genome.py::mutate_weight` + tests distribution & reset.
- Acceptation : stats conformes (moyenne ~0, SD ≈ config).

2.2 **Mutation add_conn (acyclic)**
- Livrables : `mutate_add_conn` + détection cycle.
- Acceptation : aucune arête dupliquée/cyclique.

2.3 **Mutation add_node**
- Livrables : `mutate_add_node` + tests.
- Acceptation : connexion scindée désactivée, deux connexions valides ajoutées.

2.4 **Crossover**
- Livrables : `crossover` + tests (E/D, communs, enabled prob.).
- Acceptation : parent plus fit « mène ».

## Étape 3 — Réseau (phénotype)
3.1 **Topo sort**
- Livrables : `network.py` (construction niveaux).
- Acceptation : tri correct pour graphes acycliques.

3.2 **Évaluation par niveaux**
- Livrables : exécution feed-forward + activations (`activation.py`).
- Acceptation : sorties attendues sur graphes jouets.

3.3 **Option récurrente (hooks)**
- Livrables : flags `allow_recurrent`, validation arêtes arrière (désactivée par défaut).
- Acceptation : pas d’arêtes arrière si FF only.

## Étape 4 — Spéciation & Reproduction
4.1 **Distance de compatibilité**
- Livrables : `species.py` (δ) + tests.
- Acceptation : cas synthétiques conformes.

4.2 **Clustering en espèces**
- Livrables : assignation par seuil, auto-ajustement vers `target_species`.
- Acceptation : nombre d’espèces converge vers cible.

4.3 **Allocation d’offspring / élitisme / sélection**
- Livrables : `reproduction.py`.
- Acceptation : somme offspring = `pop_size`, élitisme respecté.

## Étape 5 — Population & Boucle d’évolution
5.1 **Population**
- Livrables : `population.py` (générations, stagnation, best/champion).
- Acceptation : stagnation appliquée, arrêt sur `fitness_threshold`.

5.2 **Reporters & métriques**
- Livrables : `reporters.py`, `metrics.py` (CSV/JSON).
- Acceptation : `metrics.csv` alimenté par génération.

5.3 **Persistance**
- Livrables : `persistence.py` (save/load checkpoints).
- Acceptation : rechargement → état identique (tests equality).

## Étape 6 — Évaluation (Sync & Parallel)
6.1 **Interfaces**
- Livrables : `evaluator.py` (SyncEvaluator, ParallelEvaluator).
- Acceptation : API stable pour `env_factory` + transforms.

6.2 **Parallel (multiprocessing)**
- Livrables : pool, seeding par worker, timeouts.
- Acceptation : équivalence statistique vs sync (± tolérance), pas de deadlocks.

## Étape 7 — Flappy Bird (env & adapter)
7.1 **env_core**
- Livrables : physique/flappy, collisions, génération tuyaux, normalisation obs.
- Acceptation : tests de collisions, déterminisme seed.

7.2 **adapter**
- Livrables : `obs_transform`, `action_transform` (binaire flap).
- Acceptation : dimensions correctes, mapping fidèle.

7.3 **env_pygame** (visual)
- Livrables : rendu 60 FPS, overlay.
- Acceptation : boucle visuelle stable, aucune dépendance en headless.

## Étape 8 — CLI & Configs
8.1 **Chargement YAML → dataclasses**
- Livrables : `config.py`, fichiers `neat.yml`, `env.yml`, `run.yml`.
- Acceptation : validation de schéma, défauts raisonnables.

8.2 **CLI**
- Livrables : `cli.py` (train/play/benchmark, resume, save-every).
- Acceptation : commandes fonctionnelles, messages d’erreur clairs.

## Étape 9 — E2E & Performance
9.1 **Run E2E**
- Livrables : script test e2e (5–10 générations, pop réduite).
- Acceptation : `runs/...` généré, metrics plausibles.

9.2 **Bench headless**
- Livrables : bench steps/s (1,2,4,8 workers).
- Acceptation : ≥ seuil défini, scalabilité sublinéaire.

## Étape 10 — Documentation & Packaging
- Livrables : README complet, examples, schémas, docstrings revues.
- Acceptation : onboarding < 30 min pour un nouveau dev.

---

# Jalons & livrables cumulés
- **M1 (fin Étape 3)** : NEAT sans speciation (réseau ok) + tests
- **M2 (fin Étape 5)** : Évolution complète single-process + checkpoints
- **M3 (fin Étape 6–7)** : Parallel + Flappy jouable, headless ok
- **M4 (fin Étape 8–9)** : CLI + E2E + Bench
- **M5 (Étape 10)** : Release 0.1.0 packagée

---

# Checklists transverses
- [ ] Lint/Types/Tests/Docs verts avant merge
- [ ] Seeds et RNG vérifiés
- [ ] Aucune dépendance Pygame dans `neatlab/`
- [ ] Bench mis à jour après changements perf
- [ ] Changelog & version bump

