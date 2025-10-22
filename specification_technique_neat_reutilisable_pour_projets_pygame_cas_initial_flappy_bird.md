# 1. Contexte & objectifs
Ce document décrit une architecture et un plan de mise en œuvre pour une librairie **NEAT** (NeuroEvolution of Augmenting Topologies) codée *from scratch* en Python, réutilisable pour **n’importe quel projet Pygame** (Flappy Bird pour démarrer), avec deux modes : **visualisation** (rendu temps réel) et **entraînement headless** (sans rendu, parallélisable). L’objectif est de fournir :

- Un **cœur NEAT générique**, découplé des détails du jeu.
- Une **API d’intégration** simple pour brancher de nouveaux environnements Pygame.
- Une **CLI** et des **configs YAML** pour piloter les entraînements.
- Un **plan de tests exhaustif** (unitaire, intégration, E2E, perfs) et des **critères d’acceptation**.

---

# 2. Périmètre
## Inclus
- Implémentation complète de NEAT : gènes, innovations, topologie évolutive, spéciation, reproduction, stagnation, fitness ajustée, sauvegardes/checkpoints.
- Entraînement mono-processus et multi-processus (CPU) avec seed reproductible.
- Intégration Pygame pour Flappy Bird (environnement + adaptation d’entrées/sorties réseau).
- Instrumentation : logs, métriques CSV/JSON, graphiques facultatifs (matplotlib).

## Exclus (phase 1)
- Exécution multi-machines/distribuée (Ray/cluster) – optionnelle phase 2.
- Réseaux récurrents avancés (support crochets partiels, désactivé par défaut).
- Tableau de bord web.

---

# 3. Principes d’architecture
- **Séparation nette** NEAT ↔ Environnements : la lib `neatlab` n’importe pas Pygame et n’a pas connaissance des jeux.
- **API d’évaluation** basée sur des *factories* d’environnements et deux transformateurs (obs→input réseau, out réseau→action).
- **Headless-first** : la logique de jeu (physique, collisions) est indépendante du rendu; le rendu est branché seulement en mode visualisation.
- **Reproductibilité** : gestion centralisée des seeds et de la RNG, sérialisation des innovations et de l’état population.
- **Performance** : évaluation par *batchs* d’épisodes côté worker, absence de `pygame.time.Clock()` en headless, minimisation des copies d’objets.

---

# 4. Arborescence cible (monorepo)
```
neatlab/
  pyproject.toml
  README.md
  neatlab/
    __init__.py
    config.py
    rng.py
    genes.py           # NodeGene, ConnGene
    genome.py          # Genome: mutate/crossover/build-net
    innovations.py     # InnovationTracker
    network.py         # Phénotype: exécution (topologique), opz récurrence
    species.py         # Species, compatibilité
    reproduction.py    # offspring allocation, sélection
    population.py      # boucle d'évolution, reporters
    evaluator.py       # SyncEvaluator, ParallelEvaluator
    activation.py      # fonctions d'activation
    persistence.py     # save/load checkpoints (pkl/json)
    reporters.py       # stdout, csv, plots (option)
    metrics.py         # agrégation/flush métriques
    utils.py
  games/
    flappy/
      env_core.py      # logique jeu sans rendu
      env_pygame.py    # rendu optionnel Pygame
      adapter.py       # mapping obs/actions ↔ réseau NEAT
      configs/
        neat.yml
        env.yml
        run.yml
  tests/
    unit/
    integration/
    e2e/
  runs/                # sorties (logs, checkpoints, figures)
```

---

# 5. Spécification NEAT (lib `neatlab`)
## 5.1 Modèles de données
**NodeGene**
- `id: int` (unique), `type: {INPUT, HIDDEN, OUTPUT, BIAS}`
- `activation: str` (ex : `sigmoid`, `tanh`, `relu`, `identity`) – configurable

**ConnGene**
- `innovation: int` (global unique)
- `in_id: int`, `out_id: int`
- `weight: float`, `enabled: bool`

**Genome**
- `nodes: dict[int, NodeGene]`
- `conns: dict[int, ConnGene]` (clé : innovation)
- Opérations : `mutate_add_node`, `mutate_add_conn`, `mutate_weight`, `toggle_conn`, `crossover`

**InnovationTracker**
- Map `(in_id,out_id) -> innovation_id` persistée par run
- Fournit des IDs monotones, stable cross-générations

## 5.2 Réseau (phénotype)
- Construction **feed-forward** par tri topologique (groupes de couches *leveled*).
- Évaluation vectorisée sur un pas de temps : `y = activation(Wx + b)` par niveaux.
- Option **récurrente** (phase 2) : autoriser arêtes arrière et dérouler K itérations.

## 5.3 Distance de compatibilité (spéciation)
\( \delta = c_1\cdot\frac{E}{N} + c_2\cdot\frac{D}{N} + c_3\cdot\overline{|w_a-w_b|} \)
- E : gènes *excess*, D : *disjoint*, W : différence de poids moyenne sur gènes communs.
- `N = max(|A|, |B|)` si `N>20`, sinon `N=1`.
- Seuil de compatibilité **auto-ajusté** vers `target_species` toutes les `k` générations.

## 5.4 Fitness & reproduction
- **Fitness ajustée** : `fitness_adj = fitness / species_size`.
- Allocation d’offspring par espèce proportionnellement à la somme des `fitness_adj`.
- **Élitisme** : top `E` de chaque espèce clonés avant reproduction.
- Sélection **intra-espèce** (roulette ou tournoi pondéré par fitness).
- **Crossover** : le parent plus fit « mène » (garde E/D). Pour gènes communs : *pick* aléatoire ou moyenne des poids. Statut `enabled` hérité avec probabilité (désactivation possible si l’un est désactivé).
- **Mutations** :
  - `mutate_weight` : perturbation gaussienne (SD configurable) + *reset* aléatoire.
  - `mutate_add_conn` : échantillonner paires `(in, out)` valides, éviter doublons et cycles (si FF).
  - `mutate_add_node` : désactiver une connexion, insérer un nœud hidden + 2 connexions (1.0 et ancien poids).

## 5.5 Stagnation & arrêt
- Par espèce : piste du meilleur fitness; marquer stagnante si aucun progrès `> max_stagnation` générations.
- Extinction d’espèce stagnante (sauf espèce championne globale).
- Arrêt si `fitness_threshold` atteint ou `max_generations` dépassé.

## 5.6 Configuration (dataclass + YAML)
Attributs clés (extrait) :
- Population : `pop_size`, `elitism`, `survival_threshold`
- Mutations : `weight_mutate_rate`, `weight_perturb_sd`, `weight_reset_rate`, `add_conn_rate`, `add_node_rate`, `enable_rate`, `disable_rate`
- Spéciation : `c1,c2,c3`, `compatibility_threshold`, `target_species`, `species_adjust_period`
- Entraînement : `episodes_per_genome`, `max_generations`, `fitness_threshold`
- Réseau : `activation_default`, `allow_recurrent`
- RNG : `seed`

---

# 6. API d’intégration (réutilisable multi-jeux)
## 6.1 Contrat environnement minimal
Un environnement doit exposer :
```python
class Env:
    def reset(self, seed: int | None = None): ...  # -> obs
    def step(self, action): ...                     # -> (obs, reward, done, info)
```

## 6.2 Transformateurs I/O
- `obs_transform(obs) -> np.ndarray[float32]` : met en forme les entrées du réseau (normalisation, sélection de features, concat, etc.).
- `action_transform(net_out) -> action` : convertit la sortie réseau en action valide pour l’environnement (ex : binaire flap/no flap, ou discret à argmax, ou continu borné).

## 6.3 Évaluateurs
- `SyncEvaluator(env_factory, obs_transform, action_transform)`
- `ParallelEvaluator(env_factory, obs_transform, action_transform, workers: int, batch_size: int, timeout_s: float | None)`
- **env_factory** crée un nouvel environnement indépendant (headless ou visuel).
- Stratégie : par génome, exécuter `episodes_per_genome` épisodes et retourner la **moyenne** des fitness.

---

# 7. Flappy Bird (cas de référence)
## 7.1 Observations (par défaut)
- `y` de l’oiseau (normalisée [0,1])
- `vy` (vitesse verticale, normalisée)
- `dx` distance horizontale au prochain tuyau
- `gap_top`, `gap_bottom` (ou `gap_center`)

## 7.2 Actions
- `0 = rien`, `1 = flap` (sortie réseau ∈ [0,1] → flap si `>0.5`).

## 7.3 Récompenses
- `+1.0` par tuyau franchi
- `−0.01` par step (évite “camping”)
- `0` à la mort (ou `−1` optionnel)

## 7.4 Deux modes
- **Headless** : aucune initialisation d’écran, pas d’attente horloge, pas d’assets lourds.
- **Visualisation** : Pygame, 60 FPS, overlay (génération, best/mean fitness, score courant).

---

# 8. CLI & fichiers de configuration
## 8.1 CLI
- `python -m neatlab.cli train --config games/flappy/configs/run.yml [--headless] [--workers N] [--resume PATH] [--save-every K]`
- `python -m neatlab.cli play --checkpoint runs/.../champion.pkl --visual`
- `python -m neatlab.cli benchmark --headless --steps 200000`

## 8.2 Configs YAML (exemple synthétique)
`games/flappy/configs/neat.yml` (NEAT)
```yaml
pop_size: 200
fitness_threshold: 50
activation_default: sigmoid
add_conn_rate: 0.08
add_node_rate: 0.03
weight_mutate_rate: 0.9
weight_perturb_sd: 0.6
c1: 1.0
c2: 1.0
c3: 0.4
target_species: 12
elitism: 2
survival_threshold: 0.25
max_stagnation: 15
allow_recurrent: false
seed: 42
```
`games/flappy/configs/env.yml` (environnement)
```yaml
screen_width: 288
screen_height: 512
gravity: 0.35
flap_impulse: -6.5
pipe_speed: -3.0
pipe_gap: 110
pipe_spacing_px: 200
reward:
  alive_step: -0.01
  pipe_passed: 1.0
  death: 0.0
normalize_obs: true
```
`games/flappy/configs/run.yml` (exécution)
```yaml
headless: true
workers: 8
episodes_per_genome: 2
max_generations: 250
save_every: 5
resume: null
```

---

# 9. Détails d’implémentation (sections critiques)
## 9.1 Crossover (règles)
- Déterminer le parent **plus fit** (ou au hasard si égalité → stable grâce à seed).
- Pour chaque connexion du parent plus fit :
  - si innovation aussi présente chez l’autre parent → choisir poids au hasard (ou moyenne) ; `enabled` actif sauf probabilité de désactivation (25%).
  - sinon (E/D) → hériter tel quel.
- Ensemble de nœuds = union des nœuds référencés par les connexions héritées.

## 9.2 Mutation add_conn (prévention cycles)
- Échantillonner `(in, out)` où `in` ∈ {INPUT, HIDDEN, BIAS}, `out` ∈ {HIDDEN, OUTPUT} et `in != out`.
- **Feed-forward** : refuser arêtes qui génèrent un cycle (DFS de cycle ou numérotation de niveaux si disponible).
- Vérifier non-duplication par `(in,out)`.

## 9.3 Mutation add_node
- Sélectionner une connexion **enabled** à scinder.
- La désactiver, insérer un **nouveau nœud** HIDDEN.
- Ajouter deux connexions : `(in → new)` poids `1.0`, `(new → out)` poids = ancien poids.
- Nouvelles innovations assignées par `InnovationTracker`.

## 9.4 Topological sort & exécution
- Construire niveaux par Kahn (ou DFS) sur graphe acyclique.
- Exécution par niveaux : buffers `a[l] = act(W[l] @ a[l-1] + b[l])`.
- Biais : `BIAS` comme nœud d’entrée constant `1.0`.

## 9.5 Parallélisation
- `multiprocessing` (spawn) : un worker = un env autonome.
- *Batch d’épisodes par génome* pour limiter le coût d’IPC.
- `seed_worker = seed_base + worker_id * 10_000 + local_counter`.
- **Timeout** par épisode (optionnel) : abandon et pénalisation contrôlée.

## 9.6 Persistance & checkpoints
- Dossier `runs/<YYYY-MM-DD_HH-MM-SS>/` :
  - `config.yml` (fusion neat/env/run)
  - `neat_state.pkl` (population + espèces + innovations + RNG)
  - `champion.pkl` (meilleur génome) + `population.pkl` (option)
  - `metrics.csv` (génération, best/mean fitness, species_count, eval_time_s)
  - `events.log`

---

# 10. Qualité, métriques & perf
- **Cibles** :
  - Headless : ≥ 10k **steps/s** sur machine de référence (à préciser par l’équipe).
  - Visual : 60 FPS stables, <10 ms/frame en moyenne.
  - Entraînement Flappy : atteindre `fitness ≥ 50` en ≤ 250 générations avec configs par défaut.
- **Métriques enregistrées** : `generation, pop_size, species_count, best_fitness, mean_fitness, median_fitness, eval_time_s, steps_per_sec`.

---

# 11. Tests – Plan exhaustif
## 11.1 Unitaires (lib `neatlab`)
**genes.py**
- Création de `NodeGene`/`ConnGene`, invariants (`type` valide, poids float, enabled bool).

**innovations.py**
- Unicité `(in,out) → innovation` et monotonie des IDs.
- Persistance/chargement : stabilité inter-générations.

**genome.py**
- `mutate_weight` : distribution de perturbation (moyenne ~0, SD conforme), prob de reset.
- `mutate_add_conn` : pas de doublon, pas de cycles (FF), respect types de nœuds.
- `mutate_add_node` : désactivation de l’arête scindée, insertion de nœud + 2 arêtes correctes.
- `crossover` : héritage E/D du parent plus fit, règle d’`enabled`.

**network.py**
- Tri topologique correct pour graphes acycliques.
- Évaluation : sorties bornées selon activation, exactitude sur petits graphes connus.

**species.py**
- Distance δ sur cas synthétiques (E/D/W), auto-ajustement du seuil.

**reproduction.py**
- Allocation offspring somme = `pop_size`, élitisme appliqué, sélection pondérée.

**population.py**
- Mise à jour fitness, tri des champions, stagnation par espèce.

**evaluator.py**
- Sync : déterminisme (seed), moyenne correcte multi-épisodes.
- Parallel : équivalence statistique vs sync (± tolérance), gestion timeouts.

**persistence.py / reporters.py**
- Sauvegarde/chargement identiques (égalité d’état), append métriques et logs.

## 11.2 Intégration (Flappy)
- `env_core` : collisions (oiseaux/tuyaux/sol/plafond), génération tuyaux aléatoire avec seed.
- Normalisation des observations (bornes ∈ [0,1] si activée).
- `adapter` : mapping obs→réseau (dimension correcte), réseau→action (seuil 0.5).
- CLI `train` : lancement avec configs exemple, production des fichiers attendus.

## 11.3 End-to-End (E2E)
- Entraînement 5–10 générations **headless** sur petite population : pipeline complet, fichiers sortis, métriques plausibles.
- `play` d’un `champion.pkl` : exécution visuelle OK, pas de crash.

## 11.4 Performance & Robustesse
- **Bench** headless : mesurer steps/s pour N=1,2,4,8 workers; vérifier scalabilité sublinéaire raisonnable.
- Tests de non-régression perf (budgets temps par génération).
- **Fuzz** : variations extrêmes des hyperparamètres (taux mutation hauts/bas), pas d’exception, pas d’innombrables cycles.

## 11.5 Qualité code
- Linting `ruff`/`flake8`, typage `mypy`, couverture `pytest --cov` ≥ 85% sur `neatlab/`.

---

# 12. Critères d’acceptation
- **Réutilisabilité** : intégrer un **nouveau jeu Pygame** ne demande que :
  1) un `env_core.py` conforme (reset/step),
  2) un `adapter.py` (deux transforms),
  3) des configs YAML; **aucune** modification dans `neatlab/`.
- **Flappy** : un champion franchit ≥ 10 tuyaux en visual (60 FPS), et l’entraînement par défaut atteint `fitness ≥ 50` en ≤ 250 générations.
- **Perf** : ≥ 10k steps/s en headless avec 8 workers (machine de réf. à définir).
- **Robustesse** : reprise depuis `--resume` sans divergence (même seed → mêmes résultats).

---

# 13. Guide d’implémentation par étapes
1) **Core NEAT** : innovations, gènes, genome ops + tests unitaires.
2) **Network** : build topo + exécution + activations + tests.
3) **Species/reproduction** : distance, spéciation, allocation, crossover + tests.
4) **Population** : boucle d’évolution, stagnation, reporters/persistance + tests.
5) **Evaluators** sync/parallel + seeds + tests.
6) **Flappy env** (core + headless) + adapter + tests intégration.
7) **CLI** + configs YAML + E2E smoke test.
8) **Perf & tuning** + plots + documentation.

---

# 14. Normes & bonnes pratiques
- **Determinisme** : pas de RNG globale implicite; utiliser `rng.py` avec générateurs dédiés (global, par génération, par worker).
- **Multiprocessing** : protéger l’entrée `if __name__ == "__main__":` (Windows).
- **Pygame** : isoler l’initialisation vidéo dans `env_pygame` uniquement; ne jamais l’appeler en headless.
- **Config-driven** : toute constante tunable doit être dans YAML, chargée en dataclass.
- **Logs structurés** : JSONLines optionnel pour ingestion ultérieure.

---

# 15. Sécurité & dette technique
- Validation stricte des configs (pydantic ou schéma maison) pour éviter les états incohérents.
- Gestion des erreurs workers (timeouts, re-try limité, mise en quarantaine d’un génome fautif).
- Taille des checkpoints bornée (compression pickle/gzip optionnelle).

---

# 16. Annexes – Pseudocode clefs
**Crossover (simplifié)**
```python
def crossover(more_fit: Genome, less_fit: Genome, rng) -> Genome:
    child = Genome.empty_like(more_fit)
    map_b = {c.innovation: c for c in less_fit.conns.values()}
    for innov, ca in more_fit.conns.items():
        if innov in map_b:
            cb = map_b[innov]
            base = rng.choice([ca, cb])  # ou moyenne
            enabled = (ca.enabled and cb.enabled) or rng.random() > 0.25
            child.add_conn(base.copy(enabled=enabled))
        else:
            child.add_conn(ca.copy())  # excess/disjoint
    for nid in child.nodes_from_conns():
        child.nodes[nid] = (more_fit.nodes.get(nid) or less_fit.nodes.get(nid)).copy()
    return child
```
**Mutation add_node**
```python
def mutate_add_node(genome: Genome, innovs: InnovationTracker, rng):
    edges = [c for c in genome.conns.values() if c.enabled]
    if not edges: return False
    edge = rng.choice(edges)
    edge.enabled = False
    new_id = genome.add_hidden_node()
    in_innov  = innovs.innovation(edge.in_id, new_id)
    out_innov = innovs.innovation(new_id, edge.out_id)
    genome.add_conn(ConnGene(edge.in_id, new_id, weight=1.0, enabled=True, innovation=in_innov))
    genome.add_conn(ConnGene(new_id, edge.out_id, weight=edge.weight, enabled=True, innovation=out_innov))
    return True
```

---

# 17. Livrables attendus
- **Code source** `neatlab/` + `games/flappy/` conforme à la présente spec.
- **Jeu de tests** complet (`tests/unit`, `tests/integration`, `tests/e2e`) + rapports de couverture.
- **Jeux de configs YAML** (neat/env/run) prêts à l’emploi.
- **Documentation** : README (usage + exemples), commentaires détaillés, docstrings.
- **Run d’exemple** : un dossier `runs/...` pré-populé (quelques générations) à des fins de validation.

---

# 18. Roadmap (indicative)
- Semaine 1 : Core NEAT + tests unitaires (innovations, genome, network).
- Semaine 2 : species/reproduction/population + persistence + reporters + tests.
- Semaine 3 : Evaluators + Flappy env + adapter + intégration/CLI.
- Semaine 4 : E2E, tuning, perfs, docs, packaging.

---

*Fin de la spécification.*

