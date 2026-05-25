# Narrative and Media Layer

The narrative layer should be treated as an experiment, not a hand-waved model boost.

## Candidate Signals

- `media_mentions_count`: how often a player is mentioned in approved media sources before voting.
- `search_trend_score`: normalized Google/search interest.
- `reddit_mentions_count`: discussion volume in relevant NBA communities.
- `sentiment_score`: normalized positive/negative tone.
- `team_surprise_score`: whether the player's team exceeded expectation.
- `breakout_score`: whether this is a first major star-level leap.
- `prior_mvp_fatigue`: whether voters may discount a repeat winner.

## Modeling Rule

Avoid data leakage. A prediction should only use narrative data available before the prediction date or before MVP voting.

## Evaluation

Run an ablation:

1. Train `basketball_only`.
2. Train `basketball_plus_narrative`.
3. Compare top-5 average precision, winner accuracy, top-3 recall, MSE, and winner predicted rank.

If narrative features improve historical backtests, include them in the main model. If they do not, keep them as a dashboard-only context layer.

## Template

Create a starter CSV:

```bash
python -m ml.narrative
```

The template is saved to `data/processed/narrative_features.csv`.

Run the ablation once the CSV has real rows:

```bash
python -m ml.narrative_experiment
```

The result is saved to `data/processed/narrative_experiment.json`.
