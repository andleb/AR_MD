# AR_MD

Molecular dynamics trajectory prediction and metastable-state classification with autoregressive architectures. Final project for *STATS 700: LLMs and Transformers* (University of Michigan, Fall 2024).

- **Report:** [Modern autoregressive architectures for the collective variable problem](docs/leban_andrej-Modern%20autoregressive%20architectures%20for%20the%20collective%20variable%20problem.pdf)
- **Poster:** [STATS700LLMPoster.pdf](docs/STATS700LLMPoster.pdf)

## Summary

A BERT-style Transformer encoder is adapted to continuous trajectories (a linear projection replaces the token embedding) and trained with two heads on shared embeddings: next-step prediction and classification of the trajectory's metastable state. On Langevin simulations of the Müller-Brown potential, the medium and large models generate physically plausible trajectories and classify states with up to 99% accuracy. The attention patterns reveal a learned coarse time discretization - a few key timesteps per segment, fewer for larger models. See the report for details and limitations.

## Layout

- `src/` - model (`modelling_md.py`), plotting, utilities
- `exp/` - training script (`MB_train.py`) and analysis notebook (`MB-Bert-Analysis.ipynb`)
- `data/` - simulated Müller-Brown trajectories (`.npz`)
- `docs/` - report and poster

## Dependencies

- The architecture uses [foldingdiff](https://github.com/microsoft/foldingdiff) as a starting point.
- Requires [transformers](https://github.com/huggingface/transformers).
- Uses some plotting functions from [mlcolvar](https://github.com/luigibonati/mlcolvar).

If you use this software, please cite it by the provided [citation file](CITATION.cff).
