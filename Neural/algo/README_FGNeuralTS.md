# Feel-Good Neural Thompson Sampling (FGNeuralTS)

This module implements Feel-Good Neural Thompson Sampling (FGNeuralTS) and Smoothed Feel-Good Neural Thompson Sampling (SFGNeuralTS) as extensions to the standard Neural Thompson Sampling algorithm.

| Aspect | FGLMCTS | FGNeuralTS |
|--------|---------|------------|
| Base Algorithm | LMCTS | NeuralTS |
| Model Type | Linear | Neural Network |
| Exploration | Langevin MC | Thompson Sampling |
| Design Matrix | Not used | Diagonal approximation |

# Sarah's note to herself