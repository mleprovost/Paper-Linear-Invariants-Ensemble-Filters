
*A Julia package for preserving linear invariants in ensemble filtering methods*

This repository is a companion to the work [^1]: Le Provost, Glaubitz, and Marzouk (2024) "Preserving linear invariants in ensemble filters for data assimilation", under preparation.


Formulating dynamical models for physical phenomena is essential for understanding the interplay between the different mechanisms, predicting the evolution of physical states, and developing effective control strategies. However, a dynamical model alone is often insufficient to address these fundamental tasks, as it suffers from model errors and uncertainties. One common remedy is to rely on data assimilation, where the state estimate is updated with observations of the true system. Ensemble filters sequentially assimilate observations by updating a set of samples over time. They operate in two steps: a forecast step that propagates each sample through the dynamical model and an analysis step that updates the samples with incoming observations. For accurate and robust predictions of dynamical systems, discrete solutions must preserve their critical invariants. 
While modern numerical solvers satisfy these invariants, existing invariant-preserving analysis steps are limited to Gaussian settings and are often not compatible with classical regularization techniques of ensemble filters, e.g., inflation and covariance tapering. The present work focuses on preserving linear invariants, such as mass, stoichiometric balance of chemical species, and electrical charges. Using tools from measure transport theory (Spantini et al., 2022, SIAM Review), we introduce a generic class of nonlinear ensemble filters that automatically preserve desired linear invariants in non-Gaussian filtering problems. By specializing this framework to the Gaussian setting, we recover a constrained formulation of the Kalman filter. Then, we show how to combine existing regularization techniques for the ensemble Kalman filter (Evensen, 1994, J. Geophys. Res.) with the preservation of the linear invariants. Finally, we investigate the influence of the number of linear invariants on the performance of the unconstrained/constrained ensemble Kalman filter.

This repository contains the source code and Jupyter notebooks to reproduce the numerical experiment in Le Provost et al. [^1]


## Installation

This package works on Julia `1.6` and above. To install from the REPL, type
e.g.,
```julia
] add https://github.com/mleprovost/Paper-Linear-Invariants-Ensemble-Filters.git
```

## Correspondence email
[mleprovo@mit.edu](mailto:mleprovo@mit.edu)

## References

[^1]: Le Provost, Glaubitz, and Marzouk (2024) "Preserving linear invariants in ensemble filters for data assimilation," under preparation.

## Licence

See [LICENSE.md](https://github.com/mleprovost/Paper-Linear-Invariants-Ensemble-Filters/raw/main/LICENSE.md)

