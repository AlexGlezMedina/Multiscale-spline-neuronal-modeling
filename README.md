# Data-driven multiscale spline-based modeling of neuronal dynamics

This repository contains the MATLAB implementations used for the numerical experiments reported in the manuscript:

**Data-driven multiscale spline-based modeling of neuronal dynamics under partial observation: forward prediction and inverse current reconstruction**

The code implements forward and inverse empirical modeling for the **Morris–Lecar (ML)** and **FitzHugh–Nagumo–Rinzel (FHN–R)** neuronal models under partial state observation. In all cases, the membrane-potential trace is the only observed state variable supplied to the identified model. Hidden state variables are retained only for reference simulation and diagnostic purposes.

The proposed empirical model uses a fixed library of localized functions generated from a compactly supported quadratic spline prototype and locally normalized on nested grids. The nonlinear library is reduced from the training data using activation-energy and dynamic-activity criteria. Once the reduced library is fixed, the model is linear in the unknown coefficients and is identified by column-scaled SVD with Tikhonov regularization.

---

## Repository contents

The four main scripts are:

| Script | Neuronal model | Identification task | Regressors | Target |
|---|---|---|---|---|
| `morris_lecar_direct.m` | Morris–Lecar | Forward | delayed voltage samples + known current | next voltage sample |
| `morris_lecar_inverse.m` | Morris–Lecar | Inverse | delayed voltage samples only | applied current |
| `fitzhugh_nagumo_rinzel_direct.m` | FitzHugh–Nagumo–Rinzel | Forward | delayed voltage samples + known current | next voltage sample |
| `fitzhugh_nagumo_rinzel_inverse.m` | FitzHugh–Nagumo–Rinzel | Inverse | delayed voltage samples only | applied current |

Each file is self-contained: the auxiliary functions required by a given experiment are included at the end of the corresponding `.m` file. No external data files are required because the training and validation trajectories are generated numerically inside each script.

---

## Software and computational environment

The experiments reported in the manuscript were implemented in **MATLAB R2024a**.

The calculations reported in the study were performed on a 64-bit Windows 10 computer with:

- Intel Core i7-4790 CPU at 3.60 GHz
- 16 GB RAM
- no GPU acceleration
- no parallel computing

The scripts use standard MATLAB functionality for numerical integration, matrix operations, SVD, tables, plotting, file export, and data storage.

Because the inverse problems, particularly the FHN–R case, generate relatively large candidate libraries and regression matrices, sufficient RAM and free disk space are recommended. Execution time depends strongly on the computer used.

---

## How to run the code

1. Download or clone this repository.
2. Open MATLAB.
3. Set the current folder to the repository directory.
4. Run one of the four main scripts, for example:

```matlab
morris_lecar_direct
```

or

```matlab
fitzhugh_nagumo_rinzel_inverse
```

No input arguments are required.

Each execution creates a time-stamped directory beside the script, for example:

```text
morris_lecar_direct_output_YYYYMMDD_HHMMSS/
```

with a subdirectory:

```text
figures/
```

The scripts are independent and may be run separately.

---

## Reproducibility

The excitation signals are generated using fixed random-number seeds:

```matlab
cfg.randomSeedTraining   = 1;
cfg.randomSeedEvaluation = 2;
```

Therefore, runs performed with the same MATLAB implementation and unchanged configuration should reproduce the same training and validation excitation sequences.

In the code, variables and structures named `evaluation` correspond to the **validation data set** discussed in the manuscript.

The neuronal trajectories are integrated using a fixed-step fourth-order Runge–Kutta method.

---

## Common library and identification settings

The four experiments use the same general multiscale-library configuration:

```matlab
cfg.levels = [0 1];
cfg.boundaryPadding = 0;
cfg.energyFractionTarget = 0.97;
cfg.dynamicTopFraction = 0.02;
```

The Tikhonov parameter is:

```matlab
mu = 1e-16;
```

with the corresponding field named `cfg.muDirect` or `cfg.muInverse`.

The candidate nonlinear library is constructed on the normalized regressor domain. For each level \(r\),

\[
h_r = 2^{-r},
\qquad
m_r = 2^r+1,
\]

and the univariate functions are generated from a quadratic spline prototype, followed by local normalization so that the functions at each level form a partition of unity on the normalized domain.

Multivariate nonlinear functions are generated through tensor products of functions belonging to the same scale level.

Before coefficient estimation, candidate functions are reduced using two data-dependent scores:

1. **Activation-energy score:** favors functions that are persistently activated along the training trajectory.
2. **Dynamic-activity score:** gives additional relevance to functions activated in regions where the voltage exhibits pronounced local temporal variation.

The final nonlinear library is the union of the functions retained by both criteria.

The regression matrix also contains an intercept and the normalized linear input coordinates. Its columns are scaled to unit Euclidean norm before computing the SVD-based Tikhonov solution.

---

## Forward identification

The direct scripts identify the mapping

\[
[v(k-d_1),\ldots,v(k-d_m),I(k)]
\longrightarrow
v(k+1).
\]

The hidden neuronal states are **not** used as regressors.

Two prediction modes are evaluated.

### One-step prediction

Every prediction is formed from the reference voltage history. Errors are therefore not recursively propagated.

### Recursive free-run prediction

A measured voltage prefix is used only to initialize the delay coordinates. After the first predicted sample, estimated voltages are recursively fed back into the model, while the applied current remains known.

A persistence model,

\[
v(k+1)\approx v(k),
\]

is also evaluated as a baseline for interpreting the very high one-step accuracy obtained at the adopted sampling resolution.

---

## Inverse identification

The inverse scripts identify the mapping

\[
[v(k+1-d_1),\ldots,v(k+1-d_m)]
\longrightarrow
I(k).
\]

The temporal alignment follows the discrete transition produced by \(I(k)\): the input acts during the evolution from \(v(k)\) to \(v(k+1)\), so the inverse embedding is aligned with the endpoint of this transition.

Only voltage samples are supplied as regressors to the identified inverse model. During supervised identification, the corresponding applied-current samples are used as training targets.

After current reconstruction, the neuronal model is reintegrated using the estimated current. The short current prefix for which a complete delay window is not yet available is taken from the reference trajectory only for initialization and is excluded from the reported inverse-current evaluation.

---

## Voltage embeddings

### Morris–Lecar forward

```text
delays [samples] = [0 4 128 512]
delays [ms]      = [0 0.2 6.4 25.6]
```

Four voltage coordinates are used together with the known current, giving a total input dimension of \(N_z=5\).

### Morris–Lecar inverse

```text
delays [samples] = [0 4 8 16 32 64 128 256]
delays [ms]      = [0 0.2 0.4 0.8 1.6 3.2 6.4 12.8]
```

Eight voltage coordinates are used and no current or hidden state is included among the regressors.

### FHN–R forward

```text
delays [samples] = [0 32 128 2048 4096]
delays [model t.u.] = [0 1.6 6.4 102.4 204.8]
```

Five voltage coordinates are used together with the known current, giving \(N_z=6\).

### FHN–R inverse

```text
delays [samples] = [0 8 32 128 512 1024 2048 4096]
delays [model t.u.] = [0 0.4 1.6 6.4 25.6 51.2 102.4 204.8]
```

Eight voltage coordinates are used.

---

## Important note on the FHN–R time convention

The Morris–Lecar and FHN–R equations do **not** have the same physical interpretation of time.

For Morris–Lecar,

\[
\Delta t=0.05~\mathrm{ms},
\]

which corresponds to a sampling frequency of 20 kHz.

The FHN–R model is dimensionless. For that model, the numerical integration step is

\[
\Delta \tau=0.05
\]

in **model time units (t.u.)**, and no physical sampling frequency is assigned to the model.

The current FHN–R scripts retain some variable names such as

```matlab
fsData_Hz
Ts_s
Ts_ms
tTrain_s
tEval_s
```

for historical and implementation consistency with the Morris–Lecar scripts. In the FHN–R code these names must be interpreted as **computational bookkeeping variables, not physical SI units**.

The actual quantity passed to the FHN–R Runge–Kutta integrator is numerically equal to `0.05`, which is the intended dimensionless integration step. Consequently, the simulated trajectories and all sample-based operations use the correct numerical time discretization.

The relation between the legacy display/bookkeeping coordinate and FHN–R model time is

\[
\tau_{\mathrm{FHN-R}} = 1000\,t_{\mathrm{legacy}},
\]

where `t_legacy` is the quantity stored in variables ending in `_s`.

Therefore:

| Quantity in current FHN–R scripts | Interpretation in model time |
|---|---:|
| `cfg.integrationStep_ms = 0.05` | \(0.05\) t.u. |
| `cfg.trainingDuration_s = 10` | \(10000\) t.u. |
| `cfg.evaluationDuration_s = 5` | \(5000\) t.u. |
| `cfg.input.scanDuration_s = 1.5` | \(1500\) t.u. |
| `cfg.input.samplesPerLevelTraining = 4000` | \(200\) t.u. |
| `cfg.input.samplesPerLevelEvaluation = 5000` | \(250\) t.u. |
| `cfg.spikeMatchTolerance_s = 0.010` | \(10\) t.u. |
| `cfg.lag.maxLag_s = 0.20` | \(200\) t.u. |
| `cfg.figures.zoomWindow_s = 0.75` | \(750\) t.u. |

Because `cfg.Ts_ms = 0.05`, values stored in variables such as `delaySet_ms` happen to be **numerically identical** to the corresponding delays expressed in FHN–R model time units. The suffix `_ms` should nevertheless not be interpreted as a physical millisecond unit in the FHN–R experiments.

Similarly, any FHN–R figure or diagnostic produced with a time axis internally labeled in seconds should be understood as using this legacy rescaled coordinate. To express that axis in the model-time convention used in the manuscript, multiply it by 1000.

This bookkeeping convention was retained only to preserve the common script structure and the exact sample counts used in the reported experiments. It does **not** imply that FHN–R time has been assigned a physical unit or that FHN–R and Morris–Lecar have physically equivalent temporal scales.

---

## Data generation

Training and validation excitation signals are piecewise constant.

For each model:

1. a constant-current scan is performed to numerically locate the transition toward oscillatory behavior;
2. a training-current interval is constructed around that transition;
3. the validation interval is chosen inside the training interval;
4. uniformly spaced current levels are defined over each interval;
5. the order of the constant-current blocks is randomized.

The training and validation records contain different randomized excitation sequences.

The simulations use:

```text
Training samples:   200001
Validation samples: 100001
```

For Morris–Lecar these correspond to 10 s and 5 s, respectively.

For FHN–R they correspond to 10000 and 5000 model time units, respectively.

---

## Main outputs

Each run stores a detailed set of results in its time-stamped output directory.

Typical files include:

```text
execution_log.txt
run_summary.txt
metrics_summary.csv
selected_library.csv
singular_values.csv
library_by_level.csv
*_full_results.mat
figures/
```

The direct scripts also export:

```text
free_run_spike_times.csv
```

and the inverse scripts export:

```text
spike_times.csv
```

The `.mat` file contains the main configuration, model parameters, training and validation trajectories, identified model, selected library, diagnostics, metrics, timing information, and other quantities used during the experiment.

The `figures/` directory contains high-resolution PNG figures and, when enabled in the configuration, MATLAB `.fig` files.

---

## Reported diagnostics

Depending on the experiment, the scripts report or store:

- \(100(1-r^2)\)
- \(r^2\)
- RMSE
- MAE
- relative \(L_2\) error
- normalized RMSE
- cosine similarity
- Pearson correlation
- persistence-baseline error for forward modeling
- free-run performance
- spike precision, recall, and F1 score
- spike-timing error
- lag at maximum voltage correlation
- fraction of validation regressors inside the training normalization domain
- clipping diagnostics
- candidate and selected library sizes
- functions selected by energy and dynamic criteria
- numerical SVD rank
- effective condition number
- singular spectrum and Tikhonov filter factors
- estimated memory requirements

For inverse identification, the reconstructed current is additionally assessed indirectly by reintegrating the original neuronal equations with the estimated input and comparing the resulting voltage trajectory with the reference simulation.

---

## Notes on interpretation

The validation current interval is contained within the training interval. The reported validation therefore primarily evaluates interpolation/generalization to a different excitation realization inside the trained operating region rather than extrapolation to unseen current amplitudes.

The internal states of the neuronal models are available in the simulation code because they are required to generate the reference trajectories. They are not supplied as regressors to the identified empirical models.

The normalized nonlinear functions are generated from a quadratic spline prototype but, after local normalization on the finite domain, they should be interpreted as **spline-derived localized normalized functions**, rather than polynomial splines in the strict mathematical sense.

The Tikhonov parameter used in these experiments is very small (`1e-16`) and acts primarily as a numerical stabilization of weak singular directions after column scaling.

---

## Citation

If you use this code in academic work, please cite the associated manuscript.

---

## Contact

For questions about the implementation or reproducibility of the experiments, please contact the corresponding author listed in the manuscript.
