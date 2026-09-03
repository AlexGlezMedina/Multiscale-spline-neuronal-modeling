%% =========================================================================
%  morris_lecar_inverse.m
%  INVERSE IDENTIFICATION
%     [v(k+1-d_1), ..., v(k+1-d_m)]  --->  I(k)
%  PURPOSE
%  -------
%  This script reconstructs the applied current of the Morris--Lecar model
%  from a single voltage trace. Only delayed voltage samples are used as
%  regressors; the internal recovery variable w(t) is not supplied to the
%  inverse model.
%  TEMPORAL ALIGNMENT
%  ------------------
%  I(k) acts during the transition from v(k) to v(k+1). For this reason the
%  inverse regressor is aligned with the endpoint of that transition and is
%  written as xi(k+1) -> I(k). This convention is used consistently in the
%  training, evaluation, and exported index sets.
%
%  VOLTAGE EMBEDDING
%  -----------------
%  The inverse model uses eight nonuniform voltage coordinates:
%
%       delays = [0 4 8 16 32 64 128 256] samples
%              = [0 0.2 0.4 0.8 1.6 3.2 6.4 12.8] ms
%
%  No current value or hidden state is included among the regressors. The
%  delayed voltage coordinates provide the empirical state representation
%  from which the applied current is reconstructed.
%
%  MAIN FEATURES
%  -------------
%  - Morris--Lecar parameters are those used in Table 1 of the manuscript.
%  - A nonuniform voltage embedding is used to represent the unobserved state.
%  - The nonlinear library is built from localized functions generated from a
%    quadratic spline prototype and normalized locally on nested grids at
%    levels r=[0 1].
%  - Candidate functions are reduced using energy and dynamic activity scores.
%  - The final regression is linear in the unknown coefficients and is solved
%    by SVD with Tikhonov regularization after column scaling.
%  - Current reconstruction is also checked by reintegrating the neuronal model
%    with the estimated input.
%  - Numerical diagnostics, selected functions, metrics, and figures are saved
%    in a time-stamped output directory.
%% =========================================================================

clear; clc; close all;

overallTimer = tic;
rng(1, 'twister');

[scriptDir, scriptBase] = ...
    get_script_location('morris_lecar_inverse_multiscale_spline_full_domain');

timeStamp = datestr(now, 'yyyymmdd_HHMMSS');
outputDir = fullfile(scriptDir, [scriptBase '_output_' timeStamp]);
figureDir = fullfile(outputDir, 'figures');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
if ~exist(figureDir, 'dir')
    mkdir(figureDir);
end

diary(fullfile(outputDir, 'execution_log.txt'));
diaryCleanup = onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('============================================================\n');
fprintf('MORRIS--LECAR: INVERSE MULTISCALE SPLINE IDENTIFICATION\n');
fprintf('============================================================\n');
fprintf('Output directory:\n%s\n', outputDir);

%% =========================================================================
%% 1) CONFIGURATION
%% =========================================================================

cfg.randomSeedTraining = 1;
cfg.randomSeedEvaluation = 2;

cfg.fsData_Hz = 20000;
cfg.Ts_s = 1/cfg.fsData_Hz;
cfg.Ts_ms = 1000/cfg.fsData_Hz;
cfg.integrationStep_ms = cfg.Ts_ms;

cfg.trainingDuration_s = 10.0;
cfg.evaluationDuration_s = 5.0;

cfg.Ntrain = round(cfg.trainingDuration_s/cfg.Ts_s) + 1;
cfg.Neval = round(cfg.evaluationDuration_s/cfg.Ts_s) + 1;

cfg.tTrain_s = (0:cfg.Ntrain-1)' * cfg.Ts_s;
cfg.tEval_s = (0:cfg.Neval-1)' * cfg.Ts_s;

cfg.delaySet_samples = [0 4 8 16 32 64 128 256];
cfg.delaySet_ms = cfg.delaySet_samples * cfg.Ts_ms;

cfg.numberVoltageCoordinates = numel(cfg.delaySet_samples);
cfg.totalInputDimension = cfg.numberVoltageCoordinates;

cfg.levels = [0 1];
cfg.boundaryPadding = 0;
cfg.energyFractionTarget = 0.97;
cfg.dynamicTopFraction = 0.02;
cfg.muInverse = 1e-16;

cfg.input.coarseScanMin = 0;
cfg.input.coarseScanMax = 100;
cfg.input.coarseScanCount = 81;
cfg.input.refinedScanCount = 81;
cfg.input.scanDuration_s = 1.5;
cfg.input.scanSteadyFraction = 0.60;
cfg.input.oscillationAmplitudeThreshold = 5.0;

cfg.input.allowedMin = 0;
cfg.input.allowedMax = 100;
cfg.input.trainHalfWidth = 12;
cfg.input.evalHalfWidth = 8;

% An odd number of input levels includes the interval center exactly
% (the estimated transition when the interval is symmetric).
cfg.input.numLevelsTraining = 51;
cfg.input.numLevelsEvaluation = 31;
cfg.input.samplesPerLevelTraining = 800;
cfg.input.samplesPerLevelEvaluation = 1000;

cfg.spikeThreshold = 0;
cfg.spikeMatchTolerance_s = 0.005;

cfg.lag.maxLag_s = 0.10;
cfg.lag.downsampleFactor = 20;

cfg.figures.resolutionDpi = 600;
cfg.figures.saveMatlabFigures = true;
cfg.figures.showFigures = false;
cfg.figures.zoomWindow_s = 0.50;

fprintf('\nVoltage embedding [samples]: [%s]\n', ...
    num2str(cfg.delaySet_samples));
fprintf('Voltage embedding [ms]:      [%s]\n', ...
    num2str(cfg.delaySet_ms));
fprintf('Total input dimension:        %d (voltage only)\n', ...
    cfg.totalInputDimension);

%% =========================================================================
%% 2) MORRIS--LECAR PARAMETERS: TABLE 1
%% =========================================================================

p.Cm = 20;
p.phi = 1/15;

p.EL = -60;
p.EK = -84;
p.ECa = 120;

p.V1 = -1.2;
p.V2 = 18;
p.V3 = 12;
p.V4 = 17.4;

p.gL = 2;
p.gK = 8;
p.gCa = 4;

parameterUnits.Cm = 'microF/cm^2';
parameterUnits.phi = 'dimensionless';
parameterUnits.potentials = 'mV';
parameterUnits.conductances = 'mS/cm^2';
parameterUnits.current = 'microA/cm^2';

v0 = -60;
w0 = w_inf(v0, p);
X0 = [v0; w0];

%% =========================================================================
%% 3) OSCILLATORY-TRANSITION SCAN AND ESTIMATION
%% =========================================================================

scanTimer = tic;

coarseCurrents = linspace( ...
    cfg.input.coarseScanMin, ...
    cfg.input.coarseScanMax, ...
    cfg.input.coarseScanCount);

scanCoarse = scan_morris_lecar_currents( ...
    X0, coarseCurrents, cfg.input.scanDuration_s, ...
    cfg.fsData_Hz, cfg.integrationStep_ms, p, ...
    cfg.input.scanSteadyFraction, cfg.spikeThreshold);

transitionCoarse = estimate_transition_current( ...
    scanCoarse.current, scanCoarse.amplitude, ...
    cfg.input.oscillationAmplitudeThreshold);

if isfinite(transitionCoarse)
    coarseStep = coarseCurrents(2) - coarseCurrents(1);
    refinedMin = max( ...
        cfg.input.coarseScanMin, transitionCoarse - coarseStep);
    refinedMax = min( ...
        cfg.input.coarseScanMax, transitionCoarse + coarseStep);

    refinedCurrents = linspace( ...
        refinedMin, refinedMax, cfg.input.refinedScanCount);

    scanRefined = scan_morris_lecar_currents( ...
        X0, refinedCurrents, cfg.input.scanDuration_s, ...
        cfg.fsData_Hz, cfg.integrationStep_ms, p, ...
        cfg.input.scanSteadyFraction, cfg.spikeThreshold);

    transitionCurrent = estimate_transition_current( ...
        scanRefined.current, scanRefined.amplitude, ...
        cfg.input.oscillationAmplitudeThreshold);
else
    scanRefined.current = zeros(0,1);
    scanRefined.vMin = zeros(0,1);
    scanRefined.vMax = zeros(0,1);
    scanRefined.amplitude = zeros(0,1);
    scanRefined.spikeCount = zeros(0,1);
    transitionCurrent = 39.9632;
end

if ~isfinite(transitionCurrent)
    transitionCurrent = transitionCoarse;
end
if ~isfinite(transitionCurrent)
    transitionCurrent = 39.9632;
end

timing.scan_s = toc(scanTimer);

[trainIntervals, evalIntervals] = ...
    build_intervals_around_transition( ...
        transitionCurrent, ...
        cfg.input.allowedMin, cfg.input.allowedMax, ...
        cfg.input.trainHalfWidth, cfg.input.evalHalfWidth);

fprintf('\nEstimated oscillatory transition: I = %.6f\n', ...
    transitionCurrent);
fprintf('Training interval(s):\n');
disp(trainIntervals);
fprintf('Evaluation interval(s):\n');
disp(evalIntervals);

%% =========================================================================
%% 4) TRAINING-SET GENERATION AND SIMULATION
%% =========================================================================

rng(cfg.randomSeedTraining, 'twister');

Itrain = generate_piecewise_uniform_input_intervals( ...
    cfg.Ntrain, cfg.input.numLevelsTraining, trainIntervals, ...
    cfg.input.samplesPerLevelTraining);

trainingSimulationTimer = tic;
Xtrain = simulate_morris_lecar( ...
    X0, Itrain, cfg.integrationStep_ms, p);
timing.trainingSimulation_s = toc(trainingSimulationTimer);

vTrain = Xtrain(:,1);
wTrain = Xtrain(:,2);

[Ztrain, YtrainCurrent, inputIdxTrain, targetIdxTrain] = ...
    build_dataset_inverse(vTrain, Itrain, cfg.delaySet_samples);

[omegaTrainRaw, omegaTrainNormalized, omegaTrainMax] = ...
    build_dynamic_weights_from_signal(vTrain, inputIdxTrain);

%% =========================================================================
%% 5) INVERSE-MODEL TRAINING
%% =========================================================================

fprintf('\n============================================================\n');
fprintf('INVERSE-MODEL TRAINING\n');
fprintf('============================================================\n');

trainingTimer = tic;
modelInverse = train_scalar_multiscale_spline_dualfilter( ...
    Ztrain, YtrainCurrent, omegaTrainNormalized, ...
    cfg.levels, cfg.boundaryPadding, ...
    cfg.energyFractionTarget, cfg.dynamicTopFraction, ...
    cfg.muInverse);
timing.modelTraining_s = toc(trainingTimer);

IhatTrainValid = eval_model(modelInverse, Ztrain);
metrics.currentTraining = ...
    compute_error_metrics(YtrainCurrent, IhatTrainValid);

%% =========================================================================
%% 6) EVALUATION-SET GENERATION AND SIMULATION
%% =========================================================================

rng(cfg.randomSeedEvaluation, 'twister');

Ieval = generate_piecewise_uniform_input_intervals( ...
    cfg.Neval, cfg.input.numLevelsEvaluation, evalIntervals, ...
    cfg.input.samplesPerLevelEvaluation);

evaluationSimulationTimer = tic;
Xeval = simulate_morris_lecar( ...
    X0, Ieval, cfg.integrationStep_ms, p);
timing.evaluationSimulation_s = toc(evaluationSimulationTimer);

vEval = Xeval(:,1);
wEval = Xeval(:,2);

%% =========================================================================
%% 7) CURRENT ESTIMATION AND MODEL REINTEGRATION
%% =========================================================================

inferenceTimer = tic;
[IestEval, inputIdxEval, targetIdxEval] = estimate_inverse_current( ...
    modelInverse, vEval, cfg.delaySet_samples);
timing.inverseInference_s = toc(inferenceTimer);

currentEvalRange = targetIdxEval(:).';

IevalValid = Ieval(currentEvalRange);
IestEvalValid = IestEval(currentEvalRange);

metrics.currentEvaluation = ...
    compute_error_metrics(IevalValid, IestEvalValid);

% The inverse estimate is not available until the delay window is filled.
% For reintegration, the unavailable current prefix is therefore taken from
% the reference input only to initialize the state history. This initialization
% interval is not part of the inverse-current evaluation. From the first valid
% inverse estimate onward, the neuronal model is driven exclusively by the
% reconstructed current.
firstEstimatedCurrentIndex = targetIdxEval(1);
firstStateDrivenByEstimate = firstEstimatedCurrentIndex + 1;

% The state-evaluation range follows directly from the first current sample
% that is actually reconstructed by the inverse model.
stateEvalRange = firstStateDrivenByEstimate:cfg.Neval;

Ireintegrated = IestEval;
if firstEstimatedCurrentIndex > 1
    Ireintegrated(1:firstEstimatedCurrentIndex-1) = ...
        Ieval(1:firstEstimatedCurrentIndex-1);
end

% I(N) is not used by the discrete integration loop. A finite value is stored
% only to keep the reintegration input vector complete.
if ~isfinite(Ireintegrated(end))
    Ireintegrated(end) = IestEval(targetIdxEval(end));
end

reinjectionTimer = tic;
Xreconstructed = simulate_morris_lecar( ...
    X0, Ireintegrated, cfg.integrationStep_ms, p);
timing.reinjectionSimulation_s = toc(reinjectionTimer);

vReconstructed = Xreconstructed(:,1);
wReconstructed = Xreconstructed(:,2);

metrics.voltageEvaluation = compute_error_metrics( ...
    vEval(stateEvalRange), vReconstructed(stateEvalRange));
metrics.hiddenStateWEvaluation = compute_error_metrics( ...
    wEval(stateEvalRange), wReconstructed(stateEvalRange));

lagMetrics.voltage = compute_lag_metrics( ...
    vEval(stateEvalRange), vReconstructed(stateEvalRange), ...
    cfg.fsData_Hz, cfg.lag.maxLag_s, ...
    cfg.lag.downsampleFactor);

%% =========================================================================
%% 8) DOMAIN, SPIKE, LIBRARY, AND MEMORY DIAGNOSTICS
%% =========================================================================

[Zeval, inputIdxEvalDomain, targetIdxEvalDomain] = ...
    build_inverse_regressors_from_signal(vEval, cfg.delaySet_samples);
[Zreconstructed, inputIdxReconstructed, targetIdxReconstructed] = ...
    build_inverse_regressors_from_signal(vReconstructed, cfg.delaySet_samples);

domain.evaluation = check_domain_fraction(modelInverse, Zeval);
domain.reconstructed = ...
    check_domain_fraction(modelInverse, Zreconstructed);

domain.voltageReconstructedInsideTrainingRange = mean( ...
    vReconstructed >= min(vTrain) & ...
    vReconstructed <= max(vTrain));

spike = compute_spike_metrics( ...
    vEval(stateEvalRange), vReconstructed(stateEvalRange), ...
    cfg.tEval_s(stateEvalRange), ...
    cfg.spikeThreshold, cfg.spikeMatchTolerance_s);

library.totalCandidateFunctions = modelInverse.totalFunctions;
library.activeEnergyFunctions = numel(modelInverse.activeIdxE);
library.activeDynamicFunctions = numel(modelInverse.activeIdxD);
library.selectedEnergyFunctions = numel(modelInverse.selectedE);
library.selectedDynamicFunctions = numel(modelInverse.selectedD);
library.selectedDynamicOnlyFunctions = numel(modelInverse.selectedDynamicOnly);
library.selectedUnionFunctions = ...
    numel(modelInverse.selectedGlobalIdx);
library.reductionPercent = modelInverse.reductionPercent;
library.candidatePerLevel = modelInverse.countPerLevel;
library.selectedPerLevel = modelInverse.selectedPerLevel;
library.levels = modelInverse.levels;

memory.modelBytes = estimate_structure_bytes('modelInverse');
memory.estimatedReducedRegressorBytes = ...
    size(Ztrain,1) * modelInverse.Ntheta * 8;
memory.trainingRegressorInputBytes = numel(Ztrain) * 8;
memory.evaluationRegressorInputBytes = numel(Zeval) * 8;
memory.candidateScoreBytes = ...
    2 * modelInverse.totalFunctions * 8;

%% =========================================================================
%% 9) METRIC REPORTING
%% =========================================================================

fprintf('\n============================================================\n');
fprintf('MAIN RESULTS\n');
fprintf('============================================================\n');

print_metric_block( ...
    'Current: training', metrics.currentTraining);
print_metric_block( ...
    'Current: evaluation', metrics.currentEvaluation);
print_metric_block( ...
    'Reintegrated voltage: evaluation', metrics.voltageEvaluation);
print_metric_block( ...
    'Hidden state w: evaluation', metrics.hiddenStateWEvaluation);

fprintf('\nReintegration initialization\n');
fprintf('  Reference current samples used for initialization = %d\n', ...
    firstEstimatedCurrentIndex - 1);
fprintf('  Initialization duration                           = %.6f ms\n', ...
    (firstEstimatedCurrentIndex - 1) * cfg.Ts_ms);
fprintf('  First estimated current sample                    = %d\n', ...
    firstEstimatedCurrentIndex);
fprintf('  First state driven by an estimated current        = %d\n', ...
    firstStateDrivenByEstimate);

fprintf('\nLibrary\n');
fprintf('  Input dimension          = %d\n', modelInverse.NI);
fprintf('  Candidate functions      = %d\n', ...
    library.totalCandidateFunctions);
fprintf('  Energy-active functions  = %d\n', ...
    library.activeEnergyFunctions);
fprintf('  Dynamic-active functions = %d\n', ...
    library.activeDynamicFunctions);
fprintf('  Retained by energy       = %d\n', ...
    library.selectedEnergyFunctions);
fprintf('  Retained by dynamics     = %d\n', ...
    library.selectedDynamicFunctions);
fprintf('  Added only by dynamics   = %d\n', ...
    library.selectedDynamicOnlyFunctions);
fprintf('  Final union              = %d\n', ...
    library.selectedUnionFunctions);
fprintf('  Reduction                = %.4f %%\n', ...
    library.reductionPercent);
fprintf('  Numerical SVD rank       = %d of %d\n', ...
    modelInverse.numericalRank, numel(modelInverse.singularValues));
fprintf('  Effective condition no.  = %.6e\n', ...
    modelInverse.effectiveConditionNumber);

fprintf('\nDomain diagnostics\n');
fprintf('  Evaluation:   regressors inside = %.4f %% | clipping = %.4f %%\n', ...
    100*domain.evaluation.insideAllFraction, ...
    100*domain.evaluation.clippedAnyFraction);
fprintf('  Reintegrated: regressors inside = %.4f %% | clipping = %.4f %%\n', ...
    100*domain.reconstructed.insideAllFraction, ...
    100*domain.reconstructed.clippedAnyFraction);
fprintf('  Reintegrated voltage inside training range = %.4f %%\n', ...
    100*domain.voltageReconstructedInsideTrainingRange);

fprintf('\nSpike diagnostics\n');
fprintf('  Real = %d | reconstructed = %d | matched = %d\n', ...
    spike.nReal, spike.nReconstructed, spike.nMatched);
fprintf('  Precision = %.6f | Recall = %.6f | F1 = %.6f\n', ...
    spike.precision, spike.recall, spike.f1);
fprintf('  Timing RMSE = %.6e s\n', spike.timingRMSESeconds);

fprintf('\nLag at maximum voltage correlation\n');
fprintf('  Maximum correlation = %.8f\n', ...
    lagMetrics.voltage.maxCorrelation);
fprintf('  Lag = %.6e s (%d samples)\n', ...
    lagMetrics.voltage.bestLagSeconds, ...
    lagMetrics.voltage.bestLagSamplesOriginal);

%% =========================================================================
%% 10) EXPORTABLE TABLES
%% =========================================================================

signalNames = { ...
    'current_training', ...
    'current_evaluation', ...
    'voltage_reintegrated_evaluation', ...
    'hidden_state_w_evaluation'};

metricsCell = { ...
    metrics.currentTraining, ...
    metrics.currentEvaluation, ...
    metrics.voltageEvaluation, ...
    metrics.hiddenStateWEvaluation};

metricsTable = build_metrics_table(signalNames, metricsCell);
writetable(metricsTable, ...
    fullfile(outputDir, 'metrics_summary.csv'));

selectedLibraryTable = build_selected_library_table(modelInverse);
writetable(selectedLibraryTable, ...
    fullfile(outputDir, 'selected_library.csv'));

singularTable = table( ...
    (1:numel(modelInverse.singularValues)).', ...
    modelInverse.singularValues(:), ...
    modelInverse.filterFactors(:), ...
    'VariableNames', { ...
        'Index','SingularValue','TikhonovFilterFactor'});
writetable(singularTable, ...
    fullfile(outputDir, 'singular_values.csv'));

levelTable = table( ...
    modelInverse.levels(:), ...
    modelInverse.countPerLevel(:), ...
    modelInverse.selectedPerLevel(:), ...
    'VariableNames', { ...
        'Level','CandidateFunctions','SelectedFunctions'});
writetable(levelTable, ...
    fullfile(outputDir, 'library_by_level.csv'));

write_spike_table( ...
    spike, fullfile(outputDir, 'spike_times.csv'));

%% =========================================================================
%% 11) HIGH-RESOLUTION FIGURES
%% =========================================================================

figureTimer = tic;

% 01: transition scan, oscillation amplitude, and training input
fig = create_figure( ...
    cfg.figures.showFigures, [100 100 1500 1000]);
layout = tiledlayout(fig, 3, 1, ...
    'TileSpacing','compact','Padding','compact');

nexttile(layout);
plot(scanCoarse.current, scanCoarse.vMax, 'o-', ...
    'LineWidth',1.0,'MarkerSize',3); hold on;
plot(scanCoarse.current, scanCoarse.vMin, 's-', ...
    'LineWidth',1.0,'MarkerSize',3);
xline(transitionCurrent, '--', 'LineWidth',1.3);
grid on;
xlabel('Constant current');
ylabel('v_{max}, v_{min} [mV]', 'Interpreter','tex');
legend({'v_{max}','v_{min}','Estimated transition'}, ...
    'Location','best','Interpreter','tex');
title('Constant-current scan');

nexttile(layout);
plot(scanCoarse.current, scanCoarse.amplitude, 'o-', ...
    'LineWidth',1.0,'MarkerSize',3); hold on;
if ~isempty(scanRefined.current)
    plot(scanRefined.current, scanRefined.amplitude, '-', ...
        'LineWidth',1.3);
end
yline(cfg.input.oscillationAmplitudeThreshold, ':', ...
    'LineWidth',1.1);
xline(transitionCurrent, '--', 'LineWidth',1.3);
grid on;
xlabel('Constant current');
ylabel('Steady-state amplitude [mV]');
legend({'Coarse scan','Refined scan', ...
    'Threshold','Estimated transition'}, 'Location','best');
title('Numerical estimate of the oscillatory transition');

nexttile(layout);
stairs(cfg.tTrain_s, Itrain, 'LineWidth',1.0);
grid on;
xlabel('Time [s]');
ylabel('I [\muA/cm^2]');
title('Training current');

save_figure_high_resolution( ...
    fig, figureDir, '01_transition_scan_and_training_input', ...
    cfg.figures.resolutionDpi, ...
    cfg.figures.saveMatlabFigures);

% 02: training signals
fig = create_figure( ...
    cfg.figures.showFigures, [100 100 1500 1000]);
layout = tiledlayout(fig, 3, 1, ...
    'TileSpacing','compact','Padding','compact');

nexttile(layout);
stairs(cfg.tTrain_s, Itrain, 'LineWidth',0.9);
grid on; ylabel('I');
title('Training current');

nexttile(layout);
plot(cfg.tTrain_s, vTrain, 'LineWidth',0.9);
grid on; ylabel('v [mV]');
title('Training voltage');

nexttile(layout);
plot(cfg.tTrain_s, wTrain, 'LineWidth',0.9);
grid on; ylabel('w'); xlabel('Time [s]');
title('Recovery variable');

save_figure_high_resolution( ...
    fig, figureDir, '02_training_signals', ...
    cfg.figures.resolutionDpi, ...
    cfg.figures.saveMatlabFigures);

% 03: reconstructed current
fig = create_figure( ...
    cfg.figures.showFigures, [100 100 1500 850]);
layout = tiledlayout(fig, 2, 1, ...
    'TileSpacing','compact','Padding','compact');

nexttile(layout);
stairs(cfg.tEval_s(currentEvalRange), Ieval(currentEvalRange), ...
    'LineWidth',1.0); hold on;
plot(cfg.tEval_s(currentEvalRange), IestEval(currentEvalRange), '--', ...
    'LineWidth',1.0);
grid on; ylabel('I');
legend({'Reference','Estimated'}, 'Location','best');
title(sprintf( ...
    'Current: 100(1-r^2)=%.4f %%', ...
    metrics.currentEvaluation.oneMinusR2Percent));

nexttile(layout);
plot(cfg.tEval_s(currentEvalRange), ...
    Ieval(currentEvalRange) - IestEval(currentEvalRange), 'LineWidth',0.9);
grid on; ylabel('Error'); xlabel('Time [s]');
title('Current reconstruction error');

save_figure_high_resolution( ...
    fig, figureDir, '03_current_reconstruction', ...
    cfg.figures.resolutionDpi, ...
    cfg.figures.saveMatlabFigures);

% 04: voltage after model reintegration
fig = create_figure( ...
    cfg.figures.showFigures, [100 100 1500 850]);
layout = tiledlayout(fig, 2, 1, ...
    'TileSpacing','compact','Padding','compact');

nexttile(layout);
plot(cfg.tEval_s, vEval, 'LineWidth',1.0); hold on;
plot(cfg.tEval_s, vReconstructed, '--', 'LineWidth',1.0);
grid on; ylabel('v [mV]');
legend({'Reference','Reintegrated'}, 'Location','best');
title(sprintf( ...
    'Reintegrated voltage: 100(1-r^2)=%.4f %%', ...
    metrics.voltageEvaluation.oneMinusR2Percent));

nexttile(layout);
plot(cfg.tEval_s, vEval - vReconstructed, 'LineWidth',0.9);
grid on; ylabel('Error [mV]'); xlabel('Time [s]');
title('Voltage error');

save_figure_high_resolution( ...
    fig, figureDir, '04_voltage_reintegration', ...
    cfg.figures.resolutionDpi, ...
    cfg.figures.saveMatlabFigures);

% 05: window with the largest local error
zoomLocal = worst_error_window( ...
    vEval(stateEvalRange) - vReconstructed(stateEvalRange), ...
    cfg.figures.zoomWindow_s * cfg.fsData_Hz);
zoomIndices = stateEvalRange(zoomLocal);

fig = create_figure( ...
    cfg.figures.showFigures, [100 100 1500 1000]);
layout = tiledlayout(fig, 3, 1, ...
    'TileSpacing','compact','Padding','compact');

nexttile(layout);
stairs(cfg.tEval_s(zoomIndices), Ieval(zoomIndices), ...
    'LineWidth',1.0); hold on;
plot(cfg.tEval_s(zoomIndices), IestEval(zoomIndices), '--', ...
    'LineWidth',1.0);
grid on; ylabel('I');
legend({'Reference','Estimated'}, 'Location','best');
title('Window with the largest voltage error');

nexttile(layout);
plot(cfg.tEval_s(zoomIndices), vEval(zoomIndices), ...
    'LineWidth',1.0); hold on;
plot(cfg.tEval_s(zoomIndices), vReconstructed(zoomIndices), '--', ...
    'LineWidth',1.0);
grid on; ylabel('v [mV]');
legend({'Reference','Reintegrated'}, 'Location','best');

nexttile(layout);
plot(cfg.tEval_s(zoomIndices), ...
    vEval(zoomIndices)-vReconstructed(zoomIndices), ...
    'LineWidth',0.9);
grid on; ylabel('Error [mV]'); xlabel('Time [s]');

save_figure_high_resolution( ...
    fig, figureDir, '05_worst_error_window', ...
    cfg.figures.resolutionDpi, ...
    cfg.figures.saveMatlabFigures);

% 06: phase plane
fig = create_figure( ...
    cfg.figures.showFigures, [100 100 1100 900]);
plot(vEval, wEval, 'LineWidth',1.0); hold on;
plot(vReconstructed, wReconstructed, '--', 'LineWidth',1.0);
grid on;
xlabel('v [mV]'); ylabel('w');
legend({'Reference','Reintegrated'}, 'Location','best');
title('Morris--Lecar phase plane');

save_figure_high_resolution( ...
    fig, figureDir, '06_phase_plane', ...
    cfg.figures.resolutionDpi, ...
    cfg.figures.saveMatlabFigures);

% 07: selection scores
fig = create_figure( ...
    cfg.figures.showFigures, [100 100 1500 850]);
layout = tiledlayout(fig, 2, 1, ...
    'TileSpacing','compact','Padding','compact');

nexttile(layout);
semilogy(modelInverse.Eord, 'LineWidth',1.0);
grid on; xlabel('Sorted index'); ylabel('E_m');
title('Sorted energy score');

nexttile(layout);
if isempty(modelInverse.Dord)
    text(0.5,0.5,'No positive dynamic scores were found', ...
        'HorizontalAlignment','center');
    axis off;
else
    semilogy(modelInverse.Dord, 'LineWidth',1.0);
    grid on; xlabel('Sorted index'); ylabel('D_m');
end
title('Sorted dynamic score');

save_figure_high_resolution( ...
    fig, figureDir, '07_selection_scores', ...
    cfg.figures.resolutionDpi, ...
    cfg.figures.saveMatlabFigures);

% 08: singular spectrum
fig = create_figure( ...
    cfg.figures.showFigures, [100 100 1500 850]);
layout = tiledlayout(fig, 2, 1, ...
    'TileSpacing','compact','Padding','compact');

nexttile(layout);
semilogy(modelInverse.singularValues, 'o-', ...
    'LineWidth',1.0,'MarkerSize',3);
grid on; xlabel('Index');
ylabel('$\sigma_{\ell}$', 'Interpreter','latex');
title('Singular values of the scaled regressor matrix');

nexttile(layout);
semilogy(modelInverse.filterFactors, 'o-', ...
    'LineWidth',1.0,'MarkerSize',3);
grid on; xlabel('Index');
ylabel('$\sigma_{\ell}/(\sigma_{\ell}^2+\mu)$', ...
    'Interpreter','latex');
title('Tikhonov filter factors');

save_figure_high_resolution( ...
    fig, figureDir, '08_singular_spectrum', ...
    cfg.figures.resolutionDpi, ...
    cfg.figures.saveMatlabFigures);

% 09: measured-versus-estimated scatter plots
fig = create_figure( ...
    cfg.figures.showFigures, [100 100 1500 700]);
layout = tiledlayout(fig, 1, 2, ...
    'TileSpacing','compact','Padding','compact');

nexttile(layout);
plot(IevalValid, IestEvalValid, '.', 'MarkerSize',4); hold on;
limits = [min(IevalValid), max(IevalValid)];
plot(limits, limits, '--', 'LineWidth',1.2);
axis equal; grid on;
xlabel('Reference I'); ylabel('Estimated I');
title('Current');

nexttile(layout);
plot(vEval(stateEvalRange), ...
    vReconstructed(stateEvalRange), '.', 'MarkerSize',4); hold on;
limits = [min(vEval(stateEvalRange)), max(vEval(stateEvalRange))];
plot(limits, limits, '--', 'LineWidth',1.2);
axis equal; grid on;
xlabel('Reference v'); ylabel('Reintegrated v');
title('Voltage');

save_figure_high_resolution( ...
    fig, figureDir, '09_real_vs_estimated_scatter', ...
    cfg.figures.resolutionDpi, ...
    cfg.figures.saveMatlabFigures);

% 10: residual distributions
fig = create_figure( ...
    cfg.figures.showFigures, [100 100 1500 700]);
layout = tiledlayout(fig, 1, 2, ...
    'TileSpacing','compact','Padding','compact');

nexttile(layout);
histogram(IevalValid - IestEvalValid, 100, ...
    'Normalization','pdf');
grid on; xlabel('Current error'); ylabel('Density');
title('Current residuals');

nexttile(layout);
histogram( ...
    vEval(stateEvalRange)-vReconstructed(stateEvalRange), ...
    100, 'Normalization','pdf');
grid on; xlabel('Voltage error [mV]'); ylabel('Density');
title('Voltage residuals');

save_figure_high_resolution( ...
    fig, figureDir, '10_residual_distributions', ...
    cfg.figures.resolutionDpi, ...
    cfg.figures.saveMatlabFigures);

% 11: spikes and inter-spike intervals
fig = create_figure( ...
    cfg.figures.showFigures, [100 100 1500 850]);
layout = tiledlayout(fig, 2, 1, ...
    'TileSpacing','compact','Padding','compact');

nexttile(layout);
plot(cfg.tEval_s(stateEvalRange), vEval(stateEvalRange), ...
    'LineWidth',0.9); hold on;
plot(cfg.tEval_s(stateEvalRange), vReconstructed(stateEvalRange), '--', ...
    'LineWidth',0.9);
yline(cfg.spikeThreshold, ':');
if ~isempty(spike.timesReal)
    plot(spike.timesReal, ...
        cfg.spikeThreshold*ones(size(spike.timesReal)), ...
        'o', 'MarkerSize',4);
end
grid on; ylabel('v [mV]');
legend({'Reference','Reintegrated','Threshold','Reference spikes'}, ...
    'Location','best');
title('Spike detection');

nexttile(layout);
if ~isempty(spike.isiReal)
    histogram(spike.isiReal, 30, ...
        'Normalization','probability'); hold on;
end
if ~isempty(spike.isiReconstructed)
    histogram(spike.isiReconstructed, 30, ...
        'Normalization','probability');
end
grid on; xlabel('ISI [s]'); ylabel('Probability');
legend({'Reference','Reintegrated'}, 'Location','best');
title('Inter-spike interval distribution');

save_figure_high_resolution( ...
    fig, figureDir, '11_spikes_and_isi', ...
    cfg.figures.resolutionDpi, ...
    cfg.figures.saveMatlabFigures);

% 12: library size by scale level
fig = create_figure( ...
    cfg.figures.showFigures, [100 100 1100 800]);
bar(modelInverse.levels, ...
    [modelInverse.countPerLevel, modelInverse.selectedPerLevel], ...
    'grouped');
set(gca,'YScale','log');
grid on;
xlabel('Scale level r'); ylabel('Number of functions');
legend({'Candidate','Selected'}, 'Location','best');
title('Candidate and selected functions by scale level');

save_figure_high_resolution( ...
    fig, figureDir, '12_library_by_level', ...
    cfg.figures.resolutionDpi, ...
    cfg.figures.saveMatlabFigures);

timing.figureGeneration_s = toc(figureTimer);

%% =========================================================================
%% 12) RESULT ORGANIZATION AND STORAGE
%% =========================================================================

results.metadata.modelName = 'Morris--Lecar';
results.metadata.task = 'Inverse current reconstruction';
results.metadata.generatedAt = datestr(now, 31);
results.metadata.matlabVersion = version;
results.metadata.scriptBase = scriptBase;
results.metadata.outputDirectory = outputDir;

results.configuration = cfg;
results.parameters = p;
results.parameterUnits = parameterUnits;
results.initialState = X0;

results.transition.estimatedCurrent = transitionCurrent;
results.transition.amplitudeThreshold = ...
    cfg.input.oscillationAmplitudeThreshold;
results.transition.scanCoarse = scanCoarse;
results.transition.scanRefined = scanRefined;
results.transition.trainingIntervals = trainIntervals;
results.transition.evaluationIntervals = evalIntervals;
results.transition.excludedNeighborhood = false;

results.training.time_s = cfg.tTrain_s;
results.training.current = Itrain;
results.training.states = Xtrain;
results.training.voltage = vTrain;
results.training.hiddenStateW = wTrain;
results.training.regressors = Ztrain;
results.training.targetCurrent = YtrainCurrent;
results.training.inputIndices = inputIdxTrain;
results.training.targetCurrentIndices = targetIdxTrain;
results.training.dynamicWeightRaw = omegaTrainRaw;
results.training.dynamicWeightNormalized = omegaTrainNormalized;
results.training.dynamicWeightMaximum = omegaTrainMax;
results.training.estimatedCurrentValid = IhatTrainValid;
results.training.currentResidualValid = ...
    YtrainCurrent - IhatTrainValid;

results.evaluation.time_s = cfg.tEval_s;
results.evaluation.currentReal = Ieval;
results.evaluation.currentEstimated = IestEval;
results.evaluation.statesReal = Xeval;
results.evaluation.voltageReal = vEval;
results.evaluation.hiddenStateWReal = wEval;
results.evaluation.regressors = Zeval;
results.evaluation.domainInputIndices = inputIdxEvalDomain;
results.evaluation.domainTargetCurrentIndices = targetIdxEvalDomain;
results.evaluation.currentTargetIndices = currentEvalRange;
results.evaluation.stateEvaluationRange = stateEvalRange;
results.evaluation.inputIndices = inputIdxEval;
results.evaluation.targetCurrentIndices = targetIdxEval;
results.evaluation.currentResidual = Ieval - IestEval;

results.reintegration.currentUsed = Ireintegrated;
results.reintegration.initializationCurrentRange = ...
    1:max(firstEstimatedCurrentIndex-1,0);
results.reintegration.firstEstimatedCurrentIndex = ...
    firstEstimatedCurrentIndex;
results.reintegration.firstStateDrivenByEstimate = ...
    firstStateDrivenByEstimate;
results.reintegration.states = Xreconstructed;
results.reintegration.voltage = vReconstructed;
results.reintegration.hiddenStateW = wReconstructed;
results.reintegration.regressors = Zreconstructed;
results.reintegration.inputIndices = inputIdxReconstructed;
results.reintegration.targetCurrentIndices = targetIdxReconstructed;
results.reintegration.voltageResidual = ...
    vEval - vReconstructed;
results.reintegration.hiddenStateWResidual = ...
    wEval - wReconstructed;

results.modelInverse = modelInverse;
results.metrics = metrics;
results.lagMetrics = lagMetrics;
results.domain = domain;
results.spike = spike;
results.library = library;
results.memory = memory;
results.timing = timing;

results.files.metricsCsv = ...
    fullfile(outputDir, 'metrics_summary.csv');
results.files.selectedLibraryCsv = ...
    fullfile(outputDir, 'selected_library.csv');
results.files.singularValuesCsv = ...
    fullfile(outputDir, 'singular_values.csv');
results.files.libraryByLevelCsv = ...
    fullfile(outputDir, 'library_by_level.csv');
results.files.spikeTimesCsv = ...
    fullfile(outputDir, 'spike_times.csv');
results.files.figureDirectory = figureDir;

results.timing.totalBeforeSave_s = toc(overallTimer);

resultFile = fullfile(outputDir, ...
    'morris_lecar_inverse_full_results.mat');

save(resultFile, 'results', '-v7.3');

fileInfo = dir(resultFile);
results.files.resultMat = resultFile;
results.files.resultMatBytes = fileInfo.bytes;
results.timing.total_s = toc(overallTimer);

save(resultFile, 'results', '-v7.3');

summaryFile = fullfile(outputDir, 'run_summary.txt');
fid = fopen(summaryFile, 'w');
if fid >= 0
    fprintf(fid, 'Morris--Lecar inverse multiscale spline model\n');
    fprintf(fid, 'Generated: %s\n', results.metadata.generatedAt);
    fprintf(fid, 'Result MAT: %s\n', resultFile);
    fprintf(fid, 'Figures: %s\n\n', figureDir);
    fprintf(fid, 'Voltage delays [samples]: [%s]\n', ...
        num2str(cfg.delaySet_samples));
    fprintf(fid, 'Voltage delays [ms]: [%s]\n', num2str(cfg.delaySet_ms));
    fprintf(fid, 'Temporal alignment: xi(k+1) -> I(k)\n');
    fprintf(fid, 'First estimated current index: %d\n', ...
        firstEstimatedCurrentIndex);
    fprintf(fid, 'First state driven by estimated current: %d\n', ...
        firstStateDrivenByEstimate);
    fprintf(fid, 'Reintegration initialization duration: %.12g ms\n', ...
        (firstEstimatedCurrentIndex - 1) * cfg.Ts_ms);
    fprintf(fid, ['Reintegration initialization: reference current is used ' ...
        'only before the first estimable current sample\n']);
    fprintf(fid, 'Transition current: %.10g\n', transitionCurrent);
    fprintf(fid, 'Excluded neighborhood: no\n');
    fprintf(fid, 'Candidate functions: %d\n', ...
        library.totalCandidateFunctions);
    fprintf(fid, 'Selected functions: %d\n', ...
        library.selectedUnionFunctions);
    fprintf(fid, 'Reduction: %.6f %%\n\n', ...
        library.reductionPercent);
    fprintf(fid, 'Dynamic-only functions: %d\n', library.selectedDynamicOnlyFunctions);
    fprintf(fid, 'Tikhonov mu: %.12e\n', cfg.muInverse);
    fprintf(fid, 'Effective condition number: %.12e\n\n', ...
        modelInverse.effectiveConditionNumber);
    fprintf(fid, 'Current evaluation 100*(1-r^2): %.12e %%\n', ...
        metrics.currentEvaluation.oneMinusR2Percent);
    fprintf(fid, 'Voltage evaluation 100*(1-r^2): %.12e %%\n', ...
        metrics.voltageEvaluation.oneMinusR2Percent);
    fprintf(fid, 'Current RMSE: %.12e\n', ...
        metrics.currentEvaluation.rmse);
    fprintf(fid, 'Voltage RMSE: %.12e\n', ...
        metrics.voltageEvaluation.rmse);
    fprintf(fid, 'Spike F1: %.8f\n', spike.f1);
    fprintf(fid, 'Total runtime: %.6f s\n', results.timing.total_s);
    fclose(fid);
end

fprintf('\n============================================================\n');
fprintf('EXECUTION COMPLETED\n');
fprintf('============================================================\n');
fprintf('MAT results:\n%s\n', resultFile);
fprintf('PNG figures at %d dpi:\n%s\n', ...
    cfg.figures.resolutionDpi, figureDir);
fprintf('Total runtime: %.2f s\n', results.timing.total_s);

%% =========================================================================
%% AUXILIARY FUNCTIONS FOR THE MULTISCALE SPLINE MODEL
%% =========================================================================

function [scriptDir, scriptBase] = get_script_location(defaultBase)
    scriptFullPath = mfilename('fullpath');
    if isempty(scriptFullPath)
        scriptDir = pwd;
        scriptBase = defaultBase;
    else
        [scriptDir, scriptBase] = fileparts(scriptFullPath);
    end
end

function intervals = clean_intervals(intervals)
    if isempty(intervals)
        intervals = zeros(0,2);
        return;
    end
    intervals = intervals(all(isfinite(intervals),2),:);
    intervals = intervals(intervals(:,2) > intervals(:,1),:);
end

function [trainIntervals, evalIntervals] = build_intervals_around_transition( ...
    transitionCurrent, lowerBound, upperBound, ...
    trainHalfWidth, evalHalfWidth)

    trainLow = max(lowerBound, transitionCurrent - trainHalfWidth);
    trainHigh = min(upperBound, transitionCurrent + trainHalfWidth);

    if trainHigh <= trainLow
        error('The training interval is not valid.');
    end

    trainIntervals = [trainLow, trainHigh];

    evalLow = max(trainLow, transitionCurrent - evalHalfWidth);
    evalHigh = min(trainHigh, transitionCurrent + evalHalfWidth);

    if evalHigh <= evalLow
        error('The evaluation interval is not valid.');
    end

    evalIntervals = [evalLow, evalHigh];
end

function u = generate_piecewise_uniform_input_intervals( ...
    N, numLevelsPerInterval, intervals, samplesPerLevel)

    intervals = clean_intervals(intervals);
    if isempty(intervals)
        error('No valid intervals are available to generate the input current.');
    end

    % Build all admissible input levels over the specified intervals.
    allLevels = zeros(0,1);
    for q = 1:size(intervals,1)
        currentLevels = linspace( ...
            intervals(q,1), intervals(q,2), numLevelsPerInterval).';
        allLevels = [allLevels; currentLevels]; %#ok<AGROW>
    end
    allLevels = unique(allLevels, 'stable');

    numberOfBlocks = ceil(N / samplesPerLevel);
    numberOfLevels = numel(allLevels);

    if numberOfBlocks >= numberOfLevels
        % Ensure that every input level appears at least once.
        numberOfAdditionalBlocks = numberOfBlocks - numberOfLevels;
        if numberOfAdditionalBlocks > 0
            additionalLevels = allLevels( ...
                randi(numberOfLevels, numberOfAdditionalBlocks, 1));
            levelSequence = [allLevels; additionalLevels];
        else
            levelSequence = allLevels;
        end
        levelSequence = levelSequence(randperm(numberOfBlocks));
    else
        % If there are fewer blocks than levels, sample the domain uniformly.
        selectedIndices = round(linspace(1, numberOfLevels, numberOfBlocks));
        levelSequence = allLevels(selectedIndices);
        levelSequence = levelSequence(randperm(numberOfBlocks));
    end

    u = zeros(N,1);
    firstSample = 1;
    for blockIndex = 1:numberOfBlocks
        lastSample = min(N, firstSample + samplesPerLevel - 1);
        u(firstSample:lastSample) = levelSequence(blockIndex);
        firstSample = lastSample + 1;
        if firstSample > N
            break;
        end
    end
end

function [Z, Y, inputIdx, targetIdx] = build_dataset_inverse(v, I, delaySet)
    % Construct temporally aligned inverse-model pairs:
    %
    %   [v(k+1-d_1), ..., v(k+1-d_m)]  --->  I(k)
    %
    % I(k) drives the transition from v(k) to v(k+1). The delay vector is
    % therefore referenced to the endpoint k+1 of that transition.

    v = v(:);
    I = I(:);
    delaySet = delaySet(:).';
    maxDelay = max(delaySet);

    N = numel(v);
    M = N - maxDelay;

    if M <= 0
        error('The signal is shorter than the largest selected delay.');
    end
    if numel(I) ~= N
        error('v and I must have the same length.');
    end

    Z = zeros(M, numel(delaySet));
    Y = zeros(M, 1);
    inputIdx = zeros(M,1);
    targetIdx = zeros(M,1);

    row = 1;
    for k = maxDelay:(N - 1)
        embeddingIndex = k + 1;
        Z(row,:) = v(embeddingIndex - delaySet).';
        Y(row) = I(k);
        inputIdx(row) = embeddingIndex;
        targetIdx(row) = k;
        row = row + 1;
    end
end

function [Z, inputIdx, targetIdx] = build_inverse_regressors_from_signal(x, delaySet)
    % Build inverse-model regressors using the same xi(k+1) -> I(k)
    % alignment used during training.

    x = x(:);
    delaySet = delaySet(:).';
    maxDelay = max(delaySet);

    N = numel(x);
    M = N - maxDelay;

    if M <= 0
        error('The signal is shorter than the largest selected delay.');
    end

    Z = zeros(M, numel(delaySet));
    inputIdx = zeros(M,1);
    targetIdx = zeros(M,1);

    row = 1;
    for k = maxDelay:(N - 1)
        embeddingIndex = k + 1;
        Z(row,:) = x(embeddingIndex - delaySet).';
        inputIdx(row) = embeddingIndex;
        targetIdx(row) = k;
        row = row + 1;
    end
end

function [omegaRaw, omegaNormalized, omegaMax] = ...
    build_dynamic_weights_from_signal(x, validIdx)
    % Squared second differences emphasize regions with strong local curvature.
    % Normalization by the maximum preserves the ranking while keeping the
    % weights within [0,1].

    x = x(:);
    validIdx = validIdx(:);
    N = numel(validIdx);

    omegaRaw = zeros(N,1);
    for i = 1:N
        k = validIdx(i);
        km1 = max(k - 1, 1);
        kp1 = min(k + 1, numel(x));

        omegaRaw(i) = ...
            (x(kp1) - 2*x(k) + x(km1))^2;
    end

    omegaMax = max(omegaRaw);
    if omegaMax > 0
        omegaNormalized = omegaRaw / omegaMax;
    else
        omegaNormalized = zeros(size(omegaRaw));
    end
end

function model = train_scalar_multiscale_spline_dualfilter( ...
    Ztr, Ytr, omegaTr, ...
    levels, boundaryPadding, ...
    energyFractionTarget, dynamicTopFraction, ...
    mu)

    % Normalize every input coordinate using training data only. The same
    % limits are stored in the model and reused during evaluation.
    zmin = min(Ztr, [], 1);
    zmax = max(Ztr, [], 1);

    rangeZ = zmax - zmin;
    rangeZ(rangeZ < 1e-12) = 1;

    Ztrn = (Ztr - zmin) ./ rangeZ;
    Ztrn = min(max(Ztrn, 0), 1);

    NI = size(Ztrn,2);
    Ntrain = size(Ztrn,1);

    % Determine the number and global indexing of tensor-product spline
    % functions at each scale level before any data-driven reduction.
    [countPerLevel, levelOffsets, totalFunctions] = ...
        library_info_multiscale(NI, levels, boundaryPadding);

    fprintf('Input dimension = %d\n', NI);
    fprintf('Candidate nonlinear functions = %d\n', totalFunctions);

    % E_i measures how strongly each function is activated on the training
    % trajectory. D_i uses the same activation weighted by local voltage
    % dynamics, so rapidly changing regions can influence the selection.
    Ei = zeros(totalFunctions, 1);
    Di = zeros(totalFunctions, 1);

    for n = 1:Ntrain
        zn = Ztrn(n,:);
        omegaN = omegaTr(n);

        for il = 1:numel(levels)
            lev = levels(il);
            offset = levelOffsets(il);

            [localIdx, functionValues] = ...
                active_library_sample_level(zn, lev, boundaryPadding);
            globalIdx = offset + localIdx;

            functionSquared = functionValues.^2;
            Ei(globalIdx) = Ei(globalIdx) + functionSquared;
            Di(globalIdx) = Di(globalIdx) + omegaN * functionSquared;
        end
    end

    activeIdxE = find(Ei > 0);
    activeIdxD = find(Di > 0);

    if isempty(activeIdxE)
        error('No candidate function was active on the training data.');
    end

    % Energy selection: keep the smallest ranked subset whose cumulative
    % activation energy reaches the prescribed fraction.
    Eactive = Ei(activeIdxE);
    [Eord, permE] = sort(Eactive, 'descend');
    activeSortedE = activeIdxE(permE);

    cumulativeEnergy = cumsum(Eord) / sum(Eord);
    MkeepE = find(cumulativeEnergy >= energyFractionTarget, 1, 'first');
    if isempty(MkeepE)
        MkeepE = numel(activeSortedE);
    end
    selectedE = activeSortedE(1:MkeepE);

    % Dynamic selection: retain a fixed top fraction of the functions with
    % positive dynamic score.
    if isempty(activeIdxD)
        Dord = zeros(0,1);
        activeSortedD = zeros(0,1);
        MkeepD = 0;
        selectedD = zeros(0,1);
    else
        Dactive = Di(activeIdxD);
        [Dord, permD] = sort(Dactive, 'descend');
        activeSortedD = activeIdxD(permD);

        MkeepD = max(1, ceil(dynamicTopFraction * numel(activeSortedD)));
        selectedD = activeSortedD(1:MkeepD);
    end

    % The final library is the union of both filters. selectedDynamicOnly is
    % stored only as a diagnostic; it does not modify the union itself.
    selectedDynamicOnly = setdiff(selectedD, selectedE);

    selectedGlobalIdx = union(selectedE, selectedD);
    selectedGlobalIdx = sort(selectedGlobalIdx);

    reductionPercent = ...
        100 * (1 - numel(selectedGlobalIdx) / max(totalFunctions,1));

    fprintf('Energy-active functions = %d\n', numel(activeIdxE));
    fprintf('Dynamic-active functions = %d\n', numel(activeIdxD));
    fprintf('Retained by energy = %d\n', MkeepE);
    fprintf('Retained by dynamics = %d\n', MkeepD);
    fprintf('Final union = %d | reduction = %.2f %%\n', ...
        numel(selectedGlobalIdx), reductionPercent);
    fprintf('Added only by dynamics = %d\n', numel(selectedDynamicOnly));

    [Csel, Hsel, levelSel, localSel, jSel] = ...
        build_selected_library( ...
            selectedGlobalIdx, NI, levels, countPerLevel, ...
            levelOffsets, boundaryPadding);

    Mkeep = numel(selectedGlobalIdx);

    % Assemble the reduced nonlinear regressor block from the selected
    % localized functions. The intercept and normalized linear coordinates are
    % appended explicitly below.
    GtrSpline = zeros(Ntrain, Mkeep);
    for n = 1:Ntrain
        zn = Ztrn(n,:);
        for c = 1:Mkeep
            GtrSpline(n,c) = evaluate_selected_spline_function( ...
                zn, levelSel(c), jSel(c,:), boundaryPadding);
        end
    end

    GtrLinear = [ones(Ntrain,1), Ztrn];
    Gtr = [GtrLinear, GtrSpline];

    % Scale every regressor column before the SVD. This prevents differences
    % in column magnitude from dominating the conditioning of the fit.
    colScales = sqrt(sum(Gtr.^2, 1));
    colScales(colScales < 1e-12) = 1;

    GtrScaled = Gtr ./ colScales;

    % Tikhonov-regularized least squares in SVD coordinates. The filter
    % sigma/(sigma^2+mu) damps directions associated with small singular values.
    [U, Sigma, V] = svd(GtrScaled, 'econ');
    singularValues = diag(Sigma);
    filterFactors = singularValues ./ (singularValues.^2 + mu);

    thetaScaled = V * (diag(filterFactors) * (U' * Ytr));
    W = thetaScaled ./ colScales.';

    Wlinear = W(1:(NI+1));
    Wspline = W((NI+2):end);

    % Report numerical rank and the condition number of the resolved singular
    % subspace. These values are diagnostics and do not change the solution.
    numericalTolerance = ...
        max(size(GtrScaled)) * eps(max(singularValues));
    numericalRank = sum(singularValues > numericalTolerance);

    if numericalRank > 0
        effectiveConditionNumber = ...
            singularValues(1) / singularValues(numericalRank);
    else
        effectiveConditionNumber = Inf;
    end

    selectedPerLevel = zeros(numel(levels),1);
    for q = 1:numel(levels)
        selectedPerLevel(q) = sum(levelSel == levels(q));
    end

    model.zmin = zmin;
    model.zmax = zmax;
    model.rangeZ = rangeZ;

    model.W = W;
    model.Wlinear = Wlinear;
    model.Wspline = Wspline;
    model.thetaScaled = thetaScaled;
    model.colScales = colScales(:);

    model.Csel = Csel;
    model.Hsel = Hsel;
    model.levelSel = levelSel;
    model.localSel = localSel;
    model.jSel = jSel;
    model.selectedGlobalIdx = selectedGlobalIdx;
    model.selectedE = selectedE;
    model.selectedD = selectedD;
    model.selectedDynamicOnly = selectedDynamicOnly;

    model.Ei = Ei;
    model.Di = Di;
    model.Eord = Eord;
    model.Dord = Dord;
    model.cumulativeEnergy = cumulativeEnergy;
    model.activeIdxE = activeIdxE;
    model.activeIdxD = activeIdxD;
    model.activeSortedE = activeSortedE;
    model.activeSortedD = activeSortedD;

    model.levels = levels;
    model.boundaryPadding = boundaryPadding;
    model.countPerLevel = countPerLevel;
    model.levelOffsets = levelOffsets;
    model.totalFunctions = totalFunctions;
    model.selectedPerLevel = selectedPerLevel;

    model.energyFractionTarget = energyFractionTarget;
    model.dynamicTopFraction = dynamicTopFraction;
    model.mu = mu;

    model.singularValues = singularValues;
    model.filterFactors = filterFactors;
    model.numericalTolerance = numericalTolerance;
    model.numericalRank = numericalRank;
    model.effectiveConditionNumber = effectiveConditionNumber;

    model.Ntrain = Ntrain;
    model.NI = NI;
    model.Ntheta = numel(W);
    model.reductionPercent = reductionPercent;
end

function Yhat = eval_model(model, Z)
    G = build_reduced_design_matrix(model, Z);
    Yhat = G * model.W;
end

function G = build_reduced_design_matrix(model, Z)
    % Apply the training normalization. Values outside the training min--max
    % box are clipped to [0,1] only when the stored spline library is evaluated.
    Zn = (Z - model.zmin) ./ model.rangeZ;
    Zn = min(max(Zn,0),1);

    N = size(Zn,1);
    Mkeep = numel(model.selectedGlobalIdx);

    Gspline = zeros(N, Mkeep);
    for n = 1:N
        zn = Zn(n,:);
        for c = 1:Mkeep
            Gspline(n,c) = evaluate_selected_spline_function( ...
                zn, model.levelSel(c), model.jSel(c,:), ...
                model.boundaryPadding);
        end
    end

    Glinear = [ones(N,1), Zn];
    G = [Glinear, Gspline];
end

function [Iest, inputIdx, targetIdx] = ...
    estimate_inverse_current(modelInverse, v, delaySet)
    % Estimate I(k) from the voltage embedding referenced to k+1.
    % Samples for which the complete delay vector is unavailable are left as
    % NaN rather than being replaced by artificial estimates. The last current
    % sample is also left as NaN because I(N) would require v(N+1) and is not
    % used by the state integrator.

    [Zall, inputIdx, targetIdx] = ...
        build_inverse_regressors_from_signal(v, delaySet);
    IhatValid = eval_model(modelInverse, Zall);

    Iest = NaN(numel(v),1);
    Iest(targetIdx) = IhatValid;
end

function domain = check_domain_fraction(model, Z)
    ZnRaw = (Z - model.zmin) ./ model.rangeZ;
    insideAll = all(ZnRaw >= 0 & ZnRaw <= 1, 2);

    ZnClipped = min(max(ZnRaw,0),1);
    clippedAny = any(abs(ZnRaw - ZnClipped) > 0, 2);

    domain.insideAllFraction = mean(insideAll);
    domain.clippedAnyFraction = mean(clippedAny);
    domain.insidePerCoordinate = ...
        mean(ZnRaw >= 0 & ZnRaw <= 1, 1);
    domain.belowPerCoordinate = mean(ZnRaw < 0, 1);
    domain.abovePerCoordinate = mean(ZnRaw > 1, 1);
    domain.normalizedRawMin = min(ZnRaw, [], 1);
    domain.normalizedRawMax = max(ZnRaw, [], 1);
end

function metrics = compute_error_metrics(y, yhat)
    y = y(:);
    yhat = yhat(:);

    if numel(y) ~= numel(yhat)
        error('The compared signals must have the same length.');
    end

    residual = y - yhat;
    centered = y - mean(y);

    sse = sum(residual.^2);
    sst = sum(centered.^2);

    if sst > 0
        oneMinusR2 = sse / sst;
        r2 = 1 - oneMinusR2;
    else
        oneMinusR2 = NaN;
        r2 = NaN;
    end

    signalRange = max(y) - min(y);
    signalStd = std(y);

    metrics.N = numel(y);
    metrics.sse = sse;
    metrics.sst = sst;
    metrics.oneMinusR2 = oneMinusR2;
    metrics.oneMinusR2Percent = 100 * oneMinusR2;
    metrics.r2 = r2;
    metrics.rmse = sqrt(mean(residual.^2));
    metrics.mae = mean(abs(residual));
    metrics.maxAbsError = max(abs(residual));
    metrics.biasEstimateMinusReal = mean(yhat - y);
    metrics.relativeL2 = norm(residual) / max(norm(y), eps);
    metrics.nrmseRange = metrics.rmse / max(signalRange, eps);
    metrics.nrmseStd = metrics.rmse / max(signalStd, eps);
    metrics.cosineSimilarity = ...
        dot(y,yhat) / max(norm(y)*norm(yhat), eps);

    C = corrcoef(y,yhat);
    if all(size(C) == [2 2])
        metrics.pearsonCorrelation = C(1,2);
    else
        metrics.pearsonCorrelation = NaN;
    end

    metrics.realMean = mean(y);
    metrics.estimatedMean = mean(yhat);
    metrics.realStd = signalStd;
    metrics.estimatedStd = std(yhat);
    metrics.realMin = min(y);
    metrics.realMax = max(y);
    metrics.estimatedMin = min(yhat);
    metrics.estimatedMax = max(yhat);
end

function lagMetrics = compute_lag_metrics( ...
    y, yhat, fs, maxLagSeconds, downsampleFactor)

    y = y(:);
    yhat = yhat(:);

    downsampleFactor = max(1, round(downsampleFactor));
    y = y(1:downsampleFactor:end);
    yhat = yhat(1:downsampleFactor:end);

    y = y - mean(y);
    yhat = yhat - mean(yhat);

    fsReduced = fs / downsampleFactor;
    maxLagSamples = round(maxLagSeconds * fsReduced);
    lags = (-maxLagSamples:maxLagSamples).';

    correlations = NaN(size(lags));
    for q = 1:numel(lags)
        lag = lags(q);

        if lag >= 0
            a = y(1:end-lag);
            b = yhat(1+lag:end);
        else
            shift = -lag;
            a = y(1+shift:end);
            b = yhat(1:end-shift);
        end

        if isempty(a)
            continue;
        end

        correlations(q) = ...
            dot(a,b) / max(norm(a)*norm(b), eps);
    end

    [maxCorrelation, idx] = max(correlations);
    bestLagSamplesReduced = lags(idx);

    lagMetrics.maxCorrelation = maxCorrelation;
    lagMetrics.bestLagSeconds = bestLagSamplesReduced / fsReduced;
    lagMetrics.bestLagSamplesOriginal = ...
        bestLagSamplesReduced * downsampleFactor;
    lagMetrics.lagsSeconds = lags / fsReduced;
    lagMetrics.correlations = correlations;
    lagMetrics.downsampleFactor = downsampleFactor;
end

function spike = compute_spike_metrics( ...
    vReal, vReconstructed, t, threshold, toleranceSeconds)

    [idxReal, timesReal] = detect_upward_crossings(vReal, t, threshold);
    [idxRec, timesRec] = detect_upward_crossings( ...
        vReconstructed, t, threshold);

    [matchedReal, matchedRec] = match_spike_times( ...
        timesReal, timesRec, toleranceSeconds);

    nReal = numel(timesReal);
    nRec = numel(timesRec);
    nMatched = numel(matchedReal);

    spike.threshold = threshold;
    spike.toleranceSeconds = toleranceSeconds;
    spike.indicesReal = idxReal;
    spike.indicesReconstructed = idxRec;
    spike.timesReal = timesReal;
    spike.timesReconstructed = timesRec;
    spike.isiReal = diff(timesReal);
    spike.isiReconstructed = diff(timesRec);
    spike.matchedTimesReal = matchedReal;
    spike.matchedTimesReconstructed = matchedRec;
    spike.timingErrorsSeconds = matchedRec - matchedReal;

    spike.nReal = nReal;
    spike.nReconstructed = nRec;
    spike.nMatched = nMatched;
    spike.countDifference = nRec - nReal;

    spike.precision = nMatched / max(nRec,1);
    spike.recall = nMatched / max(nReal,1);
    spike.f1 = 2 * spike.precision * spike.recall / ...
        max(spike.precision + spike.recall, eps);

    if nMatched > 0
        spike.timingRMSESeconds = ...
            sqrt(mean(spike.timingErrorsSeconds.^2));
        spike.timingMAESeconds = ...
            mean(abs(spike.timingErrorsSeconds));
        spike.timingBiasSeconds = ...
            mean(spike.timingErrorsSeconds);
    else
        spike.timingRMSESeconds = NaN;
        spike.timingMAESeconds = NaN;
        spike.timingBiasSeconds = NaN;
    end

    spike.isiMeanReal = mean_or_nan(spike.isiReal);
    spike.isiMeanReconstructed = ...
        mean_or_nan(spike.isiReconstructed);
    spike.isiStdReal = std_or_nan(spike.isiReal);
    spike.isiStdReconstructed = ...
        std_or_nan(spike.isiReconstructed);
end

function [indices, times] = detect_upward_crossings(v, t, threshold)
    v = v(:);
    t = t(:);
    above = v > threshold;
    indices = find(diff(above) == 1) + 1;
    times = t(indices);
end

function [matchedReal, matchedRec] = match_spike_times( ...
    timesReal, timesRec, toleranceSeconds)

    timesReal = timesReal(:);
    timesRec = timesRec(:);

    matchedReal = zeros(0,1);
    matchedRec = zeros(0,1);

    i = 1;
    j = 1;

    while i <= numel(timesReal) && j <= numel(timesRec)
        dt = timesRec(j) - timesReal(i);

        if abs(dt) <= toleranceSeconds
            matchedReal(end+1,1) = timesReal(i); %#ok<AGROW>
            matchedRec(end+1,1) = timesRec(j); %#ok<AGROW>
            i = i + 1;
            j = j + 1;
        elseif timesRec(j) < timesReal(i) - toleranceSeconds
            j = j + 1;
        else
            i = i + 1;
        end
    end
end

function value = mean_or_nan(x)
    if isempty(x)
        value = NaN;
    else
        value = mean(x);
    end
end

function value = std_or_nan(x)
    if numel(x) < 2
        value = NaN;
    else
        value = std(x);
    end
end

function [countPerLevel, levelOffsets, totalFunctions] = ...
    library_info_multiscale(NI, levels, boundaryPadding)

    L = numel(levels);
    countPerLevel = zeros(L,1);
    levelOffsets = zeros(L,1);

    totalFunctions = 0;
    for i = 1:L
        lev = levels(i);
        m = (2^lev + 1) + 2*boundaryPadding;

        countPerLevel(i) = m^NI;
        levelOffsets(i) = totalFunctions;
        totalFunctions = totalFunctions + countPerLevel(i);
    end
end

function [localIdx, functionValues] = ...
    active_library_sample_level(z, lev, boundaryPadding)

    NI = numel(z);
    m = (2^lev + 1) + 2*boundaryPadding;

    idxList = 0;
    valList = 1;
    stride = 1;

    for d = 1:NI
        [idx1d, w1d] = ...
            active_quadratic_spline_1d(z(d), lev, boundaryPadding);

        newCount = numel(idxList) * numel(idx1d);
        idxNew = zeros(1, newCount);
        valNew = zeros(1, newCount);

        c = 1;
        for a = 1:numel(idxList)
            for b = 1:numel(idx1d)
                idxNew(c) = ...
                    idxList(a) + (idx1d(b)-1) * stride;
                valNew(c) = valList(a) * w1d(b);
                c = c + 1;
            end
        end

        idxList = idxNew;
        valList = valNew;
        stride = stride * m;
    end

    localIdx = idxList(:) + 1;
    functionValues = valList(:);
end

function [idx, w] = active_quadratic_spline_1d( ...
    z, lev, boundaryPadding)

    z = min(max(z,0),1);

    h = 1 / (2^lev);
    jMin = -boundaryPadding;
    jMax = 2^lev + boundaryPadding;

    s = z / h;
    j0 = floor(s);

    candidates = (j0-2):(j0+2);
    candidates = candidates( ...
        candidates >= jMin & candidates <= jMax);

    idxTemp = zeros(size(candidates));
    wTemp = zeros(size(candidates));
    c = 0;

    for k = 1:numel(candidates)
        j = candidates(k);
        center = j*h;

        t = (z - center)/h + 1.5;
        value = quadratic_spline_core(t);

        if value > 0
            c = c + 1;
            idxTemp(c) = j - jMin + 1;
            wTemp(c) = value;
        end
    end

    idx = idxTemp(1:c);
    w = wTemp(1:c);

    weightSum = sum(w);
    if weightSum > 0
        w = w / weightSum;
    else
        j = round(s);
        j = min(max(j, jMin), jMax);
        idx = j - jMin + 1;
        w = 1;
    end
end

function value = quadratic_spline_core(t)
    % Quadratic compactly supported spline prototype on [0,3].
    if t >= 0 && t < 1
        value = 0.5 * t^2;
    elseif t >= 1 && t < 2
        value = -t^2 + 3*t - 1.5;
    elseif t >= 2 && t <= 3
        value = 0.5*t^2 - 3*t + 4.5;
    else
        value = 0;
    end
end

function [Csel, Hsel, levelSel, localSel, jSel] = ...
    build_selected_library( ...
        selectedGlobalIdx, NI, levels, countPerLevel, ...
        levelOffsets, boundaryPadding)

    M = numel(selectedGlobalIdx);

    Csel = zeros(M, NI);
    Hsel = zeros(M, NI);
    levelSel = zeros(M,1);
    localSel = zeros(M,1);
    jSel = zeros(M,NI);

    for i = 1:M
        globalIndex = selectedGlobalIdx(i);

        levelPosition = find( ...
            globalIndex > levelOffsets & ...
            globalIndex <= (levelOffsets + countPerLevel), ...
            1, 'first');

        if isempty(levelPosition)
            error('Global function index is out of range.');
        end

        lev = levels(levelPosition);
        h = 1 / (2^lev);
        m = (2^lev + 1) + 2*boundaryPadding;

        localIndex = globalIndex - levelOffsets(levelPosition);
        localZeroBased = localIndex - 1;

        digits = zeros(1, NI);
        temporary = localZeroBased;
        for d = 1:NI
            digits(d) = mod(temporary, m);
            temporary = floor(temporary / m);
        end

        jPhysical = digits - boundaryPadding;
        centers = jPhysical * h;

        Csel(i,:) = centers;
        Hsel(i,:) = h * ones(1, NI);
        levelSel(i) = lev;
        localSel(i) = localIndex;
        jSel(i,:) = jPhysical;
    end
end

function value = evaluate_selected_spline_function( ...
    z, lev, jVector, boundaryPadding)
    % Evaluate one selected tensor-product localized function using exactly
    % the same one-dimensional normalization employed during library scoring.

    NI = numel(z);
    value = 1;

    for d = 1:NI
        [idx1d, w1d] = ...
            active_quadratic_spline_1d(z(d), lev, boundaryPadding);

        jMin = -boundaryPadding;
        localIndexWanted = jVector(d) - jMin + 1;

        position = find(idx1d == localIndexWanted, 1, 'first');
        if isempty(position)
            value = 0;
            return;
        end

        value = value * w1d(position);
    end
end

function T = build_metrics_table(signalNames, metricsCell)
    n = numel(signalNames);

    N = zeros(n,1);
    oneMinusR2 = zeros(n,1);
    oneMinusR2Percent = zeros(n,1);
    r2 = zeros(n,1);
    rmse = zeros(n,1);
    nrmseRange = zeros(n,1);
    nrmseStd = zeros(n,1);
    mae = zeros(n,1);
    maxAbsError = zeros(n,1);
    bias = zeros(n,1);
    relativeL2 = zeros(n,1);
    pearson = zeros(n,1);
    cosine = zeros(n,1);

    for q = 1:n
        m = metricsCell{q};
        N(q) = m.N;
        oneMinusR2(q) = m.oneMinusR2;
        oneMinusR2Percent(q) = m.oneMinusR2Percent;
        r2(q) = m.r2;
        rmse(q) = m.rmse;
        nrmseRange(q) = m.nrmseRange;
        nrmseStd(q) = m.nrmseStd;
        mae(q) = m.mae;
        maxAbsError(q) = m.maxAbsError;
        bias(q) = m.biasEstimateMinusReal;
        relativeL2(q) = m.relativeL2;
        pearson(q) = m.pearsonCorrelation;
        cosine(q) = m.cosineSimilarity;
    end

    T = table( ...
        string(signalNames(:)), N, oneMinusR2, ...
        oneMinusR2Percent, r2, rmse, nrmseRange, ...
        nrmseStd, mae, maxAbsError, bias, relativeL2, ...
        pearson, cosine, ...
        'VariableNames', { ...
            'Signal','N','OneMinusR2','OneMinusR2Percent', ...
            'R2','RMSE','NRMSERange','NRMSEStd','MAE', ...
            'MaxAbsError','BiasEstimateMinusReal','RelativeL2', ...
            'PearsonCorrelation','CosineSimilarity'});
end

function print_metric_block(name, metrics)
    fprintf('\n%s\n', name);
    fprintf('  100*(1-r^2) = %.10e %%\n', metrics.oneMinusR2Percent);
    fprintf('  r^2          = %.8f\n', metrics.r2);
    fprintf('  RMSE         = %.8e\n', metrics.rmse);
    fprintf('  Range NRMSE  = %.8e\n', metrics.nrmseRange);
    fprintf('  Std. NRMSE   = %.8e\n', metrics.nrmseStd);
    fprintf('  MAE          = %.8e\n', metrics.mae);
    fprintf('  Max |error|  = %.8e\n', metrics.maxAbsError);
    fprintf('  Est.-ref. bias = %.8e\n', ...
        metrics.biasEstimateMinusReal);
    fprintf('  Correlation  = %.8f\n', ...
        metrics.pearsonCorrelation);
    fprintf('  Cosine sim.  = %.8f\n', ...
        metrics.cosineSimilarity);
end

function selectedTable = build_selected_library_table(model)
    selectedIdx = model.selectedGlobalIdx(:);
    selectedByEnergy = ismember(selectedIdx, model.selectedE(:));
    selectedByDynamic = ismember(selectedIdx, model.selectedD(:));

    energy = model.Ei(selectedIdx);
    dynamicScore = model.Di(selectedIdx);
    coefficient = model.Wspline(:);

    selectedTable = table( ...
        selectedIdx, model.levelSel(:), model.localSel(:), ...
        selectedByEnergy, selectedByDynamic, energy, ...
        dynamicScore, coefficient, ...
        'VariableNames', { ...
            'GlobalIndex','Level','LocalIndex', ...
            'SelectedByEnergy','SelectedByDynamic', ...
            'EnergyScore','DynamicScore','Coefficient'});

    NI = size(model.Csel,2);
    centerNames = cell(1,NI);
    scaleNames = cell(1,NI);
    indexNames = cell(1,NI);

    for q = 1:NI
        centerNames{q} = sprintf('CenterZ%d', q);
        scaleNames{q} = sprintf('ScaleZ%d', q);
        indexNames{q} = sprintf('GridIndexZ%d', q);
    end

    selectedTable = [
        selectedTable, ...
        array2table(model.Csel, 'VariableNames', centerNames), ...
        array2table(model.Hsel, 'VariableNames', scaleNames), ...
        array2table(model.jSel, 'VariableNames', indexNames)
    ];
end

function write_spike_table(spike, filePath)
    nRows = max( ...
        [numel(spike.timesReal), ...
         numel(spike.timesReconstructed), ...
         numel(spike.matchedTimesReal)]);

    if nRows == 0
        T = table( ...
            zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
            'VariableNames', { ...
                'RealSpikeTime_s','ReconstructedSpikeTime_s', ...
                'MatchedRealTime_s','MatchedReconstructedTime_s'});
    else
        data = NaN(nRows,4);
        data(1:numel(spike.timesReal),1) = spike.timesReal;
        data(1:numel(spike.timesReconstructed),2) = ...
            spike.timesReconstructed;
        data(1:numel(spike.matchedTimesReal),3) = ...
            spike.matchedTimesReal;
        data(1:numel(spike.matchedTimesReconstructed),4) = ...
            spike.matchedTimesReconstructed;

        T = array2table(data, 'VariableNames', { ...
            'RealSpikeTime_s','ReconstructedSpikeTime_s', ...
            'MatchedRealTime_s','MatchedReconstructedTime_s'});
    end

    writetable(T, filePath);
end

function save_figure_high_resolution( ...
    fig, figureDirectory, baseName, resolutionDpi, saveMatlabFigure)

    pngPath = fullfile(figureDirectory, [baseName '.png']);
    exportgraphics( ...
        fig, pngPath, ...
        'Resolution', resolutionDpi, ...
        'BackgroundColor', 'white');

    if saveMatlabFigure
        savefig(fig, fullfile(figureDirectory, [baseName '.fig']));
    end

    close(fig);
end

function fig = create_figure(showFigures, position)
    if showFigures
        visibility = 'on';
    else
        visibility = 'off';
    end

    fig = figure( ...
        'Visible', visibility, ...
        'Color', 'white', ...
        'Position', position);
    
    set(fig, ...
        'DefaultTextInterpreter', 'tex', ...
        'DefaultLegendInterpreter', 'tex', ...
        'DefaultAxesTickLabelInterpreter', 'tex');
end

function indexRange = worst_error_window(errorSignal, windowSamples)
    errorSignal = errorSignal(:);
    windowSamples = max(2, min(round(windowSamples), numel(errorSignal)));

    localMSE = movmean(errorSignal.^2, windowSamples);
    [~, center] = max(localMSE);

    halfWindow = floor(windowSamples/2);
    firstIndex = max(1, center - halfWindow);
    lastIndex = min(numel(errorSignal), firstIndex + windowSamples - 1);
    firstIndex = max(1, lastIndex - windowSamples + 1);

    indexRange = firstIndex:lastIndex;
end

function value = estimate_structure_bytes(variableName)
    info = evalin('caller', ['whos(''' variableName ''')']);
    if isempty(info)
        value = NaN;
    else
        value = info.bytes;
    end
end

%% =========================================================================
%% MORRIS--LECAR MODEL FUNCTIONS
%% =========================================================================

function X = simulate_morris_lecar(X0, Iapp, integrationStepMs, p)
    N = numel(Iapp);
    X = zeros(N,2);
    X(1,:) = X0(:).';

    for k = 1:N-1
        state = X(k,:).';
        current = Iapp(k);

        k1 = morris_lecar_dynamics(state, current, p);
        k2 = morris_lecar_dynamics( ...
            state + 0.5*integrationStepMs*k1, current, p);
        k3 = morris_lecar_dynamics( ...
            state + 0.5*integrationStepMs*k2, current, p);
        k4 = morris_lecar_dynamics( ...
            state + integrationStepMs*k3, current, p);

        X(k+1,:) = ...
            (state + integrationStepMs*(k1 + 2*k2 + 2*k3 + k4)/6).';
    end
end

function derivative = morris_lecar_dynamics(state, current, p)
    v = state(1);
    w = state(2);

    mInfinity = m_inf(v, p);
    wInfinity = w_inf(v, p);
    tauW = tau_w(v, p);

    dv = ( ...
        -p.gL*(v - p.EL) ...
        -p.gK*w*(v - p.EK) ...
        -p.gCa*mInfinity*(v - p.ECa) ...
        + current) / p.Cm;

    dw = p.phi * (wInfinity - w) / tauW;

    derivative = [dv; dw];
end

function value = m_inf(v, p)
    value = 0.5 * (1 + tanh((v - p.V1)/p.V2));
end

function value = w_inf(v, p)
    value = 0.5 * (1 + tanh((v - p.V3)/p.V4));
end

function value = tau_w(v, p)
    value = 1 ./ cosh((v - p.V3)/(2*p.V4));
end

function scan = scan_morris_lecar_currents( ...
    X0, currents, simulationDurationSeconds, ...
    samplingFrequency, integrationStepMs, p, steadyFraction, spikeThreshold)

    Nscan = round(simulationDurationSeconds * samplingFrequency) + 1;
    nCurrents = numel(currents);

    scan.current = currents(:);
    scan.vMin = zeros(nCurrents,1);
    scan.vMax = zeros(nCurrents,1);
    scan.amplitude = zeros(nCurrents,1);
    scan.spikeCount = zeros(nCurrents,1);

    for q = 1:nCurrents
        constantCurrent = currents(q) * ones(Nscan,1);
        X = simulate_morris_lecar( ...
            X0, constantCurrent, integrationStepMs, p);

        firstSteadyIndex = max(1, round(steadyFraction*Nscan));
        vSteady = X(firstSteadyIndex:end,1);

        scan.vMin(q) = min(vSteady);
        scan.vMax(q) = max(vSteady);
        scan.amplitude(q) = scan.vMax(q) - scan.vMin(q);
        scan.spikeCount(q) = ...
            sum(diff(vSteady > spikeThreshold) == 1);
    end
end

function transitionCurrent = estimate_transition_current( ...
    currents, amplitudes, amplitudeThreshold)

    currents = currents(:);
    amplitudes = amplitudes(:);

    firstOscillatory = find(amplitudes >= amplitudeThreshold, 1, 'first');

    if isempty(firstOscillatory)
        transitionCurrent = NaN;
    elseif firstOscillatory == 1
        transitionCurrent = currents(1);
    else
        x1 = currents(firstOscillatory-1);
        x2 = currents(firstOscillatory);
        y1 = amplitudes(firstOscillatory-1);
        y2 = amplitudes(firstOscillatory);

        if y2 == y1
            transitionCurrent = x2;
        else
            transitionCurrent = ...
                x1 + (amplitudeThreshold-y1)*(x2-x1)/(y2-y1);
        end
    end
end
