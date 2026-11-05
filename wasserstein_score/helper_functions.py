import inspect

from river.datasets import synth
import xgboost as xgb
import numpy as np
import pandas as pd

from wasserstein_score.benchmark_protocol import (
    BIMODAL_SHIFT_SEPARATION,
    DRIFT_AT,
    N_CALIBRATION_STREAMS,
    N_TEST_STREAMS,
    NULL_STREAM_SEED_BASES,
    NULL_STREAM_SEED_OFFSETS,
    NUM_DATAPOINTS,
    NUM_MODEL_TEST_POINTS,
    NUM_TRAINING_POINTS,
)


def get_model(
    df,
    num_training_points,
    num_test_points,
    features_col,
    target_col,
    xgb_params=None,
):
    if xgb_params is None:
        xgb_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
            "random_state": 42,
            "objective": "reg:squarederror",
        }

    # Split dataset
    X = df.iloc[:num_training_points][features_col]
    Y = df.iloc[:num_training_points][target_col]

    # Create and train XGBoost model
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X, Y)

    return model


def build_model_df(X, y, target_name='target'):
    n_features = X.shape[1]
    df = pd.DataFrame(X, columns=range(n_features))
    df[target_name] = y
    return df


def _with_defaults(generator, kwargs):
    """``kwargs`` completed with ``generator``'s default argument values."""
    params = inspect.signature(generator).parameters
    return {
        name: kwargs.get(name, param.default)
        for name, param in params.items()
        if name in kwargs or param.default is not inspect.Parameter.empty
    }


def null_stream_seed_base(generator, **kwargs):
    """First calibration-branch seed of ``generator``'s held-out null streams."""
    name = generator.__name__
    if name in NULL_STREAM_SEED_OFFSETS:
        return _with_defaults(generator, kwargs)["seed_base"] + NULL_STREAM_SEED_OFFSETS[name]
    return NULL_STREAM_SEED_BASES[name]


def generator_seeds(generator, **kwargs):
    """Every RNG seed ``generator`` draws from with ``kwargs``: weights,
    training, and each calibration and test stream (including the Friedman
    generators' separate noise seeds)."""
    kw = _with_defaults(generator, kwargs)
    name = generator.__name__
    n_calib, n_test = kw["n_calibrations_streams"], kw["n_test_streams"]
    calib_base = kw.get("calibration_seed_base")

    if name == "abrupt_linear_drift_stream":
        start = 100 if calib_base is None else calib_base
        return {0, 1} | set(range(start, start + n_calib)) | set(range(200, 200 + n_test))
    if name in ("friedman_f1_f2_switch", "friedman_f2_f1_switch"):
        start = 100 if calib_base is None else calib_base
        calib = range(start, start + n_calib)
        test = range(100 + n_calib, 100 + n_calib + n_test)
        return (
            {0, 1} | set(calib) | {s + 1000 for s in calib}
            | set(test) | {s + 2000 for s in test}
        )
    if name == "river_friedman_drift_stream":
        start = 10 if calib_base is None else calib_base
        return {0} | set(range(start, start + n_calib)) | set(range(10 + n_calib, 10 + n_calib + n_test))
    if name == "bimodal_shift_streams":
        seed_base = kw["seed_base"]
        start = seed_base + 100 if calib_base is None else calib_base
        return (
            {seed_base, seed_base + 1} | set(range(start, start + n_calib))
            | set(range(seed_base + 200, seed_base + 200 + n_test))
        )
    raise ValueError(f"No seed layout for generator {name!r}")


def held_out_null_streams(generator, n_streams, **kwargs):
    """Held-out drift-free streams for the placebo detection rate.

    Draws ``n_streams`` streams from the calibration branch of ``generator``,
    starting at :func:`null_stream_seed_base`. The training data, and so the
    fitted model, do not depend on the calibration seeds, so these streams share
    the pre-drift process of the test streams generated with the same
    ``kwargs``. Raises ``ValueError`` if a null stream would reuse a seed of
    the weights, training, calibration or test streams generated with
    ``kwargs``. Returns ``(X_streams, y_streams)``.
    """
    null_kwargs = dict(
        kwargs,
        n_test_streams=0,
        n_calibrations_streams=n_streams,
        calibration_seed_base=null_stream_seed_base(generator, **kwargs),
    )
    used = generator_seeds(generator, **kwargs)
    null_calib = generator_seeds(generator, **null_kwargs) - generator_seeds(
        generator, **dict(null_kwargs, n_calibrations_streams=0)
    )
    overlap = used & null_calib
    if overlap:
        raise ValueError(
            f"{generator.__name__}: held-out null streams would reuse seeds "
            f"{sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''} of the "
            "calibration/test data; reduce the number of streams or move the "
            "seed ranges."
        )
    _, _, X_null, y_null, _, _ = generator(**null_kwargs)
    return X_null, y_null


def abrupt_linear_drift_stream(
    n, drift_at, n_features=10, noise=0.2,
    n_test_streams=5, n_calibrations_streams=10, n_training_streams=1,
    scenario_name='abrupt', calibration_seed_base=None,
):
    # One shared weight vector defines the pre-drift DGP for ALL splits.
    # Drawn from a dedicated RNG that nothing else touches.
    rng_weights = np.random.default_rng(0)
    w1 = rng_weights.uniform(-1, 1, n_features)
    w2 = rng_weights.uniform(-1, 1, n_features)  # independent post-drift weights

    # Training: same DGP as pre-drift test/calib, independent data RNG
    rng_train = np.random.default_rng(1)
    X_training = rng_train.uniform(0, 1, (n * n_training_streams, n_features))
    y_training = X_training @ w1 + rng_train.normal(0, noise, n * n_training_streams)

    # Calibration: same pre-drift DGP (w1), each stream gets its own data RNG
    calib_seed_start = 100 if calibration_seed_base is None else calibration_seed_base
    calib_seeds = np.arange(
        calib_seed_start, calib_seed_start + n_calibrations_streams
    )
    X_calib_data, y_calib_data = [], []
    for seed in calib_seeds:
        rng_c = np.random.default_rng(seed)
        X_calib = rng_c.uniform(0, 1, (n, n_features))
        y_calib = X_calib @ w1 + rng_c.normal(0, noise, n)
        X_calib_data.append(X_calib)
        y_calib_data.append(y_calib)

    # Test: pre-drift uses w1, post-drift uses w2, each stream gets its own data RNG
    test_seeds = np.arange(200, 200 + n_test_streams)
    X_data, y_data = [], []
    for seed in test_seeds:
        rng_t = np.random.default_rng(seed)
        X = rng_t.uniform(0, 1, (n, n_features))
        noise_vec = rng_t.normal(0, noise, n)
        y = np.concatenate([
            X[:drift_at] @ w1 + noise_vec[:drift_at],
            X[drift_at:] @ w2 + noise_vec[drift_at:],
        ])
        X_data.append(X)
        y_data.append(y)

    return X_training, y_training, X_calib_data, y_calib_data, {scenario_name: X_data}, {scenario_name: y_data}


def friedman_f1_f2_switch(
    n, drift_at, noise=0.1,
    n_test_streams=5, n_calibrations_streams=10, n_training_streams=1,
    scenario_name='abrupt', calibration_seed_base=None,
):
    def f1(X):
        return (
            10 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 20 * (X[:, 2] - 0.5) ** 2
            + 10 * X[:, 3]
            + 5 * X[:, 4]
        )

    def f2(X):
        # Standard F2 — x3 in [1, 11] keeps denominator safely away from zero
        x0 = X[:, 0] * 100          # rescale to [0, 100]
        x1 = X[:, 1] * (560 - 40) * np.pi + 40 * np.pi  # [40π, 560π]
        x2 = X[:, 2]                 # [0, 1]
        x3 = X[:, 3] * 10 + 1       # rescale to [1, 11]
        x4 = X[:, 4]
        raw = np.sqrt(x0 ** 2 + (x1 * x2 - 1 / (x1 * x3)) ** 2) + 5 * x4
        # raw ~ N(482.13, 378.99); rescaled to match f1's mean/std (14.414 / 4.882)
        # under X ~ U(0,1)^5, so switching f1<->f2 changes shape, not scale.
        return (raw - 482.13) * (4.882 / 378.99) + 14.414

    n_features = 5

    # Test seeds follow the default calibration range; a custom
    # calibration_seed_base moves only the calibration draw.
    calib_seed_start = 100 if calibration_seed_base is None else calibration_seed_base
    calib_seeds = np.arange(calib_seed_start, calib_seed_start + n_calibrations_streams)
    test_seeds = np.arange(100 + n_calibrations_streams, 100 + n_calibrations_streams + n_test_streams)

    rng_train = np.random.default_rng(0)
    rng_train_noise = np.random.default_rng(1)
    X_training = rng_train.uniform(0, 1, (n * n_training_streams, n_features))
    y_training = f1(X_training) + rng_train_noise.normal(0, noise, n * n_training_streams)

    X_calib_data, y_calib_data = [], []
    for seed in calib_seeds:
        rng_c = np.random.default_rng(seed)
        rng_c_noise = np.random.default_rng(seed + 1000)
        X_calib = rng_c.uniform(0, 1, (n, n_features))
        y_calib = f1(X_calib) + rng_c_noise.normal(0, noise, n)
        X_calib_data.append(X_calib)
        y_calib_data.append(y_calib)

    X_data, y_data = [], []
    for seed in test_seeds:
        rng_t = np.random.default_rng(seed)
        rng_t_noise = np.random.default_rng(seed + 2000)
        X = rng_t.uniform(0, 1, (n, n_features))
        y = np.where(
            np.arange(n) < drift_at,
            f1(X),
            f2(X),
        )
        y += rng_t_noise.normal(0, noise, n)
        X_data.append(X)
        y_data.append(y)

    return X_training, y_training, X_calib_data, y_calib_data, {scenario_name: X_data}, {scenario_name: y_data}


def friedman_f2_f1_switch(
    n, drift_at, noise=0.1,
    n_test_streams=5, n_calibrations_streams=10, n_training_streams=1,
    scenario_name='abrupt', calibration_seed_base=None,
):
    def f1(X):
        # Standard F1
        return (
            10 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 20 * (X[:, 2] - 0.5) ** 2
            + 10 * X[:, 3]
            + 5 * X[:, 4]
        )

    def f2(X):
        # Standard F2 — x3 in [1, 11] keeps denominator safely away from zero
        x0 = X[:, 0] * 100          # rescale to [0, 100]
        x1 = X[:, 1] * (560 - 40) * np.pi + 40 * np.pi  # [40π, 560π]
        x2 = X[:, 2]                 # [0, 1]
        x3 = X[:, 3] * 10 + 1       # rescale to [1, 11]
        x4 = X[:, 4]
        raw = np.sqrt(x0 ** 2 + (x1 * x2 - 1 / (x1 * x3)) ** 2) + 5 * x4
        # raw ~ N(482.13, 378.99); rescaled to match f1's mean/std (14.414 / 4.882)
        # under X ~ U(0,1)^5, so switching f1<->f2 changes shape, not scale.
        return (raw - 482.13) * (4.882 / 378.99) + 14.414

    n_features = 5

    # Test seeds follow the default calibration range; a custom
    # calibration_seed_base moves only the calibration draw.
    calib_seed_start = 100 if calibration_seed_base is None else calibration_seed_base
    calib_seeds = np.arange(calib_seed_start, calib_seed_start + n_calibrations_streams)
    test_seeds = np.arange(100 + n_calibrations_streams, 100 + n_calibrations_streams + n_test_streams)

    rng_train = np.random.default_rng(0)
    rng_train_noise = np.random.default_rng(1)
    X_training = rng_train.uniform(0, 1, (n * n_training_streams, n_features))
    y_training = f2(X_training) + rng_train_noise.normal(0, noise, n * n_training_streams)

    X_calib_data, y_calib_data = [], []
    for seed in calib_seeds:
        rng_c = np.random.default_rng(seed)
        rng_c_noise = np.random.default_rng(seed + 1000)
        X_calib = rng_c.uniform(0, 1, (n, n_features))
        y_calib = f2(X_calib) + rng_c_noise.normal(0, noise, n)
        X_calib_data.append(X_calib)
        y_calib_data.append(y_calib)

    X_data, y_data = [], []
    for seed in test_seeds:
        rng_t = np.random.default_rng(seed)
        rng_t_noise = np.random.default_rng(seed + 2000)
        X = rng_t.uniform(0, 1, (n, n_features))
        y = np.where(
            np.arange(n) < drift_at,
            f2(X),
            f1(X),
        )
        y += rng_t_noise.normal(0, noise, n)
        X_data.append(X)
        y_data.append(y)

    return X_training, y_training, X_calib_data, y_calib_data, {scenario_name: X_data}, {scenario_name: y_data}


def river_friedman_drift_stream(
    n, drift_at, n_calibrations_streams=10, n_test_streams=30,
    num_training_points=3500, num_test_points=1500,
    drift_scenarios=None, calibration_seed_base=None,
):
    if drift_scenarios is None:
        # A single mixed-concept transition: River's GSG generator draws the old
        # or new concept with probability 1/2 throughout the window, so longer
        # windows share their early paths with this one and add no scenario
        # that is distinct within the detection horizon.
        drift_scenarios = {
            'abrupt_local_lea':  {'river_type': 'lea', 'window': 0},
            'abrupt_global_gsg': {'river_type': 'gsg', 'window': 0},
            'gradual_fast':      {'river_type': 'gsg', 'window': 500},
        }

    n_total = n + num_training_points + num_test_points
    eval_start = num_training_points + num_test_points
    drift_at_absolute = eval_start + drift_at

    calibration_seed_start = 10 if calibration_seed_base is None else calibration_seed_base
    calibration_seeds = list(
        range(calibration_seed_start, calibration_seed_start + n_calibrations_streams)
    )
    test_seeds = list(range(10 + n_calibrations_streams, 10 + n_calibrations_streams + n_test_streams))

    target_col = 10
    features = list(range(10))

    # Base dataset: used for model training/validation
    base_values = []
    for x, y in synth.Friedman(seed=0).take(n_total):
        base_values.append(list(x.values()) + [y])
    df_base = pd.DataFrame(base_values)

    X_training = df_base.iloc[:num_training_points + num_test_points][features].values
    y_training = df_base.iloc[:num_training_points + num_test_points][target_col].values

    # Calibration streams: no-drift Friedman, evaluation window only
    X_calib_data, y_calib_data = [], []
    for seed in calibration_seeds:
        values = []
        for x, y in synth.Friedman(seed=seed).take(n_total):
            values.append(list(x.values()) + [y])
        df = pd.DataFrame(values).iloc[eval_start:].reset_index(drop=True)
        X_calib_data.append(df[features].values)
        y_calib_data.append(df[target_col].values)

    # Test streams: one list of streams per drift scenario, evaluation window only
    X_test_data = {name: [] for name in drift_scenarios}
    y_test_data = {name: [] for name in drift_scenarios}
    for seed in test_seeds:
        for name, scenario in drift_scenarios.items():
            river_type = scenario['river_type']
            tw = scenario['window']
            if river_type == 'lea':
                position = (drift_at_absolute, drift_at_absolute + n_total, drift_at_absolute + n_total + 1)
            else:
                position = (drift_at_absolute, n_total)
            drift_dataset = synth.FriedmanDrift(
                drift_type=river_type,
                position=position,
                transition_window=tw,
                seed=seed,
            )
            values = []
            for x, y in drift_dataset.take(n_total):
                values.append(list(x.values()) + [y])
            df_drift = pd.DataFrame(values).iloc[eval_start:].reset_index(drop=True)
            X_test_data[name].append(df_drift[features].values)
            y_test_data[name].append(df_drift[target_col].values)

    return X_training, y_training, X_calib_data, y_calib_data, X_test_data, y_test_data


def bimodal_shift_streams(
    n, drift_at, n_features=10, noise_mean=0.0, base_sigma=1.0, separation=0.6,
    n_test_streams=30, n_calibrations_streams=10, n_training_streams=1, seed_base=2000,
    calibration_seed_base=None,
):
    """Mean- and variance-preserving bimodal shift in a trained model's residuals.

    Pre-drift noise is N(noise_mean, base_sigma). Post-drift noise is a
    symmetric two-component Gaussian mixture
        0.5 * N(noise_mean - separation, b) + 0.5 * N(noise_mean + separation, b)
    with b = sqrt(base_sigma**2 - separation**2), so the mixture mean and
    variance exactly match the pre-drift regime. The squared-residual mean
    is preserved for a fixed predictor and unchanged feature distribution,
    but its variance decreases by 2 * separation**4. Equal means alone do
    not imply that a detector cannot react to the shape change.
    """
    if separation >= base_sigma:
        raise ValueError("separation must be strictly less than base_sigma")
    b = np.sqrt(base_sigma**2 - separation**2)

    def sample_bimodal(rng, size):
        signs = rng.choice([-1.0, 1.0], size=size)
        return noise_mean + signs * separation + rng.normal(0, b, size)

    rng_weights = np.random.default_rng(seed_base)
    w = rng_weights.uniform(-1, 1, n_features)

    rng_train = np.random.default_rng(seed_base + 1)
    X_training = rng_train.uniform(0, 1, (n * n_training_streams, n_features))
    y_training = X_training @ w + rng_train.normal(noise_mean, base_sigma, n * n_training_streams)

    calib_seed_start = (
        seed_base + 100 if calibration_seed_base is None else calibration_seed_base
    )
    calib_seeds = range(calib_seed_start, calib_seed_start + n_calibrations_streams)
    X_calib_data, y_calib_data = [], []
    for s in calib_seeds:
        rng_c = np.random.default_rng(s)
        X_calib = rng_c.uniform(0, 1, (n, n_features))
        y_calib = X_calib @ w + rng_c.normal(noise_mean, base_sigma, n)
        X_calib_data.append(X_calib)
        y_calib_data.append(y_calib)

    test_seeds = range(seed_base + 200, seed_base + 200 + n_test_streams)
    X_data, y_data = [], []
    for s in test_seeds:
        rng_t = np.random.default_rng(s)
        X = rng_t.uniform(0, 1, (n, n_features))
        noise = np.concatenate([
            rng_t.normal(noise_mean, base_sigma, drift_at),
            sample_bimodal(rng_t, n - drift_at),
        ])
        X_data.append(X)
        y_data.append(X @ w + noise)

    return X_training, y_training, X_calib_data, y_calib_data, {"bimodal_shift": X_data}, {"bimodal_shift": y_data}


def benchmark_datasets():
    """The main benchmark's datasets as ``(label, generator, kwargs)``.

    Diagnostic experiments reuse this list so they run on exactly the streams,
    stream lengths and calibration/test sizes of the benchmark they explain.
    """
    common = dict(
        n=NUM_DATAPOINTS,
        drift_at=DRIFT_AT,
        n_calibrations_streams=N_CALIBRATION_STREAMS,
        n_test_streams=N_TEST_STREAMS,
    )
    return [
        ("Linear Abrupt", abrupt_linear_drift_stream, dict(
            common, noise=0.1, scenario_name="abrupt_linear",
        )),
        ("Friedman F1 → F2 Switch", friedman_f1_f2_switch, dict(
            common, noise=0.1, scenario_name="abrupt_f1_f2",
        )),
        ("Friedman F2 → F1 Switch", friedman_f2_f1_switch, dict(
            common, noise=0.1, scenario_name="abrupt_f2_f1",
        )),
        ("River Friedman Dataset", river_friedman_drift_stream, dict(
            common,
            num_training_points=NUM_TRAINING_POINTS,
            num_test_points=NUM_MODEL_TEST_POINTS,
        )),
        ("Bimodal Shift", bimodal_shift_streams, dict(
            common, base_sigma=1.0, separation=BIMODAL_SHIFT_SEPARATION,
            seed_base=2000,
        )),
    ]


# ---------------------------------------------------------------------------
# Markdown report print helpers
# ---------------------------------------------------------------------------
#
# Console output from scripts like run_all_tests.py is valid Markdown, so it
# can be pasted into a Markdown editor and keep its title/subtitle structure.

def h1(text: str) -> None:
    print(f"\n# {text}\n")


def h2(text: str) -> None:
    print(f"\n## {text}\n")


def h3(text: str) -> None:
    print(f"\n### {text}\n")


def bullet(text: str) -> None:
    print(f"- {text}")


def para(text: str) -> None:
    print(f"\n{text}\n")


def _fmt_cell(value) -> str:
    if isinstance(value, float):
        return "NaN" if pd.isna(value) else f"{value:.4g}"
    return str(value)


def table(df: pd.DataFrame, index_label: str | None = None) -> None:
    """Print a DataFrame as a GitHub-flavored Markdown table."""
    label = index_label if index_label is not None else (df.index.name or "")
    headers = [label] + [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [_fmt_cell(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    print()
    print("\n".join(lines))
    print()
