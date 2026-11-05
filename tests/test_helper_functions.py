import numpy as np
import pytest

from wasserstein_score.helper_functions import (
    abrupt_linear_drift_stream,
    bimodal_shift_streams,
    build_model_df,
    friedman_f1_f2_switch,
    friedman_f2_f1_switch,
    get_model,
    held_out_null_streams,
    null_stream_seed_base,
    river_friedman_drift_stream,
)


def test_xgboost_sklearn_wrapper_is_available():
    X = np.arange(24, dtype=float).reshape(8, 3)
    y = X.sum(axis=1)

    model = get_model(
        build_model_df(X, y),
        num_training_points=6,
        num_test_points=0,
        features_col=range(X.shape[1]),
        target_col="target",
        xgb_params={
            "n_estimators": 1,
            "max_depth": 1,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
            "n_jobs": 1,
            "random_state": 42,
        },
    )

    assert model.predict(X[:1]).shape == (1,)


@pytest.mark.parametrize(
    "generator, extra_kwargs",
    [
        (abrupt_linear_drift_stream, {"noise": 0.1}),
        (friedman_f1_f2_switch, {"noise": 0.1}),
        (friedman_f2_f1_switch, {"noise": 0.1}),
        (river_friedman_drift_stream, {
            "num_training_points": 20,
            "num_test_points": 10,
            "drift_scenarios": {"lea": {"river_type": "lea", "window": 0}},
        }),
        (bimodal_shift_streams, {"separation": 0.99}),
    ],
)
def test_held_out_null_streams_are_disjoint_and_share_the_training_data(
    generator, extra_kwargs
):
    kwargs = dict(
        extra_kwargs, n=40, drift_at=30, n_calibrations_streams=20, n_test_streams=50
    )
    X_training, _, X_calib, _, X_test, _ = generator(**kwargs)
    X_null, y_null = held_out_null_streams(generator, 50, **kwargs)
    X_null_training = generator(
        **dict(kwargs, calibration_seed_base=null_stream_seed_base(generator, **kwargs))
    )[0]

    np.testing.assert_array_equal(X_null_training, X_training)
    assert len(X_null) == len(y_null) == 50
    used = {x.tobytes() for x in X_calib} | {
        x.tobytes() for streams in X_test.values() for x in streams
    }
    assert not {x.tobytes() for x in X_null} & used


@pytest.mark.parametrize("seed_base", [2_000, 2_100, 2_200])
def test_bimodal_null_streams_follow_seed_base(seed_base):
    kwargs = dict(
        n=40, drift_at=30, separation=0.9, seed_base=seed_base,
        n_calibrations_streams=20, n_test_streams=50,
    )
    _, _, X_calib, _, X_test, _ = bimodal_shift_streams(**kwargs)
    X_null, _ = held_out_null_streams(bimodal_shift_streams, 50, **kwargs)

    assert null_stream_seed_base(bimodal_shift_streams, **kwargs) == seed_base + 300
    used = {x.tobytes() for x in X_calib} | {
        x.tobytes() for streams in X_test.values() for x in streams
    }
    assert not {x.tobytes() for x in X_null} & used


def test_null_streams_reusing_a_test_seed_are_rejected():
    # Linear test seeds 200..349 run into the null range starting at 300.
    with pytest.raises(ValueError, match="reuse seeds"):
        held_out_null_streams(
            abrupt_linear_drift_stream, 10,
            n=40, drift_at=30, n_calibrations_streams=2, n_test_streams=150,
        )
