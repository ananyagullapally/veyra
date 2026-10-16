import numpy as np
import pytest

from veyra.monitoring.drift import calculate_psi


def test_identical_distributions_have_zero_psi():
    reference = np.linspace(0.001, 0.999, 1000)

    psi = calculate_psi(reference, reference.copy())

    assert psi == pytest.approx(0.0, abs=1e-10)


def test_similar_distributions_have_low_psi():
    rng = np.random.default_rng(42)

    reference = rng.beta(1, 20, 10000)
    current = rng.beta(1, 20, 10000)

    assert calculate_psi(reference, current) < 0.1


def test_shifted_distributions_have_high_psi():
    rng = np.random.default_rng(42)

    reference = rng.beta(1, 20, 10000)
    current = rng.beta(2, 12, 10000)

    assert calculate_psi(reference, current) > 0.25


def test_empty_distribution_raises_error():
    with pytest.raises(ValueError):
        calculate_psi([], [0.1, 0.2])


def test_nonfinite_values_raise_error():
    with pytest.raises(ValueError):
        calculate_psi([0.1, np.nan], [0.2, 0.3])


def test_constant_reference_raises_error():
    with pytest.raises(ValueError):
        calculate_psi([0.5] * 100, [0.4] * 100)


def test_invalid_bins_raise_error():
    with pytest.raises(ValueError):
        calculate_psi([0.1, 0.2], [0.1, 0.2], bins=1)