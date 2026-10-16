import numpy as np


def calculate_psi(
    reference,
    current,
    bins=10,
    epsilon=1e-6,
):
    """Calculate Population Stability Index using reference quantiles."""

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if reference.size == 0 or current.size == 0:
        raise ValueError("Both distributions must contain values")

    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("Distributions must contain finite values")

    if bins < 2:
        raise ValueError("bins must be at least 2")

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        raise ValueError("Reference distribution has insufficient variation")

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    reference_pct = reference_counts / reference_counts.sum()
    current_pct = current_counts / current_counts.sum()

    reference_pct = np.maximum(reference_pct, epsilon)
    current_pct = np.maximum(current_pct, epsilon)

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)