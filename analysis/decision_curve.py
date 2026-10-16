"""Thresholds at which net benefit crosses the screen-all and screen-none strategies.
"""
import sys

import numpy as np
import pandas as pd

IN = "results/primary_comparisons_dca.csv"
OUT = "results/decision_curve.csv"


def crossing(thresholds, diff):
    """First threshold (linearly interpolated) where `diff` goes from >0 to <=0,
    scanning in increasing threshold order. None if no such crossing exists."""
    for i in range(len(thresholds) - 1):
        d0, d1 = diff[i], diff[i + 1]
        if d0 > 0 and d1 <= 0:
            t0, t1 = thresholds[i], thresholds[i + 1]
            frac = d0 / (d0 - d1)
            return t0 + frac * (t1 - t0)
    return None


def main():
    df = pd.read_csv(IN)
    arms = [a for a in df["arm"].unique() if a not in ("screen_all", "screen_none")]

    screen_all = df[df["arm"] == "screen_all"].sort_values("threshold")
    sa_t = screen_all["threshold"].to_numpy()
    sa_nb = screen_all["net_benefit"].to_numpy()

    rows = []
    for arm in arms:
        sub = df[df["arm"] == arm].sort_values("threshold")
        t = sub["threshold"].to_numpy()
        nb = sub["net_benefit"].to_numpy()
        assert np.allclose(t, sa_t), f"{arm}: threshold grid does not match screen_all's"

        vs_none = crossing(t, nb)
        vs_all = crossing(t, nb - sa_nb)

        def status(diff_at_grid_lo, cross_pt):
            if cross_pt is not None:
                return "crosses_within_grid"
            return "always_exceeds_within_grid" if diff_at_grid_lo > 0 else "never_exceeds_within_grid"

        rows.append({
            "arm": arm,
            "nb_exceeds_screen_all_up_to_pt": vs_all if vs_all is not None else np.nan,
            "nb_exceeds_screen_all_status": status((nb - sa_nb)[0], vs_all),
            "nb_exceeds_screen_none_up_to_pt": vs_none if vs_none is not None else np.nan,
            "nb_exceeds_screen_none_status": status(nb[0], vs_none),
            "grid_lo": float(t.min()), "grid_hi": float(t.max()),
        })
        print(f"[{arm}] crosses screen-all at pt={vs_all}  crosses screen-none(0) at pt={vs_none}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\n[saved] -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
