"""Benchmark SWE near-field and surface-current hot paths.

Default dataset: ``tests/example.sph`` in this repository. For the intended
stress baseline, pass a larger imported feed, for example a 39 degree horn with
NMAX around 719 and MMAX around 11:

    python benchmarks/bench_near_field.py --sph path/to/feed.sph --rings 200 --phi 360

The script reports wall-clock timings for filtering, Legendre cache, Bessel
cache, the general near-field path, and the BOR path. It also writes a compact
cProfile summary for the general and BOR evaluations.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

import numpy as np

from swe import SphericalWaveExpansion
from swe.core import compute_all_modes_legendre, precompute_spherical_bessel


def _build_bor_spherical(n_rings: int, n_phi: int):
    ring_r = np.linspace(1.2, 2.4, n_rings)
    ring_theta = np.linspace(0.25, 1.25, n_rings)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    rr, pp = np.meshgrid(ring_r, phi, indexing="ij")
    tt, _ = np.meshgrid(ring_theta, phi, indexing="ij")
    return rr.ravel(), tt.ravel(), pp.ravel()


def _time(label, fn):
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    print(f"{label:28s} {elapsed:9.4f} s")
    return value, elapsed


def _profile(label, fn, lines: int = 20):
    profiler = cProfile.Profile()
    profiler.enable()
    fn()
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
    stats.print_stats(lines)
    print(f"\n--- cProfile: {label} ---")
    print(stream.getvalue())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sph", type=Path, default=Path("tests/example.sph"))
    parser.add_argument("--frequency-index", type=int, default=0)
    parser.add_argument("--rings", type=int, default=120)
    parser.add_argument("--phi", type=int, default=240)
    parser.add_argument("--no-numba", action="store_true")
    args = parser.parse_args()

    swe = SphericalWaveExpansion.from_sph_file(args.sph, normalize=False)
    freq = swe.frequencies[args.frequency_index]
    r, theta, phi = _build_bor_spherical(args.rings, args.phi)
    use_numba = not args.no_numba

    print(f"SWE file: {args.sph}")
    print(f"Frequency: {freq / 1e9:.6g} GHz")
    print(f"NMAX={swe.NMAX(freq)}, MMAX={swe.MMAX(freq)}")
    print(f"Points: {len(r)} ({args.rings} rings x {args.phi} phi)")
    print()

    q1 = swe.Q1_coeffs(freq)
    q2 = swe.Q2_coeffs(freq)
    active_tuple, _ = _time(
        "power filtering",
        lambda: swe._select_active_modes(
            q1,
            q2,
            swe.NMAX(freq),
            swe.MMAX(freq),
            power_threshold=0.999,
            azimuthal_power_threshold=1e-5,
        ),
    )
    active_modes, effective_nmax, effective_mmax, _total_power = active_tuple
    print(f"Active modes: {len(active_modes)}")

    _time(
        "Legendre cache",
        lambda: compute_all_modes_legendre(effective_nmax, effective_mmax, theta),
    )
    _time(
        "Bessel cache",
        lambda: precompute_spherical_bessel(
            effective_nmax, swe.k(freq) * r, use_numba=use_numba
        ),
    )
    _time(
        "near_field general",
        lambda: swe.near_field(
            r,
            theta,
            phi,
            frequency=freq,
            normalize=False,
            use_numba=use_numba,
            force_general=True,
        ),
    )
    _time(
        "near_field BOR",
        lambda: swe.near_field(
            r,
            theta,
            phi,
            frequency=freq,
            normalize=False,
            use_numba=use_numba,
        ),
    )

    _profile(
        "near_field general",
        lambda: swe.near_field(
            r,
            theta,
            phi,
            frequency=freq,
            normalize=False,
            use_numba=use_numba,
            force_general=True,
        ),
    )
    _profile(
        "near_field BOR",
        lambda: swe.near_field(
            r,
            theta,
            phi,
            frequency=freq,
            normalize=False,
            use_numba=use_numba,
        ),
    )


if __name__ == "__main__":
    main()
