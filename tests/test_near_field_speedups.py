import numpy as np

from swe import SphericalWaveExpansion


FREQ = 1.0e9


def _sample_swe() -> SphericalWaveExpansion:
    return SphericalWaveExpansion(
        Q1_coeffs={
            FREQ: {
                (1, 0): 0.8 + 0.1j,
                (2, 1): -0.15 + 0.05j,
                (2, -1): 0.08 - 0.02j,
                (3, 3): 1e-6 + 0.0j,
            }
        },
        Q2_coeffs={
            FREQ: {
                (1, 0): 0.2 - 0.05j,
                (2, 1): 0.04 + 0.03j,
                (3, 3): 1e-6j,
            }
        },
        NMAX={FREQ: 3},
        MMAX={FREQ: 3},
    )


def _ring_points():
    ring_r = np.array([1.25, 1.55, 1.85])
    ring_theta = np.array([0.45, 0.8, 1.15])
    phi = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    rr, pp = np.meshgrid(ring_r, phi, indexing="ij")
    tt, _ = np.meshgrid(ring_theta, phi, indexing="ij")
    return rr.ravel(), tt.ravel(), pp.ravel()


def test_bor_path_matches_general_reference():
    swe = _sample_swe()
    r, theta, phi = _ring_points()

    E_general, H_general = swe.near_field(
        r,
        theta,
        phi,
        frequency=FREQ,
        normalize=False,
        power_threshold=1.0,
        azimuthal_power_threshold=0.0,
        use_numba=False,
        force_general=True,
    )
    E_bor, H_bor = swe.near_field(
        r,
        theta,
        phi,
        frequency=FREQ,
        normalize=False,
        power_threshold=1.0,
        azimuthal_power_threshold=0.0,
        use_numba=False,
    )

    for bor, general in zip(E_bor + H_bor, E_general + H_general):
        np.testing.assert_allclose(bor, general, rtol=1e-11, atol=1e-11)


def test_numba_general_path_matches_numpy_reference():
    swe = _sample_swe()
    r = np.array([1.21, 1.43, 1.67, 1.92])
    theta = np.array([0.37, 0.71, 1.03, 1.29])
    phi = np.array([0.12, 1.17, 2.53, 4.01])

    E_numpy, H_numpy = swe.near_field(
        r,
        theta,
        phi,
        frequency=FREQ,
        normalize=False,
        power_threshold=1.0,
        azimuthal_power_threshold=0.0,
        use_numba=False,
        force_general=True,
    )
    E_numba, H_numba = swe.near_field(
        r,
        theta,
        phi,
        frequency=FREQ,
        normalize=False,
        power_threshold=1.0,
        azimuthal_power_threshold=0.0,
        use_numba=True,
        force_general=True,
    )

    for numba_value, numpy_value in zip(E_numba + H_numba, E_numpy + H_numpy):
        np.testing.assert_allclose(numba_value, numpy_value, rtol=1e-11, atol=1e-11)


def test_ring_detection_falls_back_for_perturbed_point():
    r, theta, _ = _ring_points()
    theta = theta.copy()
    theta[5] += 1e-7

    assert SphericalWaveExpansion._detect_rings(r, theta) is None


def test_ring_detection_accepts_perfect_rings():
    r, theta, _ = _ring_points()

    detected = SphericalWaveExpansion._detect_rings(r, theta)

    assert detected is not None
    ring_r, ring_theta, point_to_ring = detected
    assert len(ring_r) == 3
    assert len(ring_theta) == 3
    assert point_to_ring.shape == r.shape


def test_azimuthal_power_filter_drops_tiny_m_group():
    swe = _sample_swe()
    q1 = swe.Q1_coeffs(FREQ)
    q2 = swe.Q2_coeffs(FREQ)

    active, _nmax, effective_mmax, _total_power = SphericalWaveExpansion._select_active_modes(
        q1,
        q2,
        swe.NMAX(FREQ),
        swe.MMAX(FREQ),
        power_threshold=1.0,
        azimuthal_power_threshold=1e-5,
    )

    assert (3, 3) not in active
    assert effective_mmax == 1


def test_bor_detection_happens_after_coordinate_transform():
    swe = _sample_swe()
    r, theta, phi = _ring_points()
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    rr = np.column_stack([x, y, z])
    unr = rr / np.linalg.norm(rr, axis=1)[:, np.newaxis]
    dSr = np.ones(len(rr))

    # A translated reflector is still ring-structured after currents_on_surface
    # shifts points into the SWE frame.
    translation = np.array([0.2, -0.1, 0.35])
    shifted_rr = rr + translation
    J_bor, M_bor = swe.currents_on_surface(
        shifted_rr,
        unr,
        dSr,
        swe_origin=translation,
        frequency=FREQ,
        power_threshold=1.0,
        azimuthal_power_threshold=0.0,
        use_numba=False,
    )
    J_general, M_general = swe.currents_on_surface(
        shifted_rr,
        unr,
        dSr,
        swe_origin=translation,
        frequency=FREQ,
        power_threshold=1.0,
        azimuthal_power_threshold=0.0,
        use_numba=False,
        force_general=True,
    )

    np.testing.assert_allclose(J_bor, J_general, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(M_bor, M_general, rtol=1e-11, atol=1e-11)
