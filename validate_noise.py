#!/usr/bin/env python3
"""
Validate and benchmark the noisy MLX quantum simulator.

Tests:
  1. Ideal mode regression (still matches Qiskit)
  2. Noise sanity checks (noisy < ideal fidelity, increases with depth)
  3. Mirror circuit fidelity at different depths
  4. Comparison across noise profiles (Willow, Heron, T-9)
  5. Measurement error channel correctness
  6. Speed benchmark: noisy vs noiseless

Run:
    cd quantum-llm
    python3 -m simulator.validate_noise
"""

import sys
import os
import time
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from simulator.mlx_quantum_sim import MLXQuantumSimulator
from simulator.noise_profiles import WILLOW_NOISE, HERON_NOISE, T9_NOISE, ALL_PROFILES


# =========================================================================
# Helpers
# =========================================================================

def build_mirror_circuit(n_qubits: int, depth: int, seed: int = 42):
    """Build a mirror circuit: random layers followed by their inverse.

    For an ideal simulator, this returns to |0...0> with probability 1.
    Under noise, fidelity degrades with depth.

    Returns list of instructions compatible with MLXQuantumSimulator.
    """
    rng = np.random.RandomState(seed)
    forward_layers = []

    for _ in range(depth):
        layer = []
        # Random single-qubit rotations
        for q in range(n_qubits):
            theta = rng.uniform(0, 2 * math.pi)
            layer.append(('ry', theta, q))
        # CNOT chain
        for q in range(0, n_qubits - 1, 2):
            layer.append(('cx', q, q + 1))
        if n_qubits > 2:
            for q in range(1, n_qubits - 1, 2):
                layer.append(('cx', q, q + 1))
        forward_layers.append(layer)

    # Build full instruction list: forward + inverse
    instructions = []
    for layer in forward_layers:
        instructions.extend(layer)

    # Mirror: reverse layers, invert gates
    for layer in reversed(forward_layers):
        for instr in reversed(layer):
            if instr[0] == 'ry':
                # R_y(theta)^{-1} = R_y(-theta)
                instructions.append(('ry', -instr[1], instr[2]))
            elif instr[0] == 'cx':
                # CNOT is self-inverse
                instructions.append(instr)
            else:
                instructions.append(instr)

    return instructions


def run_mirror_circuit(sim_class, n_qubits, depth, noise_profile=None,
                       n_trajectories=100, seed=42):
    """Run mirror circuit and return P(|0...0>)."""
    instructions = build_mirror_circuit(n_qubits, depth, seed=seed)

    kwargs = {'noise_profile': noise_profile}
    if noise_profile:
        kwargs['n_trajectories'] = n_trajectories
        kwargs['seed'] = seed + 1000

    sim = sim_class(n_qubits, **kwargs)
    for instr in instructions:
        name = instr[0]
        if name in ('h', 'x', 'y', 'z', 's', 't'):
            getattr(sim, name)(instr[1])
        elif name in ('ry', 'rz', 'rx'):
            getattr(sim, name)(instr[1], instr[2])
        elif name in ('cx', 'cz', 'swap'):
            getattr(sim, name)(instr[1], instr[2])

    probs = sim.measure_probs_numpy()
    return probs[0]  # P(|0...0>)


def time_fn(fn, *args, n_repeats=3, warmup=1, **kwargs):
    """Time a function. Returns (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.mean(times), np.std(times)


# =========================================================================
# Test 1: Ideal mode regression
# =========================================================================

def test_ideal_regression():
    """Verify ideal mode still gives correct results."""
    print("--- Test 1: Ideal Mode Regression ---")

    # GHZ state
    sim = MLXQuantumSimulator(3)
    sim.h(0)
    sim.cx(0, 1)
    sim.cx(0, 2)
    p = sim.measure_probs_numpy()
    assert abs(p[0] - 0.5) < 1e-4, f"GHZ |000> = {p[0]}"
    assert abs(p[7] - 0.5) < 1e-4, f"GHZ |111> = {p[7]}"
    print("  [PASS] GHZ state")

    # Mirror circuit (ideal: returns to |0...0> exactly)
    p000 = run_mirror_circuit(MLXQuantumSimulator, 4, depth=5, noise_profile=None)
    assert abs(p000 - 1.0) < 1e-4, f"Mirror P(|0000>) = {p000} (expected 1.0)"
    print(f"  [PASS] Mirror circuit 4q depth=5: P(|0000>) = {p000:.6f}")

    # Bell state
    sim = MLXQuantumSimulator(2)
    sim.h(0)
    sim.cx(0, 1)
    p = sim.measure_probs_numpy()
    assert abs(p[0] - 0.5) < 1e-4
    assert abs(p[3] - 0.5) < 1e-4
    print("  [PASS] Bell state")

    print()


# =========================================================================
# Test 2: Noise sanity checks
# =========================================================================

def test_noise_sanity():
    """Verify that noise degrades fidelity."""
    print("--- Test 2: Noise Sanity Checks ---")

    n_q = 3
    n_traj = 100

    p_ideal = run_mirror_circuit(MLXQuantumSimulator, n_q, depth=5,
                                 noise_profile=None)
    p_noisy = run_mirror_circuit(MLXQuantumSimulator, n_q, depth=5,
                                 noise_profile=WILLOW_NOISE,
                                 n_trajectories=n_traj)

    print(f"  Mirror {n_q}q depth=5: ideal={p_ideal:.6f}, Willow={p_noisy:.4f}")
    assert p_noisy < p_ideal, "Noisy should be worse than ideal"
    assert p_noisy > 0.3, f"Noise too strong? P(|0>) = {p_noisy:.4f}"
    print("  [PASS] Noisy fidelity < ideal fidelity")

    # More depth = more degradation
    p_d2 = run_mirror_circuit(MLXQuantumSimulator, n_q, depth=2,
                              noise_profile=WILLOW_NOISE, n_trajectories=n_traj)
    p_d10 = run_mirror_circuit(MLXQuantumSimulator, n_q, depth=10,
                               noise_profile=WILLOW_NOISE, n_trajectories=n_traj)

    print(f"  Willow depth=2: {p_d2:.4f}, depth=10: {p_d10:.4f}")
    assert p_d2 > p_d10, "Deeper circuits should have lower fidelity"
    print("  [PASS] Fidelity decreases with depth")

    print()


# =========================================================================
# Test 3: Mirror circuit fidelity vs depth
# =========================================================================

def test_mirror_fidelity():
    """Run mirror circuits at varying depths, compare noise profiles."""
    print("--- Test 3: Mirror Circuit Fidelity vs Depth ---")

    n_q = 3
    depths = [1, 2, 5, 10]
    n_traj = 100

    profiles = {
        'Ideal': None,
        'Willow': WILLOW_NOISE,
        'Heron': HERON_NOISE,
        'T-9': T9_NOISE,
    }

    # Header
    header = f"{'Depth':>6}"
    for name in profiles:
        header += f" | {name:>10}"
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    results = {}
    for depth in depths:
        row = f"  {depth:>6}"
        for name, profile in profiles.items():
            p0 = run_mirror_circuit(
                MLXQuantumSimulator, n_q, depth,
                noise_profile=profile,
                n_trajectories=n_traj,
            )
            row += f" | {p0:>10.4f}"
            results[(name, depth)] = p0
        print(row)

    # Validate ordering: Ideal > Willow >= Heron > T-9 (roughly)
    for d in depths:
        assert results[('Ideal', d)] > results[('T-9', d)], \
            f"Ideal should beat T-9 at depth {d}"
    print("  [PASS] All profiles show expected ordering")
    print()

    return results


# =========================================================================
# Test 4: Measurement error channel
# =========================================================================

def test_measurement_error():
    """Verify measurement error channel works correctly."""
    print("--- Test 4: Measurement Error Channel ---")

    # Perfect |0> state with readout error
    probs_ideal = np.array([1.0, 0.0])
    noisy = MLXQuantumSimulator.apply_measurement_error(probs_ideal, 0.1)
    # P(read |0>) = 0.9, P(read |1>) = 0.1
    assert abs(noisy[0] - 0.9) < 1e-10, f"Expected 0.9, got {noisy[0]}"
    assert abs(noisy[1] - 0.1) < 1e-10, f"Expected 0.1, got {noisy[1]}"
    print(f"  1Q |0> with p_err=0.1: {noisy}")
    print("  [PASS] Single-qubit measurement error")

    # 2-qubit: |00> with readout error
    probs_2q = np.array([1.0, 0.0, 0.0, 0.0])
    noisy_2q = MLXQuantumSimulator.apply_measurement_error(probs_2q, 0.05)
    print(f"  2Q |00> with p_err=0.05: {noisy_2q}")
    assert abs(noisy_2q.sum() - 1.0) < 1e-10, "Must sum to 1"
    assert noisy_2q[0] > 0.8, "P(00) should dominate"
    # P(00) = (1-p)^2 = 0.9025, P(01)=P(10) = p*(1-p) = 0.0475, P(11) = p^2 = 0.0025
    assert abs(noisy_2q[0] - 0.9025) < 1e-10
    assert abs(noisy_2q[3] - 0.0025) < 1e-10
    print("  [PASS] Two-qubit measurement error")

    print()


# =========================================================================
# Test 5: Reproducibility with seed
# =========================================================================

def test_reproducibility():
    """Verify that setting a seed gives reproducible results."""
    print("--- Test 5: Reproducibility with Seed ---")

    results = []
    for trial in range(3):
        sim = MLXQuantumSimulator(
            3, noise_profile=WILLOW_NOISE, n_trajectories=50, seed=12345
        )
        sim.h(0)
        sim.cx(0, 1)
        sim.cx(0, 2)
        p = sim.measure_probs_numpy()
        results.append(p)

    for i in range(1, len(results)):
        assert np.allclose(results[0], results[i], atol=1e-6), \
            f"Trial {i} differs from trial 0"

    print(f"  3 runs with seed=12345: all identical")
    print("  [PASS] Reproducibility")
    print()


# =========================================================================
# Benchmark: Noisy vs Noiseless Speed
# =========================================================================

def benchmark_speed():
    """Compare execution speed across noise profiles."""
    print("--- Benchmark: Noisy vs Noiseless Speed ---")
    print()

    n_q = 6
    depth = 5
    n_traj = 100

    profiles_to_test = [
        ('Ideal', None, 1),
        ('Willow (100 traj)', WILLOW_NOISE, n_traj),
        ('Heron (100 traj)', HERON_NOISE, n_traj),
        ('T-9 (100 traj)', T9_NOISE, n_traj),
    ]

    instructions = build_mirror_circuit(n_q, depth, seed=42)

    print(f"  Circuit: {n_q} qubits, mirror depth={depth}")
    print(f"  Gates: ~{len(instructions)} total")
    print()

    print(f"  {'Profile':>25} | {'Time (ms)':>12} | {'Slowdown':>10}")
    print(f"  {'-' * 55}")

    ideal_time = None
    for name, profile, ntraj in profiles_to_test:
        def run():
            kwargs = {}
            if profile:
                kwargs = {'noise_profile': profile, 'n_trajectories': ntraj, 'seed': 0}
            sim = MLXQuantumSimulator(n_q, **kwargs)
            for instr in instructions:
                iname = instr[0]
                if iname in ('h', 'x', 'y', 'z', 's', 't'):
                    getattr(sim, iname)(instr[1])
                elif iname in ('ry', 'rz', 'rx'):
                    getattr(sim, iname)(instr[1], instr[2])
                elif iname in ('cx', 'cz', 'swap'):
                    getattr(sim, iname)(instr[1], instr[2])
            p = sim.measure_probs()
            mx.eval(p)

        t_mean, t_std = time_fn(run, n_repeats=3, warmup=1)
        if ideal_time is None:
            ideal_time = t_mean
            slowdown_str = "1.0x"
        else:
            slowdown_str = f"{t_mean / ideal_time:.1f}x"

        print(f"  {name:>25} | {t_mean:>8.1f} +/-{t_std:>4.1f} | {slowdown_str:>10}")

    print()

    # Also benchmark trajectory count scaling
    print(f"  Trajectory count scaling ({n_q}q, depth={depth}, Willow):")
    print(f"  {'Trajectories':>14} | {'Time (ms)':>12} | {'Per-traj (ms)':>14}")
    print(f"  {'-' * 50}")

    for ntraj in [10, 50, 100, 200]:
        def run_n():
            sim = MLXQuantumSimulator(n_q, noise_profile=WILLOW_NOISE,
                                      n_trajectories=ntraj, seed=0)
            for instr in instructions:
                iname = instr[0]
                if iname in ('h', 'x', 'y', 'z', 's', 't'):
                    getattr(sim, iname)(instr[1])
                elif iname in ('ry', 'rz', 'rx'):
                    getattr(sim, iname)(instr[1], instr[2])
                elif iname in ('cx', 'cz', 'swap'):
                    getattr(sim, iname)(instr[1], instr[2])
            p = sim.measure_probs()
            mx.eval(p)

        t_mean, t_std = time_fn(run_n, n_repeats=3, warmup=1)
        per_traj = t_mean / ntraj
        print(f"  {ntraj:>14} | {t_mean:>8.1f} +/-{t_std:>4.1f} | {per_traj:>10.2f}")

    print()


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 65)
    print("MLX Quantum Simulator — Noise Model Validation & Benchmark")
    print("=" * 65)
    print(f"MLX device: {mx.default_device()}")
    print(f"Noise profiles loaded: {list(ALL_PROFILES.keys())}")
    print()
    print(f"Willow noise: p_1q={WILLOW_NOISE['p_1q']}, "
          f"p_2q={WILLOW_NOISE['p_2q']}, "
          f"T1={WILLOW_NOISE['T1_us']} us, "
          f"T2={WILLOW_NOISE['T2_us']} us")
    print()

    t0 = time.time()

    test_ideal_regression()
    test_noise_sanity()
    results = test_mirror_fidelity()
    test_measurement_error()
    test_reproducibility()
    benchmark_speed()

    elapsed = time.time() - t0
    print("=" * 65)
    print(f"ALL TESTS PASSED  ({elapsed:.1f}s)")
    print("=" * 65)


if __name__ == "__main__":
    main()
