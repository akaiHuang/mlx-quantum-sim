# mlx-quantum-sim

**GPU-accelerated quantum circuit simulator for Apple Silicon, with real hardware noise models.**

Simulates quantum circuits on Apple Metal GPU via [MLX](https://github.com/ml-explore/mlx). Includes calibrated noise profiles extracted from Google Willow, IBM Heron, and QuTech Tuna-9.

## Features

- **Metal GPU acceleration**: All gate operations stay on Apple Silicon GPU — no CPU round trips
- **Stochastic noise simulation**: Trajectory-based unraveling (same approach as Google's qsim)
- **Real hardware noise**: Willow calibration data extracted from `cirq-google` (105 qubits, 364 CZ pairs)
- **Up to ~23 qubits** on M1 Max 32GB (statevector simulation, O(2^n) memory)
- **Backward compatible**: No noise profile = ideal simulation

## Quick Start

```bash
pip install mlx
```

```python
from mlx_quantum_sim import MLXQuantumSimulator
from noise_profiles import WILLOW_NOISE

# Ideal simulation
sim = MLXQuantumSimulator(3)
sim.h(0)
sim.cx(0, 1)
sim.cx(1, 2)
probs = sim.measure_probs()  # GHZ state: ~50% |000>, ~50% |111>

# Noisy simulation (Google Willow noise model)
sim_noisy = MLXQuantumSimulator(3, noise_profile=WILLOW_NOISE, n_trajectories=100)
sim_noisy.h(0)
sim_noisy.cx(0, 1)
sim_noisy.cx(1, 2)
probs_noisy = sim_noisy.measure_probs()  # Fidelity < 1 due to noise
```

## Noise Profiles

| Profile | Source | Qubits | CX Error | T1 | T2 |
|---|---|---|---|---|---|
| `WILLOW_NOISE` | Google cirq-google calibration (`willow_pink`) | 105 | 0.34% | 70 μs | 49 μs |
| `HERON_NOISE` | IBM published specs | 156 | 0.50% | 100 μs | 100 μs |
| `T9_NOISE` | QuTech Tuna-9 specs | 9 | 1.0% | 50 μs | 20 μs |

Full per-qubit Willow calibration data (T1, T2, readout errors, per-CZ-pair gate errors) available in `willow_calibration.json`.

## Noise Channels

- Depolarizing (single and two-qubit)
- Amplitude damping (T1 decay)
- Phase damping (T2 dephasing)
- Thermal noise (gate-time-dependent)
- Measurement readout error

## Benchmark (M1 Max, 3 qubits, 33 gates)

| Mode | Time per circuit |
|---|---|
| Ideal (no noise) | 2.7 ms |
| Willow (100 trajectories) | 495 ms |
| Heron (100 trajectories) | 461 ms |
| T-9 (100 trajectories) | 557 ms |

## Mirror Circuit Fidelity Validation

| Depth | Ideal | Willow | Heron | T-9 |
|---|---|---|---|---|
| 1 | 1.000 | 0.959 | 0.966 | 0.922 |
| 5 | 1.000 | 0.942 | 0.865 | 0.722 |
| 10 | 1.000 | 0.865 | 0.861 | 0.490 |

Fidelity ordering matches hardware quality: Willow > Heron > T-9.

## Gates

`H`, `X`, `Y`, `Z`, `S`, `T`, `Rx(θ)`, `Ry(θ)`, `Rz(θ)`, `CNOT/CX`, `CZ`, `SWAP`

## API Note

Gate parameter order: `sim.ry(theta, qubit)` — angle first, qubit second.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Citation

If you use this simulator in your research, please cite:

```
@software{mlx_quantum_sim,
  author = {Huang, Sheng-Kai},
  title = {mlx-quantum-sim: GPU-accelerated quantum simulator for Apple Silicon},
  year = {2026},
  url = {https://github.com/akaiHuang/mlx-quantum-sim}
}
```

## Related

- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework
- [cirq-google](https://github.com/quantumlib/Cirq) — Google's quantum SDK (noise data source)
- [qsim](https://github.com/quantumlib/qsim) — Google's C++ quantum simulator
