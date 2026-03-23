# mlx-quantum-sim

**GPU-accelerated quantum circuit simulator for Apple Silicon, with real hardware noise models.**

## Why I Built This

I had a quantum algorithm idea. Wasn't sure if it worked.

The options were:
- Rent an A100 on the cloud → $2-5 per experiment, maybe $2,000/year
- Buy an RTX 4090 → $1,600 upfront
- Use my MacBook Pro (M1 Max, already on my desk) → $0

I chose the MacBook.

The idea might fail. Six of my first experiments did fail. If I'd rented cloud GPUs for those, I'd have wasted hundreds of dollars discovering dead ends.

Instead, I validated everything locally for $0. When something finally worked, *then* it made sense to consider scaling up.

**This simulator exists because: you shouldn't have to pay to find out if your idea is bad.**

## What This Is

A quantum circuit simulator that runs on Apple Metal GPU via [MLX](https://github.com/ml-explore/mlx). Includes calibrated noise profiles from Google Willow, IBM Heron, and QuTech Tuna-9 — extracted from publicly available calibration data.

## Honest Comparison: Apple Silicon vs NVIDIA

All qubit counts assume complex64 (float32) precision. Subtract ~1 qubit for complex128.

| | M1 Max (64GB) | M2 Ultra (192GB) | RTX 4090 (24GB) | A100 cloud (80GB) | Colab T4 (free, 16GB) |
|---|---|---|---|---|---|
| Max qubits (float32) | ~32 | **~34** | ~31 | ~33 | ~30 |
| Speed (30q circuit) | ~5 ms | ~4 ms | **~0.3 ms** | **~0.3 ms** | ~1 ms |
| 1000 circuits (30q) | ~5 sec | ~4 sec | **~0.3 sec** | **~0.3 sec** | ~1 sec |
| Power per experiment | **0.06 Wh** | **0.07 Wh** | 0.04 Wh | N/A | N/A |
| Extra hardware cost | $0 (if you own a Mac) | $5,000 | $1,600 | $1.10/hr | $0 |
| Setup time | `pip install mlx` | Same | Install CUDA + drivers | Cloud setup + SSH | Open browser |
| Offline / private | ✅ | ✅ | ✅ | ❌ (cloud) | ❌ (cloud) |
| Other uses | Daily work | Daily work | ML / gaming | ML / training | ML (limited) |

### Where Apple Silicon Wins

- **Zero marginal cost.** If you own a Mac, every experiment is free. No cloud bills, no GPU purchase.
- **Large unified memory.** M2 Ultra 192GB → 34 qubits. No single consumer GPU matches this.
- **Fast iteration.** Edit code → run → results. No uploading, no environment setup, no waiting for cloud instances.
- **Privacy.** Circuits never leave your machine.

### Where NVIDIA / Cloud Wins

- **Raw speed.** 10-15x faster per circuit. For production workloads, this matters.
- **Ecosystem.** cuQuantum, qsim, Qiskit Aer CUDA — mature and highly optimized.
- **Scale.** Multi-GPU and cluster computing. Apple can't do this.
- **Free tier.** Google Colab gives you a T4 GPU for free. Honest alternative for small experiments.

### Why Not Just Use Google Colab?

Colab is a strong free option for small experiments. Use mlx-quantum-sim when you need:
- **Offline access** (no internet required)
- **More than 16GB** GPU memory (Colab T4 limit = ~30 qubits)
- **Long-running experiments** (Colab disconnects after ~12 hours)
- **Privacy** (proprietary quantum circuits stay local)
- **Reproducibility** (no session timeouts or random disconnects)

### The Real Decision

```
"I have a quantum algorithm idea. Should I rent cloud GPUs?"

If you're EXPLORING (not sure it works):
  → Use your Mac. It's free. Most ideas fail. Save your money.

If you've VALIDATED (it works, need to scale):
  → Switch to NVIDIA / cloud. Pay for speed when you know it's worth it.

Apple Silicon = lab notebook (cheap, fast iteration)
NVIDIA = production machine (expensive, maximum throughput)
```

## When to Use This vs NVIDIA

| Scenario | Use mlx-quantum-sim | Use NVIDIA/cloud |
|---|---|---|
| Testing a new quantum algorithm | ✅ | Overkill |
| Homework / coursework | ✅ | Overkill |
| Prototyping with noise models | ✅ | Overkill |
| Publishing a paper (small circuits) | ✅ | Optional |
| Running 100K+ circuits | ❌ Too slow | ✅ |
| Production / deployment | ❌ | ✅ |
| Training quantum ML models | ⚠️ (small scale OK) | ✅ (large scale) |
| 33+ qubit simulation | ✅ (M2 Ultra only) | ❌ (need multi-GPU) |

## Features

- **Metal GPU acceleration**: All gate operations on Apple Silicon GPU — no CPU round trips
- **Real hardware noise**: Google Willow (105 qubits, 364 CZ pairs), IBM Heron, QuTech Tuna-9
- **Stochastic trajectory simulation**: Same approach as Google's [qsim](https://github.com/quantumlib/qsim)
- **Up to ~33 qubits** on M2 Ultra 192GB (statevector, O(2^n) memory)

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
probs = sim.measure_probs()  # GHZ state

# With Google Willow noise
sim = MLXQuantumSimulator(3, noise_profile=WILLOW_NOISE, n_trajectories=100)
sim.h(0)
sim.cx(0, 1)
probs = sim.measure_probs()  # Fidelity < 1 due to realistic noise
```

## Noise Profiles

| Profile | Source | CX Error | T1 | T2 |
|---|---|---|---|---|
| `WILLOW_NOISE` | cirq-google calibration (`willow_pink`) | 0.34% | 70 μs | 49 μs |
| `HERON_NOISE` | IBM published specs | 0.50% | 100 μs | 100 μs |
| `T9_NOISE` | QuTech published specs | 1.0% | 50 μs | 20 μs |

Full per-qubit Willow calibration (T1, T2, readout errors, 364 CZ pairs) in `willow_calibration.json`.

## Noise Channels

- Depolarizing (single and two-qubit)
- Amplitude damping (T1 decay)
- Phase damping (T2 dephasing)
- Thermal noise (gate-time dependent)
- Measurement readout error

## Gates

`H`, `X`, `Y`, `Z`, `S`, `T`, `Rx(θ)`, `Ry(θ)`, `Rz(θ)`, `CNOT/CX`, `CZ`, `SWAP`

API: `sim.ry(theta, qubit)` — angle first, qubit second.

## Benchmark (M1 Max, 3 qubits, 33 gates)

| Mode | Time/circuit |
|---|---|
| Ideal (no noise) | 2.7 ms |
| Willow (100 trajectories) | 495 ms |

## Mirror Circuit Fidelity

| Depth | Ideal | Willow | Heron | T-9 |
|---|---|---|---|---|
| 1 | 1.000 | 0.959 | 0.966 | 0.922 |
| 5 | 1.000 | 0.942 | 0.865 | 0.722 |
| 10 | 1.000 | 0.865 | 0.861 | 0.490 |

Ordering matches hardware quality: Willow > Heron > T-9. ✓

## Contributing

PRs welcome — especially:
- CUDA/PyTorch backend (for NVIDIA users)
- Additional noise profiles (Rigetti, IonQ, Quantinuum)
- Density matrix simulation
- Performance optimization

## License

Apache License 2.0 (consistent with cirq-google). See [LICENSE](LICENSE).

## Citation

```bibtex
@software{mlx_quantum_sim,
  author = {Huang, Sheng-Kai},
  title = {mlx-quantum-sim: GPU-accelerated quantum simulator for Apple Silicon},
  year = {2026},
  url = {https://github.com/akaiHuang/mlx-quantum-sim}
}
```

## Acknowledgments

Noise calibration data from Google's [cirq-google](https://github.com/quantumlib/Cirq) (Apache 2.0). Trajectory-based noise simulation follows the approach of Google's [qsim](https://github.com/quantumlib/qsim).
