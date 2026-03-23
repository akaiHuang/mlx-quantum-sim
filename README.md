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

| | M1 Max (64GB) | M2 Ultra (192GB) | RTX 4090 | A100 (cloud) |
|---|---|---|---|---|
| Max qubits | ~31 | **~33** | ~30 | ~32 |
| Speed | 1x (baseline) | 1.3x | **15x** | **15x** |
| Power draw | **40W** | **60W** | 450W | 300W |
| Electricity/year (24/7) | **$35** | **$53** | $394 | $263 (+ rental) |
| Hardware cost | $0 (you have it) | $5,000 | $1,600 | $1.10/hr rental |
| Noise level | Silent | Silent | Loud | Datacenter |
| Also useful for | Everything else | Everything else | Gaming | Nothing else |

### Where Apple Silicon Wins

- **You already own it.** No purchase, no rental, no credit card.
- **Memory per dollar.** M2 Ultra 192GB = 33 qubits. No consumer NVIDIA card can do this.
- **Electricity.** 40W vs 450W. Run experiments overnight without guilt.
- **Iteration speed (human time).** Change code → run → see results. No uploading, no cloud setup, no SSH.
- **Privacy.** Your quantum circuits stay on your machine.

### Where NVIDIA Wins

- **Raw speed.** 15x faster per circuit. Not close.
- **Ecosystem.** cuQuantum, qsim, Qiskit Aer CUDA — mature, optimized, battle-tested.
- **Scale.** Multi-GPU, cluster computing. Apple can't do this.
- **Community.** More users, more examples, more support.

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
