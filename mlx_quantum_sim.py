"""
MLX Quantum Circuit Simulator — GPU-accelerated via Apple Metal.

Core idea:
  State vector |psi> lives as an mx.array of shape (2^n,) in complex64 on GPU.
  Gate application is done by reshaping the state into a rank-n tensor of shape
  (2, 2, ..., 2), contracting with the gate matrix along the target qubit axis,
  and reshaping back.  This avoids building full 2^n x 2^n matrices, so memory
  is O(2^n) not O(4^n).

All operations stay on Metal GPU — no CPU round-trips between gates.

Noise model (optional):
  When a noise_profile dict is provided, the simulator uses stochastic trajectory
  unraveling: after each gate, a random Pauli error is applied with the hardware's
  error probability.  Multiple trajectories are run and their measurement
  probabilities are averaged.  This matches the approach used by Google's qsim.

  Additionally, T1 (amplitude damping) and T2 (phase damping) noise is applied
  after each gate based on the gate duration and the device's coherence times.
"""

from __future__ import annotations

import math
import random
from typing import Callable, List, Optional, Sequence, Tuple

import mlx.core as mx

from simulator.gates import (
    hadamard, pauli_x, pauli_y, pauli_z,
    ry_matrix, rz_matrix, rx_matrix,
    cnot_matrix, phase_gate, t_gate, swap_gate, cz_gate,
)


class MLXQuantumSimulator:
    """GPU-accelerated statevector quantum circuit simulator.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (supports up to ~23 on 32 GB M1 Max).
    eval_interval : int
        If > 0, force GPU evaluation every N gates to bound graph size.
    noise_profile : dict or None
        Hardware noise profile (see simulator.noise_profiles).
        None = ideal noiseless simulation.
    n_trajectories : int
        Number of stochastic trajectories for noisy simulation.
        Ignored when noise_profile is None.
    seed : int or None
        Random seed for reproducible noisy simulation.

    Example
    -------
    >>> sim = MLXQuantumSimulator(3)
    >>> sim.h(0)
    >>> sim.cx(0, 1)
    >>> sim.cx(0, 2)
    >>> probs = sim.measure_probs()  # GHZ state: 50% |000>, 50% |111>

    >>> from simulator.noise_profiles import WILLOW_NOISE
    >>> sim = MLXQuantumSimulator(3, noise_profile=WILLOW_NOISE, n_trajectories=200)
    >>> sim.h(0); sim.cx(0, 1); sim.cx(0, 2)
    >>> probs = sim.measure_probs()  # Noisy GHZ: averaged over 200 trajectories
    """

    def __init__(
        self,
        n_qubits: int,
        eval_interval: int = 0,
        noise_profile: Optional[dict] = None,
        n_trajectories: int = 100,
        seed: Optional[int] = None,
    ):
        self.n_qubits = n_qubits
        self.dim = 1 << n_qubits
        self._eval_interval = eval_interval  # 0 = no periodic eval
        self._gate_count = 0

        # Noise configuration
        self.noise = noise_profile
        self.n_trajectories = n_trajectories if noise_profile else 1
        self._rng = random.Random(seed)

        # Cache fixed gate matrices (computed once, reused)
        self._h_gate = hadamard()
        self._x_gate = pauli_x()
        self._y_gate = pauli_y()
        self._z_gate = pauli_z()
        self._s_gate = phase_gate()
        self._t_gate = t_gate()
        self._cnot_gate = cnot_matrix()
        self._cz_gate = cz_gate()
        self._swap_gate = swap_gate()

        # If noisy, record circuit instructions for trajectory replay
        self._recording = noise_profile is not None
        self._instructions: List[Tuple] = []

        self.reset()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self):
        """Reset to |000...0> and clear recorded instructions."""
        # Build |0...0> = [1, 0, 0, ..., 0] via concatenation (MLX arrays are immutable)
        self.state = mx.concatenate([
            mx.array([1.0 + 0j], dtype=mx.complex64),
            mx.zeros(self.dim - 1, dtype=mx.complex64),
        ])
        self._instructions = []

    def set_state(self, state: mx.array):
        """Set an explicit state vector (must have length 2^n)."""
        assert state.shape == (self.dim,), f"Expected ({self.dim},), got {state.shape}"
        self.state = state.astype(mx.complex64)

    # ------------------------------------------------------------------
    # Low-level gate application (tensor contraction on GPU)
    # ------------------------------------------------------------------

    def _apply_single_gate(self, gate_2x2: mx.array, qubit: int):
        """Apply a single-qubit gate to the given qubit.

        Strategy: reshape state to (2, 2, ..., 2), transpose to bring
        target qubit to last axis, contract, transpose back, reshape.

        Qubit ordering follows Qiskit convention (little-endian):
          qubit 0 = least significant bit (rightmost tensor axis).
        """
        n = self.n_qubits
        # Map logical qubit to tensor axis (little-endian: qubit 0 = axis n-1)
        axis = n - 1 - qubit
        state = self.state.reshape([2] * n)  # rank-n tensor

        # Bring target qubit axis to position -1
        perm = list(range(n))
        perm.remove(axis)
        perm.append(axis)
        state = mx.transpose(state, perm)

        # Contract: state[..., j] = sum_k gate[j, k] * state[..., k]
        # Equivalent to: result = state @ gate^T
        shape_front = state.shape[:-1]
        state = state.reshape(-1, 2)           # (2^{n-1}, 2)
        gate_t = mx.transpose(gate_2x2)        # (2, 2)
        state = state @ gate_t                  # matmul on GPU

        state = state.reshape(list(shape_front) + [2])

        # Inverse permutation
        inv_perm = [0] * n
        for i, p in enumerate(perm):
            inv_perm[p] = i
        state = mx.transpose(state, inv_perm)

        self.state = state.reshape(self.dim)
        self._maybe_eval()

    def _apply_two_qubit_gate(self, gate_4x4: mx.array, qubit_a: int, qubit_b: int):
        """Apply a two-qubit gate (4x4 unitary) to qubits a, b.

        The gate is defined in the basis |ab> where a is the more significant
        bit: |00>, |01>, |10>, |11>.

        Qubit ordering follows Qiskit convention (little-endian).
        """
        n = self.n_qubits
        # Map logical qubits to tensor axes
        axis_a = n - 1 - qubit_a
        axis_b = n - 1 - qubit_b
        state = self.state.reshape([2] * n)

        # Bring axis_a and axis_b to the last two axes (in that order)
        perm = list(range(n))
        perm.remove(axis_a)
        perm.remove(axis_b)
        perm.append(axis_a)
        perm.append(axis_b)
        state = mx.transpose(state, perm)

        # Reshape: (..., 4)
        shape_front = state.shape[:-2]
        state = state.reshape(-1, 4)
        gate_t = mx.transpose(gate_4x4)
        state = state @ gate_t

        state = state.reshape(list(shape_front) + [2, 2])

        # Inverse permutation
        inv_perm = [0] * n
        for i, p in enumerate(perm):
            inv_perm[p] = i
        state = mx.transpose(state, inv_perm)

        self.state = state.reshape(self.dim)
        self._maybe_eval()

    def _maybe_eval(self):
        """Periodically evaluate to prevent excessively large compute graphs."""
        if self._eval_interval > 0:
            self._gate_count += 1
            if self._gate_count % self._eval_interval == 0:
                mx.eval(self.state)

    # ------------------------------------------------------------------
    # Noise channels (stochastic, applied per-trajectory)
    # ------------------------------------------------------------------

    def _apply_depolarizing(self, qubit: int, p: float):
        """Apply single-qubit depolarizing noise with probability p.

        With probability p, apply a random Pauli (X, Y, or Z each with p/3).
        With probability 1-p, do nothing (identity).

        For statevector simulation this is done stochastically: draw a random
        number and conditionally apply one Pauli gate.
        """
        r = self._rng.random()
        if r < p:
            pauli_choice = self._rng.random() * 3.0
            if pauli_choice < 1.0:
                self._apply_single_gate(self._x_gate, qubit)
            elif pauli_choice < 2.0:
                self._apply_single_gate(self._y_gate, qubit)
            else:
                self._apply_single_gate(self._z_gate, qubit)

    def _apply_two_qubit_depolarizing(self, qubit_a: int, qubit_b: int, p: float):
        """Apply two-qubit depolarizing noise with probability p.

        With probability p, apply a random two-qubit Pauli (one of 15
        nontrivial tensor products I*X, I*Y, ..., Z*Z each with p/15).
        """
        r = self._rng.random()
        if r < p:
            paulis = [None, self._x_gate, self._y_gate, self._z_gate]
            # Choose one of 15 non-identity Paulis
            idx = self._rng.randint(0, 14)  # 0..14
            ia = (idx // 4) + 0  # 0..3 (includes identity on a)
            ib = (idx % 4)       # 0..3
            # But we exclude (0,0) = identity x identity
            # Mapping: idx 0..14 -> (ia,ib) from (0,1),(0,2),(0,3),(1,0),...,(3,3)
            ia = (idx + 1) // 4
            ib = (idx + 1) % 4
            if ia > 0:
                self._apply_single_gate(paulis[ia], qubit_a)
            if ib > 0:
                self._apply_single_gate(paulis[ib], qubit_b)

    def _apply_amplitude_damping(self, qubit: int, gamma: float):
        """Apply amplitude damping (T1 decay) with parameter gamma.

        Kraus operators:
          E0 = [[1, 0], [0, sqrt(1-gamma)]]
          E1 = [[0, sqrt(gamma)], [0, 0]]

        For small gamma (< 0.01, typical for hardware), uses Pauli
        approximation: with prob gamma/2 apply X (bit flip toward |0>).
        This avoids expensive mx.eval() per gate.

        For large gamma, uses exact stochastic unraveling with mx.eval().
        """
        if gamma <= 0 or gamma > 1:
            return

        if gamma < 0.01:
            # Pauli approximation: amplitude damping ~ depolarizing
            # toward |0>.  Apply X with probability gamma/2.
            r = self._rng.random()
            if r < gamma / 2.0:
                self._apply_single_gate(self._x_gate, qubit)
            return

        # Exact: need to read p(|1>) from GPU
        n = self.n_qubits
        axis = n - 1 - qubit
        state_t = self.state.reshape([2] * n)

        slices_1 = [slice(None)] * n
        slices_1[axis] = 1
        p1 = mx.sum(mx.abs(state_t[tuple(slices_1)]) ** 2)
        mx.eval(p1)
        p1_val = p1.item()

        p_decay = gamma * p1_val
        r = self._rng.random()

        if r < p_decay:
            slices_0 = [slice(None)] * n
            slices_0[axis] = 0
            slices_1_t = [slice(None)] * n
            slices_1_t[axis] = 1

            amp_0 = state_t[tuple(slices_0)]
            amp_1 = state_t[tuple(slices_1_t)]
            new_0 = amp_0 + amp_1
            new_1 = mx.zeros_like(amp_1)
            state_t = mx.stack([new_0, new_1], axis=axis)
            self.state = state_t.reshape(self.dim)
            norm = mx.sqrt(mx.sum(mx.abs(self.state) ** 2))
            mx.eval(norm)
            if norm.item() > 1e-10:
                self.state = self.state / norm
        else:
            scale = math.sqrt(1.0 - gamma)
            e0 = mx.array([[1.0, 0.0], [0.0, scale]], dtype=mx.complex64)
            self._apply_single_gate(e0, qubit)
            norm = mx.sqrt(mx.sum(mx.abs(self.state) ** 2))
            mx.eval(norm)
            if norm.item() > 1e-10:
                self.state = self.state / norm

    def _apply_phase_damping(self, qubit: int, gamma: float):
        """Apply phase damping (T2 dephasing beyond T1) with parameter gamma.

        Kraus operators:
          E0 = [[1, 0], [0, sqrt(1-gamma)]]
          E1 = [[0, 0], [0, sqrt(gamma)]]

        For small gamma (< 0.01), uses Pauli Z approximation (no eval).
        """
        if gamma <= 0 or gamma > 1:
            return

        if gamma < 0.01:
            # Pauli approximation: phase damping ~ Z with prob gamma/2
            r = self._rng.random()
            if r < gamma / 2.0:
                self._apply_single_gate(self._z_gate, qubit)
            return

        # Exact: need GPU eval
        n = self.n_qubits
        axis = n - 1 - qubit
        state_t = self.state.reshape([2] * n)

        slices_1 = [slice(None)] * n
        slices_1[axis] = 1
        p1 = mx.sum(mx.abs(state_t[tuple(slices_1)]) ** 2)
        mx.eval(p1)
        p1_val = p1.item()

        p_dephase = gamma * p1_val
        r = self._rng.random()

        if r < p_dephase:
            self._apply_single_gate(self._z_gate, qubit)
        else:
            scale = math.sqrt(1.0 - gamma)
            e0 = mx.array([[1.0, 0.0], [0.0, scale]], dtype=mx.complex64)
            self._apply_single_gate(e0, qubit)
            norm = mx.sqrt(mx.sum(mx.abs(self.state) ** 2))
            mx.eval(norm)
            if norm.item() > 1e-10:
                self.state = self.state / norm

    def _apply_thermal_noise(self, qubit: int, gate_time_us: float):
        """Apply T1/T2 thermal relaxation noise based on gate duration.

        Converts T1, T2 coherence times and gate duration into amplitude
        damping (gamma_1) and phase damping (gamma_phi) parameters.

        For realistic hardware parameters (gate ~ 25-50 ns, T1 ~ 50-100 us),
        gamma values are O(1e-7), so the fast Pauli approximation is used.
        """
        if not self.noise:
            return

        T1 = self.noise['T1_us']
        T2 = self.noise['T2_us']

        if T1 <= 0:
            return

        # Amplitude damping parameter: gamma_1 = 1 - exp(-t/T1)
        gamma_1 = 1.0 - math.exp(-gate_time_us / T1)

        # Phase damping (pure dephasing beyond T1):
        # 1/T2 = 1/(2*T1) + 1/T_phi  =>  T_phi = 1/(1/T2 - 1/(2*T1))
        # gamma_phi = 1 - exp(-t/T_phi)
        rate_phi = (1.0 / T2) - (1.0 / (2.0 * T1))
        if rate_phi > 0:
            gamma_phi = 1.0 - math.exp(-gate_time_us * rate_phi)
        else:
            gamma_phi = 0.0

        if gamma_1 > 1e-12:
            self._apply_amplitude_damping(qubit, gamma_1)
        if gamma_phi > 1e-12:
            self._apply_phase_damping(qubit, gamma_phi)

    def _apply_noise_after_1q_gate(self, qubit: int):
        """Apply noise channels after a single-qubit gate (if noisy mode)."""
        if not self.noise:
            return
        # Depolarizing noise
        self._apply_depolarizing(qubit, self.noise['p_1q'])
        # Thermal relaxation
        self._apply_thermal_noise(qubit, self.noise['gate_time_1q_us'])

    def _apply_noise_after_2q_gate(self, qubit_a: int, qubit_b: int):
        """Apply noise channels after a two-qubit gate (if noisy mode)."""
        if not self.noise:
            return
        # Two-qubit depolarizing noise
        self._apply_two_qubit_depolarizing(qubit_a, qubit_b, self.noise['p_2q'])
        # Thermal relaxation on both qubits
        self._apply_thermal_noise(qubit_a, self.noise['gate_time_2q_us'])
        self._apply_thermal_noise(qubit_b, self.noise['gate_time_2q_us'])

    @staticmethod
    def apply_measurement_error(probs_np, p_error: float):
        """Apply symmetric measurement readout error to probability array.

        For each computational basis state, bit flips occur independently
        with probability p_error.  This is equivalent to applying a
        binary symmetric channel to each qubit's measurement outcome.

        Parameters
        ----------
        probs_np : numpy.ndarray
            Probability distribution of shape (2^n,).
        p_error : float
            Per-qubit readout flip probability.

        Returns
        -------
        numpy.ndarray
            Noisy probability distribution.
        """
        import numpy as np
        n_qubits = int(math.log2(len(probs_np)))
        dim = len(probs_np)
        noisy_probs = np.zeros(dim)

        for i in range(dim):
            for j in range(dim):
                # Number of differing bits = Hamming distance
                diff = i ^ j
                n_flips = bin(diff).count('1')
                n_same = n_qubits - n_flips
                # Probability of this bit-flip pattern
                p_transition = (p_error ** n_flips) * ((1 - p_error) ** n_same)
                noisy_probs[j] += probs_np[i] * p_transition

        return noisy_probs

    # ------------------------------------------------------------------
    # Standard gates (using cached gate matrices)
    # ------------------------------------------------------------------

    def h(self, qubit: int):
        """Apply Hadamard gate to qubit."""
        if self._recording:
            self._instructions.append(('h', qubit))
        else:
            self._apply_single_gate(self._h_gate, qubit)

    def x(self, qubit: int):
        """Apply Pauli-X gate."""
        if self._recording:
            self._instructions.append(('x', qubit))
        else:
            self._apply_single_gate(self._x_gate, qubit)

    def y(self, qubit: int):
        """Apply Pauli-Y gate."""
        if self._recording:
            self._instructions.append(('y', qubit))
        else:
            self._apply_single_gate(self._y_gate, qubit)

    def z(self, qubit: int):
        """Apply Pauli-Z gate."""
        if self._recording:
            self._instructions.append(('z', qubit))
        else:
            self._apply_single_gate(self._z_gate, qubit)

    def s(self, qubit: int):
        """Apply S (phase) gate."""
        if self._recording:
            self._instructions.append(('s', qubit))
        else:
            self._apply_single_gate(self._s_gate, qubit)

    def t(self, qubit: int):
        """Apply T gate."""
        if self._recording:
            self._instructions.append(('t', qubit))
        else:
            self._apply_single_gate(self._t_gate, qubit)

    def ry(self, theta: float, qubit: int):
        """Apply R_y(theta) rotation gate."""
        if self._recording:
            self._instructions.append(('ry', theta, qubit))
        else:
            self._apply_single_gate(ry_matrix(theta), qubit)

    def rz(self, theta: float, qubit: int):
        """Apply R_z(theta) rotation gate."""
        if self._recording:
            self._instructions.append(('rz', theta, qubit))
        else:
            self._apply_single_gate(rz_matrix(theta), qubit)

    def rx(self, theta: float, qubit: int):
        """Apply R_x(theta) rotation gate."""
        if self._recording:
            self._instructions.append(('rx', theta, qubit))
        else:
            self._apply_single_gate(rx_matrix(theta), qubit)

    def cx(self, control: int, target: int):
        """Apply CNOT (CX) gate."""
        if self._recording:
            self._instructions.append(('cx', control, target))
        else:
            self._apply_two_qubit_gate(self._cnot_gate, control, target)

    def cz(self, qubit_a: int, qubit_b: int):
        """Apply Controlled-Z gate."""
        if self._recording:
            self._instructions.append(('cz', qubit_a, qubit_b))
        else:
            self._apply_two_qubit_gate(self._cz_gate, qubit_a, qubit_b)

    def swap(self, qubit_a: int, qubit_b: int):
        """Apply SWAP gate."""
        if self._recording:
            self._instructions.append(('swap', qubit_a, qubit_b))
        else:
            self._apply_two_qubit_gate(self._swap_gate, qubit_a, qubit_b)

    # ------------------------------------------------------------------
    # Trajectory replay (noisy mode)
    # ------------------------------------------------------------------

    def _run_single_trajectory(self) -> mx.array:
        """Execute recorded instructions once with stochastic noise.

        Returns probability distribution for this trajectory.
        """
        # Reset state to |0...0>
        self.state = mx.concatenate([
            mx.array([1.0 + 0j], dtype=mx.complex64),
            mx.zeros(self.dim - 1, dtype=mx.complex64),
        ])

        gate_matrices = {
            'h': self._h_gate, 'x': self._x_gate, 'y': self._y_gate,
            'z': self._z_gate, 's': self._s_gate, 't': self._t_gate,
        }
        two_qubit_matrices = {
            'cx': self._cnot_gate, 'cz': self._cz_gate, 'swap': self._swap_gate,
        }
        param_gate_fns = {
            'ry': ry_matrix, 'rz': rz_matrix, 'rx': rx_matrix,
        }

        gate_count = 0
        for instr in self._instructions:
            name = instr[0]
            if name in gate_matrices:
                qubit = instr[1]
                self._apply_single_gate(gate_matrices[name], qubit)
                self._apply_noise_after_1q_gate(qubit)
            elif name in param_gate_fns:
                theta, qubit = instr[1], instr[2]
                self._apply_single_gate(param_gate_fns[name](theta), qubit)
                self._apply_noise_after_1q_gate(qubit)
            elif name in two_qubit_matrices:
                qa, qb = instr[1], instr[2]
                self._apply_two_qubit_gate(two_qubit_matrices[name], qa, qb)
                self._apply_noise_after_2q_gate(qa, qb)
            else:
                raise ValueError(f"Unknown gate in trajectory replay: {name}")

            # Periodically evaluate to bound compute graph size
            gate_count += 1
            if gate_count % 50 == 0:
                mx.eval(self.state)

        return mx.abs(self.state) ** 2

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure_probs(self) -> mx.array:
        """Return probability distribution |<i|psi>|^2 as GPU array.

        In noisy mode, this runs n_trajectories stochastic simulations
        and returns the averaged probability distribution.
        """
        if not self.noise:
            return mx.abs(self.state) ** 2

        # Noisy: run trajectories and average
        import numpy as np
        accumulated = np.zeros(self.dim, dtype=np.float64)

        for _ in range(self.n_trajectories):
            probs = self._run_single_trajectory()
            mx.eval(probs)
            accumulated += np.array(probs, dtype=np.float64)

        avg = accumulated / self.n_trajectories

        # Apply measurement readout error
        if self.noise.get('p_meas', 0) > 0:
            avg = self.apply_measurement_error(avg, self.noise['p_meas'])

        return mx.array(avg.astype(np.float32))

    def measure_probs_numpy(self):
        """Return probabilities as numpy array (triggers GPU->CPU transfer)."""
        import numpy as np
        probs = self.measure_probs()
        mx.eval(probs)
        return np.array(probs)

    def statevector(self) -> mx.array:
        """Return the full statevector (stays on GPU).

        Note: in noisy mode, returns the statevector of the last trajectory,
        not a meaningful "average" state (use measure_probs for that).
        """
        return self.state

    def statevector_numpy(self):
        """Return the full statevector as numpy array."""
        import numpy as np
        mx.eval(self.state)
        return np.array(self.state)

    # ------------------------------------------------------------------
    # Circuit execution from instruction list
    # ------------------------------------------------------------------

    def run_instructions(self, instructions: List[Tuple]):
        """Execute a list of gate instructions.

        Each instruction is a tuple:
          ('h', qubit)
          ('x', qubit)
          ('ry', theta, qubit)
          ('rz', theta, qubit)
          ('rx', theta, qubit)
          ('cx', control, target)
          ('cz', qubit_a, qubit_b)
          ('swap', qubit_a, qubit_b)
        """
        gate_map = {
            'h': self.h,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            's': self.s,
            't': self.t,
        }
        param_gate_map = {
            'ry': self.ry,
            'rz': self.rz,
            'rx': self.rx,
        }
        two_qubit_map = {
            'cx': self.cx,
            'cz': self.cz,
            'swap': self.swap,
        }

        for instr in instructions:
            name = instr[0]
            if name in gate_map:
                gate_map[name](instr[1])
            elif name in param_gate_map:
                param_gate_map[name](instr[1], instr[2])
            elif name in two_qubit_map:
                two_qubit_map[name](instr[1], instr[2])
            else:
                raise ValueError(f"Unknown gate: {name}")


# ======================================================================
# Batch simulator: run many parameter sets on the same circuit structure
# ======================================================================

class MLXBatchSimulator:
    """Run B copies of the same circuit structure with different parameters.

    The state is a (B, 2^n) matrix — all B circuits evolve in parallel
    on GPU via batched matrix operations.

    Parameters
    ----------
    n_qubits : int
    batch_size : int
        Number of parallel circuits.
    """

    def __init__(self, n_qubits: int, batch_size: int):
        self.n_qubits = n_qubits
        self.dim = 1 << n_qubits
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        """Reset all circuits to |000...0>."""
        state = mx.zeros((self.batch_size, self.dim), dtype=mx.complex64)
        idx = mx.array([0])
        for b in range(self.batch_size):
            pass
        # Faster: construct directly
        col0 = mx.ones((self.batch_size, 1), dtype=mx.complex64)
        rest = mx.zeros((self.batch_size, self.dim - 1), dtype=mx.complex64)
        self.state = mx.concatenate([col0, rest], axis=1)

    def _apply_single_gate(self, gate_2x2: mx.array, qubit: int):
        """Apply a single-qubit gate to all circuits in the batch."""
        n = self.n_qubits
        B = self.batch_size
        # (B, 2, 2, ..., 2) — n qubit dims
        state = self.state.reshape([B] + [2] * n)

        # Map logical qubit to tensor axis (little-endian: qubit 0 = axis n)
        # Axes 1..n correspond to qubits n-1..0
        qubit_axis = n - qubit  # offset by batch dim, little-endian
        perm = list(range(n + 1))
        perm.remove(qubit_axis)
        perm.append(qubit_axis)
        state = mx.transpose(state, perm)

        # Contract
        shape_front = state.shape[:-1]
        state = state.reshape(-1, 2)
        state = state @ mx.transpose(gate_2x2)
        state = state.reshape(list(shape_front) + [2])

        # Inverse permutation
        inv_perm = [0] * (n + 1)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        state = mx.transpose(state, inv_perm)

        self.state = state.reshape(B, self.dim)

    def _apply_two_qubit_gate(self, gate_4x4: mx.array, qubit_a: int, qubit_b: int):
        """Apply a two-qubit gate to all circuits in the batch."""
        n = self.n_qubits
        B = self.batch_size
        state = self.state.reshape([B] + [2] * n)

        axis_a = n - qubit_a  # little-endian
        axis_b = n - qubit_b  # little-endian
        perm = list(range(n + 1))
        perm.remove(axis_a)
        perm.remove(axis_b)
        perm.append(axis_a)
        perm.append(axis_b)
        state = mx.transpose(state, perm)

        shape_front = state.shape[:-2]
        state = state.reshape(-1, 4)
        state = state @ mx.transpose(gate_4x4)
        state = state.reshape(list(shape_front) + [2, 2])

        inv_perm = [0] * (n + 1)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        state = mx.transpose(state, inv_perm)

        self.state = state.reshape(B, self.dim)

    # --- Convenience gate methods ---

    def h(self, qubit: int):
        self._apply_single_gate(hadamard(), qubit)

    def x(self, qubit: int):
        self._apply_single_gate(pauli_x(), qubit)

    def ry(self, thetas: mx.array, qubit: int):
        """Apply R_y with per-circuit angles.  thetas shape: (B,)."""
        # For batch parametric gates we must apply per-circuit rotation.
        # Strategy: build (B, 2, 2) gate tensors and batch-contract.
        self._apply_batch_single_gate_ry(thetas, qubit)

    def rz(self, thetas: mx.array, qubit: int):
        """Apply R_z with per-circuit angles.  thetas shape: (B,)."""
        self._apply_batch_single_gate_rz(thetas, qubit)

    def rx(self, thetas: mx.array, qubit: int):
        """Apply R_x with per-circuit angles.  thetas shape: (B,)."""
        self._apply_batch_single_gate_rx(thetas, qubit)

    def cx(self, control: int, target: int):
        self._apply_two_qubit_gate(cnot_matrix(), control, target)

    def _apply_batch_single_gate_ry(self, thetas: mx.array, qubit: int):
        """Batch R_y: each circuit gets its own angle."""
        n = self.n_qubits
        B = self.batch_size
        half = thetas * 0.5                              # (B,)
        c = mx.cos(half)                                 # (B,)
        s = mx.sin(half)                                 # (B,)

        state = self.state.reshape([B] + [2] * n)
        qubit_axis = n - qubit  # little-endian
        perm = list(range(n + 1))
        perm.remove(qubit_axis)
        perm.append(qubit_axis)
        state = mx.transpose(state, perm)

        shape_front = state.shape[:-1]
        state = state.reshape(B, -1, 2)  # (B, D/2, 2)
        D_half = state.shape[1]

        # state_new[b, :, 0] = c[b]*state[b,:,0] + s[b]*state[b,:,1]
        # state_new[b, :, 1] = -s[b]*state[b,:,0] + c[b]*state[b,:,1]
        c_ = c.reshape(B, 1, 1).astype(mx.complex64)
        s_ = s.reshape(B, 1, 1).astype(mx.complex64)
        s0 = state[:, :, 0:1]
        s1 = state[:, :, 1:2]
        new_0 = c_ * s0 + s_ * s1
        new_1 = -s_ * s0 + c_ * s1
        state = mx.concatenate([new_0, new_1], axis=2)

        state = state.reshape(list(shape_front) + [2])

        inv_perm = [0] * (n + 1)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        state = mx.transpose(state, inv_perm)
        self.state = state.reshape(B, self.dim)

    def _apply_batch_single_gate_rz(self, thetas: mx.array, qubit: int):
        """Batch R_z: each circuit gets its own angle."""
        n = self.n_qubits
        B = self.batch_size
        half = thetas * 0.5

        # e^{-i*half}, e^{+i*half}
        cos_h = mx.cos(half)
        sin_h = mx.sin(half)

        state = self.state.reshape([B] + [2] * n)
        qubit_axis = n - qubit  # little-endian
        perm = list(range(n + 1))
        perm.remove(qubit_axis)
        perm.append(qubit_axis)
        state = mx.transpose(state, perm)

        shape_front = state.shape[:-1]
        state = state.reshape(B, -1, 2)

        # phase0 = cos(-half) + i*sin(-half) = cos_h - i*sin_h
        # phase1 = cos_h + i*sin_h
        phase0_real = cos_h.reshape(B, 1, 1)
        phase0_imag = (-sin_h).reshape(B, 1, 1)
        phase1_real = cos_h.reshape(B, 1, 1)
        phase1_imag = sin_h.reshape(B, 1, 1)

        s0 = state[:, :, 0:1]
        s1 = state[:, :, 1:2]

        # Multiply complex: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        # But MLX complex64 handles this if we construct the phase as complex
        one_c = mx.array(1.0 + 0j, dtype=mx.complex64)
        i_c = mx.array(0.0 + 1j, dtype=mx.complex64)

        p0 = phase0_real.astype(mx.complex64) * one_c + phase0_imag.astype(mx.complex64) * i_c
        p1 = phase1_real.astype(mx.complex64) * one_c + phase1_imag.astype(mx.complex64) * i_c

        new_0 = p0 * s0
        new_1 = p1 * s1
        state = mx.concatenate([new_0, new_1], axis=2)

        state = state.reshape(list(shape_front) + [2])
        inv_perm = [0] * (n + 1)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        state = mx.transpose(state, inv_perm)
        self.state = state.reshape(B, self.dim)

    def _apply_batch_single_gate_rx(self, thetas: mx.array, qubit: int):
        """Batch R_x: each circuit gets its own angle."""
        n = self.n_qubits
        B = self.batch_size
        half = thetas * 0.5
        c = mx.cos(half)
        s = mx.sin(half)

        state = self.state.reshape([B] + [2] * n)
        qubit_axis = n - qubit  # little-endian
        perm = list(range(n + 1))
        perm.remove(qubit_axis)
        perm.append(qubit_axis)
        state = mx.transpose(state, perm)

        shape_front = state.shape[:-1]
        state = state.reshape(B, -1, 2)

        one_c = mx.array(1.0 + 0j, dtype=mx.complex64)
        i_c = mx.array(0.0 + 1j, dtype=mx.complex64)

        c_ = c.reshape(B, 1, 1).astype(mx.complex64) * one_c
        neg_is = (-s).reshape(B, 1, 1).astype(mx.complex64) * i_c

        s0 = state[:, :, 0:1]
        s1 = state[:, :, 1:2]
        new_0 = c_ * s0 + neg_is * s1
        new_1 = neg_is * s0 + c_ * s1
        state = mx.concatenate([new_0, new_1], axis=2)

        state = state.reshape(list(shape_front) + [2])
        inv_perm = [0] * (n + 1)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        state = mx.transpose(state, inv_perm)
        self.state = state.reshape(B, self.dim)

    def measure_probs(self) -> mx.array:
        """Return (B, 2^n) probability matrix."""
        return mx.abs(self.state) ** 2

    def measure_probs_numpy(self):
        import numpy as np
        p = self.measure_probs()
        mx.eval(p)
        return np.array(p)


# ======================================================================
# Functional interface for MLX autodiff
# ======================================================================

def run_circuit_functional(
    n_qubits: int,
    instructions: List[Tuple],
    params: mx.array,
) -> mx.array:
    """Pure-functional circuit execution for use with mx.grad().

    Parameters
    ----------
    n_qubits : int
    instructions : list of tuples
        Gate instructions.  Parametric gates reference params by index:
          ('ry_p', param_index, qubit)  — uses params[param_index] as theta
          ('rz_p', param_index, qubit)
          ('rx_p', param_index, qubit)
        Non-parametric gates:
          ('h', qubit), ('cx', control, target), etc.
    params : mx.array
        1-D array of trainable parameters.

    Returns
    -------
    probs : mx.array of shape (2^n,)
    """
    dim = 1 << n_qubits
    # Build initial state |0...0>
    state = mx.zeros(dim, dtype=mx.complex64)
    # Cannot do state[0] = 1 in MLX (immutable). Concatenate instead.
    state = mx.concatenate([
        mx.array([1.0 + 0j], dtype=mx.complex64),
        mx.zeros(dim - 1, dtype=mx.complex64),
    ])

    def apply_single(st, gate, qubit):
        # Map logical qubit to tensor axis (little-endian)
        axis = n_qubits - 1 - qubit
        st_t = st.reshape([2] * n_qubits)
        perm = list(range(n_qubits))
        perm.remove(axis)
        perm.append(axis)
        st_t = mx.transpose(st_t, perm)
        shape_front = st_t.shape[:-1]
        st_t = st_t.reshape(-1, 2)
        st_t = st_t @ mx.transpose(gate)
        st_t = st_t.reshape(list(shape_front) + [2])
        inv = [0] * n_qubits
        for i, p in enumerate(perm):
            inv[p] = i
        st_t = mx.transpose(st_t, inv)
        return st_t.reshape(dim)

    def apply_two(st, gate, qa, qb):
        # Map logical qubits to tensor axes (little-endian)
        axis_a = n_qubits - 1 - qa
        axis_b = n_qubits - 1 - qb
        st_t = st.reshape([2] * n_qubits)
        perm = list(range(n_qubits))
        perm.remove(axis_a)
        perm.remove(axis_b)
        perm.append(axis_a)
        perm.append(axis_b)
        st_t = mx.transpose(st_t, perm)
        shape_front = st_t.shape[:-2]
        st_t = st_t.reshape(-1, 4)
        st_t = st_t @ mx.transpose(gate)
        st_t = st_t.reshape(list(shape_front) + [2, 2])
        inv = [0] * n_qubits
        for i, p in enumerate(perm):
            inv[p] = i
        st_t = mx.transpose(st_t, inv)
        return st_t.reshape(dim)

    # Pre-compute fixed gate matrices
    fixed = {
        'h': hadamard(), 'x': pauli_x(), 'y': pauli_y(), 'z': pauli_z(),
        's': phase_gate(), 't': t_gate(),
    }
    two_fixed = {
        'cx': cnot_matrix(), 'cz': cz_gate(), 'swap': swap_gate(),
    }

    for instr in instructions:
        name = instr[0]
        if name in fixed:
            state = apply_single(state, fixed[name], instr[1])
        elif name in two_fixed:
            state = apply_two(state, two_fixed[name], instr[1], instr[2])
        elif name == 'ry_p':
            pidx, qubit = instr[1], instr[2]
            theta = params[pidx]
            half = theta * 0.5
            c = mx.cos(half)
            s = mx.sin(half)
            gate = mx.array([[c, -s], [s, c]]).astype(mx.complex64)
            state = apply_single(state, gate, qubit)
        elif name == 'rz_p':
            pidx, qubit = instr[1], instr[2]
            theta = params[pidx]
            half = theta * 0.5
            cos_h = mx.cos(half)
            sin_h = mx.sin(half)
            one_c = mx.array(1.0 + 0j, dtype=mx.complex64)
            i_c = mx.array(0.0 + 1j, dtype=mx.complex64)
            e_neg = (cos_h.astype(mx.complex64) * one_c +
                     (-sin_h).astype(mx.complex64) * i_c)
            e_pos = (cos_h.astype(mx.complex64) * one_c +
                     sin_h.astype(mx.complex64) * i_c)
            zero_c = mx.array(0.0 + 0j, dtype=mx.complex64)
            gate = mx.array([[e_neg, zero_c], [zero_c, e_pos]])
            state = apply_single(state, gate, qubit)
        elif name == 'rx_p':
            pidx, qubit = instr[1], instr[2]
            theta = params[pidx]
            half = theta * 0.5
            c = mx.cos(half)
            s = mx.sin(half)
            one_c = mx.array(1.0 + 0j, dtype=mx.complex64)
            i_c = mx.array(0.0 + 1j, dtype=mx.complex64)
            c_cx = c.astype(mx.complex64) * one_c
            neg_is = (-s).astype(mx.complex64) * i_c
            gate = mx.array([[c_cx, neg_is], [neg_is, c_cx]])
            state = apply_single(state, gate, qubit)
        elif name in ('ry', 'rz', 'rx'):
            # Non-parametric (fixed angle) version
            theta = instr[1]
            qubit = instr[2]
            if name == 'ry':
                state = apply_single(state, ry_matrix(theta), qubit)
            elif name == 'rz':
                state = apply_single(state, rz_matrix(theta), qubit)
            elif name == 'rx':
                state = apply_single(state, rx_matrix(theta), qubit)
        else:
            raise ValueError(f"Unknown instruction: {name}")

    return mx.abs(state) ** 2
