"""
Quantum gate matrices for MLX.

All gates are returned as mx.array with dtype mx.complex64.
Convention: gates operate on the standard computational basis {|0>, |1>}.
"""

import mlx.core as mx
import math


# ---------------------------------------------------------------------------
# Single-qubit fixed gates
# ---------------------------------------------------------------------------

def hadamard() -> mx.array:
    """Hadamard gate H = (1/sqrt2) [[1,1],[1,-1]]."""
    s = 1.0 / math.sqrt(2.0)
    return mx.array([[s, s], [s, -s]], dtype=mx.complex64)


def pauli_x() -> mx.array:
    """Pauli-X (NOT) gate."""
    return mx.array([[0.0, 1.0], [1.0, 0.0]], dtype=mx.complex64)


def pauli_y() -> mx.array:
    """Pauli-Y gate."""
    return mx.array([[0.0, -1j], [1j, 0.0]], dtype=mx.complex64)


def pauli_z() -> mx.array:
    """Pauli-Z gate."""
    return mx.array([[1.0, 0.0], [0.0, -1.0]], dtype=mx.complex64)


def phase_gate() -> mx.array:
    """S (phase) gate = diag(1, i)."""
    return mx.array([[1.0, 0.0], [0.0, 1j]], dtype=mx.complex64)


def t_gate() -> mx.array:
    """T gate = diag(1, e^{i pi/4})."""
    v = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
    return mx.array([[1.0, 0.0], [0.0, v]], dtype=mx.complex64)


# ---------------------------------------------------------------------------
# Single-qubit parametric gates
# ---------------------------------------------------------------------------

def ry_matrix(theta: float) -> mx.array:
    """R_y(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]."""
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return mx.array([[c, -s], [s, c]], dtype=mx.complex64)


def rz_matrix(theta: float) -> mx.array:
    """R_z(theta) = diag(e^{-i t/2}, e^{i t/2})."""
    half = theta / 2.0
    return mx.array([
        [complex(math.cos(-half), math.sin(-half)), 0.0],
        [0.0, complex(math.cos(half), math.sin(half))],
    ], dtype=mx.complex64)


def rx_matrix(theta: float) -> mx.array:
    """R_x(theta) = [[cos(t/2), -i sin(t/2)], [-i sin(t/2), cos(t/2)]]."""
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return mx.array([
        [complex(c, 0), complex(0, -s)],
        [complex(0, -s), complex(c, 0)],
    ], dtype=mx.complex64)


# ---------------------------------------------------------------------------
# MLX-differentiable parametric gates (accept mx.array scalars)
# ---------------------------------------------------------------------------

def ry_matrix_mlx(theta: mx.array) -> mx.array:
    """R_y(theta) differentiable through MLX.

    theta must be an mx.array scalar (or 0-d array).
    Returns a (2,2) complex64 matrix on GPU.
    """
    half = theta * 0.5
    c = mx.cos(half)
    s = mx.sin(half)
    zero = mx.zeros(1, dtype=mx.float32)[0]
    # Build real part and imag part separately, then combine
    real = mx.array([[c, -s], [s, c]])
    imag = mx.zeros_like(real)
    # complex = real + 0j  — MLX doesn't have direct complex construction
    # from traced scalars, so we work in (2, dim) real representation
    # and convert at the end.
    # Actually: we can multiply real matrix with identity complex.
    return real.astype(mx.complex64)


def rz_matrix_mlx(theta: mx.array) -> mx.array:
    """R_z(theta) differentiable through MLX."""
    half = theta * 0.5
    cos_h = mx.cos(half)
    sin_h = mx.sin(half)
    # e^{-i t/2} = cos(t/2) - i sin(t/2)
    # e^{+i t/2} = cos(t/2) + i sin(t/2)
    zero = mx.array(0.0, dtype=mx.float32)
    # Build as 2x2 real and 2x2 imag
    real = mx.array([[cos_h, zero], [zero, cos_h]])
    imag = mx.array([[-sin_h, zero], [zero, sin_h]])
    # Combine: need to form complex. MLX trick: use multiplication
    # result = real_part * (1+0j) + imag_part * (0+1j)
    one_c = mx.array(1.0 + 0j, dtype=mx.complex64)
    i_c = mx.array(0.0 + 1j, dtype=mx.complex64)
    return real.astype(mx.complex64) * one_c + imag.astype(mx.complex64) * i_c


def rx_matrix_mlx(theta: mx.array) -> mx.array:
    """R_x(theta) differentiable through MLX."""
    half = theta * 0.5
    c = mx.cos(half)
    s = mx.sin(half)
    zero = mx.array(0.0, dtype=mx.float32)
    real = mx.array([[c, zero], [zero, c]])
    imag = mx.array([[zero, -s], [-s, zero]])
    one_c = mx.array(1.0 + 0j, dtype=mx.complex64)
    i_c = mx.array(0.0 + 1j, dtype=mx.complex64)
    return real.astype(mx.complex64) * one_c + imag.astype(mx.complex64) * i_c


# ---------------------------------------------------------------------------
# Two-qubit gates (4x4 matrices)
# ---------------------------------------------------------------------------

def cnot_matrix() -> mx.array:
    """CNOT (CX) gate — control=qubit0, target=qubit1 in the 4x4 space."""
    return mx.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=mx.complex64)


def swap_gate() -> mx.array:
    """SWAP gate."""
    return mx.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=mx.complex64)


def cz_gate() -> mx.array:
    """Controlled-Z gate."""
    return mx.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ], dtype=mx.complex64)
