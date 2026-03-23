"""
Hardware noise profiles for quantum device simulation.

Each profile is a dictionary containing:
  - name:             Human-readable device name
  - p_1q:             Single-qubit gate depolarizing error probability
  - p_2q:             Two-qubit gate depolarizing error probability
  - p_meas:           Measurement readout error probability (symmetric)
  - T1_us:            T1 relaxation time in microseconds
  - T2_us:            T2 dephasing time in microseconds
  - gate_time_1q_us:  Single-qubit gate duration in microseconds
  - gate_time_2q_us:  Two-qubit gate duration in microseconds

The Willow profile was extracted from cirq-google 1.6.1 calibration data
(processor_id='willow_pink') using median values across all qubits.

Usage:
    from simulator.noise_profiles import WILLOW_NOISE, HERON_NOISE, T9_NOISE
    sim = MLXQuantumSimulator(5, noise_profile=WILLOW_NOISE)
"""

# =========================================================================
# Google Willow  (extracted from cirq_google willow_pink calibration)
# =========================================================================
WILLOW_NOISE = {
    'name': 'Google Willow',
    'p_1q': 0.000405,          # median PhasedXZ Pauli error
    'p_2q': 0.003443,          # median CZ Pauli error
    'p_meas': 0.007298,        # median (P(1|0)+P(0|1))/2
    'T1_us': 70.2,             # median T1
    'T2_us': 49.0,             # median T2 = 1/(1/(2*T1) + 1/Tphi)
    'gate_time_1q_us': 0.025,  # 25 ns (PhasedXZGate)
    'gate_time_2q_us': 0.042,  # 42 ns (CZPowGate)
}

# =========================================================================
# IBM Heron  (published values, Eagle/Heron R2 class)
# =========================================================================
HERON_NOISE = {
    'name': 'IBM Heron',
    'p_1q': 0.0003,
    'p_2q': 0.005,
    'p_meas': 0.005,
    'T1_us': 100.0,
    'T2_us': 100.0,
    'gate_time_1q_us': 0.025,
    'gate_time_2q_us': 0.050,
}

# =========================================================================
# QuTech Tuna-9  (representative spin-qubit values)
# =========================================================================
T9_NOISE = {
    'name': 'QuTech Tuna-9',
    'p_1q': 0.001,
    'p_2q': 0.01,
    'p_meas': 0.01,
    'T1_us': 50.0,
    'T2_us': 20.0,
    'gate_time_1q_us': 0.020,
    'gate_time_2q_us': 0.200,
}

# =========================================================================
# Ideal (no noise) — convenience sentinel
# =========================================================================
IDEAL = None

# =========================================================================
# All profiles for iteration
# =========================================================================
ALL_PROFILES = {
    'ideal': IDEAL,
    'willow': WILLOW_NOISE,
    'heron': HERON_NOISE,
    't9': T9_NOISE,
}
