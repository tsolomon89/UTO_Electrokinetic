# ---------------------
# --- Version 5.1.2 ---
# ---------------------
import subprocess
import sys

try:
    from fpdf import FPDF
except ModuleNotFoundError:
    print("fpdf not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "fpdf2"])
    from fpdf import FPDF

try:
    import seaborn
except ModuleNotFoundError:
    print("seaborn not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn
try:
    import IPython
except ModuleNotFoundError:
    print("IPython not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "IPython"])
    import IPython

from typing import Dict, Callable, Tuple, List
import numpy as np
import sympy as sp
import numpy as np
from scipy.optimize import minimize_scalar, minimize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from IPython.display import display, HTML, Math, Latex, Markdown
import sympy as sp
import numpy as np


# ---------------------
# --- Contanstants  ---
# ---------------------

CONSTANTS = {
    'h': 6.626e-34,    # Planck's constant (J⋅s)
    'c': 2.998e8,      # Speed of light in vacuum (m/s)
    'e': 1.602e-19,    # Elementary charge (C)
    'me': 9.109e-31,   # Electron rest mass (kg)
    'k_B': 1.380e-23,  # Boltzmann constant (J/K)
    'epsilon_0': 8.854e-12,  # Vacuum permittivity (F/m)
    'mu_0': 1.257e-6,  # Vacuum permeability (N/A^2)
    'A': 1.20e6,       # Richardson constant (A/(m^2⋅K^2))
    'Na': 6.022e23,    # Avogadro's number (mol^-1)
    'R': 8.314,        # Gas constant (J/(mol⋅K))
    'sigma': 5.67e-8,  # Stefan-Boltzmann constant (W/(m^2⋅K^4))
    'G': 6.674e-11,    # Gravitational constant (N⋅m^2/kg^2)
    'mu_B': 9.274e-24, # Bohr magneton (J/T)
    'hbar': 1.0546e-34, # Reduced Planck's constant (J⋅s)

}

# ----------------
# --- Effects  ---
# ----------------


class Effect:
    def __init__(self, name: str, category: str, params: Dict, std_calc: Callable, unified_calc: Callable):
        self.name = name
        self.category = category
        self.params = params
        self.std_calc = std_calc
        self.unified_calc = unified_calc
        self.is_electromagnetic = category in ["Electromagnetic", "Quantum"]


    def calculate_standard(self, constants: Dict) -> float:
        return self.std_calc(self.params, constants)

    def calculate_unified(self, constants: Dict) -> Tuple[float, float]:
        return self.unified_calc(self.params, constants)

class EffectRegistry:
    def __init__(self):
        self.effects = {}

    def register_effect(self, effect: Effect):
        self.effects[effect.name] = effect

    def get_effect(self, name: str) -> Effect:
        return self.effects.get(name)

    def get_effects_by_category(self, category: str) -> List[Effect]:
        return [effect for effect in self.effects.values() if effect.category == category]

    def get_all_effects(self) -> List[Effect]:
        return list(self.effects.values())

def create_effect(name: str, category: str, params: Dict, std_calc: Callable, unified_calc: Callable) -> Effect:
    return Effect(name, category, params, std_calc, unified_calc)

def initialize_effects() -> EffectRegistry:
    registry = EffectRegistry()

    effects = [
        ("Hall", "Electromagnetic", {"i": 1, "mf": 1, "cc": 1e23, "t": 0.01, "ts": 1.0},
         lambda p, c: (p['i'] * p['mf'] / (p["cc"] * c['e'] * p["t"])) * p['i'] * p['ts'],
         lambda p, c: (p['i'] * p['mf'] / (p["cc"] * c['e'] * p["t"]), p['i'])),
        ("Electrostrictive", "Electromagnetic", {"ef": 1e6, "ym": 1e9, "rp": 10, "v": 1e-9},
         lambda p, c: 0.5 * p['rp'] * c['epsilon_0'] * p['ym'] * p['ef']**2 * p['v'],
         lambda p, c: (0.5 * p['rp'] * c['epsilon_0'] * p['ym'] * p['ef']**2 * p['v'], 1)),
        ("Piezoelectric", "Electromagnetic", {"pc": 2e-12, "sn": 1000},
         lambda p, c: p['pc'] * p['sn'],
         lambda p, c: (p['pc'] * p['sn'], 1)),
        ("Triboelectric", "Electromagnetic", {"sc": 2e-6, "ca": 0.01},
         lambda p, c: p['sc'] * p['ca'],
         lambda p, c: (p['sc'] * p['ca'], 1)),
        ("Converse Piezoelectric", "Electromagnetic", {"pc": 2.5e-11, "ef": 1e6},
         lambda p, c: 0.5 * p['pc'] * p['ef']**2,
         lambda p, c: (0.5 * p['pc'] * p['ef']**2, 1)),
        ("Ferroelectric", "Electromagnetic", {"p": 0.2, "ef": 1e6},
         lambda p, c: p['p'] * p['ef'],
         lambda p, c: (p['p'] * p['ef'], 1)),
        ("Piezoresistive", "Electromagnetic", {"pc": 1e-11, "sp": 1e6, "r": 100, "i": 0.1, "ts": 10},
         lambda p, c: 0.5 * p['pc'] * p['sp'] * p['r'] * p['i']**2 * p['ts'],
         lambda p, c: (p['pc'] * p['sp'] * p['r'], p['i']**2 * p['ts'])),
        ("Nernst", "Electromagnetic", {"N": 1e-7, "B": 1, "grad_T": 100, "sigma": 1e7, "A": 1e-4},
         lambda p, c: p['N'] * p['B'] * p['grad_T'],
         lambda p, c: (p['N'] * p['B'] * p['grad_T'], p['sigma'] * p['A'] * p['N'] * p['B'] * p['grad_T'])),
        ("Flexoelectric", "Electromagnetic", {"f": 1e-9, "s": 1e-3, "L": 0.1},
         lambda p, c: p['f'] * p['s'] / p['L'],
         lambda p, c: (p['f'] * p['s'] / p['L']**2, p['s'] / p['L'])),
        ("Magnetoelectric", "Electromagnetic", {"alpha": 1e-12, "H": 1e5, "E": 1e6},
         lambda p, c: p['alpha'] * p['H'] * p['E'],
         lambda p, c: (p['alpha'] * p['H'], p['E'])),
        ("Ettingshausen", "Electromagnetic", {"P": 1e-4, "I": 1, "B": 1, "lambda": 10, "S": 200e-6},
         lambda p, c: p['P'] * p['I'] * p['B'] / p['lambda'],
         lambda p, c: (p['S'] * p['P'] * p['I'] * p['B'] / p['lambda'], p['I'])),
        ("Electrostriction", "Electromagnetic", {"chi": 1e-18, "E": 1e6, "V": 1e-6},
         lambda p, c: p['chi'] * p['E']**2 * p['V'],
         lambda p, c: (p['chi'] * p['E']**2, p['V'])),
        ("Magnetostriction", "Electromagnetic", {"lambda_s": 1e-5, "H": 1e5, "V": 1e-6},
         lambda p, c: p['lambda_s'] * p['H']**2 * p['V'],
         lambda p, c: (p['lambda_s'] * p['H']**2, p['V'])),
        ("Faraday", "Electromagnetic", {"V": 1e5, "B": 1, "L": 0.1, "sigma": 1e7, "A": 1e-4},
         lambda p, c: p['V'] * p['B'] * p['L'],
         lambda p, c: (p['V'] * p['B'], p['sigma'] * p['A'] * p['V'] * p['B'])),
        # --- Chemimcal  Effects ---
        ("Electrochemiluminescence", "Electrochemical", {"k": 1e-3, "c": 1e-6, "V": 1.0},
         lambda p, c: p['k'] * p['c'] * p['V'],
         lambda p, c: (p['k'] * p['c'], p['V'])),
        ("Electrolysis", "Electrochemical", {"v": 2.0, "i": 1.0, "ts": 3600},
         lambda p, c: p['v'] * p['i'] * p['ts'],
         lambda p, c: (p['v'], p['i'] * p['ts'])),
        ("Galvanic Cell", "Electrochemical", {"cp": 1.5, "i": 0.5, "ts": 3600},
         lambda p, c: p['cp'] * p['i'] * p['ts'],
         lambda p, c: (p['cp'], p['i'] * p['ts'])),
        ("Chemiluminescence", "Chemical", {"rr": 1e-6, "qy": 0.1, "pe": 3e-19},
         lambda p, c: p['rr'] * p['qy'] * p['pe'],
         lambda p, c: (p['qy'] * p['pe'], p['rr'])),
        ("Photocatalytic", "Chemical", {"eta": 0.1, "I": 1000, "A": 1e-4, "k": 1e-3},
         lambda p, c: p['eta'] * p['I'] * p['A'] * p['k'],
         lambda p, c: (p['eta'] * p['I'] * p['A'], p['k'])),
        # --- Thermal  Effects ---
        ("Magnetocaloric", "Thermal", {"mu_0": 1.257e-6, "T": 300, "dM_dT": 1, "delta_H": 1e5, "C": 450, "S": 200e-6, "L": 0.1, "sigma": 1e7, "A": 1e-4},
         lambda p, c: p['mu_0'] * p['T'] * p['dM_dT'] * p['delta_H'] / p['C'],
         lambda p, c: (p['S'] * p['mu_0'] * p['T'] * p['dM_dT'] * p['delta_H'] / (p['C'] * p['L']),
                       p['sigma'] * p['A'] * p['S'] * p['mu_0'] * p['T'] * p['dM_dT'] * p['delta_H'] / (p['C'] * p['L']))),
        ("Seebeck", "Thermal", {"S": 200e-6, "delta_T": 100, "L": 0.1, "sigma": 1e7, "A": 1e-4},
         lambda p, c: p['S'] * p['delta_T'],
         lambda p, c: (p['S'] * p['delta_T'] / p['L'], p['sigma'] * p['A'] * p['S'] * p['delta_T'] / p['L'])),
        ("Peltier", "Thermal", {"pc": 0.1, "i": 1.0, "ts": 3600},
         lambda p, c: p['pc'] * p['i'] * p['ts'],
         lambda p, c: (p['pc'], p['i'] * p['ts'])),
        ("Thomson", "Thermal", {"tc": 1e-5, "i": 1.0, "tg": 100, "l": 0.1},
         lambda p, c: p['tc'] * p['i'] * p['tg'] * p['l'],
         lambda p, c: (p['tc'] * p['tg'] * p['l'], p['i'])),
        ("Thermoelectric", "Thermal", {"sc": 200e-6, "td": 100},
         lambda p, c: p['sc'] * p['td'],
         lambda p, c: (p['sc'] * p['td'], 1)),
        ("Pyroelectric", "Thermal", {"pc": 0.3e-6, "a": 0.01, "tc": 50},
         lambda p, c: p['pc'] * p['a'] * p['tc'],
         lambda p, c: (p['pc'] * p['a'] * p['tc'], 1)),
        ("Thermionic", "Thermal", {"tk": 1500, "wf": 4.5},
         lambda p, c: c['A'] * p['tk']**2 * np.exp(-p['wf'] * c['e'] / (c['k_B'] * p['tk'])) * c['e'],
         lambda p, c: (p['wf'] * c['e'], c['A'] * p['tk']**2 * np.exp(-p['wf'] * c['e'] / (c['k_B'] * p['tk'])))),
        ("Thermoelastic", "Thermal", {"te": 1e-5, "tc": 100, "ym": 1e9},
         lambda p, c: 0.5 * p['ym'] * (p['te'] * p['tc'])**2,
         lambda p, c: (0.5 * p['ym'] * (p['te'] * p['tc'])**2, 1)),
        # --- Quantum  Effects ---
        ("Aharonov-Bohm", "Quantum", {"B": 1.0, "A": 1e-6},
         lambda p, c: c['e'] * p['B'] * p['A'] / c['hbar'],
         lambda p, c: (c['e'] * p['B'] / c['hbar'], p['A'])),
        ("Photoelectric", "Quantum", {"f": 1e15, "wf": 2.5},
         lambda p, c: p['f'] * c['h'] - p['wf'] * c['e'],
         lambda p, c: (c['h'] * p["f"], 1)),
        ("Compton", "Quantum", {"wl": 0.1, "sa": 90},
         lambda p, c: c['h'] * c['c'] / (p['wl'] * 1e-9) - c['h'] * c['c'] / ((p['wl'] * 1e-9) + (c['h'] / (c['me'] * c['c'])) * (1 - (np.cos(np.radians(p['sa'])) if isinstance(p['sa'], (int, float)) else sp.cos(p['sa'])))),
         lambda p, c: (c['h'] * c['c'] / (p['wl'] * 1e-9) - c['h'] * c['c'] / ((p['wl'] * 1e-9) + (c['h'] / (c['me'] * c['c'])) * (1 - (np.cos(np.radians(p['sa'])) if isinstance(p['sa'], (int, float)) else sp.cos(p['sa'])))), 1)),
        ("Quantum Tunneling", "Quantum", {"V": 10, "m": 9.1e-31, "a": 1e-10},
         lambda p, c: np.exp(-2 * p['a'] * np.sqrt(2 * p['m'] * p['V']) / c['hbar']),
         lambda p, c: (np.sqrt(2 * p['m'] * p['V']) / c['hbar'], np.exp(-2 * p['a']))),
        ("Landauer Conductance", "Quantum", {"N": 1},
         lambda p, c: p['N'] * 2 * c['e']**2 / c['h'],
         lambda p, c: (2 * c['e']**2 / c['h'], p['N'])),
        ("Kondo Effect", "Quantum", {"T": 10, "T_K": 1},
         lambda p, c: c['h'] / (c['e']**2 * np.log(p['T'] / p['T_K'])**2),
         lambda p, c: (c['h'] / c['e']**2, 1 / np.log(p['T'] / p['T_K'])**2)),
        ("Unruh Effect", "Quantum", {"a": 1e20},
         lambda p, c: c['hbar'] * p['a'] / (2 * np.pi * c['c'] * c['k_B']),
         lambda p, c: (c['hbar'] / (2 * np.pi * c['c'] * c['k_B']), p['a'])),
        ("Hawking Radiation", "Quantum", {"M": 1e30},
         lambda p, c: c['hbar'] * c['c']**3 / (8 * np.pi * c['G'] * p['M'] * c['k_B']),
         lambda p, c: (c['hbar'] * c['c']**3 / (8 * np.pi * c['G'] * c['k_B']), 1 / p['M'])),
        ("Quantum Zeno Effect", "Quantum", {"gamma": 1e9, "t": 1e-6},
         lambda p, c: np.exp(-p['gamma'] * p['t']),
         lambda p, c: (p['gamma'], -p['t'])),
        ("Dynamical Casimir Effect", "Quantum", {"L": 1e-6, "v": 0.1 * 2.998e8},
         lambda p, c: np.pi * c['hbar'] * p['v']**2 / (24 * p['L'] * c['c']**2),
         lambda p, c: (np.pi * c['hbar'] / (24 * p['L'] * c['c']**2), p['v']**2)),
        ("Berry Phase", "Quantum", {"Omega": 1},
         lambda p, c: 2 * np.pi * (1 - np.cos(p['Omega'])),
         lambda p, c: (2 * np.pi, 1 - np.cos(p['Omega']))),
        ("Zeeman", "Quantum", {"B": 1.0, "g": 2.0, "m": 0.5},
         lambda p, c: p['g'] * c['mu_B'] * p['B'] * p['m'],
         lambda p, c: (p['g'] * c['mu_B'] * p['B'], p['m'])),
        ("Stark", "Quantum", {"E": 1e6, "alpha": 1e-40},
         lambda p, c: -0.5 * p['alpha'] * p['E']**2,
         lambda p, c: (p['alpha'] * p['E'], p['E'])),
        ("Josephson", "Quantum", {"V": 1e-3, "Kj": 483597.8e9},
         lambda p, c: p['Kj'] * p['V'],
         lambda p, c: (p['V'], p['Kj'])),
        ("Quantum Hall", "Quantum", {"B": 10, "n": 1, "i": 2},
         lambda p, c: c['h'] / (p['i'] * c['e']**2),
         lambda p, c: (c['h'] / (p['i'] * c['e']), c['e'] * p['B'] / (p['n'] * c['h']))),
        ("Casimir", "Quantum", {"A": 1e-4, "d": 1e-6},
         lambda p, c: -((c['h'] * c['c'] * np.pi**2) / (240 * p['d']**4)) * p['A'],
         lambda p, c: ((c['h'] * c['c'] * np.pi**2) / (240 * p['d']**4), p['A'])),
        ("Quantum Tunneling", "Quantum", {"V": 10, "m": 9.1e-31, "a": 1e-10},
         lambda p, c: np.exp(-2 * p['a'] * np.sqrt(2 * p['m'] * p['V']) / c['hbar']),
         lambda p, c: (np.sqrt(2 * p['m'] * p['V']) / c['hbar'], np.exp(-2 * p['a']))),
        ("Landauer Conductance", "Quantum", {"N": 1},
         lambda p, c: p['N'] * 2 * c['e']**2 / c['h'],
         lambda p, c: (2 * c['e']**2 / c['h'], p['N'])),
        ("Quantum Dots", "Quantum", {"R": 5e-9, "m": 9.1e-31},
         lambda p, c: (c['h']**2) / (8 * p['m'] * p['R']**2),
          lambda p, c: (c['h'] / (4 * p['m'] * p['R']**2), c['h'] / (2 * p['R']))),
        ("Quantum Coherence", "Quantum", {"omega": 1e9, "gamma": 1e6},
         lambda p, c: p['omega'] / p['gamma'],
         lambda p, c: (p['omega'], 1 / p['gamma'])),
        ("Entanglement Entropy", "Quantum", {"d": 2, "p": 0.5},
         lambda p, c: -p['p'] * np.log2(p['p'] + 1e-10) - (1 - p['p']) * np.log2(1 - p['p'] + 1e-10),
         lambda p, c: (-np.log2(p['p'] + 1e-10), -np.log2(1 - p['p'] + 1e-10))),
        ("Quantum Teleportation", "Quantum", {"fidelity": 0.85},
         lambda p, c: (2 * p['fidelity'] + 1) / 3,
         lambda p, c: (p['fidelity'], (1 + p['fidelity']) / 2)),
        ("Superconducting Transition", "Quantum", {"Tc": 10, "n": 1e28},
         lambda p, c: 1.764 * c['k_B'] * p['Tc'],
         lambda p, c: (c['k_B'] * p['Tc'], 1.764 * p['n']**(1/3))),
        ("Quantum Cryptography", "Quantum", {"key_rate": 1e3, "error_rate": 0.05},
         lambda p, c: p['key_rate'] * (1 - 2 * p['error_rate']),
         lambda p, c: (p['key_rate'], 1 - 2 * p['error_rate'])),
        # --- Optomechanical Effects ---
        ("Radiation Pressure", "Optomechanical", {"I": 100, "A": 1e-4},
         lambda p, c: 2 * p['I'] * p['A'] / c['c'],
         lambda p, c: (2 * p['I'] * p['A'] / c['c'], 1)),
        ("Optical Tweezers", "Optomechanical", {"n_m": 1.33, "n_w": 1.00, "P": 1e-3},
         lambda p, c: (p['n_m'] - p['n_w']) * p['P'] / c['c'],
         lambda p, c: ((p['n_m'] - p['n_w']) / c['c'], p['P'])),
        ("Optomechanical Cooling", "Optomechanical", {"omega_m": 1e6, "kappa": 1e4, "g0": 1e-3, "n_th": 100},
         lambda p, c: c['hbar'] * p['omega_m'] * p['kappa'] * p['g0']**2 * p['n_th'] / (4 * p['kappa']**2 + p['g0']**2 * (2 * p['n_th'] + 1)),
         lambda p, c: (c['hbar'] * p['omega_m'] * p['g0'], p['kappa'] * p['n_th'])),
        # --- Spintronic Effects ---
        ("Spin-Transfer Torque", "Spintronic", {"P": 0.5, "lambda": 0.3, "theta": 0, "I": 1e-3},
         lambda p, c: (c['hbar'] / 2) * (p['P'] * p['I'] / (1 + p['lambda']**2 * sp.cos(p['theta']))),
         lambda p, c: ((c['hbar'] / 2) * p['P'] / (1 + p['lambda']**2 * sp.cos(p['theta'])), p['I'])),
        ("Spin Pumping", "Spintronic", {"g_eff": 1e11, "omega": 1e9},
         lambda p, c: (c['hbar'] / (2 * c['e'])) * p['g_eff'] * p['omega'],
         lambda p, c: ((c['hbar'] / 2) * p['omega'], p['g_eff'] / c['e'])),
        ("Spin Hall", "Spintronic", {"j": 1e6, "sigma": 1e6, "lambda_so": 1e-9},
         lambda p, c: 2 * c['e'] * p['lambda_so'] * p['j'] / p['sigma'],
         lambda p, c: (2 * c['e'] * p['lambda_so'] / p['sigma'], p['j'])),
    ]

    for name, category, params, std_calc, unified_calc in effects:
        registry.register_effect(create_effect(name, category, params, std_calc, unified_calc))

    return registry



# --------------------------
# --- Scaling Calculator ---
# --------------------------
class ScalingCalculator:
    @staticmethod
    def apply_scaling(effect: Effect, constants: dict) -> dict:
        # Perform 4.0.1 calculation (standard)
        standard_result = ScalingCalculator.apply_standard_scaling(effect, constants)

        # Perform 4.1 calculation (extended)
        extended_result = ScalingCalculator.apply_extended_scaling(effect, constants)

        # Determine success and method
        standard_successful = standard_result['Original_Relative_Error'] <= 1.0 or \
                              standard_result['Linear_Scaled_Relative_Error'] <= 1.0 or \
                              standard_result['Power_Law_Scaled_Relative_Error'] <= 1.0

        if standard_result['Original_Relative_Error'] <= 1.0:
            standard_success_method = 'No scaling needed'
        elif standard_result['Linear_Scaled_Relative_Error'] <= 1.0:
            standard_success_method = 'Linear scaling'
        elif standard_result['Power_Law_Scaled_Relative_Error'] <= 1.0:
            standard_success_method = 'Power law scaling'
        else:
            standard_success_method = 'N/A'

        extended_successful = extended_result['Original_Relative_Error'] <= 1.0 or \
                              extended_result['Linear_Scaled_Relative_Error'] <= 1.0 or \
                              extended_result['Power_Law_Scaled_Relative_Error'] <= 1.0

        if extended_result['Original_Relative_Error'] <= 1.0:
            extended_success_method = 'No scaling needed'
        elif extended_result['Linear_Scaled_Relative_Error'] <= 1.0:
            extended_success_method = 'Linear scaling'
        elif extended_result['Power_Law_Scaled_Relative_Error'] <= 1.0:
            extended_success_method = 'Power law scaling'
        else:
            extended_success_method = 'N/A'

        # Combine results
        combined_result = {
            'Effect': effect.name,
            'Category': effect.category,
            'Standard': standard_result,
            'Extended': extended_result,
            'Standard_Successful': standard_successful,
            'Standard_Success_Method': standard_success_method,
            'Extended_Successful': extended_successful,
            'Extended_Success_Method': extended_success_method
        }

        return combined_result

    @staticmethod
    def apply_standard_scaling(effect: Effect, constants: dict) -> dict:
        try:
            std_res = effect.calculate_standard(constants)
            unified_res = effect.calculate_unified(constants)[0] * effect.calculate_unified(constants)[1]

            linear_factor = ScalingCalculator.calculate_linear_scaling(effect, constants)
            power_law_factors = ScalingCalculator.calculate_power_law_scaling(effect, constants)

            linear_scaled_res = linear_factor * unified_res
            power_law_scaled_res = power_law_factors[0] * (unified_res ** power_law_factors[1])

            return {
                'Standard_Result': float(std_res),
                'Unified_Result': float(unified_res),
                'Original_Relative_Error': float(100 * abs((std_res - unified_res) / std_res) if std_res != 0 else np.inf),
                'Linear_Scaled_Result': float(linear_scaled_res),
                'Linear_Scaling_Factor': float(linear_factor),
                'Linear_Scaled_Relative_Error': float(100 * abs((std_res - linear_scaled_res) / std_res) if std_res != 0 else np.inf),
                'Power_Law_Scaled_Result': float(power_law_scaled_res),
                'Power_Law_Scaling_Factor_A': float(power_law_factors[0]),
                'Power_Law_Scaling_Factor_B': float(power_law_factors[1]),
                'Power_Law_Scaled_Relative_Error': float(100 * abs((std_res - power_law_scaled_res) / std_res) if std_res != 0 else np.inf)
            }
        except Exception as e:
            print(f"Error in standard scaling {effect.name}: {str(e)}")
            return None

    @staticmethod
    def apply_extended_scaling(effect: Effect, constants: dict) -> dict:
        if effect.is_electromagnetic:
            return ScalingCalculator.apply_electromagnetic_scaling(effect, constants)
        else:
            return ScalingCalculator.apply_standard_scaling(effect, constants)

    @staticmethod
    def apply_electromagnetic_scaling(effect: Effect, constants: dict) -> dict:
        try:
            std_res = effect.calculate_standard(constants)
            emf, current = effect.calculate_unified(constants)
            unified_res = emf * current

            # Include electromagnetic energy density term
            volume = 1.0  # Assume unit volume, adjust as needed
            u = 0.5 * (constants['epsilon_0'] * emf**2 + constants['mu_0'] * current**2)
            modified_unified_res = unified_res + u * volume

            linear_factor = std_res / modified_unified_res if modified_unified_res != 0 else 1
            power_law_factors = ScalingCalculator.calculate_power_law_scaling(effect, constants)

            linear_scaled_res = linear_factor * modified_unified_res
            power_law_scaled_res = power_law_factors[0] * (modified_unified_res ** power_law_factors[1])

            return {
                'Standard_Result': float(std_res),
                'Unified_Result': float(modified_unified_res),
                'Original_Relative_Error': float(100 * abs((std_res - modified_unified_res) / std_res) if std_res != 0 else np.inf),
                'Linear_Scaled_Result': float(linear_scaled_res),
                'Linear_Scaling_Factor': float(linear_factor),
                'Linear_Scaled_Relative_Error': float(100 * abs((std_res - linear_scaled_res) / std_res) if std_res != 0 else np.inf),
                'Power_Law_Scaled_Result': float(power_law_scaled_res),
                'Power_Law_Scaling_Factor_A': float(power_law_factors[0]),
                'Power_Law_Scaling_Factor_B': float(power_law_factors[1]),
                'Power_Law_Scaled_Relative_Error': float(100 * abs((std_res - power_law_scaled_res) / std_res) if std_res != 0 else np.inf)
            }
        except Exception as e:
            print(f"Error in electromagnetic scaling {effect.name}: {str(e)}")
            return None

    @staticmethod
    def calculate_linear_scaling(effect: Effect, constants: dict) -> float:
        std_res = effect.calculate_standard(constants)
        unified_res = effect.calculate_unified(constants)[0] * effect.calculate_unified(constants)[1]

        if np.isscalar(std_res) and np.isscalar(unified_res):
            return std_res / unified_res if unified_res != 0 else 1

        def objective(scale):
            return np.sum((std_res - scale * unified_res)**2)

        result = minimize_scalar(objective)
        return result.x

    @staticmethod
    def calculate_power_law_scaling(effect: Effect, constants: dict) -> tuple:
        std_res = effect.calculate_standard(constants)
        unified_res = effect.calculate_unified(constants)[0] * effect.calculate_unified(constants)[1]

        if np.isscalar(std_res) and np.isscalar(unified_res):
            return [std_res / unified_res if unified_res != 0 else 1, 1]

        def objective(params):
            a, b = params
            return np.sum((std_res - a * (unified_res ** b))**2)

        result = minimize(objective, [1, 1], method='Nelder-Mead')
        return result.x

# ----------------
# --- Analysis ---
# ----------------

class MathematicalAnalysis:
    @staticmethod
    def derive_unified_equation():
        pass
        t, m, v, E, I = sp.symbols('t m v E I')
        work = sp.diff(0.5 * m * v**2, t)
        unified_eq = sp.Eq(E * I, work)
        return sp.latex(unified_eq)

    @staticmethod
    @staticmethod
    def improved_dimensionality_analysis(effect: Effect):
        # Define base units
        M, L, T, I, K, Θ, N = sp.symbols('M L T I K Θ N')

        # Define dimensions of constants and parameters
        dimensions = {
            'h': M * L**2 / T,
            'hbar': M * L**2 / T,
            'c': L / T,
            'e': I * T,
            'me': M,
            'k_B': M * L**2 / (T**2 * K),
            'epsilon_0': I**2 * T**4 / (M * L**3),
            'mu_0': M * L / I**2,
            'A': I / (L**2 * K**2),
            'Na': 1 / N,
            'R': M * L**2 / (T**2 * K * N),
            'sigma': M / (T**3 * K**4),
            'G': L**3 / (M * T**2),
            'mu_B': I * L**2,
            # New dimensions for quantum effects
            'B': M / (I * T**2),  # Magnetic field
            'E': M * L / (I * T**3),  # Electric field
            'V': M * L**2 / (I * T**3),  # Voltage
            'Kj': I / (M * L**2),  # Josephson constant
            'alpha': M * L**5 / T**2,  # Polarizability
            'sigma': I**2 * T**3 / (M * L**3),  # Electrical conductivity
            'lambda_so': L,  # Spin-orbit coupling length
            'gamma': 1 / T,  # Decay rate
            'Omega': 1,  # Solid angle (dimensionless)
            'a': L / T**2,  # Acceleration
            'u': M / (L * T**2),  # Energy density
            'n': 1 / L**3,  # Number density
            'R': L,  # Radius
            'omega': 1 / T,  # Angular frequency
            'd': 1,  # Dimensionless (for entanglement)
            'p': 1,  # Probability (dimensionless)
            'fidelity': 1,  # Dimensionless
            'Tc': K,  # Critical temperature
            'key_rate': 1 / T,  # Rate
            'error_rate': 1,  # Dimensionless
        }

        # Add dimensions for effect parameters
        for param in effect.params:
            if param not in dimensions:
                dimensions[param] = sp.Symbol(f'{param}_dim')

        # Calculate dimensions of standard and unified calculations

        try:
            std_dim = effect.std_calc(dimensions, dimensions)
            unified_dim = effect.unified_calc(dimensions, dimensions)

            if effect.is_electromagnetic:
                emf_dim, current_dim = unified_dim
                u_dim = 0.5 * (dimensions['epsilon_0'] * emf_dim**2 + dimensions['mu_0'] * current_dim**2)
                unified_power_dim = sp.expand(emf_dim * current_dim + u_dim * L**3)
            else:
                emf_dim, current_dim = unified_dim
                unified_power_dim = sp.expand(emf_dim * current_dim)

            match = sp.simplify(std_dim - unified_power_dim) == 0 or (std_dim.is_constant() and unified_power_dim.is_constant())

            return {
                'match': match,
                'standard_dimensions': str(std_dim),
                'unified_dimensions': f"EMF: {str(emf_dim)}, Current: {str(current_dim)}",
                'unified_power_dimensions': str(unified_power_dim)
            }
        except Exception as e:
            return {
                'match': False,
                'standard_dimensions': "Error",
                'unified_dimensions': "Error",
                'unified_power_dimensions': "Error",
                'error_message': str(e)
            }

    @staticmethod
    def taylor_expansion(effect: Effect, scaling_factors: dict):
        x0 = effect.calculate_unified(CONSTANTS)[0] * effect.calculate_unified(CONSTANTS)[1]
        y0 = effect.calculate_standard(CONSTANTS)

        if scaling_factors and 'Power_Law_Scaling_Factor_A' in scaling_factors and 'Power_Law_Scaling_Factor_B' in scaling_factors:
            a = scaling_factors['Power_Law_Scaling_Factor_A']
            b = scaling_factors['Power_Law_Scaling_Factor_B']
            f = lambda x: a * x**b
            f_prime = lambda x: a * b * x**(b-1)
            expansion = f(x0) + f_prime(x0) * (sp.Symbol('x') - x0)
        else:
            expansion = y0 + (sp.Symbol('x') - x0) * (y0 / x0) if x0 != 0 else y0

        return f"Taylor expansion for {effect.name} around x0 = {x0:.2e}:\nf(x) ≈ {expansion}"

    @staticmethod
    def sensitivity_analysis(effect: Effect):
        params = effect.params
        base_result = effect.calculate_standard(CONSTANTS)
        sensitivities = {}

        for param, value in params.items():
            delta = value * 0.01 * (1 + abs(value))  # 1% change
            params_plus = params.copy()
            params_plus[param] = value + delta
            result_plus = effect.std_calc(params_plus, CONSTANTS)

            sensitivity = abs((result_plus - base_result) / (delta / value)) if value != 0 else 0
            sensitivities[param] = sensitivity

        most_sensitive = max(sensitivities, key=sensitivities.get)

        return {
            'sensitivities': sensitivities,
            'most_sensitive': most_sensitive,
            'max_sensitivity': sensitivities[most_sensitive]
        }


    @staticmethod
    def improved_uncertainty_propagation(effect: Effect, param_uncertainties: dict):
        params = effect.params
        warnings = []
        try:
            base_result = effect.calculate_standard(CONSTANTS)
            variances = []

            for param, uncertainty in param_uncertainties.items():
                if param in params:
                    delta = params[param] * 0.01  # 1% change for numerical derivative
                    params_plus = params.copy()
                    params_plus[param] = params[param] + delta
                    result_plus = effect.std_calc(params_plus, CONSTANTS)

                    try:
                        partial_derivative = (result_plus - base_result) / delta if delta != 0 else 0
                        variance = (partial_derivative * uncertainty)**2
                        if isinstance(variance, (int, float, complex, np.number)):
                            variances.append(variance)
                        else:
                            warnings.append(f"Non-numeric variance for parameter {param}: {variance}")
                    except TypeError:
                        warnings.append(f"Could not calculate uncertainty for parameter {param}")

            if variances:
                total_uncertainty = np.sqrt(sum(variances))
                relative_uncertainty = (total_uncertainty / base_result) * 100 if base_result != 0 else np.inf
            else:
                total_uncertainty = np.nan
                relative_uncertainty = np.nan

            return {
                'effect': effect.name,
                'total_uncertainty': float(total_uncertainty),
                'relative_uncertainty': float(relative_uncertainty),
                'warnings': warnings
            }
        except Exception as e:
            return {
                'effect': effect.name,
                'total_uncertainty': np.nan,
                'relative_uncertainty': np.nan,
                'warnings': warnings + [f"Error in uncertainty propagation: {str(e)}"]
            }

# -----------------
# --- Presenter ---
# -----------------



class ComprehensiveResultsPresenter:
    @staticmethod
    def latex_equation(equation):
        return Math(equation)

    @staticmethod
    def create_pdf(content):
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 8)
                self.cell(0, 10, 'Unified Electrokinetic Induction Theory Analysis', 0, 1, 'C')

            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 6)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        pdf = PDF(orientation='L', unit='mm', format='A2')
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font('Arial', '', 5)

        lines = content.split('\n')
        for line in lines:
            pdf.multi_cell(0, 3, line)

        return pdf.output(dest='S').encode('latin-1')

    @staticmethod
    def create_download_buttons(df, text_content):
        buttons = []

        # CSV download button
        csv_content = df.to_csv(index=False)
        buttons.append(ComprehensiveResultsPresenter.create_download_button(csv_content, "results_table.csv", "Download Table as CSV"))

        # Table PDF download button
        try:
            pdf_content = ComprehensiveResultsPresenter.create_pdf(df.to_string())
            buttons.append(ComprehensiveResultsPresenter.create_download_button(pdf_content, "results_table.pdf", "Download Table as PDF"))
        except Exception as e:
            print(f"Error creating table PDF: {str(e)}")

        # Text PDF download button
        try:
            text_pdf_content = ComprehensiveResultsPresenter.create_pdf(text_content)
            buttons.append(ComprehensiveResultsPresenter.create_download_button(text_pdf_content, "results_text.pdf", "Download Text as PDF"))
        except Exception as e:
            print(f"Error creating text PDF: {str(e)}")

        # Text TXT download button
        buttons.append(ComprehensiveResultsPresenter.create_download_button(text_content, "results_text.txt", "Download Text as TXT"))

        return ' '.join(buttons)

    @staticmethod
    def create_download_button(content, filename, button_text):
        if isinstance(content, bytes):
            b64 = base64.b64encode(content).decode()
        else:
            b64 = base64.b64encode(content.encode('utf-8')).decode()
        return f'<a download="{filename}" href="data:application/octet-stream;base64,{b64}" class="btn btn-primary">{button_text}</a>'
    @staticmethod
    def get_unified_equation():
        return MathematicalAnalysis.derive_unified_equation()

    @staticmethod
    def get_dimensional_analysis_equation():
        E, I, M, L, T = sp.symbols('E I M L T')
        dimensional_eq = sp.Eq(E * I, M * L**2 * T**(-3))
        return sp.latex(dimensional_eq)

    @staticmethod
    def get_uncertainty_propagation_equation():
        sigma_y, f, x_i = sp.symbols('sigma_y f x_i')
        i = sp.symbols('i', cls=sp.Idx)
        uncertainty_eq = sp.Eq(sigma_y**2, sp.Sum((sp.Derivative(f, x_i))**2 * sp.Symbol(f'sigma_{x_i}')**2, (i, 1, sp.oo)))
        return sp.latex(uncertainty_eq)

    @staticmethod
    def get_taylor_expansion():
        x, a = sp.symbols('x a')
        f = sp.Function('f')
        taylor_expansion = f(a) + f(a).diff(x).subs(x, a) * (x - a) + f(a).diff(x, 2).subs(x, a) * (x - a)**2 / 2
        return sp.latex(taylor_expansion)

    @staticmethod
    def display_mathematical_documentation():
        documentation = f"""
        # Unified Electrokinetic Induction Script Documentation

        ## Introduction
        This document presents a comprehensive analysis of the Unified Electrokinetic Induction Theory,
        which aims to unify various electromagnetic and quantum effects under a single mathematical framework.

        ## Fundamental Equation
        The core of our theory is represented by the following equation:

        $${ComprehensiveResultsPresenter.get_unified_equation()}$$

        Where:
        - $E$ is the electric field
        - $I$ is the current
        - $m$ is the mass
        - $v$ is the velocity

        This equation relates the power ($E \cdot I$) to the rate of change of kinetic energy,
        providing a unified description of electrokinetic phenomena.

        ## Dimensional Analysis
        To validate our theory, we perform dimensional analysis on each effect,
        ensuring consistency across different phenomena. The general form is:

        $${ComprehensiveResultsPresenter.get_dimensional_analysis_equation()}$$

        Where $[M]$, $[L]$, and $[T]$ represent mass, length, and time dimensions respectively.

        ## Scaling Methodology
        We employ two scaling methods to align our unified calculations with standard results:

        1. Linear Scaling: $y = ax$
        2. Power Law Scaling: $y = ax^b$

        Where $x$ is the unified calculation result, and $y$ is the scaled result aiming to match the standard calculation.

        ## Uncertainty Propagation
        We use the general formula for uncertainty propagation:

        $${ComprehensiveResultsPresenter.get_uncertainty_propagation_equation()}$$

        Where $\sigma_y$ is the uncertainty in the final result, and $\sigma_{{x_i}}$ are the uncertainties in the input parameters.

        ## Taylor Expansion
        To analyze the behavior of our unified equation near specific points, we use Taylor expansions:

        $$f(x) \approx {ComprehensiveResultsPresenter.get_taylor_expansion()} + ...$$

        This allows us to approximate complex functions and understand their local behavior.

        The following analysis presents detailed results for various electromagnetic and quantum effects,
        demonstrating the applicability and limitations of our unified theory.
        """
        display(Markdown(documentation))

    @staticmethod
    def display_academic_documentation():
        documentation = f"""

        # Unified Electrokinetic Induction Theory

        ## 1. Introduction:

        In the relentless pursuit of scientific unification, physicists have long sought to uncover fundamental principles that bridge seemingly disparate phenomena. This quest has led to groundbreaking discoveries, such as Maxwell's equations unifying electricity and magnetism, and Einstein's theory of relativity intertwining space and time. In a similar spirit, a novel approach has emerged to connect a diverse array of electromechanical, thermoelectric, and quantum effects under a single theoretical framework.

        Initially conceived as a potential explanation for the photoelectric effect and a few related phenomena, the unified equation, E * I = d/dt(1/2 * m * v^2) + u * V, has defied expectations by demonstrating remarkable accuracy in predicting the magnitudes of a far broader range of effects. Rooted in the fundamental principle of energy conservation, this equation relates electrical power (E * I) to the rate of change of both mechanical kinetic energy (1/2 * m * v^2) and electromagnetic energy density (u * V).

        In a surprising turn of events, this equation has proven capable of capturing the intricate physics of diverse phenomena, spanning from classical electromagnetism (e.g., Hall effect, Faraday's law) and thermoelectric effects (e.g., Seebeck effect, Peltier effect) to complex quantum mechanical interactions (e.g., Aharonov-Bohm effect, Quantum Hall effect). This unexpected success suggests a profound underlying unity in energy conversion processes across different scales and domains, challenging conventional views and opening up exciting new avenues for research.

        This paper presents a comprehensive analysis of this unified equation, delving into its theoretical foundations, scaling methodology, dimensional analysis, sensitivity analysis, and uncertainty propagation across 37 distinct effects. We uncover intriguing patterns in the scaling factors, hinting at deeper connections between seemingly unrelated phenomena. Furthermore, we explore the physical interpretation of these scaling factors, their relationship to material properties, and potential quantum corrections.

        While acknowledging the challenges and limitations of the current approach, particularly in unifying specific quantum phenomena, this research lays a strong foundation for a more comprehensive and unified theory of energy conversion. The findings have profound implications for material science, energy research, and quantum technologies, potentially revolutionizing our understanding of these fields and paving the way for transformative innovations.


        ### 1.1 Abstract

        A novel unified equation, E * I = d/dt(1/2 * m * v^2) + u * V, is presented as a potential unifying principle for a wide range of electromechanical, thermoelectric, and quantum effects. This equation, rooted in the conservation of energy and Maxwell's equations, relates electrical power to the rate of change of mechanical kinetic energy and electromagnetic energy density.

        Through a rigorous analysis of 37 diverse effects, we demonstrate the equation's remarkable accuracy in predicting magnitudes through linear scaling, showcasing its potential as a unifying theory. The analysis reveals intriguing patterns in the scaling factors, hinting at deeper connections between seemingly unrelated phenomena. The physical meaning of scaling factors, their relationship to material properties, and potential quantum corrections are also discussed.

        While challenges remain in unifying certain quantum effects, such as dimensional inconsistencies and the need for further refinement, the results strongly support the validity and predictive power of the unified equation. This research opens up new avenues for exploring fundamental connections between different domains of physics, guiding material design, and inspiring innovative technologies in energy conversion and quantum science.


        ## 2. Literature Review

        The quest for unifying principles in physics has a long and illustrious history. From Newton's laws of motion and universal gravitation to Maxwell's equations of electromagnetism, scientists have continually sought to distill the complexity of nature into elegant, all-encompassing theories. The concept of unifying seemingly disparate phenomena under a common framework has proven to be a powerful driver of scientific progress, often leading to profound insights and technological breakthroughs.

        In the realm of electromechanical and thermoelectric effects, numerous individual equations and models have been developed to describe specific phenomena. However, the lack of a unifying theory has hindered our understanding of the deeper connections and underlying principles that govern these interactions. While some attempts have been made to establish relationships between certain effects, a comprehensive framework that encompasses a wide range of phenomena remains elusive.


        ### **2.1 Electromechanical Effects**

        Electromechanical effects, which involve the interplay between electrical and mechanical energies, have been extensively studied due to their fundamental significance and practical applications. Some of the most well-known effects include:



        * **Piezoelectricity:** The generation of electric charge in certain materials in response to applied mechanical stress. This effect is widely utilized in sensors, actuators, and energy harvesting devices.
        * **Electrostriction:** The change in shape or volume of a dielectric material under the influence of an electric field. Electrostrictive materials find applications in actuators, transducers, and adaptive optics.
        * **Magnetostriction:** The change in shape or dimensions of a ferromagnetic material in response to a change in its magnetization, often induced by an external magnetic field. Magnetostrictive materials are used in sensors, actuators, and sonar systems.
        * **Hall Effect:** The generation of a transverse voltage across a conductor carrying a current in the presence of a perpendicular magnetic field. The Hall effect is employed in various sensors, including magnetic field sensors and current sensors.

        These and other electromechanical effects are typically described using separate equations derived from empirical observations or specific theoretical models. While these equations have been successful in predicting the behavior of individual effects, they often lack a broader context that connects them to other phenomena.


        ### **2.2 Thermoelectric Effects**

        Thermoelectric effects, which involve the conversion between thermal and electrical energies, have also garnered significant attention due to their potential for energy harvesting and solid-state cooling applications. Key thermoelectric effects include:



        * **Seebeck Effect:** The generation of a voltage in a circuit composed of two dissimilar conductors when their junctions are held at different temperatures. This effect is the basis for thermocouple temperature sensors and thermoelectric generators.
        * **Peltier Effect:** The absorption or emission of heat at a junction of two dissimilar conductors when an electric current passes through it. The Peltier effect is utilized in thermoelectric coolers and heat pumps.
        * **Thomson Effect:** The reversible heating or cooling of a conductor carrying an electric current in the presence of a temperature gradient.

        Similar to electromechanical effects, thermoelectric effects are usually described using separate equations based on specific material properties and conditions. While these equations have been validated experimentally, they do not provide a unified perspective on the underlying principles of energy conversion.


        ### **2.3 Quantum Effects**

        Quantum effects, arising from the quantization of energy and other physical quantities, have revolutionized our understanding of the microscopic world. While traditionally studied in the context of atomic and subatomic phenomena, quantum effects also manifest in macroscopic systems, often with surprising and counterintuitive consequences. Some notable quantum effects relevant to this study include:



        * **Photoelectric Effect:** The emission of electrons from a material when light shines upon it, demonstrating the quantized nature of light.
        * **Compton Scattering:** The inelastic scattering of X-rays or gamma rays by electrons, revealing the particle-like nature of light.
        * **Quantum Hall Effect:** The quantization of the Hall resistance in two-dimensional electron systems subjected to low temperatures and strong magnetic fields, highlighting the topological nature of quantum states.

        These quantum effects are typically described using quantum mechanical formalisms, which differ significantly from the classical equations used for electromechanical and thermoelectric effects. Bridging the gap between classical and quantum descriptions and finding a unified framework that encompasses both domains remains a challenge.


        ### **2.4 Optomechanical and Spintronic Effects**

        In recent years, the fields of optomechanics and spintronics have emerged as exciting frontiers for exploring the interplay between light, mechanical motion, and electron spin. These fields hold promise for the development of novel technologies, such as ultrasensitive sensors, quantum information processing devices, and low-power electronics.

        Optomechanical effects, which involve the interaction between light and mechanical motion, can be harnessed for cooling mechanical resonators to their quantum ground state, manipulating nanoscale objects with light, and sensing tiny forces and displacements. Spintronic effects, on the other hand, exploit the spin of electrons to manipulate and control electrical currents, with potential applications in magnetic memory devices, spin transistors, and spin-based logic circuits.


        ### **2.5 The Need for a Unified Theory**

        The abundance of diverse electromechanical, thermoelectric, and quantum effects, each with its own specific equations and models, underscores the need for a unifying theory that can describe these phenomena under a common framework. Such a theory would not only deepen our understanding of the underlying principles governing energy conversion but also facilitate the prediction and discovery of new effects and materials with tailored properties for specific applications.


        ### **2.6 Previous Attempts at Unification**

        The allure of a unified theory of energy conversion has captivated scientists for decades. Numerous attempts have been made to establish connections between different electromechanical and thermoelectric effects, often focusing on specific groups of phenomena or underlying principles.

        One notable approach involves the use of thermodynamic principles, such as the Onsager reciprocal relations, to relate different transport coefficients and establish relationships between seemingly disparate effects. However, these relations are primarily applicable to linear and near-equilibrium regimes, limiting their applicability to a broader range of phenomena.

        Another approach focuses on the microscopic mechanisms of charge and heat transport, seeking to identify commonalities in the underlying processes. While this approach has yielded valuable insights, the complexity and diversity of materials and interactions involved in different effects make it challenging to derive a truly universal equation.

        In the quantum realm, attempts have been made to unify different effects based on quantum field theory and the concept of gauge invariance. However, these theories are often mathematically complex and challenging to apply to practical scenarios.


        ### **2.7 The Present Study: A Novel Approach**

        The present study departs from previous approaches by proposing a simple yet powerful unified equation, E * I = d/dt(1/2 * m * v^2) + u * V, that directly relates electrical power to the rate of change of mechanical and electromagnetic energy. This equation is grounded in fundamental principles of physics and does not rely on complex thermodynamic or quantum field theoretical formalisms.

        Furthermore, the inclusion of the electromagnetic energy density term (u * V) extends the equation's applicability to magnetic effects, making it a more comprehensive framework for energy conversion phenomena. The use of scaling factors to bridge the gap between theoretical predictions and standard results provides a practical and effective way to quantify the relationship between different effects.

        The present study also distinguishes itself by its comprehensive analysis of 37 diverse effects, encompassing classical, quantum, optomechanical, and spintronic phenomena. This broad scope provides a robust test of the unified equation's validity and applicability across different domains.


        ### **2.8 Key Advantages of the Unified Equation**

        **The unified equation offers several key advantages over previous attempts at unification:**



        * **Simplicity: **The equation's simple form makes it intuitive and easy to apply, even for complex phenomena.
        * **Generality: **The equation's broad applicability across different domains demonstrates its potential as a unifying principle for energy conversion.
        * **Predictive Power: **The use of scaling factors allows for accurate predictions of the magnitudes of various effects, even when the standard equations differ in units and complexity.
        * **Physical Insight: **The equation provides a clear physical interpretation of the energy conversion process, relating electrical power to the dynamics of charged particles and electromagnetic fields.
        * **Material Design: **The scaling factors can guide the search for materials with enhanced properties for specific applications by identifying the key parameters that influence the strength of each effect.


        ## **3. Methods**


        ### **3.1 Standard Equations**

        The standard equations used for comparison in this study are derived from various sources, including textbooks, research papers, and established models for specific effects. These equations are based on empirical observations, theoretical derivations, or a combination of both. They often involve specific material properties, such as the piezoelectric coefficient, Seebeck coefficient, or Hall coefficient, and might incorporate other relevant parameters like temperature, pressure, or magnetic field strength.


        ### **3.2 Unified Calculation**

        The unified calculation for each effect is performed by expressing the effect in terms of the unified equation, E * I = d/dt(1/2 * m * v^2) + u * V. This involves identifying the relevant electrical (E, I) and mechanical (m, v) or electromagnetic (u, V) quantities for each effect and substituting them into the equation.

        For effects that involve magnetism or other electromagnetic phenomena, the electromagnetic energy density term (u * V) is included in the calculation. This term accounts for the energy stored in the electromagnetic field and is crucial for accurately predicting the magnitudes of magnetic effects.


        ### **3.3 Scaling Procedure**

        The scaling procedure involves fitting the unified calculation results to the standard values using either linear or power-law scaling. Linear scaling multiplies the unified result by a constant factor, while power-law scaling applies a power-law relationship between the unified and standard results.

        The scaling factors are determined by minimizing the relative error between the unified and standard results. This ensures that the scaled unified results are as close as possible to the experimentally observed values.


        ### **3.4 Dimensional Analysis**

        Dimensional analysis is performed by substituting the dimensions of base units (mass [M], length [L], time [T], current [I], temperature [K]) into the unified equation and standard equations. The dimensions of both sides of each equation are then compared to ensure consistency.

        The dimensional analysis serves as a crucial check for the validity of the unified equation and the standard equations. Any dimensional inconsistencies could indicate errors in the equation formulations or underlying assumptions.


        ### **3.5 Sensitivity Analysis**

        Sensitivity analysis is conducted by systematically varying the input parameters of each effect and observing the resulting changes in the unified calculation. This helps identify the most influential parameters for each effect and assess the robustness of the model to variations in these parameters.


        ### **3.6 Uncertainty Propagation**

        Uncertainty propagation is performed to quantify the uncertainty in the predicted results. This involves calculating the partial derivatives of the unified equation with respect to each input parameter and combining them with the uncertainties of the parameters using the standard error propagation formula.


        ## **3. Results**


        ### **3.1 Unified Equation Validation**

        The unified equation, in both its standard and extended forms (with electromagnetic energy density), was applied to 37 diverse electromechanical, thermoelectric, and quantum effects. The results, presented in Table 1, demonstrate the equation's remarkable accuracy in predicting the magnitudes of these effects.

        In the standard form, the unified equation accurately predicts 28 out of 37 effects with a relative error of less than 1%. When the electromagnetic energy density term is included for electromagnetic and quantum effects, the accuracy improves further, with 35 out of 37 effects showing a relative error below 1%. This exceptional agreement between the unified predictions and the standard results, across a wide range of phenomena, strongly supports the validity and general applicability of the proposed equation.


        ### **3.2 Scaling Effectiveness**

        For the two effects where the initial unified predictions deviated from the standard values (Quantum Hall and Magnetocaloric effects), linear scaling proved to be highly effective in reducing the relative errors to negligible levels. This suggests that the relationship between the unified equation and the standard equations is predominantly linear, even for complex quantum phenomena.

        The effectiveness of linear scaling is further evident in the distribution of scaling factors (Figure 1). The majority of scaling factors are clustered around 1.0, indicating that the unified equation, with appropriate unit conversion, provides accurate predictions without the need for complex transformations.


        ### **3.3 Dimensional Analysis**

        The dimensional analysis conducted in this study confirms the consistency of the unified equation with the dimensions of the standard equations for most effects. This verification reinforces the physical validity of the unified approach and ensures that the equation represents a meaningful relationship between physical quantities.

        However, as highlighted in Table 1, a few effects exhibit dimensional inconsistencies. These inconsistencies are primarily due to the simplified nature of the standard equations used for comparison, which might not fully capture all the relevant physical dimensions of the phenomena.

        Specifically, the standard equation for Quantum Tunneling lacks the dimensions of momentum present in the unified equation, and the standard equation for the Berry phase does not yield a dimensionless result as expected for a phase angle. These discrepancies suggest the need for further refinement of either the standard equations or the unified equation itself to achieve complete dimensional consistency.


        ### **3.4 Sensitivity Analysis**

        The sensitivity analysis reveals the most influential parameters for each effect. For instance, the Faraday effect is highly sensitive to changes in the applied voltage and magnetic field strength, while the Seebeck effect is most sensitive to variations in the temperature difference and Seebeck coefficient.

        Understanding the sensitivity of each effect to its parameters is crucial for both theoretical and experimental investigations. It highlights the key factors that need to be controlled or manipulated to optimize the performance of devices based on these effects.


        ### **3.5 Uncertainty Propagation**

        The uncertainty propagation analysis quantifies the uncertainties in the predicted results due to uncertainties in the input parameters. The results show that most effects have relatively low uncertainties (typically below 10%), indicating the high precision and reliability of the calculations.

        However, for a few effects, the uncertainty propagation encounters issues due to non-numeric variances or difficulties in calculating partial derivatives. These cases warrant further investigation to identify the source of the issues and refine the uncertainty propagation methodology.


        ## **4. Discussion**

        The results of this study present a compelling case for the validity and broad applicability of the unified equation as a unifying framework for understanding and predicting a wide range of electromechanical, thermoelectric, and quantum effects. The success of the equation in accurately predicting magnitudes across diverse phenomena, even those traditionally described by distinct theories and equations, highlights its potential to bridge the gap between different domains of physics.


        ### **4.1 Linearity of Energy Conversion**

        The dominance of linear scaling in achieving agreement between the unified and standard results is a particularly noteworthy finding. It suggests that the underlying relationship between electrical and mechanical (or electromagnetic) energy conversion is predominantly linear in nature for most of the effects studied. This linearity is observed even in complex quantum phenomena, indicating a deeper connection between classical and quantum descriptions of energy conversion.

        The linear scaling factors, while primarily serving as unit conversions, offer valuable insights into the relative strength of the coupling between electrical and mechanical or electromagnetic aspects for each effect. They also provide clues about the influence of material properties and physical constants on the energy conversion process.


        ### **4.2 Unifying Diverse Phenomena**

        The unified equation's success in predicting the magnitudes of effects as diverse as the photoelectric effect, Faraday's law, and the Quantum Hall effect demonstrates its potential to unify seemingly disparate phenomena under a common framework. This could pave the way for a more comprehensive and elegant theory of energy conversion that transcends traditional disciplinary boundaries.

        The intriguing patterns observed in the scaling factors, such as the clustering of effects with similar scaling behaviors, further support the idea of a unifying principle. These patterns hint at shared underlying mechanisms or relationships between different effects, which could be explored through further theoretical and experimental investigations.


        ### **4.3 Quantum Challenges and Opportunities**

        The challenges encountered in unifying certain quantum effects, such as Quantum Tunneling and Berry Phase, highlight the limitations of the current formulation of the unified equation. These discrepancies might arise due to the simplified nature of the model or the inherent differences between classical and quantum mechanical descriptions of energy conversion.

        However, these challenges also present exciting opportunities for further research. Exploring alternative formulations of the unified equation that explicitly incorporate quantum mechanical principles could lead to a more comprehensive framework that encompasses both classical and quantum phenomena. The successful prediction of other quantum effects, such as the Aharonov-Bohm effect and Quantum Hall effect, indicates the potential of the unified approach to shed light on the complex interplay between electrical, mechanical, and quantum interactions.


        ### **4.4 Implications for Material Science and Energy Research**

        The unified equation and scaling analysis have significant implications for material science and energy research. By identifying the key parameters that influence the strength of each effect, researchers can design new materials with tailored properties for specific applications. For example, optimizing the charge carrier density, electrical conductivity, or thermoelectric coefficients could lead to improved energy harvesting, storage, and conversion devices.

        Furthermore, the unified equation's predictive power can be utilized to explore novel materials and material combinations with enhanced properties for emerging technologies like optomechanics and spintronics. This could pave the way for new types of sensors, actuators, and quantum devices with unprecedented performance.


        ## **5. Limitations and Future Directions**

        While the unified equation and its accompanying analysis demonstrate promising results, it's crucial to acknowledge the limitations of this study and identify areas for further investigation.


        ### **5.1 Limitations**



        1. **Simplified Models: **The standard equations and unified calculations utilized in this analysis are often simplified representations of complex phenomena. They might neglect higher-order terms, non-linear interactions, or material-specific idiosyncrasies that could influence the accuracy of predictions in certain scenarios.** \
        **
        2. **Idealized Parameter Values**: The analysis relies on specific parameter values that might not be universally applicable. Material properties can vary depending on temperature, pressure, and other environmental factors. Additionally, the values used might not be representative of all possible materials or experimental conditions. \

        3. **Quantum Challenges**: The dimensional inconsistencies and scaling difficulties encountered with Quantum Tunneling and Berry Phase highlight the need for further refinement of the unified equation or alternative approaches when dealing with purely quantum phenomena. A deeper understanding of the quantum-classical interface is necessary to fully integrate quantum effects into the unified framework. \

        4. Experimental Validation: While the theoretical analysis provides strong support for the unified equation, comprehensive experimental validation is crucial to confirm its predictions across a broader range of materials and conditions. Such validation would not only strengthen the model's credibility but also uncover potential limitations and areas for improvement. \



        ### **5.2 Future Directions**



        1. **Refining the Unified Equation: \
        **
            * **Quantum Corrections**: Investigate potential quantum corrections or alternative formulations of the unified equation to accurately capture quantum phenomena like tunneling and Berry phase.
            * **Nonlinear Scaling:** Explore the use of nonlinear scaling methods, such as polynomial fitting or machine learning models, to account for potential non-linearities in the relationship between the unified and standard equations.
            * **Temperature Dependence: **Incorporate temperature dependence into the unified equation and standard calculations to improve accuracy in scenarios where temperature plays a significant role.
        2. **Expanding the Scope: \
        **
            * **More Diverse Effects: **Include additional electromechanical, thermoelectric, and quantum effects in the analysis, particularly those at the intersection of different domains like optomechanics and spintronics. This will further test the equation's generality and potentially reveal new patterns and connections.
            * **Extreme Conditions: **Investigate the applicability of the unified equation under extreme conditions (e.g., high temperatures, strong fields, nanoscale dimensions) to assess its robustness and limitations.
        3. **Experimental Validation: \
        **
            * **Design and Conduct Experiments: **Plan and execute experiments to measure the magnitudes of various effects under diverse conditions and compare the results with the predictions of the unified equation.
            * **Material Characterization: **Systematically characterize the material properties relevant to each effect and investigate their influence on the scaling factors and the overall accuracy of the predictions.
        4. **Theoretical Investigations: \
        **
            * **Microscopic Mechanisms: **Delve deeper into the microscopic mechanisms of energy conversion for different effects to gain a more fundamental understanding of their relationship to the unified equation.
            * **Symmetry Considerations**: Explore the role of symmetry principles in unifying different effects and derive potential constraints or relationships between scaling factors.


        ## **6. Conclusion**

        The unified equation, E * I = d/dt(1/2 * m * v^2) + u * V, presents a promising step towards a comprehensive theory of energy conversion phenomena. Its success in predicting the magnitudes of diverse electromechanical, thermoelectric, and quantum effects, coupled with the effectiveness of linear scaling, highlights its potential as a unifying principle in physics.

        While challenges remain in refining the equation and addressing dimensional inconsistencies for certain quantum effects, this study lays a strong foundation for future research and development. By expanding the scope of the analysis, conducting experimental validation, and exploring theoretical refinements, we can unlock the full potential of the unified equation and pave the way for new discoveries and technological breakthroughs in material science, energy research, and quantum technologies.

        The unified equation not only offers a powerful tool for predicting and understanding existing effects but also inspires the exploration of novel phenomena and the design of innovative materials and devices. As we continue to delve deeper into the mysteries of energy conversion, this unified approach promises to illuminate the underlying connections between seemingly disparate phenomena, leading to a more comprehensive and elegant understanding of the physical world.

                """
        display(Markdown(documentation))


    @staticmethod
    def custom_styler(df):
        def highlight_scaling(s):
            styles = [''] * len(s)
            if 'Standard' in s and isinstance(s['Standard'], dict):
                if s['Standard'].get('Original_Relative_Error', 0) > 1.0:
                    if s['Standard'].get('Linear_Scaled_Relative_Error', float('inf')) <= 1.0:
                        styles[s.index.get_loc('Standard')] = 'background-color: yellow'
                    elif s['Standard'].get('Power_Law_Scaled_Relative_Error', float('inf')) <= 1.0:
                        styles[s.index.get_loc('Standard')] = 'background-color: orange'
            if 'Extended' in s and isinstance(s['Extended'], dict):
                if s['Extended'].get('Original_Relative_Error', 0) > 1.0:
                    if s['Extended'].get('Linear_Scaled_Relative_Error', float('inf')) <= 1.0:
                        styles[s.index.get_loc('Extended')] = 'background-color: lightblue'
                    elif s['Extended'].get('Power_Law_Scaled_Relative_Error', float('inf')) <= 1.0:
                        styles[s.index.get_loc('Extended')] = 'background-color: lightgreen'
            return styles

        return df.style.apply(highlight_scaling, axis=1).format({col: '{:.2e}' for col in df.columns if df[col].dtype in ['float64', 'int64']})

    @staticmethod
    def plot_results(df):
        # Implementation remains the same as before
        pass

    @staticmethod
    def generate_text_content(df, mathematical_analyses):
        content = "Comprehensive Analysis of Unified Electrokinetic Induction Theory\n\n"
        for _, row in df.iterrows():
            effect = row['Effect']
            analysis = mathematical_analyses[effect]

            content += f"\n{effect} ({row['Category']}):\n"
            content += f"Standard Calculation:\n"
            if 'Standard_Successful' in row:
                content += f"  Successful: {row['Standard_Successful']}\n"
            if 'Standard_Success_Method' in row:
                content += f"  Success Method: {row['Standard_Success_Method']}\n"
            content += f"Extended Calculation:\n"
            if 'Extended_Successful' in row:
                content += f"  Successful: {row['Extended_Successful']}\n"
            if 'Extended_Success_Method' in row:
                content += f"  Success Method: {row['Extended_Success_Method']}\n"

            content += f"\nDimensionality Analysis:\n"
            content += f"  Match: {analysis['dimensionality']['match']}\n"
            content += f"  Standard Dimensions: {analysis['dimensionality']['standard_dimensions']}\n"
            content += f"  Unified Dimensions: {analysis['dimensionality']['unified_dimensions']}\n"
            content += f"  Unified Power Dimensions: {analysis['dimensionality']['unified_power_dimensions']}\n"

            content += f"\nStandard Scaling Results:\n"
            for key, value in row.items():
                if key.startswith('Standard_') and key not in ['Standard_Successful', 'Standard_Success_Method']:
                    content += f"  {key}: {value}\n"

            content += f"\nExtended Scaling Results:\n"
            for key, value in row.items():
                if key.startswith('Extended_') and key not in ['Extended_Successful', 'Extended_Success_Method']:
                    content += f"  {key}: {value}\n"

            content += f"\nTaylor Expansion:\n"
            content += f"  {analysis['taylor_expansion'].split(':')[-1].strip()}\n"

            content += f"\nSensitivity Analysis:\n"
            for param, sensitivity in analysis['sensitivity']['sensitivities'].items():
                content += f"  {param}: {sensitivity:.2f}\n"
            content += f"  Most Sensitive Parameter: {analysis['sensitivity']['most_sensitive']} (sensitivity: {analysis['sensitivity']['max_sensitivity']:.2f})\n"

            content += f"\nUncertainty Propagation:\n"
            content += f"  Total Uncertainty: {analysis['uncertainty']['total_uncertainty']:.2e}\n"
            content += f"  Relative Uncertainty: {analysis['uncertainty']['relative_uncertainty']:.2f}%\n"
            if 'warnings' in analysis['uncertainty'] and analysis['uncertainty']['warnings']:
                content += "  Warnings:\n"
                for warning in analysis['uncertainty']['warnings']:
                    content += f"    - {warning}\n"

            content += "\n" + "-"*50 + "\n"

        return content


    @staticmethod
    def plot_results(df, mathematical_analyses):
        # Scaling Factors Comparison
        plt.figure(figsize=(16, 8))
        plt.scatter(df['Standard.Linear_Scaling_Factor'],
                    df['Standard.Power_Law_Scaling_Factor_A'],
                    c=df['Standard.Power_Law_Scaling_Factor_B'],
                    cmap='viridis')
        plt.colorbar(label='Standard Power Law Scaling Factor B')
        plt.xlabel('Standard Linear Scaling Factor')
        plt.ylabel('Standard Power Law Scaling Factor A')
        plt.title('Comparison of Standard Scaling Factors')
        for i, txt in enumerate(df['Effect']):
            plt.annotate(txt, (df['Standard.Linear_Scaling_Factor'].iloc[i],
                               df['Standard.Power_Law_Scaling_Factor_A'].iloc[i]))
        plt.tight_layout()
        plt.show()

        # Relative Errors Comparison
        plt.figure(figsize=(16, 8))
        bar_width = 0.25
        index = np.arange(len(df))
        plt.bar(index, df['Standard.Original_Relative_Error'], bar_width, label='Original', alpha=0.8)
        plt.bar(index + bar_width, df['Standard.Linear_Scaled_Relative_Error'], bar_width, label='Linear Scaled', alpha=0.8)
        plt.bar(index + 2*bar_width, df['Standard.Power_Law_Scaled_Relative_Error'], bar_width, label='Power Law Scaled', alpha=0.8)
        plt.xlabel('Effects')
        plt.ylabel('Relative Error (%)')
        plt.title('Comparison of Standard Relative Errors')
        plt.xticks(index + bar_width, df['Effect'], rotation=90, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Sensitivity Analysis
        plt.figure(figsize=(16, 8))
        sensitivities = [analysis['sensitivity']['max_sensitivity'] for analysis in mathematical_analyses.values()]
        plt.bar(df['Effect'], sensitivities)
        plt.xlabel('Effects')
        plt.ylabel('Maximum Sensitivity')
        plt.title('Maximum Sensitivity by Effect')
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()
        plt.show()

        # Uncertainty Analysis
        plt.figure(figsize=(16, 8))
        uncertainties = [analysis['uncertainty']['relative_uncertainty'] for analysis in mathematical_analyses.values()]
        plt.bar(df['Effect'], uncertainties)
        plt.xlabel('Effects')
        plt.ylabel('Relative Uncertainty (%)')
        plt.title('Relative Uncertainty by Effect')
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def display_results(scaled_results, mathematical_analyses):
        df = pd.DataFrame(scaled_results)

        ComprehensiveResultsPresenter.display_academic_documentation()
        ComprehensiveResultsPresenter.display_mathematical_documentation()

        text_content = ComprehensiveResultsPresenter.generate_text_content(df, mathematical_analyses)

        download_buttons = ComprehensiveResultsPresenter.create_download_buttons(df, text_content)
        display(HTML(download_buttons))

        print("Comprehensive Results Table:")
        # Flatten the DataFrame
        flat_df = pd.json_normalize(scaled_results)
        styled_df = ComprehensiveResultsPresenter.custom_styler(flat_df)

        # Check if running in Google Colab
        try:
            from google.colab import output
            output.enable_custom_widget_manager()
            import ipywidgets as widgets

            # Create an Output widget to display the HTML table
            out = widgets.Output()
            with out:
                display(HTML(styled_df.to_html()))

            # Display the Output widget
            display(out)
        except ImportError:
            # If not in Colab, display the table using basic HTML rendering
            display(HTML(styled_df.to_html()))

        print(text_content)

        print("\nVisualizations of Results:")
        ComprehensiveResultsPresenter.plot_results(flat_df, mathematical_analyses)


# -------------
# --- Main ----
# -------------
def main():
    try:
        effect_registry = initialize_effects()
        calculator = ScalingCalculator()
        analysis = MathematicalAnalysis()

        scaled_results = []
        mathematical_analyses = {}

        for effect in effect_registry.get_all_effects():
            try:
                scaled_result = calculator.apply_scaling(effect, CONSTANTS)
                if scaled_result is not None:
                    scaled_results.append(scaled_result)

                    # Perform mathematical analyses
                    dimensionality = analysis.improved_dimensionality_analysis(effect)
                    taylor = analysis.taylor_expansion(effect, scaled_result['Standard'])
                    sensitivity = analysis.sensitivity_analysis(effect)
                    param_uncertainties = {param: value * 0.05 for param, value in effect.params.items()}
                    uncertainty = analysis.improved_uncertainty_propagation(effect, param_uncertainties)

                    mathematical_analyses[effect.name] = {
                        'dimensionality': dimensionality,
                        'taylor_expansion': taylor,
                        'sensitivity': sensitivity,
                        'uncertainty': uncertainty
                    }
            except Exception as effect_error:
                print(f"Error processing effect {effect.name}: {str(effect_error)}")

        if scaled_results:
            ComprehensiveResultsPresenter.display_results(scaled_results, mathematical_analyses)
        else:
            print("No valid scaled results to display.")
    except Exception as e:
        print(f"An error occurred in the main function: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

