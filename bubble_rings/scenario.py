"""Reference parameters.

Every value here is illustrative.  The gas volume, ring radius and
circulation of a diver-blown bubble ring have not, to the authors'
knowledge, been measured systematically; the reports state which
results depend on them and which do not.  All quantities are SI: lengths
in meters, times in seconds, pressures in pascals, volumes in cubic
meters, circulation in m^2 s^-1.  The vertical coordinate z points up and
depth is d = -z.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["Params", "PRM"]


@dataclass(frozen=True)
class Params:
    """Physical and numerical constants shared by all scripts."""

    # sea water at about 25 C and the atmosphere
    rho: float = 1025.0          # liquid density (kg m^-3)
    g: float = 9.81              # gravity (m s^-2)
    gamma: float = 0.073         # air-sea water surface tension (N m^-1)
    nu: float = 1.0e-6           # kinematic viscosity (m^2 s^-1), Re only
    p_atm: float = 101325.0      # surface pressure (Pa)
    kappa_gas: float = 2.0e-5    # thermal diffusivity of air (m^2 s^-1)

    # the ring at release
    d0: float = 10.0             # release depth (m)
    R0: float = 0.05             # ring radius (m)
    V0: float = 10.0e-6          # gas volume at release depth (m^3)
    Gamma: float = 0.06          # circulation about the core (m^2 s^-1)
    poly: float = 1.0            # polytropic exponent, 1 = isothermal
    d_end: float = 0.5           # depth at which integration stops (m)

    # filament discretization
    n_nodes: int = 256           # Lagrangian nodes per ring
    dt: float = 1.0e-3           # RK4 step (s)

    def p_inf(self, d):
        """Hydrostatic pressure at depth d (Pa)."""
        return self.p_atm + self.rho * self.g * np.asarray(d, float)

    @property
    def p0(self):
        """Ambient pressure at the release depth (Pa)."""
        return float(self.p_inf(self.d0))

    @property
    def Pi(self):
        """Release pressure number rho g d0 / p_atm."""
        return self.rho * self.g * self.d0 / self.p_atm

    @property
    def a0(self):
        """Core radius at release from V0 = 2 pi^2 R0 a0^2 (m)."""
        return float(np.sqrt(self.V0 / (2.0 * np.pi ** 2 * self.R0)))

    @property
    def eps0(self):
        """Core slenderness a0/R0 at release."""
        return self.a0 / self.R0

    @property
    def Re(self):
        """Circulation Reynolds number Gamma/nu."""
        return self.Gamma / self.nu


PRM = Params()
