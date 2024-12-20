#!/usr/bin/env python

"""distpersion_relations.py: absorption is calucated from dispersion relations."""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.integrate import quad

from pynasonde.model.absorption.constants import pconst


@dataclass
class AbsorptionProfiles:
    ah: SimpleNamespace = SimpleNamespace(ft=SimpleNamespace(O=[]))


# ===================================================================================
# These are special function dedicated to the Sen-Wyller absorption calculation.
#
# Sen, H. K., and Wyller, A. A. ( 1960), On the Generalization of Appleton-Hartree magnetoionic Formulas
# J. Geophys. Res., 65( 12), 3931- 3950, doi:10.1029/JZ065i012p03931.
#
# ===================================================================================
def C(p, y):

    def gamma_factorial(N):
        n = int(str(N).split(".")[0])
        f = N - n
        if f > 0.0:
            fact = math.factorial(n) * math.gamma(f)
        else:
            fact = math.factorial(n)
        return fact

    func = lambda t: t**p * np.exp(-t) / (t**2 + y**2)
    cy, abserr = quad(func, 0, np.inf)
    return cy / gamma_factorial(p)


def calculate_sw_RL_abs(Bo, Ne, nu, fo=30e6, nu_sw_r=2.5):
    if (
        Ne > 0.0
        and Bo > 0.0
        and nu > 0.0
        and (not np.isnan(Ne))
        and (not np.isnan(Bo))
        and (not np.isnan(nu))
    ):
        k = (2 * np.pi * fo) / pconst["c"]
        w = 2 * np.pi * fo
        nu_sw = nu * nu_sw_r
        wh = pconst["q_e"] * Bo / pconst["m_e"]
        yo, yx = (w + wh) / nu_sw, (w - wh) / nu_sw
        nL = 1 - (
            (Ne * pconst["q_e"] ** 2 / (2 * pconst["m_e"] * w * pconst["eps0"] * nu_sw))
            * np.complex(yo * C(1.5, yo), 2.5 * C(2.5, yo))
        )
        nR = 1 - (
            (Ne * pconst["q_e"] ** 2 / (2 * pconst["m_e"] * w * pconst["eps0"] * nu_sw))
            * np.complex(yx * C(1.5, yx), 2.5 * C(2.5, yx))
        )
        R, L = np.abs(nR.imag * 8.68 * k * 1e3), np.abs(nL.imag * 8.68 * k * 1e3)
    else:
        R, L = np.nan, np.nan
    return R, L


def calculate_sw_OX_abs(Bo, Ne, nu, fo=30e6, nu_sw_r=2.5):
    if (
        Ne > 0.0
        and Bo > 0.0
        and nu > 0.0
        and (not np.isnan(Ne))
        and (not np.isnan(Bo))
        and (not np.isnan(nu))
    ):
        k = (2 * np.pi * fo) / pconst["c"]
        w = 2 * np.pi * fo
        nu_sw = nu * nu_sw_r
        wh = pconst["q_e"] * Bo / pconst["m_e"]
        wo2 = Ne * pconst["q_e"] ** 2 / (pconst["m_e"] * pconst["eps0"])
        yo, yx = (w) / nu_sw, (w) / nu_sw
        y = (w) / nu_sw

        ajb = (wo2 / (w * nu_sw)) * ((y * C(1.5, y)) + 1.0j * (2.5 * C(2.5, y)))
        c = (wo2 / (w * nu_sw)) * yx * C(1.5, yx)
        d = 2.5 * (wo2 / (w * nu_sw)) * C(1.5, yx)
        e = (wo2 / (w * nu_sw)) * yo * C(1.5, yo)
        f = 2.5 * (wo2 / (w * nu_sw)) * C(1.5, yo)

        eI = 1 - ajb
        eII = 0.5 * ((f - d) + (c - e) * 1.0j)
        eIII = ajb - (0.5 * ((c + e) + 1.0j * (d + f)))

        Aa = 2 * eI * (eI + eIII)
        Bb = (eIII * (eI + eII)) + eII**2
        Cc = 2 * eI * eII
        Dd = 2 * eI
        Ee = 2 * eIII

        nO = np.sqrt(Aa / (Dd + Ee))
        nX = np.sqrt((Aa + Bb) / (Dd + Ee))
        O, X = np.abs(nO.imag * 8.68 * k * 1e3), np.abs(nX.imag * 8.68 * k * 1e3)
    else:
        O, X = np.nan, np.nan
    return O, X


# ===================================================================================
# This class is used to estimate O,X,R & L mode absorption height profile.
# ===================================================================================
class CalculateAbsorption(object):
    """
    This class is used to estimate O,X,R & L mode absorption height profile.

    Bo = geomagnetic field
    coll = collision frequency
    Ne = electron density
    fo = operating frequency
    """
