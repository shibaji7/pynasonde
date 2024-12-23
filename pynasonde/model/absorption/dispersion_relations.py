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
    ah: SimpleNamespace = None
    sw: SimpleNamespace = None

    def _initialize_(self):
        self.ah = SimpleNamespace(
            ft=SimpleNamespace(O=None, X=None, L=None, R=None, no=None),
            sn=SimpleNamespace(O=None, X=None, L=None, R=None, no=None),
            av_cc=SimpleNamespace(O=None, X=None, L=None, R=None, no=None),
            av_mb=SimpleNamespace(O=None, X=None, L=None, R=None, no=None),
        )
        self.sw = SimpleNamespace(
            ft=SimpleNamespace(O=None, X=None, L=None, R=None),
        )
        return


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
            * np.complex128((yo * C(1.5, yo)) + (1j * 2.5 * C(2.5, yo)))
        )
        nR = 1 - (
            (Ne * pconst["q_e"] ** 2 / (2 * pconst["m_e"] * w * pconst["eps0"] * nu_sw))
            * np.complex128((yx * C(1.5, yx)) + (1j * 2.5 * C(2.5, yx)))
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

    def __init__(
        self,
        Bo,
        col_freq,
        Ne,
        fo=2e6,
        nu_sw_r=2.5,
    ):
        self.Bo = Bo
        self.col_freq = col_freq
        self.Ne = Ne
        self.fo = fo
        self.nu_sw_r = nu_sw_r
        self.w = 2 * np.pi * fo
        self.k = (2 * np.pi * fo) / pconst["c"]
        self.abs_profiles = AbsorptionProfiles()
        self.abs_profiles._initialize_()
        return

    def estimate_AH(self):
        # =========================================================
        # Using FT collision frequency
        # =========================================================
        X, Z = (self.Ne * pconst["q_e"] ** 2) / (
            pconst["eps0"] * pconst["m_e"] * self.w**2
        ), self.col_freq.nu.ft / self.w
        x, jz = X, Z * 1.0j
        n = np.sqrt(1 - (x / (1 - jz)))
        self.abs_profiles.ah.ft.no = np.abs(8.68 * self.k * 1e3 * n.imag)

        YL, YT = 0, (pconst["q_e"] * self.Bo) / (pconst["m_e"] * self.w)
        nO, nX = (
            np.sqrt(1 - (x / (1 - jz))),
            np.sqrt(
                1
                - (
                    (2 * x * (1 - x - jz))
                    / ((2 * (1 - x - jz) * (1 - jz)) - (2 * YT**2))
                )
            ),
        )
        self.abs_profiles.ah.ft.O, self.abs_profiles.ah.ft.X = (
            np.abs(8.68 * self.k * 1e3 * nO.imag),
            np.abs(8.68 * self.k * 1e3 * nX.imag),
        )

        YL, YT = (pconst["q_e"] * self.Bo) / (pconst["m_e"] * self.w), 0
        nL, nR = np.sqrt(1 - (x / ((1 - jz) + YL))), np.sqrt(1 - (x / ((1 - jz) - YL)))
        self.abs_profiles.ah.ft.R, self.abs_profiles.ah.ft.L = (
            np.abs(8.68 * self.k * 1e3 * nR.imag),
            np.abs(8.68 * self.k * 1e3 * nL.imag),
        )

        # ========================================================
        # Using SN collision frequency  quite_model
        # ========================================================
        Z = self.col_freq.nu.sn.total / self.w
        jz = Z * 1.0j
        n = np.sqrt(1 - (x / (1 - jz)))
        self.abs_profiles.ah.sn.no = np.abs(8.68 * self.k * 1e3 * n.imag)

        YL, YT = 0, (pconst["q_e"] * self.Bo) / (pconst["m_e"] * self.w)
        nO, nX = (
            np.sqrt(1 - (x / (1 - jz))),
            np.sqrt(
                1
                - (
                    (2 * x * (1 - x - jz))
                    / ((2 * (1 - x - jz) * (1 - jz)) - (2 * YT**2))
                )
            ),
        )
        self.abs_profiles.ah.sn.O, self.abs_profiles.ah.sn.X = (
            np.abs(8.68 * self.k * 1e3 * nO.imag),
            np.abs(8.68 * self.k * 1e3 * nX.imag),
        )

        YL, YT = (pconst["q_e"] * self.Bo) / (pconst["m_e"] * self.w), 0
        nL, nR = (
            np.sqrt(1 - (x / ((1 - jz) + YL))),
            np.sqrt(1 - (x / ((1 - jz) - YL))),
        )
        self.abs_profiles.ah.sn.R, self.abs_profiles.ah.sn.L = (
            np.abs(8.68 * self.k * 1e3 * nR.imag),
            np.abs(8.68 * self.k * 1e3 * nL.imag),
        )

        # =========================================================
        # Using AV_CC collision frequency quite_model
        # =========================================================
        Z = self.col_freq.nu.av_cc / self.w
        jz = Z * 1.0j
        n = np.sqrt(1 - (x / (1 - jz)))
        self.abs_profiles.ah.av_cc.no = np.abs(8.68 * self.k * 1e3 * n.imag)

        YL, YT = 0, (pconst["q_e"] * self.Bo) / (pconst["m_e"] * self.w)
        nO, nX = (
            np.sqrt(1 - (x / (1 - jz))),
            np.sqrt(
                1
                - (
                    (2 * x * (1 - x - jz))
                    / ((2 * (1 - x - jz) * (1 - jz)) - (2 * YT**2))
                )
            ),
        )
        self.abs_profiles.ah.av_cc.O, self.abs_profiles.ah.av_cc.X = (
            np.abs(8.68 * self.k * 1e3 * nO.imag),
            np.abs(8.68 * self.k * 1e3 * nX.imag),
        )

        YL, YT = (pconst["q_e"] * self.Bo) / (pconst["m_e"] * self.w), 0
        nL, nR = (
            np.sqrt(1 - (x / ((1 - jz) + YL))),
            np.sqrt(1 - (x / ((1 - jz) - YL))),
        )
        self.abs_profiles.ah.av_cc.R, self.abs_profiles.ah.av_cc.L = (
            np.abs(8.68 * self.k * 1e3 * nR.imag),
            np.abs(8.68 * self.k * 1e3 * nL.imag),
        )

        # =========================================================
        # Using AV_MB collision frequency quite_model
        # =========================================================
        Z = self.col_freq.nu.av_mb / self.w
        jz = Z * 1.0j
        n = np.sqrt(1 - (x / (1 - jz)))
        self.abs_profiles.ah.av_mb.no = np.abs(8.68 * self.k * 1e3 * n.imag)

        YL, YT = 0, (pconst["q_e"] * self.Bo) / (pconst["m_e"] * self.w)
        nO, nX = (
            np.sqrt(1 - (x / (1 - jz))),
            np.sqrt(
                1
                - (
                    (2 * x * (1 - x - jz))
                    / ((2 * (1 - x - jz) * (1 - jz)) - (2 * YT**2))
                )
            ),
        )
        self.abs_profiles.ah.av_mb.O, self.abs_profiles.ah.av_mb.X = (
            np.abs(8.68 * self.k * 1e3 * nO.imag),
            np.abs(8.68 * self.k * 1e3 * nX.imag),
        )

        YL, YT = (pconst["q_e"] * self.Bo) / (pconst["m_e"] * self.w), 0
        nL, nR = (
            np.sqrt(1 - (x / ((1 - jz) + YL))),
            np.sqrt(1 - (x / ((1 - jz) - YL))),
        )
        self.abs_profiles.ah.av_mb.R, self.abs_profiles.ah.av_mb.L = (
            np.abs(8.68 * self.k * 1e3 * nR.imag),
            np.abs(8.68 * self.k * 1e3 * nL.imag),
        )
        return

    def estimate_SW(self):
        I, J, K = self.Bo.shape
        (
            self.abs_profiles.sw.ft.O,
            self.abs_profiles.sw.ft.X,
            self.abs_profiles.sw.ft.L,
            self.abs_profiles.sw.ft.R,
        ) = (
            np.zeros_like(self.Bo) * np.nan,
            np.zeros_like(self.Bo) * np.nan,
            np.zeros_like(self.Bo) * np.nan,
            np.zeros_like(self.Bo) * np.nan,
        )
        # ===================================================
        # Using FT collistion frequency
        # ===================================================
        nu = self.col_freq.nu.ft
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    (
                        self.abs_profiles.sw.ft.O[i, j, k],
                        self.abs_profiles.sw.ft.X[i, j, k],
                    ) = calculate_sw_OX_abs(
                        self.Bo[i, j, k],
                        self.Ne[i, j, k],
                        nu[i, j, k],
                        self.fo,
                        nu_sw_r=self.nu_sw_r,
                    )
                    (
                        self.abs_profiles.sw.ft.R[i, j, k],
                        self.abs_profiles.sw.ft.L[i, j, k],
                    ) = calculate_sw_RL_abs(
                        self.Bo[i, j],
                        self.Ne[i, j],
                        nu[i, j],
                        self.fo,
                        nu_sw_r=self.nu_sw_r,
                    )
        return
