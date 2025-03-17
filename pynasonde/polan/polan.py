import math

from loguru import logger


class Polan(object):
    """
    ** This is an python implementation of the code in https://github.com/space-physics/POLAN

    A generalised POLynomial real-height ANalysis for ionograms.


    This implements a complex algorithm for ionospheric analysis
    based on polynomial fitting of virtual and real height data. It manages
    different modes of analysis, deals with x-ray data, and can perform
    least-squares fitting for Chapman layer peaks.

    Consult the POLAN.TXT documentation for a detailed explanation of
    the parameters and the underlying algorithm. This Python code is a
    direct translation of the original Fortran subroutine and retains
    much of its structure.
    """
    def __init__(self, e: ScaledEntries):
        return