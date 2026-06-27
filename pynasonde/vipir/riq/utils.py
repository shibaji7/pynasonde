"""Utility helpers for processing VIPIR RIQ (raw IQ) records.

This module bundles a few small helpers that are shared across the RIQ
parsers and datatypes:

- `odd` / `even` check integer parity without treating zero as odd.
- `trim_null` and `len_trim_null` handle null-terminated strings that appear
  in VIPIR binary structures.
- `unwrap` keeps phase angles within the ``[-pi, pi]`` interval so downstream
  routines can reason about angular differences coherently.
"""

import math


# Determine if the number is odd
def odd(i):
    """
    Determine if the argument is odd. Zero is considered even.

    Parameters:
        i: int
            Integer to check for oddness.

    Returns:
        True if odd, False if even.
    """
    return i % 2 != 0


# Determine if the number is even
def even(i):
    """
    Determine if the argument is even. Zero is considered even.

    Parameters:
        i: int
            Integer to check for evenness.

    Returns:
        True if even, False if odd.
    """
    return i % 2 == 0


# Function to remove null characters and return a trimmed string
def trim_null(string):
    """
    Remove null characters (ASCII 0) from the string.

    Parameters:
        string: str
            Input string to trim.

    Returns:
        Trimmed string with null characters replaced by spaces.
    """
    return string.replace("\x00", "").strip()


# Function to return the length of the string without null characters
def len_trim_null(string):
    """
    Get the length of the string after stripping null characters.

    Parameters:
        string: str
            Input string to evaluate.

    Returns:
        Length of the string after nulls are stripped.
    """
    return len(trim_null(string))


# Function to unwrap phase to the [-PI, PI] range
def unwrap(phase):
    """
    Unwrap radian phase to the range [-PI, PI].

    Parameters:
        phase: float
            Input phase in radians.

    Returns:
        Unwrapped phase.
    """
    if abs(phase) <= math.pi:
        return phase
    elif phase > math.pi:
        return phase - 2 * math.pi
    elif phase < -math.pi:
        return phase + 2 * math.pi
    else:
        return math.nan  # Error case
