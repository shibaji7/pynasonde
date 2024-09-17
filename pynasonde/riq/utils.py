import math

# Constants for PI and TWOPI
PII = math.pi
TWOPII = 2 * math.pi


# Determine if the number is odd
def odd(i):
    """
    Determine if the argument is odd. Zero is considered even.

    Parameters:
    -----------
    i : int
        Integer to check for oddness.

    Returns:
    --------
    bool :
        True if odd, False if even.
    """
    return i % 2 != 0


# Determine if the number is even
def even(i):
    """
    Determine if the argument is even. Zero is considered even.

    Parameters:
    -----------
    i : int
        Integer to check for evenness.

    Returns:
    --------
    bool :
        True if even, False if odd.
    """
    return i % 2 == 0


# Function to remove null characters and return a trimmed string
def trim_null(string):
    """
    Remove null characters (ASCII 0) from the string.

    Parameters:
    -----------
    string : str
        Input string to trim.

    Returns:
    --------
    str :
        Trimmed string with null characters replaced by spaces.
    """
    return string.replace("\x00", "").strip()


# Function to return the length of the string without null characters
def len_trim_null(string):
    """
    Get the length of the string after stripping null characters.

    Parameters:
    -----------
    string : str
        Input string to evaluate.

    Returns:
    --------
    int :
        Length of the string after nulls are stripped.
    """
    return len(trim_null(string))


# Function to unwrap phase to the [-PI, PI] range
def unwrap(phase):
    """
    Unwrap radian phase to the range [-PI, PI].

    Parameters:
    -----------
    phase : float
        Input phase in radians.

    Returns:
    --------
    float :
        Unwrapped phase.
    """
    if abs(phase) <= PII:
        return phase
    elif phase > PII:
        return phase - TWOPII
    elif phase < -PII:
        return phase + TWOPII
    else:
        return 9999.0  # Error case
