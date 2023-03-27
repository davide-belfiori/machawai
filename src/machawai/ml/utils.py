"""
Set of utility functions.
"""

# --------------
# --- IMPORT ---
# --------------

import numpy as np

# -----------------
# --- FUNCTIONS ---
# -----------------

def flatten_dictionary(dictionary: dict) -> np.ndarray:
    items = list(map(lambda item: item[1], dictionary.items()))
    return np.hstack(items)