"""
Global constants.
"""

# --------------
# --- LABELS ---
# --------------

# >>> Test Data
#
TIME = "TIME"
DISP = "DISP"
LOAD = "LOAD"
EXTS = "EXTS"
DISP_STRAIN = "DISP_STRAIN"
EXTS_STRAIN = "EXTS_STRAIN"
STRESS = "STRESS"

# >>> Specimen Properties
#
WIDTH = "WIDTH"
THICKNESS = "THICKNESS"
INTERAXIS = "INTERAXIS"
EXTS_LENGTH = "EXTS_LENGTH"
CS_LENGTH = "CS_LENGTH"

# ----------------------------
# --- TEST SETUP CONSTANTS ---
# ----------------------------

MOTOR = "MOTOR"
ENCODER = "ENCO"
EXTENSOMETER = "EXTENSO"

ACCEPTED_REF_DISPLACEMENT_LOAD = [MOTOR, ENCODER, EXTENSOMETER]
ACCEPTED_REF_PARAM_STRAIN = [MOTOR, ENCODER, EXTENSOMETER]

RP02 = "Rp02"
RP05 = "Rp05"
RP1 = "Rp1"
RP2 = "Rp2"

LINEARITY_DEV_METHODS = [RP02, RP05, RP1, RP2]

# ---------------------------
# --- WORKBOOKS CONSTANTS ---
# ---------------------------

# >>> Sheets name
#
RAW = "(1) RAW"
ELAB = "(2) ELAB"

# >>> Data col. name
#
TIME = "TIME"
DISP = "DISP"
LOAD = "LOAD"
EXTS = "EXTS"

# >>> Specimen properties table
#
SPROP_START = "G13"
SPROP_END = "I17"

# >>> Setup table
#
SETUP_START = "I24"
SETUP_END = "I29"

# >>> Cut and Offset
TAIL_P = "I38"
FOOT_OFFSET = "I43"

# >>> Linear Section
BOTTOM_CUTOUT = "G57"
UPPER_CUTOUT = "H57"