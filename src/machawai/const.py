"""
Global constants.
"""

import json

class Const():

    # ----------------------------
    # --- TEST SETUP CONSTANTS ---
    # ----------------------------

    MOTOR = "MOTOR"
    ENCODER = "ENCO"
    EXTENSO = "EXTENSO"

    ACCEPTED_REF_DISPLACEMENT_LOAD = [MOTOR, ENCODER, EXTENSO]
    ACCEPTED_REF_PARAM_STRAIN = [MOTOR, ENCODER, EXTENSO]

    RP02 = "Rp02"
    RP05 = "Rp05"
    RP1 = "Rp1"
    RP2 = "Rp2"

    LINEARITY_DEV_METHODS = [RP02, RP05, RP1, RP2]

    # ---------------------------
    # --- WORKBOOKS CONSTANTS ---
    # ---------------------------

    # >>> Sheets name
    RAW = "(1) RAW"
    ELAB = "(2) ELAB"

    # >>> Data col. name
    TIME_COL = "TIME"
    DISP_COL = "DISP"
    LOAD_COL = "LOAD"
    EXTS_COL = "EXTS"

    # >>> Specimen properties table
    SPROP_START = "G13"
    SPROP_END = "I17"

    # >>> Setup table
    SETUP_START = "I24"
    SETUP_END = "I29"

    # >>> Cut and Offset
    TAIL_P_CELL = "I38"
    FOOT_OFFSET_CELL = "I43"

    # >>> Linear Section
    BOTTOM_CUTOUT_CELL = "G57"
    UPPER_CUTOUT_CELL = "H57"

    @classmethod
    def read_contants(cls, file: str) -> None:
        """
        Read the constant values from a json config file.

        Arguments
        ---------

        file: str
            Config file path
        """
        try:
            with open(file, "r") as config_file:
                config = json.load(config_file)
        except:
            raise ValueError("Invalid config file path")
        
        cls.MOTOR = config["MOTOR"]
        cls.ENCODER = config["ENCODER"]
        cls.EXTENSO = config["EXTENSO"]

        cls.RP02 = config["RP02"]
        cls.RP05 = config["RP05"]
        cls.RP1 = config["RP1"]
        cls.RP2 = config["RP2"]

        cls.RAW = config["RAW"]
        cls.ELAB = config["ELAB"]

        cls.TIME_COL = config["TIME_COL"]
        cls.DISP_COL = config["DISP_COL"]
        cls.LOAD_COL = config["LOAD_COL"]
        cls.EXTS_COL = config["EXTS_COL"]

        cls.SPROP_START = config["SPROP_START"]
        cls.SPROP_END = config["SPROP_END"]
        
        cls.SETUP_START = config["SETUP_START"]
        cls.SETUP_END = config["SETUP_END"]

        cls.TAIL_P_CELL = config["TAIL_P_CELL"]
        cls.FOOT_OFFSET_CELL = config["FOOT_OFFSET_CELL"]

        cls.BOTTOM_CUTOUT_CELL = config["BOTTOM_CUTOUT_CELL"]
        cls.UPPER_CUTOUT_CELL = config["UPPER_CUTOUT_CELL"]
