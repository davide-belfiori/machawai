"""
    Global variables and objects.
"""

# ------------------
# --- EXCEPTIONS ---
# ------------------

class MissingTimeError(Exception):

    def __init__(self, message="No Time values provided") -> None:
        self.message = message
        super().__init__(self.message)

class MissingExtsError(Exception):

    def __init__(self, message="No Extensometer values provided") -> None:
        self.message = message
        super().__init__(self.message)

# ---------------
# --- CLASSES ---
# ---------------

class LabelledObject():

    def __init__(self) -> None:
        pass

    def labels(self) -> 'list[str]':
        raise NotImplementedError()
    
    def get_by_label(self, label:str):
        raise NotImplementedError()
    
    def get_by_labels(self, labels:'list[str]'):
        raise NotImplementedError()
