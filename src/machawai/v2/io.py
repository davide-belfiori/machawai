import os
import pandas as pd
from openpyxl import load_workbook
from dataclasses import dataclass
from machawai.utils import string2bool

@dataclass
class TensileTestData():

    curve: pd.DataFrame
    width: 'list[float]'
    thickness: 'list[float]'
    bottom_cutout: float
    upper_cutout: float
    interaxis: float
    extensometer_acquired: bool
    extensometer_gage_length: float = None
    filename: str = None

def parse_curve_df(curve_df: pd.DataFrame,
                   comment_col_idx: int = 0,
                   comment_startswith: str = "+"):
    """
        ...
    """
    # 1) Clean header
    if isinstance(curve_df.columns, pd.MultiIndex):
        curve_df.columns = map(lambda col: col[0], curve_df.columns)
    # 2) Drop comment
    comment_idx = curve_df.iloc[:,comment_col_idx].str.startswith(comment_startswith).replace(pd.NA, False).argmax()
    if comment_idx > 0:
        curve_df = curve_df[:comment_idx]
    # 3) Replace blank with NaN
    curve_df = curve_df.replace(r'^\s*$', pd.NA, regex=True)
    # 4) Drop rows with all NaN values
    curve_df = curve_df.dropna(axis=0, how="all")
    # 5) Cast to float
    curve_df = curve_df.astype(float)

    return curve_df

def read_excel(file: str, 
               data_sheet_name: str,
               sprop_sheet_name: str,
               lin_sheet_name: str,
               data_header: 'int | list[int]' = [0,1],
               width_start: str = "J13",
               width_end: str = "L13",
               thickness_start: str = "J14",
               thickness_end: str = "L14",
               interaxis_cell: str = "L18",
               exts_acquired_cell: str = "L26",
               exts_gage_length_cell: str = "L20",
               upper_cutout_cell: str = "L8",
               bottom_cutout_cell: str = "L12"):
    # Read curve data
    curve_df = pd.read_excel(io = file,
                            sheet_name = data_sheet_name,
                            header = data_header)
    curve_df = parse_curve_df(curve_df)
    # Load workbook
    wb = load_workbook(filename=file, data_only=True)
    # Read specimen dimension
    sprop_sheet = wb[sprop_sheet_name]
    width = sprop_sheet[width_start: width_end]
    width = list(map(lambda c: c.value, width[0]))
    thickness = sprop_sheet[thickness_start: thickness_end]
    thickness = list(map(lambda c: c.value, thickness[0]))
    interaxis = sprop_sheet[interaxis_cell].value
    exts_acquired = string2bool(sprop_sheet[exts_acquired_cell].value)
    exts_gage_length = None
    if exts_acquired:
        exts_gage_length = sprop_sheet[exts_gage_length_cell].value
    # Read linear section boundaries
    lin_sheet = wb[lin_sheet_name]
    upper_cutout = lin_sheet[upper_cutout_cell].value
    bottom_cutout = lin_sheet[bottom_cutout_cell].value
    if upper_cutout < bottom_cutout:
        upper_cutout, bottom_cutout = bottom_cutout, upper_cutout

    filename = os.path.basename(file)
    filename = os.path.splitext(filename)[0]

    return TensileTestData(curve=curve_df,
                           width=width,
                           thickness=thickness,
                           upper_cutout=upper_cutout,
                           bottom_cutout=bottom_cutout,
                           interaxis=interaxis,
                           extensometer_acquired=exts_acquired,
                           extensometer_gage_length=exts_gage_length,
                           filename=filename)
