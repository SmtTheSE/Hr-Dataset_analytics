import pandas as pd

excel_path = "HR dataset.xlsx"
xls = pd.ExcelFile(excel_path)
print(xls.sheet_names)
