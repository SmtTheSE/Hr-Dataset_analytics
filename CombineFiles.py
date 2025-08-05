import pandas as pd

csv_sheets = {
    "Staff": "final_clean_staff.csv",
    "Overall_Satisfaction": "clean_overall_satisfaction.csv",
    "Satisfaction_Private": "clean_satisfaction_private.csv",
    "Organization_Content": "clean_organization_content.csv"
}

with pd.ExcelWriter("HR_Dataset_Final.xlsx", engine="xlsxwriter") as writer:
    for sheet_name, file_path in csv_sheets.items():
        df = pd.read_csv(file_path)
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(" Combined Excel file created: HR_Dataset_Final.xlsx")
