import pandas as pd

file_paths = {
    "Staff": "clean_staff.csv",
    "Overall satisfaction": "clean_overall_satisfaction.csv",
    "Satisfaction private": "clean_satisfaction_private.csv",
    "Organization content": "clean_organization_content.csv"
}


def debug_check(df, sheet_name):
    print(f"\n====  Analyzing Sheet: {sheet_name} ====\n")

    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

    print(" Shape:", df.shape)
    print(" Column Types:\n", df.dtypes)

    print("\n Missing Values:")
    print(df.isnull().sum())

    print("\n Missing Data (%):")
    print((df.isnull().mean() * 100).round(2))

    print(f"\n Duplicate Rows: {df.duplicated().sum()}")

    print("\n Numeric Summary:")
    print(df.describe())

    print("\n Unique Text Values:")
    for col in df.select_dtypes(include='object').columns:
        uniques = df[col].unique()
        print(f"{col} ({len(uniques)} unique): {uniques[:10]}")


for name, path in file_paths.items():
    df = pd.read_csv(path)
    debug_check(df, name)
