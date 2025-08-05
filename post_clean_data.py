import pandas as pd

# Load final cleaned file
staff = pd.read_csv("clean_staff.csv")  # You can repeat for others or automate

# Clean column names
staff.columns = staff.columns.str.strip().str.replace(" ", "_").str.lower()

print(" Step-by-step Post-Cleaning Validation for: Staff\n")

# 1. Check for out-of-range values (e.g., age)
print("\n Age range check (should be between 18 and 100):")
print(staff[(staff['age'] < 18) | (staff['age'] > 100)][['employee_id', 'age']])

# 2. Check for negative values in any numeric columns
print("\n Negative Values Check:")
numeric_cols = staff.select_dtypes(include=['number']).columns
for col in numeric_cols:
    if (staff[col] < 0).any():
        print(f"{col} has negative values:")
        print(staff[staff[col] < 0][['employee_id', col]])

# 3. Check for unexpected values in categorical columns
print("\n Unexpected Category Values:")
expected_floors = [0, 1]
unexpected_floors = staff[~staff['floor'].isin(expected_floors)]
if not unexpected_floors.empty:
    print(" Floor column has unexpected values:")
    print(unexpected_floors[['employee_id', 'floor']])

# 4. Check for duplicates
print(f"\n Duplicate rows: {staff.duplicated().sum()}")

# 5. Zero variance check
print("\n Zero Variance Columns:")
zero_var_cols = [col for col in numeric_cols if staff[col].nunique() == 1]
print("Columns with same value in every row:", zero_var_cols)

# 6. Abnormal strings (long, special chars, etc.)
print("\n Text Columns Abnormality Check (length > 50 or special characters):")
text_cols = staff.select_dtypes(include='object').columns
for col in text_cols:
    long_vals = staff[staff[col].str.len() > 50]
    special_vals = staff[staff[col].str.contains(r'[^a-zA-Z\s]', na=False)]
    if not long_vals.empty:
        print(f" Column `{col}` has unusually long values:")
        print(long_vals[[col]].drop_duplicates().head())
    if not special_vals.empty:
        print(f" Column `{col}` has special characters:")
        print(special_vals[[col]].drop_duplicates().head())
