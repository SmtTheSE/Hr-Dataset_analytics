import pandas as pd
import numpy as np

# Load Excel file
excel_path = "./HR dataset.xlsx"
df_staff = pd.read_excel(excel_path, sheet_name=" Staff")
df_satisfaction = pd.read_excel(excel_path, sheet_name=" Overall satisfaction")
df_private = pd.read_excel(excel_path, sheet_name=" Satisfaction private")
df_importance = pd.read_excel(excel_path, sheet_name=" Organization and content of ac")

# Step 1: Create Work Experience Band based on 'Work experience at the institute'
def get_experience_band(years):
    if years <= 2:
        return "Entry level (0–2 yrs)"
    elif years <= 5:
        return "Junior level (3–5 yrs)"
    elif years <= 10:
        return "Mid level (6–10 yrs)"
    elif years <= 20:
        return "Senior level (11–20 yrs)"
    else:
        return "Expert level (21+ yrs)"

df_staff["Work Experience Band"] = df_staff["Work experience at the institute"].apply(get_experience_band)

# Step 2: Impute missing 'Experience in the industry' for Medical Staff
def fill_missing_experience(row):
    if pd.isna(row['Experience in the industry']) and row['Category'] == 'Medical staff':
        band = row['Work Experience Band']
        filtered = df_staff[(df_staff['Category'] == 'Medical staff') &
                            (df_staff['Work Experience Band'] == band) &
                            (~df_staff['Experience in the industry'].isna())]
        mean_val = filtered['Experience in the industry'].mean()
        if not pd.isna(mean_val):
            return int(round(mean_val))  # Ensure whole number
    return row['Experience in the industry']

df_staff['Experience in the industry'] = df_staff.apply(fill_missing_experience, axis=1)

# Step 3: Impute missing Age values
def impute_missing_age(row):
    if pd.isna(row['Age']):
        if row['Category'] == 'Doctors':
            return int(round(18 + 7 + row['Experience in the industry']))  # Whole number
        elif row['Category'] == 'Administration':
            avg_start_age = df_staff.loc[df_staff['Category'] == 'Administration']\
                                      .assign(StartAge=lambda df: df['Age'] - df['Experience in the industry'])\
                                      ['StartAge'].mean()
            return int(round(avg_start_age + row['Experience in the industry']))  # Whole number
    return row['Age']

df_staff['Age'] = df_staff.apply(impute_missing_age, axis=1)

# Ensure Age & Experience columns are integers in final dataset
df_staff['Age'] = df_staff['Age'].round().astype('Int64')
df_staff['Experience in the industry'] = df_staff['Experience in the industry'].round().astype('Int64')

# Step 4: Impute missing Place of residence for Doctors
doctor_mode_city = df_staff[df_staff['Category'] == 'Doctors']['Place of residence'].mode()[0]
df_staff['Place of residence'] = df_staff.apply(
    lambda row: doctor_mode_city if pd.isna(row['Place of residence']) and row['Category'] == 'Doctors' else row['Place of residence'],
    axis=1
)

# Step 5: Mark abnormal satisfaction values
df_satisfaction['Response Type'] = df_satisfaction.iloc[:, 5].apply(lambda x: 'Abnormal' if x in [0, 6] else 'Normal')
df_private['Response Type'] = df_private.iloc[:, 5].apply(lambda x: 'Abnormal' if x in [0, 6] else 'Normal')

# Step 6: Analyze importance vs satisfaction
importance_scores = df_importance.iloc[:, 5]
satisfaction_scores = df_satisfaction.iloc[:, 5]
importance_response_type = importance_scores.apply(lambda x: 'Zero' if x == 0 else ('High' if x >= 4 else 'Mid'))
satisfaction_response_type = satisfaction_scores.apply(lambda x: 'Zero' if x == 0 else ('High' if x == 6 else 'Mid'))
importance_vs_satisfaction = pd.DataFrame({
    'Importance': importance_response_type,
    'Satisfaction': satisfaction_response_type
})
summary = importance_vs_satisfaction.value_counts().unstack().fillna(0)

# Step 7: Build relationships between tables using Employee ID
for df in [df_satisfaction, df_private, df_importance]:
    df.rename(columns={df.columns[0]: 'Employee ID'}, inplace=True)

df_satisfaction = pd.merge(df_satisfaction, df_staff, on='Employee ID', how='left')
df_private = pd.merge(df_private, df_staff, on='Employee ID', how='left')
df_importance = pd.merge(df_importance, df_staff, on='Employee ID', how='left')

# Step 8: Check for missing values
def check_missing(df, name):
    print(f"\nMissing values in {name}:")
    print(df.isnull().sum())

check_missing(df_staff, "Cleaned Staff")
check_missing(df_satisfaction, "Cleaned Satisfaction")
check_missing(df_private, "Cleaned Private")
check_missing(df_importance, "Cleaned Importance")

# Step 9: Descriptive Analytics
print("\n--- Descriptive Insights ---")
print("\nStaff count by category:")
print(df_staff['Category'].value_counts())

print("\nAbnormal vs Normal responses (Overall Satisfaction):")
print(df_satisfaction['Response Type'].value_counts())

print("\nTop 5 most common places of residence:")
print(df_staff['Place of residence'].value_counts().head(5))

print("\nAge distribution (bins of 10 years):")
print(pd.cut(df_staff['Age'], bins=[20, 30, 40, 50, 60, 70, 80]).value_counts())

# Step 10: Diagnostic Analytics
print("\n--- Diagnostic Insights ---")
print("\nAbnormal responses by Job Category:")
print(df_satisfaction[df_satisfaction['Response Type'] == 'Abnormal']['Category'].value_counts())

print("\nAbnormal responses by Age Group:")
print(pd.cut(df_satisfaction[df_satisfaction['Response Type'] == 'Abnormal']['Age'], bins=[20, 30, 40, 50, 60, 70, 80]).value_counts())

print("\nAbnormal responses by Work Experience Band:")
print(df_satisfaction[df_satisfaction['Response Type'] == 'Abnormal']['Work Experience Band'].value_counts())

# Step 11: Deep Insights - Abnormal response ratios
print("\n--- Deep Insight Ratios ---")
print("\nAbnormal Rate by Job Category (%):")
print((df_satisfaction[df_satisfaction['Response Type'] == 'Abnormal']['Category'].value_counts() / df_satisfaction['Category'].value_counts() * 100).round(2))

print("\nAbnormal Rate by Experience Band (%):")
print((df_satisfaction[df_satisfaction['Response Type'] == 'Abnormal']['Work Experience Band'].value_counts() / df_satisfaction['Work Experience Band'].value_counts() * 100).round(2))

# Step 12: Correlation check
print("\n--- Correlation with Abnormal Responses ---")
df_satisfaction['Is Abnormal'] = df_satisfaction['Response Type'].apply(lambda x: 1 if x == 'Abnormal' else 0)
print(df_satisfaction[['Age', 'Experience in the industry', 'Is Abnormal']].corr())

# Step 13: Most problematic questions
print("\n--- Most Problematic Questions (Answer = 0 or 6) ---")
problem_q = df_satisfaction[df_satisfaction['Answer'].isin([0, 6])]
print(problem_q['Question'].value_counts().head(5))

# Step 14: Export cleaned data
output_path = "Cleaned_HR_Dataset.xlsx"
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    df_staff.to_excel(writer, sheet_name='Cleaned Staff', index=False)
    df_satisfaction.to_excel(writer, sheet_name='Cleaned Satisfaction', index=False)
    df_private.to_excel(writer, sheet_name='Cleaned Private', index=False)
    df_importance.to_excel(writer, sheet_name='Cleaned Importance', index=False)

print(f"\n✅ Cleaned Excel file saved to: {output_path}")
