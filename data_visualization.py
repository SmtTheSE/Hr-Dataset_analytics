# hr_data_visuals.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load cleaned data
excel_path = "Cleaned_HR_Dataset.xlsx"
df_staff = pd.read_excel(excel_path, sheet_name="Cleaned Staff")
df_satisfaction = pd.read_excel(excel_path, sheet_name="Cleaned Satisfaction")

# Pie chart: Staff Composition
plt.figure(figsize=(6, 6))
df_staff['Category'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
plt.title('Staff Composition by Category')
plt.ylabel('')
plt.tight_layout()
plt.show()

# Bar: Place of residence
plt.figure(figsize=(10, 5))
df_staff['Place of residence'].value_counts().head(10).plot(kind='barh')
plt.title('Top 10 Places of Residence')
plt.xlabel('Count')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Age Distribution
plt.figure(figsize=(8, 5))
age_bins = pd.cut(df_staff['Age'], bins=[20, 30, 40, 50, 60, 70, 80])
age_bins.value_counts().sort_index().plot(kind='bar')
plt.title('Age Distribution (Decades)')
plt.xlabel('Age Range')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Satisfaction Response Types
plt.figure(figsize=(6, 4))
df_satisfaction['Response Type'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Satisfaction Response Types')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Top 5 abnormal questions
plt.figure(figsize=(8, 5))
abnormal = df_satisfaction[df_satisfaction['Answer'].isin([0, 6])]
abnormal['Question'].value_counts().head(5).plot(kind='barh', color='orange')
plt.title('Top 5 Questions with Abnormal Ratings')
plt.xlabel('Abnormal Count')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Grouped bar: Abnormal by Category
plt.figure(figsize=(6, 4))
df_satisfaction[df_satisfaction['Response Type'] == 'Abnormal']['Category'].value_counts().plot(kind='bar', color='purple')
plt.title('Abnormal Responses by Job Category')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Grouped bar: Abnormal by Age Group
plt.figure(figsize=(8, 5))
age_groups = pd.cut(df_satisfaction[df_satisfaction['Response Type'] == 'Abnormal']['Age'], bins=[20,30,40,50,60,70,80])
age_groups.value_counts().sort_index().plot(kind='bar', color='green')
plt.title('Abnormal Responses by Age Group')
plt.xlabel('Age Range')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Abnormal by Work Experience Band
plt.figure(figsize=(6, 4))
df_satisfaction[df_satisfaction['Response Type'] == 'Abnormal']['Work Experience Band'].value_counts().plot(kind='bar', color='teal')
plt.title('Abnormal Responses by Experience Band')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6, 4))
df_satisfaction['Is Abnormal'] = df_satisfaction['Response Type'].apply(lambda x: 1 if x == 'Abnormal' else 0)
sns.heatmap(df_satisfaction[['Age', 'Experience in the industry', 'Is Abnormal']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Clustering based on satisfaction answers (assume answer columns 6–end are Qs)
satisfaction_values = df_satisfaction.iloc[:, 6:]
satisfaction_values = satisfaction_values.select_dtypes(include='number').dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(satisfaction_values)

# Reduce to 2D for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Plot Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=kmeans_labels, palette='Set2')
plt.title('KMeans Clustering of Satisfaction Responses')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()
