import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay
import shap
import numpy as np
import streamlit as st

# ---------------------------------------
# Streamlit Dashboard Setup
# ---------------------------------------
st.set_page_config(page_title="HR Analytics Dashboard", layout="wide")
st.title(" HR Satisfaction Analytics")

# ---------------------------------------
# Load cleaned data
# ---------------------------------------
excel_path = "Cleaned_HR_Dataset.xlsx"
df_staff = pd.read_excel(excel_path, sheet_name="Cleaned Staff")
df_satisfaction = pd.read_excel(excel_path, sheet_name="Cleaned Satisfaction")
df_private = pd.read_excel(excel_path, sheet_name="Cleaned Private")
df_importance = pd.read_excel(excel_path, sheet_name="Cleaned Importance")

# ---------------------------------------
# Preprocessing
# ---------------------------------------
df_satisfaction['Is Abnormal'] = df_satisfaction['Response Type'].apply(lambda x: 1 if x == 'Abnormal' else 0)

# ---------------------------------------
# Sidebar filters
# ---------------------------------------
st.sidebar.header("Filters")
selected_category = st.sidebar.selectbox("Select Job Category:", ["All"] + df_staff['Category'].unique().tolist())

# Apply filter to all datasets
if selected_category != "All":
    df_satisfaction_filtered = df_satisfaction[df_satisfaction['Category'] == selected_category]
    df_staff_filtered = df_staff[df_staff['Category'] == selected_category]
    df_private_filtered = df_private[df_private['Category'] == selected_category]
    df_importance_filtered = df_importance[df_importance['Category'] == selected_category]
else:
    df_satisfaction_filtered = df_satisfaction.copy()
    df_staff_filtered = df_staff.copy()
    df_private_filtered = df_private.copy()
    df_importance_filtered = df_importance.copy()

# ---------------------------------------
# Clustering prep
# ---------------------------------------
satisfaction_values = df_satisfaction_filtered.iloc[:, 6:].select_dtypes(include='number').dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(satisfaction_values)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# ---------------------------------------
# Utility function for saving SVG
# ---------------------------------------
def save_svg(fig, filename):
    fig.savefig(filename, format='svg')
    with open(filename, "rb") as f:
        st.download_button(" Download SVG", f, file_name=filename, mime="image/svg+xml")

# ---------------------------------------
# Tabs
# ---------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    " Satisfaction Overview",
    " Clusters",
    " Abnormal Predictor",
    " Descriptive & Diagnostic",
    " Predictive & Prescriptive",
    " Private Satisfaction",
    " Importance vs Satisfaction"
])

# ---------------------------------------
# Tab 1: Satisfaction Overview
# ---------------------------------------
with tab1:
    st.subheader("Satisfaction Response Types")
    fig, ax = plt.subplots()
    df_satisfaction_filtered['Response Type'].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)
    save_svg(fig, "satisfaction_response_types.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: This chart shows how staff responded to the satisfaction survey. 
    Normal responses suggest general satisfaction, while abnormal responses can indicate dissatisfaction or unusual feedback patterns._
    """)

    st.subheader("Abnormal Responses by Age Group")
    age_groups = pd.cut(
        df_satisfaction_filtered[df_satisfaction_filtered['Response Type'] == 'Abnormal']['Age'],
        bins=[20, 30, 40, 50, 60, 70, 80],
        labels=["21–30", "31–40", "41–50", "51–60", "61–70", "71–80"]
    )
    age_group_counts = age_groups.value_counts().reindex(age_groups.cat.categories, fill_value=0)
    fig, ax = plt.subplots()
    age_group_counts.plot(kind="bar", ax=ax)
    st.pyplot(fig)
    save_svg(fig, "abnormal_responses_by_age.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: This chart identifies which age groups have the highest share of abnormal survey responses. 
    HR can use this insight to target engagement strategies for specific age demographics._
    """)

# ---------------------------------------
# Tab 2: Clusters
# ---------------------------------------
with tab2:
    st.subheader("t-SNE Clustering")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    tsne_df = pd.DataFrame(X_tsne, columns=['t-SNE 1', 't-SNE 2'])
    tsne_df['Cluster'] = kmeans_labels
    fig, ax = plt.subplots()
    sns.scatterplot(data=tsne_df, x='t-SNE 1', y='t-SNE 2', hue='Cluster', palette='Set2', ax=ax)
    st.pyplot(fig)
    save_svg(fig, "tsne_clusters.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: This diagram uses machine learning (t‑SNE + KMeans) to group employees with similar satisfaction patterns. 
    Each color represents a cluster of staff with similar survey responses, helping HR detect common satisfaction profiles._
    """)

# ---------------------------------------
# Tab 3: Decision Tree Predictor
# ---------------------------------------
with tab3:
    st.subheader("Decision Tree: Abnormal Response Predictor")
    features = ['Age', 'Experience in the industry']
    X = df_satisfaction_filtered[features]
    y = df_satisfaction_filtered['Is Abnormal']
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_tree(tree, feature_names=features, class_names=['Normal', 'Abnormal'], filled=True, ax=ax)
    st.pyplot(fig)
    save_svg(fig, "decision_tree.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: This decision tree predicts whether a response is abnormal based on age and industry experience. 
    It visually shows how certain thresholds in age or experience relate to dissatisfaction._
    """)

# ---------------------------------------
# Tab 4: Descriptive & Diagnostic
# ---------------------------------------
with tab4:
    st.subheader("Staff Composition")
    fig, ax = plt.subplots()
    df_staff_filtered['Category'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig)
    save_svg(fig, "staff_composition.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: This pie chart shows the proportion of staff in each job category. 
    Understanding staff composition helps in tailoring HR programs for each group._
    """)

    st.markdown("**Top 10 Places of Residence**")
    fig, ax = plt.subplots()
    df_staff_filtered['Place of residence'].value_counts().head(10).plot(kind='barh', ax=ax)
    st.pyplot(fig)
    save_svg(fig, "residence_distribution.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: This chart shows where staff live. 
    Understanding residential distribution helps HR plan commuting support or location‑based policies._
    """)

    st.markdown("**Age Distribution (Decades)**")
    fig, ax = plt.subplots()
    age_bins = pd.cut(df_staff_filtered['Age'], bins=[20, 30, 40, 50, 60, 70, 80])
    age_bins.value_counts().sort_index().plot(kind='bar', ax=ax)
    st.pyplot(fig)
    save_svg(fig, "age_distribution.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: This chart shows staff age distribution by decade, useful for understanding workforce demographics._
    """)

    st.subheader("Diagnostic Analytics")
    fig, ax = plt.subplots()
    df_satisfaction_filtered[df_satisfaction_filtered['Response Type'] == 'Abnormal']['Category'].value_counts().plot(kind='bar', color='purple', ax=ax)
    st.pyplot(fig)
    save_svg(fig, "abnormal_by_category.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: Shows which job categories report the most dissatisfaction (abnormal responses)._
    """)

    fig, ax = plt.subplots()
    abnormal = df_satisfaction_filtered[df_satisfaction_filtered['Answer'].isin([0, 6])]
    abnormal['Question'].value_counts().head(5).plot(kind='barh', color='orange', ax=ax)
    st.pyplot(fig)
    save_svg(fig, "top_problematic_questions.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: Highlights the top survey questions with the most extreme responses, indicating key dissatisfaction drivers._
    """)

    fig, ax = plt.subplots()
    sns.heatmap(df_satisfaction_filtered[['Age', 'Experience in the industry', 'Is Abnormal']].corr(),
                annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    save_svg(fig, "correlation_heatmap.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: This heatmap shows correlations between age, experience, and dissatisfaction._
    """)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='Set2', ax=ax)
    st.pyplot(fig)
    save_svg(fig, "kmeans_pca_clusters.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: Similar to the t‑SNE plot but using PCA for simpler visualization of satisfaction clusters._
    """)

# ---------------------------------------
# Tab 5: Predictive & Prescriptive
# ---------------------------------------
with tab5:
    #  Hiring Patterns Over Time
    st.markdown("### Employee Hiring Patterns Over Years")
    current_year = pd.Timestamp.now().year
    if 'Work experience at the institute' in df_staff_filtered.columns:
        df_staff_filtered['Year Joined'] = current_year - df_staff_filtered['Work experience at the institute']
        hiring_counts = df_staff_filtered.groupby('Year Joined').size().reset_index(name='Hires')
        hiring_counts = hiring_counts[hiring_counts['Year Joined'] > 1990]

        fig, ax = plt.subplots()
        sns.lineplot(data=hiring_counts, x='Year Joined', y='Hires', marker='o', ax=ax)
        st.pyplot(fig)
        save_svg(fig, "hiring_patterns.svg")
        plt.close(fig)
        st.markdown("""
        _Explanation: This time series chart shows how many employees were hired each year. 
        HR can identify growth trends, hiring peaks, or slowdowns over time._
        """)

        #  Linear Regression Forecast
        X_years = hiring_counts['Year Joined'].values.reshape(-1, 1)
        y_hires = hiring_counts['Hires'].values
        model_lr = LinearRegression()
        model_lr.fit(X_years, y_hires)

        future_years = np.arange(hiring_counts['Year Joined'].min(), current_year + 6).reshape(-1, 1)
        predicted_hires = model_lr.predict(future_years)

        fig, ax = plt.subplots()
        ax.plot(hiring_counts['Year Joined'], hiring_counts['Hires'], marker='o', label='Actual Hires')
        ax.plot(future_years.flatten(), predicted_hires, linestyle='--', color='red', label='Predicted Hires')
        ax.legend()
        st.pyplot(fig)
        save_svg(fig, "hiring_forecast.svg")
        plt.close(fig)
        st.markdown("""
        _Explanation: This linear regression model predicts future hiring trends based on past data. 
        The dotted red line shows the expected number of hires in the coming years._
        """)

    # 3️ Logistic Regression for Abnormal Responses
    features = ['Age', 'Experience in the industry', 'Floor', 'Availability of part-time jobs']
    X = df_satisfaction_filtered[features]
    y = df_satisfaction_filtered['Is Abnormal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    st.markdown("**ROC Curve & AUC**")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.legend()
    st.pyplot(fig)
    save_svg(fig, "roc_curve.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: ROC curve measures how well the model can distinguish between satisfied and dissatisfied employees._
    """)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    st.pyplot(fig)
    save_svg(fig, "confusion_matrix.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: Confusion matrix shows prediction accuracy — how many cases were correctly or incorrectly classified._
    """)

    coefs = pd.Series(model.coef_[0], index=features).sort_values(ascending=False)
    st.write("Top features driving abnormal prediction:")
    st.write(coefs)
    st.info("Suggest targeting staff with high 'Floor' or low 'Experience' for engagement programs.")
    st.markdown("""
    _Explanation: Lists the most influential features affecting dissatisfaction likelihood._
    """)

    # 4️⃣ SHAP Analysis
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, show=False)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("""
    _Explanation: SHAP values explain how each feature impacts predictions for each individual._
    """)

    # 5️⃣ Prescriptive: Top Areas for Improvement
    importance_scores = df_importance_filtered['Evaluation of the importance of factors']
    satisfaction_scores = df_importance_filtered['Satisfaction rating']
    gap = importance_scores - satisfaction_scores
    improvement_df = df_importance_filtered.copy()
    improvement_df['Gap'] = gap
    top_improvement_areas = improvement_df.groupby('Question')['Gap'].mean().sort_values(ascending=False).head(5)

    fig, ax = plt.subplots()
    top_improvement_areas.plot(kind='barh', color='crimson', ax=ax)
    st.pyplot(fig)
    save_svg(fig, "top_improvement_areas.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: This chart highlights the top factors where importance is high but satisfaction is low. 
    Improving these areas is likely to produce the greatest increase in employee satisfaction._
    """)

# ---------------------------------------
# Tab 6: Private Satisfaction
# ---------------------------------------
with tab6:
    st.subheader(" Private Satisfaction Breakdown")

    # Ensure both Normal and Abnormal appear even if one is missing
    all_types = ["Normal", "Abnormal"]
    counts = df_private_filtered['Response Type'].value_counts().reindex(all_types, fill_value=0)

    fig, ax = plt.subplots()
    counts.plot(kind='bar', color=['skyblue', 'salmon'], ax=ax)

    # Set a consistent y-axis so very small bars are visible
    max_val = counts.max()
    ax.set_ylim(0, max(max_val * 1.3, 5))  # Minimum height ensures visibility

    # Add value labels on top of bars
    for i, value in enumerate(counts):
        ax.text(i, value + 0.1, str(value), ha='center', va='bottom', fontsize=10)

    ax.set_ylabel("Number of Responses")
    ax.set_xlabel("Response Type")
    ax.set_title("Private Staff Satisfaction Breakdown")

    st.pyplot(fig)
    save_svg(fig, "private_response_type.svg")
    plt.close(fig)

    st.markdown("""
    _Explanation: Shows how private staff responded to satisfaction surveys — all categories 
    are shown even if counts are very small, and exact counts are labeled for clarity._
    """)


    fig, ax = plt.subplots()
    df_private_filtered[df_private_filtered['Response Type'] == 'Abnormal']['Category'].value_counts().plot(kind='bar', color='salmon', ax=ax)
    st.pyplot(fig)
    save_svg(fig, "private_abnormal_by_category.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: Highlights which job categories inside private staff have the most dissatisfaction._
    """)

    fig, ax = plt.subplots()
    abnormal_answers = df_private_filtered[df_private_filtered['Evaluation of the importance of factors'].isin([0, 6])]
    abnormal_answers['Question'].value_counts().head(5).plot(kind='barh', color='orange', ax=ax)
    st.pyplot(fig)
    save_svg(fig, "private_top_abnormal_questions.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: Shows the top questions where private staff gave extreme answers, revealing specific problem areas._
    """)

# ---------------------------------------
# Tab 7: Importance vs Satisfaction
# ---------------------------------------
with tab7:
    st.subheader(" Importance vs Satisfaction")
    importance_scores = df_importance_filtered['Evaluation of the importance of factors']
    satisfaction_scores = df_importance_filtered['Satisfaction rating']
    gap = importance_scores - satisfaction_scores

    fig, ax = plt.subplots()
    sns.histplot(gap, bins=20, kde=True, ax=ax, color='teal')
    st.pyplot(fig)
    save_svg(fig, "importance_satisfaction_gap.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: A histogram showing how much higher importance scores are compared to satisfaction — 
    large gaps point to areas where expectations aren't met._
    """)

    st.markdown("**High Importance, Low Satisfaction**")
    high_gap_df = df_importance_filtered[(importance_scores >= 4) & (satisfaction_scores <= 2)]
    top_questions = high_gap_df['Question'].value_counts().head(5)
    st.write(top_questions)

    fig, ax = plt.subplots()
    top_questions.plot(kind='barh', color='red', ax=ax)
    st.pyplot(fig)
    save_svg(fig, "top_gap_questions.svg")
    plt.close(fig)
    st.markdown("""
    _Explanation: Focuses on the top problem areas where staff rated factors as very important but were highly dissatisfied._
    """)
