import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

def save_svg(fig, filename):
    fig.savefig(filename, format='svg')
    with open(filename, "rb") as f:
        st.download_button("📥 Download SVG", f, file_name=filename, mime="image/svg+xml")

def tab_private_satisfaction(df_private):
    st.subheader("🔐 Private Satisfaction Breakdown")

    fig1, ax1 = plt.subplots()
    df_private['Response Type'].value_counts().plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_title("Private Satisfaction Response Type Count")
    st.pyplot(fig1)
    save_svg(fig1, "private_response_type.svg")
    st.markdown("""
    _Explanation: Shows how private staff responded to satisfaction surveys — 
    a quick view of satisfaction vs dissatisfaction within the private category._
    """)

    fig2, ax2 = plt.subplots()
    df_private[df_private['Response Type'] == 'Abnormal']['Category'].value_counts().plot(kind='bar', color='salmon', ax=ax2)
    ax2.set_title("Private Abnormal Responses by Category")
    st.pyplot(fig2)
    save_svg(fig2, "private_abnormal_by_category.svg")
    st.markdown("""
    _Explanation: Highlights which job categories inside private staff have the most dissatisfaction._
    """)

    fig3, ax3 = plt.subplots()
    abnormal_answers = df_private[df_private['Evaluation of the importance of factors'].isin([0, 6])]
    abnormal_answers['Question'].value_counts().head(5).plot(kind='barh', color='orange', ax=ax3)
    ax3.set_title("Top Abnormal Private Questions")
    st.pyplot(fig3)
    save_svg(fig3, "private_top_abnormal_questions.svg")
    st.markdown("""
    _Explanation: Shows the top questions where private staff gave extreme answers, revealing specific problem areas._
    """)

def tab_importance_gap(df_importance):
    st.subheader("🎯 Importance vs Satisfaction")

    importance_scores = df_importance['Evaluation of the importance of factors']
    satisfaction_scores = df_importance['Satisfaction rating']
    gap = importance_scores - satisfaction_scores

    fig4, ax4 = plt.subplots()
    sns.histplot(gap, bins=20, kde=True, ax=ax4, color='teal')
    ax4.set_title("Gap between Importance and Satisfaction")
    ax4.set_xlabel("Importance - Satisfaction")
    st.pyplot(fig4)
    save_svg(fig4, "importance_satisfaction_gap.svg")
    st.markdown("""
    _Explanation: A histogram showing how much higher importance scores are compared to satisfaction — 
    large gaps point to areas where expectations aren't met._
    """)

    st.markdown("**High Importance, Low Satisfaction**")
    high_gap_df = df_importance[(importance_scores >= 4) & (satisfaction_scores <= 2)]
    top_questions = high_gap_df['Question'].value_counts().head(5)
    st.write(top_questions)

    fig5, ax5 = plt.subplots()
    top_questions.plot(kind='barh', color='red', ax=ax5)
    ax5.set_title("Top Gap Areas to Address")
    st.pyplot(fig5)
    save_svg(fig5, "top_gap_questions.svg")
    st.markdown("""
    _Explanation: Focuses on the top problem areas where staff rated factors as very important but were highly dissatisfied._
    """)

def show_tab6_and_tab7(df_private, df_importance):
    tab6, tab7 = st.tabs(["🔐 Private Satisfaction", "🎯 Importance vs Satisfaction"])
    with tab6:
        tab_private_satisfaction(df_private)
    with tab7:
        tab_importance_gap(df_importance)
