import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
from source import *

# Suppress warnings for cleaner output in Streamlit
warnings.filterwarnings('ignore')

st.set_page_config(page_title="QuLab: Lab 53: Bond Rating Prediction", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 53: Bond Rating Prediction")
st.divider()

# Initialize Session State
def initialize_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    if 'n_issuers' not in st.session_state:
        st.session_state.n_issuers = 1000 
    if 'corp_data' not in st.session_state:
        st.session_state.corp_data = None
    if 'financial_features' not in st.session_state:
        st.session_state.financial_features = None
    if 'latent_quality' not in st.session_state:
        st.session_state.latent_quality = None
    if 'nlp_features' not in st.session_state:
        st.session_state.nlp_features = None
    if 'all_features' not in st.session_state:
        st.session_state.all_features = None
    if 'model_full' not in st.session_state:
        st.session_state.model_full = None
    if 'model_fin_only' not in st.session_state:
        st.session_state.model_fin_only = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'test_idx' not in st.session_state:
        st.session_state.test_idx = None
    if 'pred_full_numeric' not in st.session_state:
        st.session_state.pred_full_numeric = None
    if 'pred_fin_numeric' not in st.session_state:
        st.session_state.pred_fin_numeric = None
    if 'mismatches_df' not in st.session_state:
        st.session_state.mismatches_df = None
    if 'explainer_full' not in st.session_state:
        st.session_state.explainer_full = None
    if 'nlp_helped' not in st.session_state:
        st.session_state.nlp_helped = None
    if 'nlp_hurt' not in st.session_state:
        st.session_state.nlp_hurt = None
    if 'downgrade_watchlist' not in st.session_state:
        st.session_state.downgrade_watchlist = None

initialize_session_state()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", [
    "Home",
    "1. Data Acquisition & Feature Engineering",
    "2. Model Training & Evaluation",
    "3. Rating Mismatch Analysis",
    "4. Explainable AI: Rating Rationales",
    "5. NLP Contribution & Downgrade Watchlist"
])
st.session_state.page = page

# --- PAGE: HOME ---
if st.session_state.page == "Home":
    st.header("Bond Rating Prediction with ML and NLP: A Credit Analyst's Workflow")
    st.markdown(f"### Persona: Sri Krishnamurthy, CFA")
    st.markdown(f"A seasoned Credit Analyst at \"Alpha Asset Management,\" a fixed income investment firm.")
    st.markdown("### Scenario")
    st.markdown(f"Sri's firm manages a significant portfolio of corporate bonds. Traditional credit analysis, involving meticulous review of financial statements and earnings call transcripts, is highly time-consuming. Sri needs a data-driven approach to rapidly screen hundreds of corporate bonds for creditworthiness, identify potential rating changes (upgrades or downgrades) ahead of rating agencies, and provide a competitive edge in portfolio management. This application outlines a real-world workflow to integrate quantitative financial data with qualitative management commentary using machine learning.")
    st.markdown("---")

    st.header("Application Workflow Overview")
    st.markdown(f"""
    This application guides you through a comprehensive workflow for corporate credit rating prediction, mimicking the process a CFA Charterholder like Sri Krishnamurthy would follow.
    """)
    st.markdown(f"**Steps:**")
    st.markdown(f"1.  **Data Acquisition & Feature Engineering:** Generate synthetic corporate financial data and enhance it with NLP-derived sentiment features.")
    st.markdown(f"2.  **Model Training & Evaluation:** Train a multimodal XGBoost model and a financials-only baseline, comparing their performance, especially focusing on 'within-1-notch accuracy'.")
    st.markdown(f"3.  **Rating Mismatch Analysis:** Identify potential downgrade risks and upgrade candidates by comparing model-implied ratings against actual agency ratings.")
    st.markdown(f"4.  **Explainable AI: Rating Rationales:** Utilize SHAP to understand the drivers behind individual rating predictions, distinguishing between financial and NLP feature contributions.")
    st.markdown(f"5.  **NLP Contribution & Downgrade Watchlist:** Quantify the incremental value of NLP sentiment features and generate a prioritized watchlist of high-risk issuers.")

# --- PAGE 1: DATA ACQUISITION ---
elif st.session_state.page == "1. Data Acquisition & Feature Engineering":
    st.header("1. Setting the Stage: Data Acquisition & Initial Credit Assessment")
    st.markdown(f"**Story**: As a CFA Charterholder, Sri understands that the foundation of any credit assessment is robust financial data. Before diving into sophisticated modeling, he needs a representative dataset of corporate financials, reflecting various aspects of creditworthiness. For this simulation, he'll generate a synthetic dataset, ensuring the data's underlying structure mirrors real-world correlations between financial health and credit ratings. This initial step provides the quantitative basis for his analysis.")
    st.markdown(f"**Real-World Relevance**: Generating or acquiring a clean, comprehensive dataset is the first critical step for any data-driven analyst. Understanding the input features (financial ratios) is paramount, as they directly influence the model's ability to assess a company's financial risk and capacity to meet its obligations.")
    st.markdown(f"**Quantitative Logic**: The generation process uses a latent `quality` factor. This `quality` factor is a hidden variable that drives both the financial ratios and the ultimate credit rating. This ensures that the simulated financial ratios are naturally correlated with the credit rating, mimicking real-world data where stronger financial metrics generally lead to higher credit ratings.")

    st.markdown(r"For example, `interest_coverage` is positively correlated with `quality`, meaning higher quality companies tend to have better interest coverage:")
    st.markdown(r"$$ \text{{interest\_coverage}} = \text{{clip}}(2 + \text{{quality}} \times 3 + \text{{noise}}, 0.5, 30) $$")
    st.markdown(r"where $\text{interest\_coverage}$ is a measure of a company's ability to pay interest expenses, $\text{quality}$ is a latent credit quality factor, $\text{noise}$ is random perturbation, and $\text{clip}$ limits the values to a reasonable range.")

    st.markdown(r"Conversely, `debt_to_ebitda` is negatively correlated, meaning higher quality companies tend to have lower debt:")
    st.markdown(r"$$ \text{{debt\_to\_ebitda}} = \text{{clip}}(4 - \text{{quality}} \times 1.5 + \text{{noise}}, 0.5, 15) $$")
    st.markdown(r"where $\text{debt\_to\_ebitda}$ is a leverage ratio, $\text{quality}$ is the latent credit quality factor, $\text{noise}$ is random perturbation, and $\text{clip}$ limits the values to a reasonable range.")

    st.markdown(f"The final `rating_score` is also derived from `quality` with added noise, then binned to create the ordinal credit ratings.")
    
    n_issuers = st.slider("Number of Issuers", min_value=100, max_value=5000, value=st.session_state.n_issuers, step=100)
    st.session_state.n_issuers = n_issuers
    
    if st.button("Generate Corporate Data"):
        st.session_state.corp_data, st.session_state.financial_features, st.session_state.latent_quality = generate_corporate_data(st.session_state.n_issuers)
        st.success("Corporate data generated successfully.")

    if st.session_state.corp_data is not None:
        st.dataframe(st.session_state.corp_data.head())
        st.markdown(f"**Explanation of Execution**: The output shows a snapshot of the generated corporate data, including `issuer_id`, `sector`, various financial ratios, and the assigned numerical credit rating (`rating`). The distribution of credit ratings indicates a realistic spread across different credit tiers, which is crucial for training a robust prediction model. This initial dataset forms the quantitative foundation for Sri's analysis.")

    st.header("Incorporating Qualitative Insights: NLP Sentiment Features")
    st.markdown(f"**Story**: Sri knows that financial ratios are often lagging indicators, reflecting past performance. To gain a competitive edge and identify early warning signs of credit deterioration or improvement, he needs forward-looking insights. He decides to simulate NLP-derived sentiment scores from earnings call transcripts, such as management tone, guidance sentiment, and risk language frequency. These qualitative signals can often capture management's outlook and potential operational shifts before they manifest in financial statements.")
    st.markdown(f"**Real-World Relevance**: Multimodal feature engineering—combining structured numerical data with unstructured text-derived features—is a hallmark of advanced credit analysis. Sentiment analysis of earnings calls provides a crucial 'temporal lead,' often signaling credit quality trajectory 1-2 quarters before financial ratios reflect the change. This proactive insight is invaluable for portfolio managers making timely investment or risk management decisions.")
    st.markdown(f"**Mathematical Formulation: Multimodal Feature Fusion**")
    st.markdown(r"The feature vector for each issuer $i$ is: $x_i = [ X_{{i,1}}, ..., X_{{i,8}}, S_{{i,1}}, S_{{i,2}}, S_{{i,3}} ]$")
    st.markdown(r"where $X_{{i,j}}$ are the financial ratios and $S_{{i,k}}$ are the NLP sentiment features.")
    st.markdown(r"-   **Financial ratios** ($X_{{i,1}}, ..., X_{{i,8}}$) capture the *current* financial position: leverage (`debt/EBITDA`), coverage (`interest coverage`), profitability (`margins`), liquidity (`current ratio`), and size (`total assets`).")
    st.markdown(r"-   **NLP sentiment** ($S_{{i,1}}, S_{{i,2}}, S_{{i,3}}$) captures the *trajectory* and management confidence: a CEO who speaks optimistically about future cash flows signals improving credit quality; one who hedges with risk language signals deterioration.")
    st.markdown(f"The NLP signal is leading, not lagging. Financial ratios reflect last quarter's results. Earnings call sentiment reflects management's view of next quarter. When sentiment deteriorates before ratios do, it is an early warning of a potential downgrade—giving credit analysts a temporal edge over purely ratio-based analysis.")
    
    if st.button("Add NLP Sentiment Features"):
        if st.session_state.corp_data is not None:
            st.session_state.corp_data, st.session_state.nlp_features = add_sentiment_features(st.session_state.corp_data, st.session_state.latent_quality, st.session_state.n_issuers)
            # Update all features list
            st.session_state.all_features = st.session_state.financial_features + st.session_state.nlp_features
            st.success("NLP sentiment features added.")
        else:
            st.error("Please generate corporate data first.")
            
    if st.session_state.corp_data is not None and st.session_state.all_features is not None:
        st.dataframe(st.session_state.corp_data.head())
        st.markdown(f"**Explanation of Execution**: The `corp_data` DataFrame now includes three new NLP-derived features: `mgmt_sentiment`, `guidance_sentiment`, and `risk_language_score`. These features are designed to mimic real-world sentiment analysis, where positive values generally indicate better credit quality and negative/higher risk scores indicate potential deterioration. By fusing these qualitative signals with the financial ratios, Sri can build a more comprehensive and forward-looking credit risk model.")

# --- PAGE 2: MODEL TRAINING ---
elif st.session_state.page == "2. Model Training & Evaluation":
    st.header("2. Building Our Predictive Engine: Ordinal Credit Rating Model")
    st.markdown(f"**Story**: With all the necessary features (financial ratios and NLP sentiment) prepared, Sri is ready to train his predictive model. He understands that credit ratings are inherently ordinal (AAA > AA > A, etc.), meaning there's a natural order to the classes, and misclassifying AAA as CCC is far worse than misclassifying it as AA. Therefore, he chooses an XGBoost classifier with an objective suitable for multi-class classification, but he will also pay close attention to 'within-1-notch accuracy,' a more relevant metric for ordinal predictions. He will train two models: one using only financial features and another using the full multimodal feature set, to quantify the value of NLP.")
    st.markdown(f"**Real-World Relevance**: Implementing an ordinal classification model is crucial for financial applications where target variables have a natural order. Evaluating model performance not just by exact accuracy but by within-1-notch accuracy provides a more nuanced and practically useful measure of a model's utility in a credit risk context. The comparison between financials-only and multimodal models directly answers the question of whether adding NLP brings incremental value.")

    st.markdown(f"**Quantitative Logic: Within-1-Notch Accuracy**")
    st.markdown(r"Given predicted numerical rating $\hat{{y}}$ and actual numerical rating $y$, the 'within-1-notch accuracy' is defined as the proportion of predictions where the absolute difference between the predicted and actual rating is less than or equal to 1:")
    st.markdown(r"$$ \text{{Within-1-Notch Accuracy}} = \frac{{1}}{{N}} \sum_{{i=1}}^{{N}} \mathbb{{I}}(|\hat{{y}}_i - y_i| \le 1) $$")
    st.markdown(r"where $N$ is the number of samples, $\hat{y}_i$ is the predicted numerical rating for issuer $i$, $y_i$ is the actual numerical rating for issuer $i$, and $\mathbb{{I}}(\cdot)$ is the indicator function which is 1 if the condition is true and 0 otherwise. This metric is more relevant for credit ratings than exact accuracy because predicting an 'A' when the actual is 'AA' (1-notch error) is far less consequential than predicting 'CCC' when the actual is 'AA' (5-notch error).")

    if st.button("Train and Evaluate Models"):
        if st.session_state.corp_data is not None and st.session_state.all_features is not None:
            with st.spinner("Training models..."):
                model_full, model_fin_only, X_test, y_test, test_idx, pred_full_numeric, pred_fin_numeric = train_and_evaluate_models(
                    st.session_state.corp_data,
                    st.session_state.financial_features,
                    st.session_state.nlp_features,
                    st.session_state.all_features,
                    RATING_MAP
                )
                
                # Store results in session state
                st.session_state.model_full = model_full
                st.session_state.model_fin_only = model_fin_only
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.test_idx = test_idx
                st.session_state.pred_full_numeric = pred_full_numeric
                st.session_state.pred_fin_numeric = pred_fin_numeric
                
            st.pyplot(plt.gcf())
            st.success("Models trained and evaluated.")
        else:
            st.error("Data not ready. Please complete Step 1 first.")
            
    if st.session_state.model_full is not None:
        st.markdown(f"**Explanation of Execution**: The performance metrics clearly show the exact accuracy and, more importantly, the 'within 1-notch accuracy' for both models. The 'NLP contribution' quantifies the improvement gained by including sentiment features. A higher within-1-notch accuracy is crucial for Sri, as it indicates the model is robust enough to provide actionable insights without frequently making severe misclassifications. The confusion matrix visually confirms how well the multimodal model aligns predictions with actual ratings, with most predictions concentrated on or near the diagonal, demonstrating its ability to distinguish between different credit tiers.")

# --- PAGE 3: MISMATCH ANALYSIS ---
elif st.session_state.page == "3. Rating Mismatch Analysis":
    st.header("3. Identifying Opportunities & Risks: Implied vs. Actual Rating Mismatches")
    st.markdown(f"**Story**: A model's prediction, or 'implied rating,' is not an end in itself. Sri's main objective is to compare these implied ratings against the actual agency ratings to identify discrepancies. These mismatches are critical signals: a model-implied rating significantly below the actual rating suggests a potential 'downgrade risk,' while a higher implied rating points to an 'upgrade candidate.' He needs a structured way to classify these issuers to prioritize his deep-dive analysis.")
    st.markdown(f"**Real-World Relevance**: This mismatch analysis is the core deliverable for a fixed income PM. It transforms raw model predictions into actionable intelligence, enabling proactive risk management (e.g., reducing exposure to downgrade candidates) or identifying alpha opportunities (e.g., increasing exposure to upgrade candidates before spread tightening). It complements, rather than replaces, traditional credit analysis.")

    st.markdown(f"**Quantitative Logic: Rating Mismatch Analysis**")
    st.markdown(r"For each issuer, the `notch_diff` is calculated as the numerical implied rating minus the numerical actual rating:")
    st.markdown(r"$$ \text{{notch\_diff}} = \text{{implied\_numeric\_rating}} - \text{{actual\_numeric\_rating}} $$")
    st.markdown(r"where $\text{implied\_numeric\_rating}$ is the model's predicted credit rating on a numerical scale, and $\text{actual\_numeric\_rating}$ is the official agency credit rating on the same numerical scale.")
    st.markdown(f"Based on `notch_diff`, issuers are categorized:")
    st.markdown(f"-   **DOWNGRADE RISK**: $\text{{notch\_diff}} \le -2$ (model predicts at least 2 notches lower)")
    st.markdown(f"-   **UPGRADE CANDIDATE**: $\text{{notch\_diff}} \ge 2$ (model predicts at least 2 notches higher)")
    st.markdown(f"-   **WATCH (1 notch)**: $|\text{{notch\_diff}}| = 1$")
    st.markdown(f"-   **ALIGNED**: $\text{{notch\_diff}} = 0$")

    if st.button("Perform Mismatch Analysis"):
        if st.session_state.model_full is not None:
            mismatches_df = mismatch_analysis(
                st.session_state.model_full,
                st.session_state.X_test,
                st.session_state.y_test,
                st.session_state.corp_data,
                st.session_state.test_idx,
                RATING_MAP,
                RATING_NUM_TO_STR
            )
            st.session_state.mismatches_df = mismatches_df
            st.pyplot(plt.gcf())
        else:
            st.error("Models not trained. Please complete Step 2 first.")
            
    if st.session_state.mismatches_df is not None:
        st.subheader("Practitioner Warning")
        st.markdown(f"Implied ratings are analytical tools, not official ratings. Presenting model-implied ratings as equivalent to agency ratings (Moody's, S&P, Fitch) would violate CFA Standard V(B) on communication with clients. The implied rating must be clearly labeled as 'model-generated, for internal analytical use only.' Sri uses it as one input among many—alongside the agency rating, his own fundamental analysis, market-implied ratings from CDS spreads, and peer comparison.")
        st.markdown(f"**Explanation of Execution**: The mismatch analysis output provides a summary of how many issuers fall into each category (Aligned, Watch, Downgrade Risk, Upgrade Candidate). The specific lists of top downgrade risks and upgrade candidates immediately highlight where Sri's attention should be focused for deeper research. The histogram of `notch_diff` visually represents the distribution of prediction errors. A skewed distribution towards negative notches would indicate a bias towards predicting lower ratings or that the model is effectively identifying potential downgrade risks. This information is invaluable for Sri to prioritize his workload and alert portfolio managers to actionable insights.")

# --- PAGE 4: EXPLAINABLE AI ---
elif st.session_state.page == "4. Explainable AI: Rating Rationales":
    st.header("4. Unpacking the 'Why': Explainable AI for Rating Rationales")
    st.markdown(f"**Story**: For a CFA Charterholder like Sri, simply knowing *what* the model predicts isn't enough; he needs to understand *why*. Regulatory scrutiny and internal decision-making demand transparency. He will use SHAP (SHapley Additive exPlanations) values to dissect individual predictions. This will allow him to see which specific financial ratios and NLP sentiment features most strongly contributed to an issuer's implied rating, providing clear, interpretable rationales for stakeholders.")
    st.markdown(f"**Real-World Relevance**: Explainable AI (XAI) is critical in finance. SHAP values allow Sri to convert complex model outputs into understandable feature contributions, fostering trust and enabling him to justify investment decisions or risk assessments. By distinguishing between financial and NLP drivers, he can provide a richer narrative, emphasizing the leading indicators identified by sentiment.")

    st.markdown(f"**Quantitative Logic: SHAP Values**")
    st.markdown(f"SHAP values explain the prediction of an instance by computing the contribution of each feature to the prediction. For a tree-based model like XGBoost, SHAP values are calculated efficiently using `shap.TreeExplainer`. The SHAP value for a feature represents the average marginal contribution of that feature value across all possible coalitions of features. A positive SHAP value for a feature means it pushes the prediction higher, while a negative value pushes it lower.")

    if st.session_state.mismatches_df is None:
        st.warning("Please perform Mismatch Analysis (Step 3) first to identify issuers for explanation.")
    else:
        categories = sorted(st.session_state.mismatches_df['mismatch_category'].unique())
        selected_category = st.selectbox("Select Mismatch Category", categories)
        
        issuers_in_cat = st.session_state.mismatches_df[st.session_state.mismatches_df['mismatch_category'] == selected_category]['issuer_id'].unique()
        selected_issuer_id = st.selectbox("Select Issuer ID", issuers_in_cat)
        
        if st.button(f"Explain {selected_issuer_id}'s Rating"):
            # Initialize explainer if not already done
            if st.session_state.explainer_full is None:
                st.session_state.explainer_full = shap.TreeExplainer(st.session_state.model_full)
            
            # Retrieve data for the selected issuer
            # We need the features for this specific issuer from X_test
            # We can find the row index in X_test corresponding to the selected issuer
            # We can map back via corp_data and test_idx
            
            row_idx = st.session_state.corp_data[st.session_state.corp_data['issuer_id'] == selected_issuer_id].index[0]
            
            # Check if this row_idx is in test_idx (it should be if selected from mismatches_df which comes from test set)
            if row_idx in st.session_state.test_idx:
                # Find location in X_test
                # X_test is a DataFrame/array subset. We need to match the index.
                # Assuming X_test carries the original index if it's a dataframe, or we re-slice corp_data
                
                # Safer way: Re-extract features for this specific issuer
                issuer_features = st.session_state.corp_data.loc[[row_idx], st.session_state.all_features]
                actual_numeric = st.session_state.corp_data.loc[row_idx, 'rating']
                
                # Predict again to get implied numeric
                implied_numeric = st.session_state.model_full.predict(issuer_features)[0]
                
                explain_rating(
                    st.session_state.model_full,
                    st.session_state.explainer_full,
                    issuer_features,
                    st.session_state.all_features,
                    implied_numeric,
                    actual_numeric,
                    selected_issuer_id,
                    RATING_MAP,
                    st.session_state.nlp_features
                )
                
                st.pyplot(plt.gcf())
                plt.close('all')
                
                st.markdown(f"**Explanation of Execution**: For the selected issuer, the model's rationale is displayed. This would include specific financial metrics (e.g., high `debt_to_ebitda` or low `interest_coverage`) and NLP sentiment signals (e.g., negative `mgmt_sentiment` or high `risk_language_score`) that are driving the implied rating. The SHAP waterfall plot offers a powerful visualization, showing how each feature pushes the prediction from the base value to the final output. This level of transparency is essential for Sri to validate the model's logic, present findings to colleagues, and support deeper human analysis.")
            else:
                st.error("Selected issuer not found in test set.")

# --- PAGE 5: NLP CONTRIBUTION & WATCHLIST ---
elif st.session_state.page == "5. NLP Contribution & Downgrade Watchlist":
    st.header("5. Quantifying NLP's Edge & Prioritizing Action: Watchlist & Contribution Analysis")
    st.markdown(f"**Story**: Sri is convinced NLP sentiment adds value, but he needs to quantify this benefit empirically for his firm's stakeholders. By comparing how many predictions were 'corrected' by NLP (i.e., became accurate only with NLP features) versus 'worsened,' he can demonstrate its net contribution. Finally, he'll synthesize all insights to create a prioritized 'Downgrade Watchlist' – a concrete, actionable deliverable for portfolio managers to review, complete with relevant financial and sentiment data for high-risk issuers.")
    st.markdown(f"**Real-World Relevance**: Quantifying the incremental value of advanced features like NLP sentiment is crucial for justifying model complexity and data acquisition costs. The Downgrade Watchlist is a direct output for risk management and alpha generation, enabling portfolio managers to focus their attention on the most critical cases and potentially adjust holdings or hedging strategies. This represents the final, actionable step in Sri's workflow.")

    if st.button("Perform NLP Contribution Analysis"):
        if st.session_state.pred_full_numeric is not None and st.session_state.y_test is not None:
            # Internal logic to calculate helped/hurt
            diff_full = np.abs(st.session_state.pred_full_numeric - st.session_state.y_test)
            diff_fin = np.abs(st.session_state.pred_fin_numeric - st.session_state.y_test)
            
            nlp_helped = np.sum(diff_full < diff_fin)
            nlp_hurt = np.sum(diff_full > diff_fin)
            
            st.session_state.nlp_helped = nlp_helped
            st.session_state.nlp_hurt = nlp_hurt
            
            st.metric("Predictions Improved by NLP", nlp_helped)
            st.metric("Predictions Worsened by NLP", nlp_hurt)
            st.metric("Net Improvement", nlp_helped - nlp_hurt)
        else:
             st.error("Model predictions not available. Please complete Step 2.")

    if st.button("Build Downgrade Watchlist"):
        if st.session_state.mismatches_df is not None:
            downgrade_watchlist = build_watchlist(st.session_state.mismatches_df, notch_threshold=-2, top_n=10)
            st.session_state.downgrade_watchlist = downgrade_watchlist
            st.dataframe(downgrade_watchlist)
        else:
            st.error("Mismatch analysis not performed. Please complete Step 3.")
            
    if st.session_state.nlp_helped is not None and st.session_state.downgrade_watchlist is not None:
        st.markdown(f"**Explanation of Execution**: The NLP contribution analysis provides a quantitative summary of how much the sentiment features improved the model's predictions, indicating a net benefit in 'corrected' predictions. This data point is crucial for Sri to articulate the value of incorporating unstructured data. The generated 'Downgrade Watchlist' table then presents the most critical issuers, prioritized by their `notch_diff` (how far below the actual rating the model predicted), along with their sector, actual and implied ratings, and key NLP sentiment scores (`mgmt_sentiment`, `risk_language_score`). This direct, actionable list allows Sri and his portfolio managers to immediately identify and investigate companies at the highest risk of a downgrade, enabling timely strategic adjustments to their bond holdings.")