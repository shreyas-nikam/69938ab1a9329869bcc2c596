
# Streamlit Application Specification: Bond Rating Prediction with ML and NLP

## 1. Application Overview

**Purpose of the Application:**
The "Bond Rating Prediction with ML and NLP" Streamlit application provides a robust, interactive platform for CFA Charterholders and Investment Professionals (like Sri Krishnamurthy) to perform advanced credit analysis. It demonstrates a real-world workflow for predicting corporate credit ratings by combining structured financial data with unstructured NLP sentiment features from earnings calls. The application facilitates the identification of potential rating changes, quantifies the value of NLP insights, and provides explainable AI rationales for predictions, all while adhering to CFA ethical standards regarding model-implied ratings.

**High-Level Story Flow of the Application (Persona: Sri Krishnamurthy, CFA):**

1.  **Home / Workflow Overview:** Sri starts by understanding the application's purpose, the persona, and the overall workflow steps.
2.  **Data Acquisition & Feature Engineering:** Sri generates a synthetic dataset of corporate financials, mirroring real-world credit drivers. He then enhances this dataset by adding NLP-derived sentiment features, recognizing their importance as leading indicators for credit quality.
3.  **Model Training & Evaluation:** Sri trains two XGBoost models: one using only financial features (baseline) and another using the full multimodal dataset (financials + NLP). He evaluates their performance, emphasizing "within-1-notch accuracy" as a key metric for ordinal credit ratings.
4.  **Rating Mismatch Analysis:** Sri compares the model-implied ratings against actual agency ratings to identify discrepancies. This analysis categorizes issuers into "Downgrade Risks," "Upgrade Candidates," "Watch," or "Aligned," allowing him to prioritize further investigation. A crucial "Practitioner Warning" ensures ethical communication regarding model results.
5.  **Explainable AI: Rating Rationales:** To understand the "why" behind the model's predictions, Sri uses SHAP values. He selects specific issuers (e.g., a downgrade risk, an upgrade candidate) to visualize feature contributions from both financial and NLP drivers, providing clear, interpretable rationales.
6.  **NLP Contribution & Downgrade Watchlist:** Sri quantifies the incremental value of NLP sentiment by comparing cases where NLP features corrected versus worsened predictions. Finally, he generates a prioritized "Downgrade Watchlist" for high-risk issuers, providing an actionable deliverable for portfolio managers to inform timely investment or risk management decisions.

## 2. Code Requirements

### Import Statement

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings

# Suppress warnings for cleaner output in Streamlit
warnings.filterwarnings('ignore')

# Import all functions and constants from the source.py file
import source
```

### `st.session_state` Design

`st.session_state` will be used to preserve the state of generated data, trained models, and analysis results across different "pages" (conditional renders) and interactions within the Streamlit application.

**Initialization:**
All session state keys are initialized at the start of `app.py` to `None` or default values, ensuring the application has a consistent starting point.

```python
def initialize_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    if 'n_issuers' not in st.session_state:
        st.session_state.n_issuers = 1000 # Default for data generation
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
    if 'nlp_helped' not in st.session_state: # For NLP contribution analysis
        st.session_state.nlp_helped = None
    if 'nlp_hurt' not in st.session_state: # For NLP contribution analysis
        st.session_state.nlp_hurt = None
    if 'downgrade_watchlist' not in st.session_state:
        st.session_state.downgrade_watchlist = None

initialize_session_state()
```

**Updates and Reads:**

*   **`st.session_state.page`**: Updated by the sidebar `st.selectbox` to control conditional rendering.
*   **`st.session_state.n_issuers`**: Updated by the user's slider input in "Data Acquisition".
*   **`st.session_state.corp_data`**:
    *   **Updated:** When `source.generate_corporate_data()` is called (Page 1).
    *   **Read:** Across all subsequent pages for model training, analysis, and display.
*   **`st.session_state.financial_features`, `st.session_state.latent_quality`**:
    *   **Updated:** When `source.generate_corporate_data()` is called (Page 1).
    *   **Read:** By `source.add_sentiment_features()` (Page 1) and `source.train_and_evaluate_models()` (Page 2).
*   **`st.session_state.nlp_features`, `st.session_state.all_features`**:
    *   **Updated:** When `source.add_sentiment_features()` is called (Page 1).
    *   **Read:** By `source.train_and_evaluate_models()` (Page 2), `source.explain_rating()` (Page 4).
*   **`st.session_state.model_full`, `st.session_state.model_fin_only`**:
    *   **Updated:** When `source.train_and_evaluate_models()` is called (Page 2).
    *   **Read:** By `source.mismatch_analysis()` (Page 3), `source.explain_rating()` (Page 4), and `source.nlp_contribution_analysis()` (Page 5).
*   **`st.session_state.X_test`, `st.session_state.y_test`, `st.session_state.test_idx`**:
    *   **Updated:** When `source.train_and_evaluate_models()` is called (Page 2).
    *   **Read:** By `source.mismatch_analysis()` (Page 3), `source.explain_rating()` (Page 4), and `source.nlp_contribution_analysis()` (Page 5).
*   **`st.session_state.pred_full_numeric`, `st.session_state.pred_fin_numeric`**:
    *   **Updated:** When `source.train_and_evaluate_models()` is called (Page 2).
    *   **Read:** Internally for performance display (Page 2), or for debugging if needed. Not directly passed to later *source* functions but used in calculation of `nlp_helped`/`nlp_hurt`.
*   **`st.session_state.mismatches_df`**:
    *   **Updated:** When `source.mismatch_analysis()` is called (Page 3).
    *   **Read:** By `source.explain_rating()` (Page 4) for selecting example issuers, and by `source.build_watchlist()` (Page 5).
*   **`st.session_state.explainer_full`**:
    *   **Updated:** When `shap.TreeExplainer()` is initialized (Page 4).
    *   **Read:** By `source.explain_rating()` (Page 4).
*   **`st.session_state.nlp_helped`, `st.session_state.nlp_hurt`**:
    *   **Updated:** After `source.nlp_contribution_analysis()` is conceptually performed (Page 5). (Note: The `source.nlp_contribution_analysis` function *prints* these values; they need to be re-calculated or captured if intended for explicit `st.write` display.)
    *   **Read:** For display on Page 5.
*   **`st.session_state.downgrade_watchlist`**:
    *   **Updated:** When `source.build_watchlist()` is called (Page 5).
    *   **Read:** For display on Page 5.

### UI Interactions and `source.py` Function Calls

**Global Constants (from `source.py`):**
*   `source.RATING_MAP`
*   `source.RATING_NUM_TO_STR`

**Sidebar Navigation:**

```python
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
```

**Page: Home / Workflow Overview**
*   No function calls. Displays introductory markdown.

**Page: 1. Data Acquisition & Feature Engineering**

*   **Widget:** `st.slider` for `n_issuers`.
*   **Widget:** "Generate Corporate Data" `st.button`
    *   **Calls:** `st.session_state.corp_data, st.session_state.financial_features, st.session_state.latent_quality = source.generate_corporate_data(st.session_state.n_issuers)`
*   **Widget:** "Add NLP Sentiment Features" `st.button`
    *   **Calls:** `st.session_state.corp_data, st.session_state.nlp_features = source.add_sentiment_features(st.session_state.corp_data, st.session_state.latent_quality, st.session_state.n_issuers)`

**Page: 2. Model Training & Evaluation**

*   **Widget:** "Train and Evaluate Models" `st.button`
    *   **Calls:** `model_full, model_fin_only, X_test, y_test, test_idx, pred_full_numeric, pred_fin_numeric = source.train_and_evaluate_models(st.session_state.corp_data, st.session_state.financial_features, st.session_state.nlp_features, st.session_state.all_features, source.RATING_MAP)`
    *   Displays performance metrics (accuracy, within-1-notch) and a confusion matrix (`st.pyplot(fig)`).

**Page: 3. Rating Mismatch Analysis**

*   **Widget:** "Perform Mismatch Analysis" `st.button`
    *   **Calls:** `mismatches_df = source.mismatch_analysis(st.session_state.model_full, st.session_state.X_test, st.session_state.y_test, st.session_state.corp_data, st.session_state.test_idx, source.RATING_MAP, source.RATING_NUM_TO_STR)`
    *   Displays signal counts and a histogram of notch differences (`st.pyplot(fig)`).

**Page: 4. Explainable AI: Rating Rationales**

*   **Widget:** `st.selectbox` for `selected_category` (e.g., 'DOWNGRADE RISK').
*   **Widget:** `st.selectbox` for `selected_issuer_id` within the chosen category.
*   **Widget:** "Explain [Issuer ID]'s Rating" `st.button`
    *   **Calls:** `st.session_state.explainer_full = shap.TreeExplainer(st.session_state.model_full)` (initialized once per session)
    *   **Calls:** `source.explain_rating(st.session_state.model_full, st.session_state.explainer_full, issuer_features, st.session_state.all_features, implied_numeric, actual_numeric, selected_issuer_id, source.RATING_MAP, st.session_state.nlp_features)`
    *   Displays text output (from `explain_rating`'s `print` statements, described generically in markdown) and the SHAP waterfall plot (`st.pyplot(plt.gcf())`, followed by `plt.close('all')`).

**Page: 5. NLP Contribution & Downgrade Watchlist**

*   **Widget:** "Perform NLP Contribution Analysis" `st.button`
    *   **Calls:** Internal logic to calculate `nlp_helped` and `nlp_hurt` (mimicking `source.nlp_contribution_analysis`'s output logic but using session state for models/data to display via `st.write`). The `source.nlp_contribution_analysis` function itself is designed to print; Streamlit requires values for `st.write`.
*   **Widget:** "Build Downgrade Watchlist" `st.button`
    *   **Calls:** `downgrade_watchlist = source.build_watchlist(st.session_state.mismatches_df, notch_threshold=-2, top_n=10)`
    *   Displays the downgrade watchlist as a `st.dataframe`.

### Markdown Content

The following markdown blocks, including mathematical formulas, will be used within the Streamlit application at the specified locations.

---

**Application Title and Overall Structure**

```python
st.set_page_config(layout="wide", page_title="Bond Rating Prediction")
st.title("Bond Rating Prediction with ML and NLP")
st.markdown("---")
```

**Page: Home / Workflow Overview**

```python
st.header("Bond Rating Prediction with ML and NLP: A Credit Analyst's Workflow")
st.markdown("### Persona: Sri Krishnamurthy, CFA")
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
```

**Page: 1. Data Acquisition & Feature Engineering**

```python
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

# After data generation and display:
if st.session_state.corp_data is not None:
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

# After adding sentiment features and display:
if st.session_state.corp_data is not None and st.session_state.all_features is not None:
    st.markdown(f"**Explanation of Execution**: The `corp_data` DataFrame now includes three new NLP-derived features: `mgmt_sentiment`, `guidance_sentiment`, and `risk_language_score`. These features are designed to mimic real-world sentiment analysis, where positive values generally indicate better credit quality and negative/higher risk scores indicate potential deterioration. By fusing these qualitative signals with the financial ratios, Sri can build a more comprehensive and forward-looking credit risk model.")
```

**Page: 2. Model Training & Evaluation**

```python
st.header("2. Building Our Predictive Engine: Ordinal Credit Rating Model")
st.markdown(f"**Story**: With all the necessary features (financial ratios and NLP sentiment) prepared, Sri is ready to train his predictive model. He understands that credit ratings are inherently ordinal (AAA > AA > A, etc.), meaning there's a natural order to the classes, and misclassifying AAA as CCC is far worse than misclassifying it as AA. Therefore, he chooses an XGBoost classifier with an objective suitable for multi-class classification, but he will also pay close attention to 'within-1-notch accuracy,' a more relevant metric for ordinal predictions. He will train two models: one using only financial features and another using the full multimodal feature set, to quantify the value of NLP.")
st.markdown(f"**Real-World Relevance**: Implementing an ordinal classification model is crucial for financial applications where target variables have a natural order. Evaluating model performance not just by exact accuracy but by within-1-notch accuracy provides a more nuanced and practically useful measure of a model's utility in a credit risk context. The comparison between financials-only and multimodal models directly answers the question of whether adding NLP brings incremental value.")

st.markdown(f"**Quantitative Logic: Within-1-Notch Accuracy**")
st.markdown(r"Given predicted numerical rating $\hat{{y}}$ and actual numerical rating $y$, the 'within-1-notch accuracy' is defined as the proportion of predictions where the absolute difference between the predicted and actual rating is less than or equal to 1:")
st.markdown(r"$$ \text{{Within-1-Notch Accuracy}} = \frac{{1}}{{N}} \sum_{{i=1}}^{{N}} \mathbb{{I}}(|\hat{{y}}_i - y_i| \le 1) $$")
st.markdown(r"where $N$ is the number of samples, $\hat{y}_i$ is the predicted numerical rating for issuer $i$, $y_i$ is the actual numerical rating for issuer $i$, and $\mathbb{{I}}(\cdot)$ is the indicator function which is 1 if the condition is true and 0 otherwise. This metric is more relevant for credit ratings than exact accuracy because predicting an 'A' when the actual is 'AA' (1-notch error) is far less consequential than predicting 'CCC' when the actual is 'AA' (5-notch error).")

# After model training and evaluation display:
if st.session_state.model_full is not None:
    st.markdown(f"**Explanation of Execution**: The performance metrics clearly show the exact accuracy and, more importantly, the 'within 1-notch accuracy' for both models. The 'NLP contribution' quantifies the improvement gained by including sentiment features. A higher within-1-notch accuracy is crucial for Sri, as it indicates the model is robust enough to provide actionable insights without frequently making severe misclassifications. The confusion matrix visually confirms how well the multimodal model aligns predictions with actual ratings, with most predictions concentrated on or near the diagonal, demonstrating its ability to distinguish between different credit tiers.")
```

**Page: 3. Rating Mismatch Analysis**

```python
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

# After mismatch analysis display:
if st.session_state.mismatches_df is not None:
    st.subheader("Practitioner Warning")
    st.markdown(f"Implied ratings are analytical tools, not official ratings. Presenting model-implied ratings as equivalent to agency ratings (Moody's, S&P, Fitch) would violate CFA Standard V(B) on communication with clients. The implied rating must be clearly labeled as 'model-generated, for internal analytical use only.' Sri uses it as one input among many—alongside the agency rating, his own fundamental analysis, market-implied ratings from CDS spreads, and peer comparison.")
    st.markdown(f"**Explanation of Execution**: The mismatch analysis output provides a summary of how many issuers fall into each category (Aligned, Watch, Downgrade Risk, Upgrade Candidate). The specific lists of top downgrade risks and upgrade candidates immediately highlight where Sri's attention should be focused for deeper research. The histogram of `notch_diff` visually represents the distribution of prediction errors. A skewed distribution towards negative notches would indicate a bias towards predicting lower ratings or that the model is effectively identifying potential downgrade risks. This information is invaluable for Sri to prioritize his workload and alert portfolio managers to actionable insights.")
```

**Page: 4. Explainable AI: Rating Rationales**

```python
st.header("4. Unpacking the 'Why': Explainable AI for Rating Rationales")
st.markdown(f"**Story**: For a CFA Charterholder like Sri, simply knowing *what* the model predicts isn't enough; he needs to understand *why*. Regulatory scrutiny and internal decision-making demand transparency. He will use SHAP (SHapley Additive exPlanations) values to dissect individual predictions. This will allow him to see which specific financial ratios and NLP sentiment features most strongly contributed to an issuer's implied rating, providing clear, interpretable rationales for stakeholders.")
st.markdown(f"**Real-World Relevance**: Explainable AI (XAI) is critical in finance. SHAP values allow Sri to convert complex model outputs into understandable feature contributions, fostering trust and enabling him to justify investment decisions or risk assessments. By distinguishing between financial and NLP drivers, he can provide a richer narrative, emphasizing the leading indicators identified by sentiment.")

st.markdown(f"**Quantitative Logic: SHAP Values**")
st.markdown(f"SHAP values explain the prediction of an instance by computing the contribution of each feature to the prediction. For a tree-based model like XGBoost, SHAP values are calculated efficiently using `shap.TreeExplainer`. The SHAP value for a feature represents the average marginal contribution of that feature value across all possible coalitions of features. A positive SHAP value for a feature means it pushes the prediction higher, while a negative value pushes it lower.")

# After SHAP explanation generation and display:
if st.session_state.explainer_full is not None:
    # Textual output will be from the source.explain_rating function. Provide a generic explanation.
    st.markdown(f"**Explanation of Execution**: For the selected issuer, the model's rationale is displayed. This would include specific financial metrics (e.g., high `debt_to_ebitda` or low `interest_coverage`) and NLP sentiment signals (e.g., negative `mgmt_sentiment` or high `risk_language_score`) that are driving the implied rating. The SHAP waterfall plot offers a powerful visualization, showing how each feature pushes the prediction from the base value to the final output. This level of transparency is essential for Sri to validate the model's logic, present findings to colleagues, and support deeper human analysis.")
```

**Page: 5. NLP Contribution & Downgrade Watchlist**

```python
st.header("5. Quantifying NLP's Edge & Prioritizing Action: Watchlist & Contribution Analysis")
st.markdown(f"**Story**: Sri is convinced NLP sentiment adds value, but he needs to quantify this benefit empirically for his firm's stakeholders. By comparing how many predictions were 'corrected' by NLP (i.e., became accurate only with NLP features) versus 'worsened,' he can demonstrate its net contribution. Finally, he'll synthesize all insights to create a prioritized 'Downgrade Watchlist' – a concrete, actionable deliverable for portfolio managers to review, complete with relevant financial and sentiment data for high-risk issuers.")
st.markdown(f"**Real-World Relevance**: Quantifying the incremental value of advanced features like NLP sentiment is crucial for justifying model complexity and data acquisition costs. The Downgrade Watchlist is a direct output for risk management and alpha generation, enabling portfolio managers to focus their attention on the most critical cases and potentially adjust holdings or hedging strategies. This represents the final, actionable step in Sri's workflow.")

# After NLP contribution and watchlist display:
if st.session_state.nlp_helped is not None and st.session_state.downgrade_watchlist is not None:
    st.markdown(f"**Explanation of Execution**: The NLP contribution analysis provides a quantitative summary of how much the sentiment features improved the model's predictions, indicating a net benefit in 'corrected' predictions. This data point is crucial for Sri to articulate the value of incorporating unstructured data. The generated 'Downgrade Watchlist' table then presents the most critical issuers, prioritized by their `notch_diff` (how far below the actual rating the model predicted), along with their sector, actual and implied ratings, and key NLP sentiment scores (`mgmt_sentiment`, `risk_language_score`). This direct, actionable list allows Sri and his portfolio managers to immediately identify and investigate companies at the highest risk of a downgrade, enabling timely strategic adjustments to their bond holdings.")
```
