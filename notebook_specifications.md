
# Bond Rating Prediction with ML and NLP: A Credit Analyst's Workflow

**Persona**: Sri Krishnamurthy, CFA – A seasoned Credit Analyst at "Alpha Asset Management," a fixed income investment firm.

**Scenario**: Sri's firm manages a significant portfolio of corporate bonds. Traditional credit analysis, involving meticulous review of financial statements and earnings call transcripts, is highly time-consuming. Sri needs a data-driven approach to rapidly screen hundreds of corporate bonds for creditworthiness, identify potential rating changes (upgrades or downgrades) ahead of rating agencies, and provide a competitive edge in portfolio management. This notebook outlines a real-world workflow to integrate quantitative financial data with qualitative management commentary using machine learning.

---

## 1. Environment Setup

### Installing Required Libraries

This step ensures all necessary Python packages for data manipulation, machine learning, and explainable AI are available in the environment.

```python
!pip install numpy pandas scikit-learn xgboost shap matplotlib seaborn
```

### Importing Dependencies

Importing the libraries needed for data generation, model building, evaluation, and visualization.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define rating map for consistency
RATING_MAP = {
    0: 'CCC', 1: 'B', 2: 'BB', 3: 'BBB',
    4: 'A', 5: 'AA', 6: 'AAA'
}
RATING_NUM_TO_STR = {v: k for k, v in RATING_MAP.items()}
```

---

## 2. Setting the Stage: Data Acquisition & Initial Credit Assessment

**Story**: As a CFA Charterholder, Sri understands that the foundation of any credit assessment is robust financial data. Before diving into sophisticated modeling, he needs a representative dataset of corporate financials, reflecting various aspects of creditworthiness. For this simulation, he'll generate a synthetic dataset, ensuring the data's underlying structure mirrors real-world correlations between financial health and credit ratings. This initial step provides the quantitative basis for his analysis.

**Real-World Relevance**: Generating or acquiring a clean, comprehensive dataset is the first critical step for any data-driven analyst. Understanding the input features (financial ratios) is paramount, as they directly influence the model's ability to assess a company's financial risk and capacity to meet its obligations.

**Quantitative Logic**: The generation process uses a latent `quality` factor. This `quality` factor is a hidden variable that drives both the financial ratios and the ultimate credit rating. This ensures that the simulated financial ratios are naturally correlated with the credit rating, mimicking real-world data where stronger financial metrics generally lead to higher credit ratings.
For example, `interest_coverage` is positively correlated with `quality`, meaning higher quality companies tend to have better interest coverage:
$$ \text{interest\_coverage} = \text{clip}(2 + \text{quality} \times 3 + \text{noise}, 0.5, 30) $$
Conversely, `debt_to_ebitda` is negatively correlated, meaning higher quality companies tend to have lower debt:
$$ \text{debt\_to\_ebitda} = \text{clip}(4 - \text{quality} \times 1.5 + \text{noise}, 0.5, 15) $$
The final `rating_score` is also derived from `quality` with added noise, then binned to create the ordinal credit ratings.

```python
def generate_corporate_data(n_issuers, random_state=42):
    """
    Simulates corporate financial data and credit ratings.
    A latent credit quality factor drives both financial ratios and ratings.
    """
    np.random.seed(random_state)

    # Latent credit quality (drives both ratios and ratings)
    quality = np.random.randn(n_issuers) * 1.5

    data = pd.DataFrame({
        'issuer_id': [f'ISSUER_{i:04d}' for i in range(n_issuers)],
        'sector': np.random.choice(['Industrials', 'Technology', 'Healthcare', 'Financials', 'Energy', 'Utilities', 'Consumer'], n_issuers)
    })

    # Generate financial ratios with rating-correlated structure
    data['interest_coverage'] = np.clip(2 + quality * 3 + np.random.randn(n_issuers) * 0.5, 0.5, 30)
    data['debt_to_ebitda'] = np.clip(4 - quality * 1.5 + np.random.randn(n_issuers) * 0.8, 0.5, 15)
    data['net_debt_to_equity'] = np.clip(1.0 - quality * 0.3 + np.random.randn(n_issuers) * 0.3, -0.2, 5)
    data['profit_margin'] = np.clip(0.10 + quality * 0.05 + np.random.randn(n_issuers) * 0.03, -0.1, 0.5)
    data['revenue_growth'] = np.random.randn(n_issuers) * 0.08 + 0.03
    data['current_ratio'] = np.clip(1.5 + quality * 0.3 + np.random.randn(n_issuers) * 0.3, 0.3, 4)
    data['free_cash_flow_yield'] = np.clip(0.05 + quality * 0.02 + np.random.randn(n_issuers) * 0.02, -0.1, 0.2)
    data['total_assets_log'] = np.random.lognormal(10, 1, n_issuers) # Proxy for size

    # Rating based on quality + noise
    rating_score = np.clip(quality * 1.2 + np.random.randn(n_issuers) * 0.5, -3, 3)
    data['rating'] = pd.cut(rating_score,
                            bins=[-np.inf, -2, -1, 0, 1, 1.5, 2.2, np.inf],
                            labels=list(RATING_MAP.keys())).astype(int)

    financial_features = [
        'interest_coverage', 'debt_to_ebitda', 'net_debt_to_equity',
        'profit_margin', 'revenue_growth', 'current_ratio',
        'free_cash_flow_yield', 'total_assets_log'
    ]
    return data, financial_features, quality

# Execute data generation
n_issuers = 1000
corp_data, financial_features, latent_quality = generate_corporate_data(n_issuers)

print(f"Generated data for {n_issuers} issuers with {len(financial_features)} financial features.")
print("\nFirst 5 rows of the generated financial data:")
print(corp_data.head())

print("\nDistribution of actual credit ratings:")
rating_counts = corp_data['rating'].map(RATING_MAP).value_counts().sort_index()
print(rating_counts)

# Mapping numerical ratings back to strings for display
corp_data['actual_rating_str'] = corp_data['rating'].map(RATING_MAP)
```

**Explanation of Execution**:
The output shows a snapshot of the generated corporate data, including `issuer_id`, `sector`, various financial ratios, and the assigned numerical credit rating (`rating`). The distribution of credit ratings indicates a realistic spread across different credit tiers, which is crucial for training a robust prediction model. This initial dataset forms the quantitative foundation for Sri's analysis.

---

## 3. Incorporating Qualitative Insights: NLP Sentiment Features

**Story**: Sri knows that financial ratios are often lagging indicators, reflecting past performance. To gain a competitive edge and identify early warning signs of credit deterioration or improvement, he needs forward-looking insights. He decides to simulate NLP-derived sentiment scores from earnings call transcripts, such as management tone, guidance sentiment, and risk language frequency. These qualitative signals can often capture management's outlook and potential operational shifts before they manifest in financial statements.

**Real-World Relevance**: Multimodal feature engineering—combining structured numerical data with unstructured text-derived features—is a hallmark of advanced credit analysis. Sentiment analysis of earnings calls provides a crucial "temporal lead," often signaling credit quality trajectory 1-2 quarters before financial ratios reflect the change. This proactive insight is invaluable for portfolio managers making timely investment or risk management decisions.

**Mathematical Formulation: Multimodal Feature Fusion**
The feature vector for each issuer $i$ is: $x_i = [ X_{i,1}, ..., X_{i,8}, S_{i,1}, S_{i,2}, S_{i,3} ]$
where $X_{i,j}$ are the financial ratios and $S_{i,k}$ are the NLP sentiment features.
-   **Financial ratios** ($X_{i,1}, ..., X_{i,8}$) capture the *current* financial position: leverage (`debt/EBITDA`), coverage (`interest coverage`), profitability (`margins`), liquidity (`current ratio`), and size (`total assets`).
-   **NLP sentiment** ($S_{i,1}, S_{i,2}, S_{i,3}$) captures the *trajectory* and management confidence: a CEO who speaks optimistically about future cash flows signals improving credit quality; one who hedges with risk language signals deterioration.

The NLP signal is leading, not lagging. Financial ratios reflect last quarter's results. Earnings call sentiment reflects management's view of next quarter. When sentiment deteriorates before ratios do, it is an early warning of a potential downgrade—giving credit analysts a temporal edge over purely ratio-based analysis.

```python
def add_sentiment_features(data, latent_quality, n_issuers, random_state=42):
    """
    Simulates FinBERT-like sentiment scores from earnings calls.
    Sentiment correlates with credit quality but adds independent signal.
    """
    np.random.seed(random_state)

    # Management tone (positive=confident, negative=cautious)
    data['mgmt_sentiment'] = np.clip(
        0.5 + latent_quality * 0.15 + np.random.randn(n_issuers) * 0.15, -1, 1)

    # Forward guidance sentiment (optimistic vs. hedging)
    data['guidance_sentiment'] = np.clip(
        0.3 + latent_quality * 0.10 + np.random.randn(n_issuers) * 0.20, -1, 1)

    # Risk language frequency (higher = more risk discussion = worse credit)
    data['risk_language_score'] = np.clip(
        0.3 - latent_quality * 0.08 + np.random.randn(n_issuers) * 0.10, 0, 1)

    nlp_features = ['mgmt_sentiment', 'guidance_sentiment', 'risk_language_score']
    return data, nlp_features

# Execute adding sentiment features
corp_data, nlp_features = add_sentiment_features(corp_data, latent_quality, n_issuers)
all_features = financial_features + nlp_features

print(f"\nAdded {len(nlp_features)} NLP sentiment features.")
print(f"Total features for multimodal model: {len(all_features)}.")
print("\nFirst 5 rows of data with new sentiment features:")
print(corp_data.head())
```

**Explanation of Execution**:
The `corp_data` DataFrame now includes three new NLP-derived features: `mgmt_sentiment`, `guidance_sentiment`, and `risk_language_score`. These features are designed to mimic real-world sentiment analysis, where positive values generally indicate better credit quality and negative/higher risk scores indicate potential deterioration. By fusing these qualitative signals with the financial ratios, Sri can build a more comprehensive and forward-looking credit risk model.

---

## 4. Building Our Predictive Engine: Ordinal Credit Rating Model

**Story**: With all the necessary features (financial ratios and NLP sentiment) prepared, Sri is ready to train his predictive model. He understands that credit ratings are inherently ordinal (AAA > AA > A, etc.), meaning there's a natural order to the classes, and misclassifying AAA as CCC is far worse than misclassifying it as AA. Therefore, he chooses an XGBoost classifier with an objective suitable for multi-class classification, but he will also pay close attention to "within-1-notch accuracy," a more relevant metric for ordinal predictions. He will train two models: one using only financial features and another using the full multimodal feature set, to quantify the value of NLP.

**Real-World Relevance**: Implementing an ordinal classification model is crucial for financial applications where target variables have a natural order. Evaluating model performance not just by exact accuracy but by within-1-notch accuracy provides a more nuanced and practically useful measure of a model's utility in a credit risk context. The comparison between financials-only and multimodal models directly answers the question of whether adding NLP brings incremental value.

**Quantitative Logic: Within-1-Notch Accuracy**
Given predicted numerical rating $\hat{y}$ and actual numerical rating $y$, the "within-1-notch accuracy" is defined as the proportion of predictions where the absolute difference between the predicted and actual rating is less than or equal to 1:
$$ \text{Within-1-Notch Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(|\hat{y}_i - y_i| \le 1) $$
where $N$ is the number of samples and $\mathbb{I}(\cdot)$ is the indicator function. This metric is more relevant for credit ratings than exact accuracy because predicting an 'A' when the actual is 'AA' (1-notch error) is far less consequential than predicting 'CCC' when the actual is 'AA' (5-notch error).

```python
def train_and_evaluate_models(corp_data, financial_features, nlp_features, all_features, RATING_MAP, random_state=42):
    """
    Trains two XGBoost models (financials-only and multimodal) and evaluates their performance.
    """
    X = corp_data[all_features]
    y = corp_data['rating']

    # Split data into training and testing sets, stratified by rating
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, corp_data.index, test_size=0.2, random_state=random_state, stratify=y
    )

    # Convert numerical ratings back to string labels for mapping
    y_test_str = y_test.map(RATING_MAP)

    # --- Financials-only model ---
    model_fin_only = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softprob', # Using multi:softprob for probabilistic output in multi-class
        num_class=len(RATING_MAP),
        random_state=random_state,
        eval_metric='mlogloss'
    )
    model_fin_only.fit(X_train[financial_features], y_train)
    pred_fin_numeric = model_fin_only.predict(X_test[financial_features])

    # --- Multimodal (financials + NLP) model ---
    model_full = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=len(RATING_MAP),
        random_state=random_state,
        eval_metric='mlogloss'
    )
    model_full.fit(X_train, y_train)
    pred_full_numeric = model_full.predict(X_test)

    # --- Evaluation ---
    acc_fin = accuracy_score(y_test, pred_fin_numeric)
    acc_full = accuracy_score(y_test, pred_full_numeric)

    within1_fin = np.mean(np.abs(pred_fin_numeric - y_test) <= 1)
    within1_full = np.mean(np.abs(pred_full_numeric - y_test) <= 1)

    print("RATING PREDICTION PERFORMANCE")
    print("=" * 55)
    print(f"{'Metric':<30s}{'Financials Only':>15s}{'+ NLP Sentiment':>15s}")
    print(f"{'Exact accuracy':<30s}{acc_fin:>15.1%}{acc_full:>15.1%}")
    print(f"{'Within 1-notch accuracy':<30s}{within1_fin:>15.1%}{within1_full:>15.1%}")
    nlp_contribution = acc_full - acc_fin
    print(f"{'NLP contribution':<30s}{'---':>15s}{nlp_contribution:>+15.1%}")

    # Plot Confusion Matrix for the full model
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, pred_full_numeric)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[RATING_MAP[i] for i in sorted(RATING_MAP.keys())],
                yticklabels=[RATING_MAP[i] for i in sorted(RATING_MAP.keys())])
    plt.xlabel('Predicted Rating')
    plt.ylabel('Actual Rating')
    plt.title('Confusion Matrix (Multimodal Model)')
    plt.show()

    return model_full, model_fin_only, X_test, y_test, test_idx, pred_full_numeric, pred_fin_numeric

# Execute model training and evaluation
model_full, model_fin_only, X_test, y_test, test_idx, pred_full_numeric, pred_fin_numeric = \
    train_and_evaluate_models(corp_data, financial_features, nlp_features, all_features, RATING_MAP)
```

**Explanation of Execution**:
The performance metrics clearly show the exact accuracy and, more importantly, the "within 1-notch accuracy" for both models. The "NLP contribution" quantifies the improvement gained by including sentiment features. A higher within-1-notch accuracy is crucial for Sri, as it indicates the model is robust enough to provide actionable insights without frequently making severe misclassifications. The confusion matrix visually confirms how well the multimodal model aligns predictions with actual ratings, with most predictions concentrated on or near the diagonal, demonstrating its ability to distinguish between different credit tiers.

---

## 5. Identifying Opportunities & Risks: Implied vs. Actual Rating Mismatches

**Story**: A model's prediction, or "implied rating," is not an end in itself. Sri's main objective is to compare these implied ratings against the actual agency ratings to identify discrepancies. These mismatches are critical signals: a model-implied rating significantly below the actual rating suggests a potential "downgrade risk," while a higher implied rating points to an "upgrade candidate." He needs a structured way to classify these issuers to prioritize his deep-dive analysis.

**Real-World Relevance**: This mismatch analysis is the core deliverable for a fixed income PM. It transforms raw model predictions into actionable intelligence, enabling proactive risk management (e.g., reducing exposure to downgrade candidates) or identifying alpha opportunities (e.g., increasing exposure to upgrade candidates before spread tightening). It complements, rather than replaces, traditional credit analysis.

**Quantitative Logic: Rating Mismatch Analysis**
For each issuer, the `notch_diff` is calculated as the numerical implied rating minus the numerical actual rating:
$$ \text{notch\_diff} = \text{implied\_numeric\_rating} - \text{actual\_numeric\_rating} $$
Based on `notch_diff`, issuers are categorized:
-   **DOWNGRADE RISK**: $\text{notch\_diff} \le -2$ (model predicts at least 2 notches lower)
-   **UPGRADE CANDIDATE**: $\text{notch\_diff} \ge 2$ (model predicts at least 2 notches higher)
-   **WATCH (1 notch)**: $|\text{notch\_diff}| = 1$
-   **ALIGNED**: $\text{notch\_diff} = 0$

```python
def mismatch_analysis(model, X_test, y_test, corp_data, test_idx, RATING_MAP, RATING_NUM_TO_STR):
    """
    Compares model-implied ratings to actual agency ratings and classifies mismatches.
    """
    implied_numeric = model.predict(X_test)
    actual_numeric = y_test.values

    test_issuers_df = corp_data.loc[test_idx].copy()
    test_issuers_df['implied_numeric'] = implied_numeric
    test_issuers_df['actual_numeric'] = actual_numeric

    test_issuers_df['implied_rating_str'] = test_issuers_df['implied_numeric'].map(RATING_MAP)
    test_issuers_df['actual_rating_str'] = test_issuers_df['actual_numeric'].map(RATING_MAP)

    test_issuers_df['notch_diff'] = test_issuers_df['implied_numeric'] - test_issuers_df['actual_numeric']

    # Classify mismatches
    def classify_signal(d):
        if d <= -2:
            return 'DOWNGRADE RISK'
        elif d >= 2:
            return 'UPGRADE CANDIDATE'
        elif abs(d) == 1:
            return 'WATCH (1 notch)'
        else:
            return 'ALIGNED'

    test_issuers_df['signal'] = test_issuers_df['notch_diff'].apply(classify_signal)

    print("\nIMPLIED RATING MISMATCH ANALYSIS")
    print("=" * 60)
    signal_counts = test_issuers_df['signal'].value_counts()
    for signal, count in signal_counts.items():
        print(f" {signal}: {count} issuers ({count/len(test_issuers_df):.0%})")

    # Display top downgrade risks
    downgrade_risks = test_issuers_df[test_issuers_df['notch_diff'] <= -2].nlargest(5, 'notch_diff', keep='all')
    if not downgrade_risks.empty:
        print("\nTOP DOWNGRADE RISKS (Model predicts much lower than actual):")
        for _, row in downgrade_risks.iterrows():
            print(f" - {row['issuer_id']}: Actual={row['actual_rating_str']}, Implied={row['implied_rating_str']} ({row['notch_diff']:+d} notches)")

    # Display top upgrade candidates
    upgrade_candidates = test_issuers_df[test_issuers_df['notch_diff'] >= 2].nlargest(5, 'notch_diff', keep='all')
    if not upgrade_candidates.empty:
        print("\nTOP UPGRADE CANDIDATES (Model predicts much higher than actual):")
        for _, row in upgrade_candidates.iterrows():
            print(f" - {row['issuer_id']}: Actual={row['actual_rating_str']}, Implied={row['implied_rating_str']} ({row['notch_diff']:+d} notches)")

    # Plot histogram of notch differences
    plt.figure(figsize=(10, 6))
    sns.histplot(test_issuers_df['notch_diff'], bins=np.arange(-6.5, 7.5, 1), kde=False, palette='viridis')
    plt.title('Distribution of Rating Notch Differences (Implied - Actual)')
    plt.xlabel('Notch Difference (Implied Rating - Actual Rating)')
    plt.ylabel('Number of Issuers')
    plt.xticks(np.arange(-6, 7, 1))
    plt.axvline(x=-2, color='r', linestyle='--', label='Downgrade Risk Threshold (-2)')
    plt.axvline(x=2, color='g', linestyle='--', label='Upgrade Candidate Threshold (+2)')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    return test_issuers_df

# Execute mismatch analysis
mismatches_df = mismatch_analysis(model_full, X_test, y_test, corp_data, test_idx, RATING_MAP, RATING_NUM_TO_STR)
```

**Practitioner Warning**:
Implied ratings are analytical tools, not official ratings. Presenting model-implied ratings as equivalent to agency ratings (Moody's, S&P, Fitch) would violate CFA Standard V(B) on communication with clients. The implied rating must be clearly labeled as "model-generated, for internal analytical use only." Sri uses it as one input among many—alongside the agency rating, his own fundamental analysis, market-implied ratings from CDS spreads, and peer comparison.

**Explanation of Execution**:
The mismatch analysis output provides a summary of how many issuers fall into each category (Aligned, Watch, Downgrade Risk, Upgrade Candidate). The specific lists of top downgrade risks and upgrade candidates immediately highlight where Sri's attention should be focused for deeper research. The histogram of `notch_diff` visually represents the distribution of prediction errors. A skewed distribution towards negative notches would indicate a bias towards predicting lower ratings or that the model is effectively identifying potential downgrade risks. This information is invaluable for Sri to prioritize his workload and alert portfolio managers to actionable insights.

---

## 6. Unpacking the "Why": Explainable AI for Rating Rationales

**Story**: For a CFA Charterholder like Sri, simply knowing *what* the model predicts isn't enough; he needs to understand *why*. Regulatory scrutiny and internal decision-making demand transparency. He will use SHAP (SHapley Additive exPlanations) values to dissect individual predictions. This will allow him to see which specific financial ratios and NLP sentiment features most strongly contributed to an issuer's implied rating, providing clear, interpretable rationales for stakeholders.

**Real-World Relevance**: Explainable AI (XAI) is critical in finance. SHAP values allow Sri to convert complex model outputs into understandable feature contributions, fostering trust and enabling him to justify investment decisions or risk assessments. By distinguishing between financial and NLP drivers, he can provide a richer narrative, emphasizing the leading indicators identified by sentiment.

**Quantitative Logic: SHAP Values**
SHAP values explain the prediction of an instance by computing the contribution of each feature to the prediction. For a tree-based model like XGBoost, SHAP values are calculated efficiently using `shap.TreeExplainer`. The SHAP value for a feature represents the average marginal contribution of that feature value across all possible coalitions of features. A positive SHAP value for a feature means it pushes the prediction higher, while a negative value pushes it lower.

```python
def explain_rating(model, explainer, issuer_features, feature_names,
                   predicted_rating_numeric, actual_rating_numeric, issuer_id,
                   rating_map, nlp_features):
    """
    Generates SHAP-based rationale for a rating prediction.
    """
    print(f"\nRATING RATIONALE: {issuer_id}")
    print(f"Actual: {rating_map.get(actual_rating_numeric, '?')}, Implied: {rating_map.get(predicted_rating_numeric, '?')}")
    print("-" * 50)

    # SHAP values for the predicted class
    shap_values = explainer.shap_values(issuer_features.reshape(1, -1))
    class_shap = shap_values[predicted_rating_numeric][0] if isinstance(shap_values, list) else shap_values[0]

    contributions = pd.Series(class_shap, index=feature_names).sort_values(key=abs, ascending=False)

    # Separate financial and NLP drivers
    fin_drivers = contributions[[f for f in contributions.index if f in financial_features]]
    nlp_drivers_contributions = contributions[[f for f in contributions.index if f in nlp_features]]

    print("Financial drivers:")
    if not fin_drivers.empty:
        for feat, val in fin_drivers.head(3).items():
            direction = "supports higher rating" if val > 0 else "drags rating lower"
            print(f"  - {feat}: {direction} (SHAP={val:+.3f})")
    else:
        print("  No significant financial drivers found.")


    if not nlp_drivers_contributions.empty:
        print("\nNLP sentiment drivers:")
        for feat, val in nlp_drivers_contributions.head(3).items():
            direction = "positive signal" if val > 0 else "negative signal"
            print(f"  - {feat}: {direction} (SHAP={val:+.3f})")
    else:
        print("  No significant NLP sentiment drivers found.")

    # Waterfall plot
    shap.initjs()
    # Need to convert single instance to DataFrame with feature_names
    instance_df = pd.DataFrame([issuer_features], columns=feature_names)
    shap_values_to_plot = explainer(instance_df)
    # Filter shap_values for the specific predicted class
    class_index_to_plot = predicted_rating_numeric
    if isinstance(shap_values_to_plot.values, list): # Multi-output case
        shap_values_to_plot.values = shap_values_to_plot.values[class_index_to_plot]
        shap_values_to_plot.data = shap_values_to_plot.data # Keep the data aligned
        shap_values_to_plot.base_values = shap_values_to_plot.base_values[class_index_to_plot]
    
    plt.figure(figsize=(10,6))
    shap.plots.waterfall(shap_values_to_plot[0], max_display=10, show=False)
    plt.title(f"SHAP Waterfall Plot for {issuer_id} (Predicted: {RATING_MAP.get(predicted_rating_numeric, '?')})")
    plt.show()

# Initialize SHAP explainer for the full multimodal model
explainer_full = shap.TreeExplainer(model_full)

# Select an example downgrade risk for explanation
downgrade_idx = mismatches_df[mismatches_df['signal'] == 'DOWNGRADE RISK'].index
if not downgrade_idx.empty:
    example_downgrade_issuer_id = mismatches_df.loc[downgrade_idx[0], 'issuer_id']
    example_downgrade_features = X_test.loc[downgrade_idx[0]].values
    example_downgrade_implied = mismatches_df.loc[downgrade_idx[0], 'implied_numeric']
    example_downgrade_actual = mismatches_df.loc[downgrade_idx[0], 'actual_numeric']

    explain_rating(model_full, explainer_full, example_downgrade_features, all_features,
                   example_downgrade_implied, example_downgrade_actual,
                   example_downgrade_issuer_id, RATING_MAP, nlp_features)
else:
    print("No downgrade risk examples found to explain.")

# Select an example upgrade candidate for explanation
upgrade_idx = mismatches_df[mismatches_df['signal'] == 'UPGRADE CANDIDATE'].index
if not upgrade_idx.empty:
    example_upgrade_issuer_id = mismatches_df.loc[upgrade_idx[0], 'issuer_id']
    example_upgrade_features = X_test.loc[upgrade_idx[0]].values
    example_upgrade_implied = mismatches_df.loc[upgrade_idx[0], 'implied_numeric']
    example_upgrade_actual = mismatches_df.loc[upgrade_idx[0], 'actual_numeric']

    explain_rating(model_full, explainer_full, example_upgrade_features, all_features,
                   example_upgrade_implied, example_upgrade_actual,
                   example_upgrade_issuer_id, RATING_MAP, nlp_features)
else:
    print("No upgrade candidate examples found to explain.")
```

**Explanation of Execution**:
For the selected downgrade risk and upgrade candidate, the output provides a detailed rationale. Sri can see precisely which financial metrics (e.g., high `debt_to_ebitda` or low `interest_coverage`) and NLP sentiment signals (e.g., negative `mgmt_sentiment` or high `risk_language_score`) are driving the implied rating. The SHAP waterfall plots offer a powerful visualization, showing how each feature pushes the prediction from the base value to the final output. This level of transparency is essential for Sri to validate the model's logic, present findings to colleagues, and support deeper human analysis.

---

## 7. Quantifying NLP's Edge & Prioritizing Action: Watchlist & Contribution Analysis

**Story**: Sri is convinced NLP sentiment adds value, but he needs to quantify this benefit empirically for his firm's stakeholders. By comparing how many predictions were "corrected" by NLP (i.e., became accurate only with NLP features) versus "worsened," he can demonstrate its net contribution. Finally, he'll synthesize all insights to create a prioritized "Downgrade Watchlist" – a concrete, actionable deliverable for portfolio managers to review, complete with relevant financial and sentiment data for high-risk issuers.

**Real-World Relevance**: Quantifying the incremental value of advanced features like NLP sentiment is crucial for justifying model complexity and data acquisition costs. The Downgrade Watchlist is a direct output for risk management and alpha generation, enabling portfolio managers to focus their attention on the most critical cases and potentially adjust holdings or hedging strategies. This represents the final, actionable step in Sri's workflow.

```python
def nlp_contribution_analysis(model_full, model_fin_only, X_test, y_test, financial_features):
    """
    Measures how much NLP features improve rating prediction by comparing full vs financials-only models.
    """
    pred_full = model_full.predict(X_test)
    pred_fin = model_fin_only.predict(X_test[financial_features])

    # Cases where NLP flipped the prediction to correct
    nlp_helped = ((pred_full == y_test) & (pred_fin != y_test)).sum()
    # Cases where NLP worsened prediction
    nlp_hurt = ((pred_full != y_test) & (pred_fin == y_test)).sum()

    print("\nNLP CONTRIBUTION ANALYSIS")
    print("=" * 50)
    print(f"Cases where NLP corrected prediction: {nlp_helped}")
    print(f"Cases where NLP worsened prediction: {nlp_hurt}")
    net_nlp_benefit = nlp_helped - nlp_hurt
    print(f"Net NLP benefit: {net_nlp_benefit:+d} correct predictions")
    print(f"NLP is {'BENEFICIAL' if nlp_helped > nlp_hurt else 'DETRIMENTAL'}")

nlp_contribution_analysis(model_full, model_fin_only, X_test, y_test, financial_features)

def build_watchlist(mismatches_df, notch_threshold=-2, top_n=10):
    """
    Builds a prioritized downgrade watchlist for the credit portfolio.
    """
    watchlist = mismatches_df[mismatches_df['notch_diff'] <= notch_threshold].copy()
    if watchlist.empty:
        print(f"\nNo issuers found with notch difference <= {notch_threshold}.")
        return pd.DataFrame()

    watchlist = watchlist.sort_values('notch_diff', ascending=True).head(top_n)

    print(f"\nTOP {top_n} DOWNGRADE WATCHLIST ({len(watchlist)} issuers)")
    print("=" * 70)
    print(f"{'Issuer':<15s}{'Sector':<15s}{'Actual':>8s}{'Implied':>8s}{'Notch':>8s}{'MgmtSent':>10s}{'RiskLang':>10s}")
    for _, row in watchlist.iterrows():
        print(f"{row['issuer_id']:<15s}{row['sector']:<15s}"
              f"{row['actual_rating_str']:>8s}{row['implied_rating_str']:>8s}"
              f"{row['notch_diff']:>+8d}{row.get('mgmt_sentiment', 0):>10.2f}{row.get('risk_language_score', 0):>10.2f}")
    return watchlist

# Execute building the downgrade watchlist
downgrade_watchlist = build_watchlist(mismatches_df, notch_threshold=-2, top_n=10)
```

**Explanation of Execution**:
The NLP contribution analysis provides a quantitative summary of how much the sentiment features improved the model's predictions, indicating a net benefit in "corrected" predictions. This data point is crucial for Sri to articulate the value of incorporating unstructured data. The generated "Downgrade Watchlist" table then presents the most critical issuers, prioritized by their `notch_diff` (how far below the actual rating the model predicted), along with their sector, actual and implied ratings, and key NLP sentiment scores (`mgmt_sentiment`, `risk_language_score`). This direct, actionable list allows Sri and his portfolio managers to immediately identify and investigate companies at the highest risk of a downgrade, enabling timely strategic adjustments to their bond holdings.

---
