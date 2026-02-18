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

# Define rating map for consistency (Global constant)
RATING_MAP = {
    0: 'CCC', 1: 'B', 2: 'BB', 3: 'BBB',
    4: 'A', 5: 'AA', 6: 'AAA'
}
RATING_NUM_TO_STR = {v: k for k, v in RATING_MAP.items()} # Global constant

def generate_corporate_data(n_issuers: int, random_state: int = 42) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """
    Simulates corporate financial data and credit ratings.
    A latent credit quality factor drives both financial ratios and ratings.

    Args:
        n_issuers (int): Number of corporate issuers to simulate.
        random_state (int): Seed for random number generation.

    Returns:
        tuple[pd.DataFrame, list[str], np.ndarray]:
            - data (pd.DataFrame): DataFrame containing simulated corporate data.
            - financial_features (list[str]): List of column names representing financial features.
            - quality (np.ndarray): Latent credit quality factor for each issuer.
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

def add_sentiment_features(data: pd.DataFrame, latent_quality: np.ndarray, n_issuers: int, random_state: int = 42) -> tuple[pd.DataFrame, list[str]]:
    """
    Simulates FinBERT-like sentiment scores from earnings calls.
    Sentiment correlates with credit quality but adds independent signal.

    Args:
        data (pd.DataFrame): Existing DataFrame with corporate data.
        latent_quality (np.ndarray): Latent credit quality factor, used to correlate sentiment.
        n_issuers (int): Number of corporate issuers.
        random_state (int): Seed for random number generation.

    Returns:
        tuple[pd.DataFrame, list[str]]:
            - data (pd.DataFrame): DataFrame with added NLP sentiment features.
            - nlp_features (list[str]): List of column names representing NLP features.
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

def train_and_evaluate_models(
    corp_data: pd.DataFrame,
    financial_features: list[str],
    nlp_features: list[str],
    all_features: list[str],
    rating_map: dict[int, str],
    random_state: int = 42
) -> tuple[xgb.XGBClassifier, xgb.XGBClassifier, pd.DataFrame, pd.Series, pd.Index, np.ndarray, np.ndarray]:
    """
    Trains two XGBoost models (financials-only and multimodal) and evaluates their performance.

    Args:
        corp_data (pd.DataFrame): The complete corporate data.
        financial_features (list[str]): List of financial feature names.
        nlp_features (list[str]): List of NLP feature names.
        all_features (list[str]): List of all feature names (financial + NLP).
        rating_map (dict[int, str]): Mapping from numerical ratings to string labels.
        random_state (int): Seed for random number generation and model training.

    Returns:
        tuple[xgb.XGBClassifier, xgb.XGBClassifier, pd.DataFrame, pd.Series, pd.Index, np.ndarray, np.ndarray]:
            - model_full (xgb.XGBClassifier): The trained multimodal XGBoost model.
            - model_fin_only (xgb.XGBClassifier): The trained financials-only XGBoost model.
            - X_test (pd.DataFrame): Test features.
            - y_test (pd.Series): Test labels.
            - test_idx (pd.Index): Original index of test samples.
            - pred_full_numeric (np.ndarray): Predictions from the multimodal model.
            - pred_fin_numeric (np.ndarray): Predictions from the financials-only model.
    """
    X = corp_data[all_features]
    y = corp_data['rating']

    # Split data into training and testing sets, stratified by rating
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, corp_data.index, test_size=0.2, random_state=random_state, stratify=y
    )

    # --- Financials-only model ---
    model_fin_only = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=len(rating_map),
        random_state=random_state,
        eval_metric='mlogloss',
        use_label_encoder=False # Suppress warning for newer XGBoost versions
    )
    model_fin_only.fit(X_train[financial_features], y_train)
    pred_fin_numeric = model_fin_only.predict(X_test[financial_features])

    # --- Multimodal (financials + NLP) model ---
    model_full = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=len(rating_map),
        random_state=random_state,
        eval_metric='mlogloss',
        use_label_encoder=False # Suppress warning for newer XGBoost versions
    )
    model_full.fit(X_train, y_train)
    pred_full_numeric = model_full.predict(X_test)

    # --- Evaluation ---
    acc_fin = accuracy_score(y_test, pred_fin_numeric)
    acc_full = accuracy_score(y_test, pred_full_numeric)

    within1_fin = np.mean(np.abs(pred_fin_numeric - y_test) <= 1)
    within1_full = np.mean(np.abs(pred_full_numeric - y_test) <= 1)

    print("\nRATING PREDICTION PERFORMANCE")
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
                xticklabels=[rating_map[i] for i in sorted(rating_map.keys())],
                yticklabels=[rating_map[i] for i in sorted(rating_map.keys())])
    plt.xlabel('Predicted Rating')
    plt.ylabel('Actual Rating')
    plt.title('Confusion Matrix (Multimodal Model)')
    plt.show()

    return model_full, model_fin_only, X_test, y_test, test_idx, pred_full_numeric, pred_fin_numeric

def mismatch_analysis(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    corp_data: pd.DataFrame,
    test_idx: pd.Index,
    rating_map: dict[int, str],
    rating_num_to_str: dict[str, int] # Not used in current implementation, but kept for signature consistency
) -> pd.DataFrame:
    """
    Compares model-implied ratings to actual agency ratings and classifies mismatches.

    Args:
        model (xgb.XGBClassifier): The trained classification model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Actual test labels.
        corp_data (pd.DataFrame): The original complete corporate data.
        test_idx (pd.Index): Original index of test samples.
        rating_map (dict[int, str]): Mapping from numerical ratings to string labels.
        rating_num_to_str (dict[str, int]): Reverse mapping (string to numeric).

    Returns:
        pd.DataFrame: DataFrame of test issuers with actual, implied ratings, and mismatch signals.
    """
    implied_numeric = model.predict(X_test)
    actual_numeric = y_test.values

    test_issuers_df = corp_data.loc[test_idx].copy()
    test_issuers_df['implied_numeric'] = implied_numeric
    test_issuers_df['actual_numeric'] = actual_numeric

    test_issuers_df['implied_rating_str'] = test_issuers_df['implied_numeric'].map(rating_map)
    test_issuers_df['actual_rating_str'] = test_issuers_df['actual_numeric'].map(rating_map)

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

    # Display top downgrade risks (most negative notch_diff)
    downgrade_risks = test_issuers_df[test_issuers_df['notch_diff'] <= -2].nsmallest(5, 'notch_diff', keep='all')
    if not downgrade_risks.empty:
        print("\nTOP DOWNGRADE RISKS (Model predicts much lower than actual):")
        for _, row in downgrade_risks.iterrows():
            print(f" - {row['issuer_id']}: Actual={row['actual_rating_str']}, Implied={row['implied_rating_str']} ({row['notch_diff']:+d} notches)")

    # Display top upgrade candidates (most positive notch_diff)
    upgrade_candidates = test_issuers_df[test_issuers_df['notch_diff'] >= 2].nlargest(5, 'notch_diff', keep='all')
    if not upgrade_candidates.empty:
        print("\nTOP UPGRADE CANDIDATES (Model predicts much higher than actual):")
        for _, row in upgrade_candidates.iterrows():
            print(f" - {row['issuer_id']}: Actual={row['actual_rating_str']}, Implied={row['implied_rating_str']} ({row['notch_diff']:+d} notches)")

    # Plot histogram of notch differences
    plt.figure(figsize=(10, 6))
    sns.histplot(test_issuers_df['notch_diff'], bins=np.arange(-6.5, 7.5, 1), kde=False, color='#4CAF50')
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

def explain_rating(
    explainer: shap.TreeExplainer,
    issuer_features: np.ndarray,
    feature_names: list[str],
    predicted_rating_numeric: int,
    actual_rating_numeric: int,
    issuer_id: str,
    rating_map: dict[int, str],
    financial_features: list[str],
    nlp_features: list[str]
):
    """
    Generates SHAP-based rationale for a rating prediction and visualizes it.

    Args:
        explainer (shap.TreeExplainer): The SHAP explainer object for the model.
        issuer_features (np.ndarray): Feature vector for the specific issuer.
        feature_names (list[str]): Names of all features in the correct order.
        predicted_rating_numeric (int): The model's predicted numerical rating.
        actual_rating_numeric (int): The actual numerical rating.
        issuer_id (str): Identifier for the issuer.
        rating_map (dict[int, str]): Mapping from numerical ratings to string labels.
        financial_features (list[str]): List of financial feature names.
        nlp_features (list[str]): List of NLP feature names.
    """
    print(f"\nRATING RATIONALE: {issuer_id}")
    print(f"Actual: {rating_map.get(actual_rating_numeric, '?')}, Implied: {rating_map.get(predicted_rating_numeric, '?')}")
    print("-" * 50)

    # SHAP values for the predicted class
    shap_values_raw = explainer.shap_values(issuer_features.reshape(1, -1))

    if isinstance(shap_values_raw, list):
        # Standard behavior: list of arrays (num_samples, num_features)
        class_shap = shap_values_raw[predicted_rating_numeric][0]
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        # Non-standard observed behavior: 3D array (num_samples, num_features, num_classes)
        class_shap = shap_values_raw[0, :, predicted_rating_numeric]
    else:
        # Fallback for single-output or other unexpected structures
        print(f"Warning: Unexpected shap_values_raw type/shape: {type(shap_values_raw)}, {getattr(shap_values_raw, 'shape', 'N/A')}. Attempting default index.")
        class_shap = shap_values_raw[0]

    if class_shap.ndim > 1:
        print(f"Warning: class_shap is still multi-dimensional after processing: {class_shap.shape}. Attempting to flatten.")
        class_shap = class_shap.flatten()

    contributions = pd.Series(class_shap, index=feature_names).sort_values(key=abs, ascending=False)

    # Separate financial and NLP drivers
    fin_drivers = contributions[[f for f in contributions.index if f in financial_features]]
    nlp_drivers_contributions = contributions[[f for f in contributions.index if f in nlp_features]]

    print("Financial drivers:")
    if not fin_drivers.empty:
        for feat, val in fin_drivers.head(3).items():
            direction = "supports higher rating" if val > 0 else "drags rating lower"
            feat_value = issuer_features[feature_names.index(feat)]
            print(f"  - {feat} ({feat_value:.2f}): {direction} (SHAP={val:+.3f})")
    else:
        print("  No significant financial drivers found.")

    if not nlp_drivers_contributions.empty:
        print("\nNLP sentiment drivers:")
        for feat, val in nlp_drivers_contributions.head(3).items():
            direction = "positive signal" if val > 0 else "negative signal"
            feat_value = issuer_features[feature_names.index(feat)]
            print(f"  - {feat} ({feat_value:.2f}): {direction} (SHAP={val:+.3f})")
    else:
        print("  No significant NLP sentiment drivers found.")

    # Waterfall plot
    shap.initjs()
    instance_df = pd.DataFrame([issuer_features], columns=feature_names)
    shap_values_to_plot = explainer(instance_df)

    # Get the specific explanation for plotting
    explanation_for_plot = shap_values_to_plot[0, predicted_rating_numeric]

    # Set the base_values for the specific explanation to the correct class-specific base value.
    explanation_for_plot.base_values = float(explainer.expected_value[predicted_rating_numeric])
    explanation_for_plot.data = issuer_features

    plt.figure(figsize=(10,6))
    shap.plots.waterfall(explanation_for_plot, max_display=10, show=False)
    plt.title(f"SHAP Waterfall Plot for {issuer_id} (Predicted: {rating_map.get(predicted_rating_numeric, '?')})")
    plt.tight_layout()
    plt.show()

def nlp_contribution_analysis(
    model_full: xgb.XGBClassifier,
    model_fin_only: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    financial_features: list[str]
):
    """
    Measures how much NLP features improve rating prediction by comparing full vs financials-only models.

    Args:
        model_full (xgb.XGBClassifier): The multimodal model.
        model_fin_only (xgb.XGBClassifier): The financials-only model.
        X_test (pd.DataFrame): Test features for the full model.
        y_test (pd.Series): Actual test labels.
        financial_features (list[str]): List of financial feature names (for the financials-only model).
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

def build_watchlist(mismatches_df: pd.DataFrame, notch_threshold: int = -2, top_n: int = 10) -> pd.DataFrame:
    """
    Builds a prioritized downgrade watchlist for the credit portfolio.

    Args:
        mismatches_df (pd.DataFrame): DataFrame containing mismatch analysis results.
        notch_threshold (int): The notch difference threshold to consider for downgrade risk.
        top_n (int): Number of top downgrade risks to display.

    Returns:
        pd.DataFrame: A DataFrame of issuers on the downgrade watchlist.
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
              f"{row['notch_diff']:+8d}{row.get('mgmt_sentiment', 0):>10.2f}{row.get('risk_language_score', 0):>10.2f}")
    return watchlist

def main():
    """
    Main function to run the credit rating prediction and analysis pipeline.
    This orchestrates data generation, feature engineering, model training,
    evaluation, mismatch analysis, NLP contribution analysis, watchlist generation,
    and SHAP explanations.
    """
    # --- 1. Data Generation ---
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

    # --- 2. Add NLP Sentiment Features ---
    corp_data, nlp_features = add_sentiment_features(corp_data, latent_quality, n_issuers)
    all_features = financial_features + nlp_features

    print(f"\nAdded {len(nlp_features)} NLP sentiment features.")
    print(f"Total features for multimodal model: {len(all_features)}.")
    print("\nFirst 5 rows of data with new sentiment features:")
    print(corp_data.head())

    # --- 3. Train and Evaluate Models ---
    model_full, model_fin_only, X_test, y_test, test_idx, pred_full_numeric, pred_fin_numeric = \
        train_and_evaluate_models(corp_data, financial_features, nlp_features, all_features, RATING_MAP)

    # --- 4. Mismatch Analysis ---
    mismatches_df = mismatch_analysis(model_full, X_test, y_test, corp_data, test_idx, RATING_MAP, RATING_NUM_TO_STR)

    # --- 5. NLP Contribution Analysis ---
    nlp_contribution_analysis(model_full, model_fin_only, X_test, y_test, financial_features)

    # --- 6. Build Downgrade Watchlist ---
    downgrade_watchlist = build_watchlist(mismatches_df, notch_threshold=-2, top_n=10)

    # --- 7. Explain Rating Examples (SHAP) ---
    print("\nINITIATING SHAP EXPLANATION EXAMPLES...")
    explainer_full = shap.TreeExplainer(model_full)

    # Select an example downgrade risk for explanation
    downgrade_idx_candidates = mismatches_df[mismatches_df['signal'] == 'DOWNGRADE RISK'].index
    if not downgrade_idx_candidates.empty:
        # Get the first issuer from the sorted downgrade risks for explanation
        example_downgrade_issuer_id = mismatches_df.loc[downgrade_idx_candidates, 'issuer_id'].iloc[0]
        # X_test rows are already filtered to test_idx in train_and_evaluate_models
        example_downgrade_features = X_test.loc[downgrade_idx_candidates].iloc[0].values
        example_downgrade_implied = mismatches_df.loc[downgrade_idx_candidates, 'implied_numeric'].iloc[0]
        example_downgrade_actual = mismatches_df.loc[downgrade_idx_candidates, 'actual_numeric'].iloc[0]

        explain_rating(explainer_full, example_downgrade_features, all_features,
                       example_downgrade_implied, example_downgrade_actual,
                       example_downgrade_issuer_id, RATING_MAP, financial_features, nlp_features)
    else:
        print("No downgrade risk examples found to explain.")

    # Select an example upgrade candidate for explanation
    upgrade_idx_candidates = mismatches_df[mismatches_df['signal'] == 'UPGRADE CANDIDATE'].index
    if not upgrade_idx_candidates.empty:
        # Get the first issuer from the sorted upgrade candidates for explanation
        example_upgrade_issuer_id = mismatches_df.loc[upgrade_idx_candidates, 'issuer_id'].iloc[0]
        example_upgrade_features = X_test.loc[upgrade_idx_candidates].iloc[0].values
        example_upgrade_implied = mismatches_df.loc[upgrade_idx_candidates, 'implied_numeric'].iloc[0]
        example_upgrade_actual = mismatches_df.loc[upgrade_idx_candidates, 'actual_numeric'].iloc[0]

        explain_rating(explainer_full, example_upgrade_features, all_features,
                       example_upgrade_implied, example_upgrade_actual,
                       example_upgrade_issuer_id, RATING_MAP, financial_features, nlp_features)
    else:
        print("No upgrade candidate examples found to explain.")

if __name__ == "__main__":
    main()
