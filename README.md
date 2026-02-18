# QuLab: Lab 53: Bond Rating Prediction - A Credit Analyst's Workflow with ML and NLP

## Project Title

**QuLab: Lab 53: Bond Rating Prediction - A Credit Analyst's Workflow with ML and NLP**

## Description

This Streamlit application, "QuLab: Lab 53: Bond Rating Prediction," simulates a real-world workflow for a seasoned Credit Analyst (Persona: Sri Krishnamurthy, CFA) at a fixed income investment firm. The objective is to leverage machine learning and natural language processing (NLP) to rapidly screen hundreds of corporate bonds for creditworthiness, identify potential rating changes (upgrades or downgrades), and provide a competitive edge in portfolio management.

The application guides users through a comprehensive, step-by-step process, demonstrating how to integrate quantitative financial data with qualitative management commentary for robust credit risk assessment. It showcases how a data-driven approach can augment traditional credit analysis, moving from raw data to actionable insights.

## Features

The application is structured into five sequential modules, each addressing a critical aspect of the credit analysis workflow:

1.  **Data Acquisition & Feature Engineering:**
    *   **Synthetic Data Generation:** Generates a diverse synthetic dataset of corporate financial metrics (e.g., interest coverage, debt-to-EBITDA, profitability, liquidity, size) based on a latent "quality" factor.
    *   **NLP Sentiment Integration:** Enhances the dataset by adding simulated NLP-derived sentiment features (e.g., `mgmt_sentiment`, `guidance_sentiment`, `risk_language_score`) from earnings call transcripts.
    *   **Multimodal Feature Fusion:** Demonstrates the combination of numerical financial data with qualitative text-based insights.

2.  **Model Training & Evaluation:**
    *   **Ordinal Classification:** Trains two XGBoost models: one using only financial features (baseline) and another using the full multimodal feature set. The objective is tailored for ordinal credit rating prediction.
    *   **Performance Metrics:** Evaluates models using standard accuracy and, crucially, "within-1-notch accuracy," a more relevant metric for credit ratings.
    *   **Visual Performance:** Presents confusion matrices to visualize classification performance across credit rating tiers.

3.  **Rating Mismatch Analysis:**
    *   **Implied vs. Actual:** Compares model-implied ratings against actual agency ratings to identify discrepancies.
    *   **Categorization:** Classifies issuers into actionable categories: "Downgrade Risk," "Upgrade Candidate," "Watch (1 notch)," and "Aligned."
    *   **Actionable Insights:** Identifies and lists top downgrade risks and upgrade candidates, providing a focus for deeper human analysis.

4.  **Explainable AI: Rating Rationales:**
    *   **SHAP Value Explanations:** Utilizes SHAP (SHapley Additive exPlanations) values to explain individual rating predictions for selected issuers.
    *   **Feature Contribution Analysis:** Clearly shows which specific financial ratios and NLP sentiment features most strongly contributed to an issuer's implied rating.
    *   **Transparency:** Provides clear, interpretable rationales for model predictions, essential for regulatory compliance and stakeholder communication.

5.  **NLP Contribution & Downgrade Watchlist:**
    *   **Quantifying NLP Value:** Quantifies the incremental net contribution of NLP sentiment features by comparing predictions improved versus worsened by their inclusion.
    *   **Prioritized Watchlist:** Generates a prioritized "Downgrade Watchlist" for high-risk issuers, complete with relevant financial and sentiment data, serving as a concrete deliverable for portfolio managers.
    *   **Practitioner Warning:** Includes a critical warning regarding the appropriate use and labeling of model-implied ratings in a professional context (CFA Standard V(B)).

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Ensure you have Python installed (version 3.8+ recommended).

*   Python 3.8+
*   pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/quslab53-bond-rating-prediction.git
    cd quslab53-bond-rating-prediction
    ```
    *(Note: Replace `your-username/quslab53-bond-rating-prediction.git` with the actual repository URL once available.)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The application relies on several Python libraries. Install them using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

    *(Hypothetical `requirements.txt` content, based on `import` statements in `app.py` and inferred ML libraries):*
    ```
    streamlit>=1.0.0
    pandas>=1.0.0
    numpy>=1.0.0
    matplotlib>=3.0.0
    seaborn>=0.11.0
    shap>=0.39.0
    xgboost>=1.4.0 # Implied by shap.TreeExplainer and model type
    scikit-learn>=1.0.0 # For train_test_split etc., likely used in source.py
    ```

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  **Access the application:**
    Your web browser should automatically open to `http://localhost:8501` (or a similar address). If not, navigate there manually.

3.  **Navigate the workflow:**
    *   Use the **sidebar navigation** to move through the different modules of the lab project.
    *   Follow the instructions and click the buttons on each page to progress through data generation, model training, analysis, and explanation. The lab is designed to be completed sequentially from "Home" through "5. NLP Contribution & Downgrade Watchlist."

## Project Structure

```
.
├── app.py                     # Main Streamlit application file
├── source.py                  # Contains all helper functions for data generation, model training, analysis, etc.
├── requirements.txt           # List of Python dependencies
├── README.md                  # This file
└── .gitignore                 # Specifies intentionally untracked files to ignore
```

*   `app.py`: This is the entry point for the Streamlit application. It handles the UI layout, session state management, and calls functions from `source.py` to perform the core logic.
*   `source.py`: This file encapsulates all the backend logic, including:
    *   `generate_corporate_data()`: Simulates financial data.
    *   `add_sentiment_features()`: Adds NLP features.
    *   `train_and_evaluate_models()`: Trains XGBoost models and evaluates them.
    *   `mismatch_analysis()`: Performs the implied vs. actual rating comparison.
    *   `explain_rating()`: Uses SHAP to explain individual predictions.
    *   `build_watchlist()`: Generates the downgrade watchlist.
    *   `RATING_MAP`, `RATING_NUM_TO_STR`: Global constants for rating conversions.

## Technology Stack

*   **Application Framework:** [Streamlit](https://streamlit.io/)
*   **Programming Language:** [Python 3.x](https://www.python.org/)
*   **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Machine Learning:** [XGBoost](https://xgboost.readthedocs.io/en/stable/) (for classification), [scikit-learn](https://scikit-learn.org/stable/) (for data splitting and preprocessing)
*   **Explainable AI (XAI):** [SHAP](https://shap.readthedocs.io/en/latest/) (SHapley Additive exPlanations)
*   **Data Visualization:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

1.  **Fork** the repository.
2.  **Create a new branch** for your feature or fix.
3.  **Implement** your changes.
4.  **Submit a Pull Request** with a clear description of your contributions.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (or add one if not present).

## Contact

This project is part of QuLab (QuantUniversity Labs). For inquiries or further information, please contact:

*   **QuantUniversity:** [https://www.quantuniversity.com/](https://www.quantuniversity.com/)
*   **Sri Krishnamurthy, CFA:** [sri@quantuniversity.com](mailto:sri@quantuniversity.com) (or relevant contact for the lab project)
*   **GitHub Issues:** For technical questions or bug reports related to this specific lab, please use the [Issues section](https://github.com/your-username/quslab53-bond-rating-prediction/issues) of the GitHub repository.