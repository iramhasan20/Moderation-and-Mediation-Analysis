# Moderation-and-Mediation-Analysis

This repository contains code and analysis exploring the moderation and mediation relationships governing the link between social and financial performance. The moderation analysis is conducted using panel data random effects regression with interaction terms, while the mediation analysis measures direct and indirect effects, identifying key pathways between dependent and independent variables using the Baron and Kenny (1986) approach.

The code is structured under the following sections for both moderation and mediation analyses:

1. Descriptive Statistics
2. Model Diagnostic Checks
3. Moderation Analysis
4. Mediation Analysis
5. Robustness Assessment

The following Python libraries are utilized in this analysis:

1. pandas: Handles data manipulation, including reading datasets, renaming variables, and computing correlation matrices.
2. numpy: Supports numerical operations, including statistical computations for hypothesis testing.
3. statsmodels.api: Provides statistical modeling and regression analysis.
4. statsmodels.formula.api: Enables formula-based regression modeling.
5. statsmodels.stats.outliers_influence: Computes Variance Inflation Factor (VIF) for multicollinearity checks.
6. linearmodels.panel: Implements panel data econometric models, including Random Effects regression.
7. scipy.stats: Facilitates statistical hypothesis testing, including the Sobel test for mediation analysis.
