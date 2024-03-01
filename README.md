# Money Laundering Detection in SAML-D Synthetic Dataset
## Project Overview

This project focuses on leveraging transaction features and graph analytics to detect money laundering transactions in the SAML-D synthetic dataset. By employing advanced techniques such as feature engineering, ensemble learning, and local interpretable model-agnostic explanations (LIME), I aim to develop a robust system for identifying suspicious transactions.

## Project Highlights

1. Feature Engineering: Graph features of sender and receiver accounts, along with risk calculations based on bank locations and payment types, were extracted to enhance the dataset.

2. Imbalanced Learning: Employed an undersampled ensemble approach to address the class imbalance problem commonly encountered in money laundering detection tasks.

3. Model Training: Utilized the XGBoost Voting Classifier to train the model, achieving an impressive F1 Score of 99%.

4. Interpretability with LIME: Leveraged LIME explainer to gain insights into feature contributions, aiding in the understanding of model predictions.

5. Development Module: Designed a development module to generate reports for AML (Anti-Money Laundering) Compliance Officers, providing detailed explanations for flagged transactions.
