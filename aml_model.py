import xgboost as xgb
import numpy as np

class XGBoostVotingClassifier():
    def __init__(self, n_estimators=3):
        self.n_estimators = n_estimators
        self.estimators_ = []
        self.is_fitted = False

    def fit(self, X_list, y_list, verbose=False):
        for i in range(self.n_estimators):
            X = X_list[i]
            y = y_list[i]
            model = xgb.XGBClassifier()
            model.fit(X, y)
            self.estimators_.append(model)

            if verbose:
                print(f"Model {i+1} is fitted.")

        self.is_fitted = True

    def predict(self, X, voting='hard'):
        if self.is_fitted:
            predictions = []
            for model in self.estimators_:
                if voting == 'hard':
                    predictions.append(model.predict(X))
                elif voting == 'soft':
                    predictions.append(model.predict_proba(X))
                else:
                    raise ValueError("Invalid voting type. Please choose 'hard' or 'soft'.")

            if voting == 'hard':
                predictions = np.array(predictions)
                ensemble_predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)
            elif voting == 'soft':
                predictions = np.mean(predictions, axis=0)
                ensemble_predictions = np.argmax(predictions, axis=1)

            return ensemble_predictions
        else:
            raise ValueError("Model is not fitted yet. Call fit method first.")
            
    def predict_proba(self, X):
        if self.is_fitted:
            # Collect probabilities from each model
            probabilities = []
            for model in self.estimators_:
                probabilities.append(model.predict_proba(X))
            
            # Average probabilities across all models
            avg_probabilities = np.mean(probabilities, axis=0)
            return avg_probabilities
        else:
            raise ValueError("Model is not fitted yet. Call fit method first.")