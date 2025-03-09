from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, KFold

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=5, random_state=42, n_jobs=-1):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs
        )

    def cross_validate(self, X_train, y_train, cv=10):
        cv_results = cross_validate(
            self.model, X_train, y_train, cv=cv, scoring='roc_auc',
            return_train_score=True
        )
        return cv_results

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]