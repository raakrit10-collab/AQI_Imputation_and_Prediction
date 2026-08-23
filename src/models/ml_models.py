from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor, XGBClassifier

# Return dictionary of baseline and ensemble regression models
def get_regression_models(seed=42):
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=seed),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=7),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=-1, verbosity=0),
        "MLP Regressor": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed),
    }

# Return dictionary of classification models for AQI category prediction
def get_classification_models(seed=42):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, max_iter=200, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=-1, verbosity=0, use_label_encoder=False, eval_metric="mlogloss"),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed),
    }
