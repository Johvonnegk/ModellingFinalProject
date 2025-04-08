from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from tune import run_simulation_with_tuning
from simulate import L


def train_random_forest(df=None, estimators=100):
    if df is None:
        print("DataFrame not found, running simulation 100 times")
        df = run_simulation_with_tuning(100)

    feature_cols = [
        "mass", "damping", "gravity", "angle_deg",  # original
        "damping_ratio", "omega_n", "y0",  # engineered
    ]

    # Filter successful trials
    df_success = df[df['success']].dropna()

    df_success["damping_ratio"] = df_success["damping"] / (2 * np.sqrt(df_success["best_k"] * df_success["mass"]))
    df_success["omega_n"] = np.sqrt(df_success["best_k"] / df_success["mass"])
    df_success["theta_rad"] = np.radians(df_success["angle_deg"])
    df_success["y0"] = L * df_success["theta_rad"]

    # Features and target
    X = df_success[feature_cols]
    y = df_success['best_k']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest
    rf_model = RandomForestRegressor(n_estimators=estimators, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate
    y_pred = rf_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Random Forest R² score: {r2:.3f}")

    # Feature importances
    importances = rf_model.feature_importances_
    for feature, importance in zip(X.columns, importances):
        print(f"{feature}: {importance:.3f}")

    print("")
    return rf_model

# d = run_balanced_success_sampler(target_successes=40, min_per_bin=3, max_trials=2000)
# d = run_simulation_with_tuning(100)
# train_random_forest(d, 100)
# Optional: Plot predicted vs actual
# plt.scatter(y_test, y_pred, alpha=0.7)
# plt.xlabel("Actual spring_k")
# plt.ylabel("Predicted spring_k")
# plt.title("Random Forest: Predicted vs Actual spring_k")
# plt.grid(True)
# plt.show()
