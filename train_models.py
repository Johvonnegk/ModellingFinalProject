from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from tune import run_simulation_with_tuning
from simulate import L
import matplotlib.pyplot as plt
from tabulate import tabulate

feature_cols = [
    "mass", "damping", "gravity", "angle_deg",  # original
    "damping_ratio", "omega_n", "y0",  # engineered
]

def get_estimators(data_len, model_type):
    if model_type == "rf":
        # 1 tree per 5–10 samples, capped at 300
        estimators = min(max(data_len // 5, 50), 300)
        print(f"Running Random Forest with {estimators} estimators")
        return min(max(data_len // 5, 50), 300)
    elif model_type == "gb":
        # 1 tree per 3–4 samples, capped at 1000
        estimators = min(max(data_len // 3, 100), 1000)
        print(f"Running Gradient Boost with {estimators} estimators")
        return estimators


def collect_data(df=None):
    if df is None:
        print("DataFrame not found, running simulation 100 times")
        df = run_simulation_with_tuning(100)



    # Filter successful trials
    df_success = df[df['success']].dropna()

    df_success["damping_ratio"] = df_success["damping"] / (2 * np.sqrt(df_success["best_k"] * df_success["mass"]))
    df_success["omega_n"] = np.sqrt(df_success["best_k"] / df_success["mass"])
    df_success["theta_rad"] = np.radians(df_success["angle_deg"])
    df_success["y0"] = L * df_success["theta_rad"]

    # Features and target
    x = df_success[feature_cols]
    y = df_success['best_k']

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test, x, y

def train_random_forest(df=None, save=False, show=True):    
    x_train, x_test, y_train, y_test, x, y = collect_data(df)
    rf_model = RandomForestRegressor(n_estimators=get_estimators(len(x_train), 'rf'), random_state=42)
    rf_model.fit(x_train, y_train)

    # Evaluate
    y_pred = rf_model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mae, rmse = calculate_mae_rmse(y_test, y_pred)

    # Feature importances
    importances = rf_model.feature_importances_
    importances_percent = importances / importances.sum() * 100
    xi = sorted(zip(x.columns, importances_percent), key=lambda i: i[1], reverse=True)

    print(tabulate([
        ["R² Score", f"{r2:.3f}"],
        ["R² (as %)", f"{r2 * 100:.1f} %"],
        ["MAE", f"{mae:.2f} N/m"],
        ["RMSE", f"{rmse:.2f} N/m"]
    ], headers=["Metric", "Random Forest"], tablefmt="fancy_grid"))
    print(tabulate(xi, headers=["Feature", "Importance %"], tablefmt="fancy_grid", floatfmt=".3f"))
    print("")
    display_importances(xi, model="Random Forest", save=save, show=show)
    display_predicted_vs_actual(y_test, y_pred, y, model="Random Forest", save=save, show=show)
    display_distribution_err(y_test, y_pred, model="Random Forest", save=save, show=show)
    return rf_model

def train_gradient_boost(df=None, save=False, show=True):
    x_train, x_test, y_train, y_test, x, y = collect_data(df)

    # Train a random forest
    gb_model = GradientBoostingRegressor(
        n_estimators=get_estimators(len(x_train), 'gb'),
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    gb_model.fit(x_train, y_train)

    # Evaluate
    y_pred = gb_model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mae, rmse = calculate_mae_rmse(y_test, y_pred)

    # Feature importances
    importances = gb_model.feature_importances_
    importances_percent = importances / importances.sum() * 100
    xi = sorted(zip(x.columns, importances_percent), key=lambda i: i[1], reverse=True)

    print(tabulate([
        ["R² Score", f"{r2:.3f}"],
        ["R² (as %)", f"{r2 * 100:.1f} %"],
        ["MAE", f"{mae:.2f} N/m"],
        ["RMSE", f"{rmse:.2f} N/m"]
    ], headers=["Metric", "Gradient Boost"], tablefmt="fancy_grid"))
    print(tabulate(xi, headers=["Feature", "Importance %"], tablefmt="fancy_grid", floatfmt=".3f"))
    print("")
    display_importances(xi, model="Gradient Boosting", save=save, show=show)
    display_predicted_vs_actual(y_test, y_pred, y, model="Gradient Boosting", save=save, show=show)
    display_distribution_err(y_test, y_pred, model="Gradient Boosting", save=save, show=show)
    return gb_model

def calculate_mae_rmse(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, rmse

def display_predicted_vs_actual(y_test, y_pred, y, model="Model", save=False, show=True):
    plt.scatter(y_test, y_pred, color='dodgerblue', alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Prediction')
    plt.xlabel("Actual Spring k")
    plt.ylabel("Predicted Spring k")
    plt.title(f"{model}: Predicted vs Actual spring_k")
    plt.grid(True)
    if save:
        plt.savefig(f"{model}_predicted_vs_actual.png")
    
    if show:
        plt.show()

def display_distribution_err(y_test, y_pred, model="Model", save=False, show=True):
    errors = y_pred - y_test
    plt.hist(errors, bins=20)
    plt.title(f"{model}: Prediction Error Distribution")
    plt.xlabel("Prediction Error", color='salmon')
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(f"{model}_prediction_error.png")
    
    if show:
        plt.show()

def display_importances(xi, model="Model", save=False, show=True):
    features, importances = zip(*xi)
    plt.barh(features, importances)
    plt.xlabel("Importance")
    plt.title(f"{model}: Feature Importances")
    plt.tight_layout()

    if save:
        plt.savefig(f"{model}_feature_importances.png")
    if show:
        plt.show()

