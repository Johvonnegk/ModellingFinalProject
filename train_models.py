from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from tune import run_success_based_sampling_simulation, run_monte_carlo_simulation
from simulate import L
import matplotlib.pyplot as plt
from tabulate import tabulate

# These are the columns that will be used as features for the model.
# The original columns are mass, damping, gravity, and angle_deg.
# The engineered columns are damping_ratio, omega_n, and y0.
feature_cols = [
    "mass", "damping", "gravity", "angle_deg",  # original
    "damping_ratio", "omega_n", "y0",  # engineered
]

# This function will be used to calculate the number of estimators for the Random Forest and Gradient Boosting models.
# The number of estimators is based on the length of the data and the type of model.
def get_estimators(data_len, model_type):
    if model_type == "rf":
        # 1 tree per 5–10 samples, capped at 300 ~ Random Forest
        estimators = min(max(data_len // 5, 50), 300)
        print(f"Running Random Forest with {estimators} estimators")
        return min(max(data_len // 5, 50), 300)
    
    elif model_type == "gb":
        # 1 tree per 3–4 samples, capped at 1000 ~ Gradient Boosting
        estimators = min(max(data_len // 3, 100), 1000)
        print(f"Running Gradient Boost with {estimators} estimators")
        return estimators

# This function will be used to collect the data needed to train the model.
def collect_data(df=None):
    
    # Will generate its own data if no DataFrame is provided.
    if df is None:
        print("DataFrame not found, cannot train model.")
        return None



    # Filter only the successful trials
    df_success = df[df['success']].dropna()

    # Feature engineering, gathering more fields from the simulation to train the model.
    df_success["damping_ratio"] = df_success["damping"] / (2 * np.sqrt(df_success["best_k"] * df_success["mass"]))
    df_success["omega_n"] = np.sqrt(df_success["best_k"] / df_success["mass"])
    df_success["theta_rad"] = np.radians(df_success["angle_deg"])
    df_success["y0"] = L * df_success["theta_rad"]

    # Rows and columns to be used for training the model.
    x = df_success[feature_cols]
    y = df_success['best_k']

    #  Splitting the data into training and testing sets, and returning them
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test, x, y

# Function used to train the model using Random Forest 
def train_random_forest(df=None, mode=None, save=False, show=True): 
    
    # Collect the data necessary to train the model.
    collected_data = collect_data(df)
    
    if collected_data is None:
        print("No data to train the model.")
        return None
    
    x_train, x_test, y_train, y_test, x, y = collected_data   
    
    # Training the model using Random Forest.
    rf_model = RandomForestRegressor(n_estimators=get_estimators(len(x_train), 'rf'), random_state=42)
    rf_model.fit(x_train, y_train)

    # Evaluate
    y_pred = rf_model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mae, rmse = calculate_mae_rmse(y_test, y_pred)

    # Calculate feature importances/weights (how important each feature is for the model).
    importances = rf_model.feature_importances_
    importances_percent = importances / importances.sum() * 100
    xi = sorted(zip(x.columns, importances_percent), key=lambda i: i[1], reverse=True)

    # Prints the results of the model evaluation in a nice tabular output
    # Including R² score, MAE, RMSE, and feature importances.
    print(f"{mode} ~ Random Forest:")
    print(tabulate([
        ["R² Score", f"{r2:.3f}"],
        ["R² (as %)", f"{r2 * 100:.1f} %"],
        ["MAE", f"{mae:.2f} N/m"],
        ["RMSE", f"{rmse:.2f} N/m"],
        ["Number of successes", f"{len(x)} samples"],
        ["Total number of samples", f"{len(df)} samples"],
        ["Success rate", f"{len(x) / len(df) * 100:.1f} %"],
    ], headers=["Metric", "Random Forest"], tablefmt="fancy_grid"))
    print(tabulate(xi, headers=["Feature", "Importance %"], tablefmt="fancy_grid", floatfmt=".3f"))
    print("")
    
    # Display the feature importances, predicted vs actual values, and distribution of errors.
    # Returns the trained model.
    display_importances(xi, mode, model="Random Forest", save=save, show=show)
    display_predicted_vs_actual(y_test, y_pred, y, mode, model="Random Forest", save=save, show=show)
    display_distribution_err(y_test, y_pred, mode, model="Random Forest", save=save, show=show)
    return rf_model

# Function used to train the model using Gradient Boosting
def train_gradient_boost(df=None, mode="", save=False, show=True):
    
    # Collect the data necessary to train the model.
    collected_data = collect_data(df)
    
    if collected_data is None:
        print("No data to train the model.")
        return None
    
    x_train, x_test, y_train, y_test, x, y = collected_data 

    # Training the model using Gradient Boosting.
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

    # Feature importances/weights (how important each feature is for the model).
    importances = gb_model.feature_importances_
    importances_percent = importances / importances.sum() * 100
    xi = sorted(zip(x.columns, importances_percent), key=lambda i: i[1], reverse=True)

    # Prints the results of the model evaluation in a nice tabular output
    # Including R² score, MAE, RMSE, and feature importances.
    print(f"{mode} ~ Gradient Boost:")
    print(tabulate([
        ["R² Score", f"{r2:.3f}"],
        ["R² (as %)", f"{r2 * 100:.1f} %"],
        ["MAE", f"{mae:.2f} N/m"],
        ["RMSE", f"{rmse:.2f} N/m"],
        ["Number of successes", f"{len(x)} samples"],
        ["Total number of samples", f"{len(df)} samples"],
        ["Success rate", f"{len(x) / len(df) * 100:.1f} %"],
    ], headers=["Metric", "Gradient Boost"], tablefmt="fancy_grid"))
    print(tabulate(xi, headers=["Feature", "Importance %"], tablefmt="fancy_grid", floatfmt=".3f"))
    print("")
    
    # Display the feature importances, predicted vs actual values, and distribution of errors.
    # Returns the trained model.
    display_importances(xi, mode, model="Gradient Boosting", save=save, show=show)
    display_predicted_vs_actual(y_test, y_pred, y, mode, model="Gradient Boosting", save=save, show=show)
    display_distribution_err(y_test, y_pred, mode, model="Gradient Boosting", save=save, show=show)
    return gb_model

# Function used to calculate the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
def calculate_mae_rmse(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, rmse

# Function used to display the predicted vs actual values of the model.
# It will plot a scatter plot of the predicted vs actual values.
# It will also plot a red dashed line representing the line of best fit.
def display_predicted_vs_actual(y_test, y_pred, y, mode, model="Model", save=False, show=True):
    plt.scatter(y_test, y_pred, color='dodgerblue', alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Prediction')
    plt.xlabel("Actual Spring k")
    plt.ylabel("Predicted Spring k")
    plt.title(f"{mode} ~ {model}: Predicted vs Actual spring_k")
    plt.grid(True)
    if save:
        plt.savefig(f"{mode}_{model}_predicted_vs_actual.png")
    
    if show:
        plt.show()

# Function used to display the distribution of errors.
# It will plot a histogram of the errors (predicted - actual values).
def display_distribution_err(y_test, y_pred, mode, model="Model", save=False, show=True):
    errors = y_pred - y_test
    plt.hist(errors, bins=20)
    plt.title(f"{mode} ~ {model}: Prediction Error Distribution")
    plt.xlabel("Prediction Error", color='salmon')
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(f"{mode}_{model}_prediction_error.png")
    
    if show:
        plt.show()

# Function used to display the feature importances.
# It will plot a bar chart of the feature importances.
def display_importances(xi, mode, model="Model", save=False, show=True):
    features, importances = zip(*xi)
    plt.barh(features, importances)
    plt.xlabel("Importance")
    plt.title(f"{mode} ~ {model}: Feature Importances")
    plt.tight_layout()

    if save:
        plt.savefig(f"{mode}_{model}_feature_importances.png")
    if show:
        plt.show()

