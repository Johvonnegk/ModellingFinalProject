from multiprocessing import Pool, cpu_count
from tune import run_simulation_with_tuning
import pandas as pd
import time
from train_models import train_random_forest, train_gradient_boost
import joblib


def generate_data(workers=1, success_count=40):
    try:
        print("Generating data with parallel processing...")
        with Pool(workers) as pool:
            args = [(success_count, i+1) for i in range(workers)]

            results = pool.starmap(run_simulation_with_tuning, args)

        df = pd.concat(results, ignore_index=True)
        print("\nCollected data from all workers. Data generation successful")
        return df
    except KeyboardInterrupt as e:
        print("\nData generation interrupted. Exiting...")
        raise e


if __name__ == "__main__":
    start_time = time.time()
    try:
        data_set = generate_data(workers=5, success_count=50)
        data_set.to_csv("parallel_successes.csv", index=False)
        print(f"Data generation completed in {time.time() - start_time:.2f} seconds")
        print("Data saved to 'parallel_successes.csv'")
        print("Training models...")
        rf_model = train_random_forest(data_set, 200)
        gb_model = train_gradient_boost(data_set, 200)
        joblib.dump(rf_model, "rf_model.pkl")
        joblib.dump(gb_model, "gb_model.pkl")
        print("Saved models to 'rf_model.pkl' and 'gb_model.pkl'")

    except KeyboardInterrupt:
        print("Interrupted by user, exiting main program...")