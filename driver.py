from multiprocessing import Pool, cpu_count
from tune import run_simulation_with_tuning
import pandas as pd
import time
from train_models import train_random_forest, train_gradient_boost
import joblib
import argparse


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
    parser = argparse.ArgumentParser(description="Spring Constant Predictor")

    # Arguments
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--successes", type=int, default=40, help="Target successes per worker")
    parser.add_argument("--models", 
                        nargs="+", 
                        choices=["rf", "gb"], 
                        default=["rf", "gb"], 
                        help="Which model(s) to train: rf, gb, or both (e.g. --models rf gb)")
    parser.add_argument("--output", type=str, default="parallel_successes", help="Output CSV filename, you do not have to include the file extension.")

    args = parser.parse_args()

    start_time = time.time()
    try:
        data_set = generate_data(workers=args.workers, success_count=args.successes)
        data_set.to_csv(f"{args.output}.csv", index=False)
        elapsed = time.time() - start_time
        if elapsed < 60:
            print(f"⏱️ Data generation completed in {elapsed:.2f} seconds")
        else:
            mins, secs = divmod(elapsed, 60)
            print(f"⏱️ Data generation completed in {int(mins)}m {int(secs)}s")

        print("Data saved to 'parallel_successes.csv'")
        print("Training models...")

        if 'rf' in args.models:
            rf_model = train_random_forest(data_set)
            joblib.dump(rf_model, "rf_model.pkl")
            print("Saved model to 'rf_model.pkl'")

        if 'gb' in args.models:
            gb_model = train_gradient_boost(data_set)
            joblib.dump(gb_model, "gb_model.pkl")
            print("Saved model to 'gb_model.pkl'")

    except KeyboardInterrupt:
        print("Interrupted by user, exiting main program...")