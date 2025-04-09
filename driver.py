from multiprocessing import Pool, cpu_count
from tune import run_simulation_with_tuning
import pandas as pd
import time
from train_models import train_random_forest, train_gradient_boost
import joblib
import argparse
from tabulate import tabulate

# This function is responsible for generating the data using parallel processing.
# It will create a pool of workers and run the simulation for each worker.
# Each worker will run the simulation for a specified number of successful trials.
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

# Main function to run the script.
# It will parse the arguments, generate the data, train the models, and save the results.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spring Constant Predictor")

    # CLI Arguments
    parser.add_argument("--workers", type=int, default=5, 
                        help="Number of parallel workers") # Specifies the number of workers each given their own process
    parser.add_argument("--successes", type=int, default=40, 
                        help="Target successes per worker"
                        ) # Specifies the number of successes each worker needs to achieve
    parser.add_argument("--models", 
                        nargs="+", 
                        choices=["rf", "gb"], 
                        default=["rf", "gb"], 
                        help="Which model(s) to train: rf, gb, or both (e.g. --models rf gb). Will run both by default.") # Model training options
    parser.add_argument("--output", type=str, 
                        default="parallel_successes", 
                        help="Output CSV filename, you do not have to include the file extension.") # Output filename of generated data
    parser.add_argument("--save-figures", action="store_true", help="Save all figures to disk") # Save figures option
    parser.add_argument("--hide-figures", action="store_false", help="Hide all figures") # Hide figures option
    args = parser.parse_args()

    print("Running with the following parameters:")
    
    # Prints the input parameters in a table format for better readability.
    print(tabulate([
        ["Workers", args.workers],
        ["Successes per worker", args.successes],
        ["Total number of data points", args.workers * args.successes],
        ["Models to train", ", ".join(args.models)],
        ["Output filename", f"{args.output}.csv"],
        ["Save figures", args.save_figures],
        ["Hide figures", args.hide_figures]
    ], headers=["Parameter", "Value"], tablefmt="fancy_grid"))

    # Timer to measure the time taken for data generation.
    start_time = time.time()
    try:
        
        # Generates the data using the specified number of workers and successes. 
        data_set = generate_data(workers=args.workers, success_count=args.successes)
        data_set.to_csv(f"{args.output}.csv", index=False)
           
        # Prints elapsed time
        elapsed = time.time() - start_time
        if elapsed < 60:
            print(f"Data generation completed in {elapsed:.2f} seconds")
        else:
            mins, secs = divmod(elapsed, 60)
            print(f"Data generation completed in {int(mins)}m {int(secs)}s")

        print("Data saved to 'parallel_successes.csv'")
        print("Training models...\n")

        # Train the models based on the specified arguments.
        # The models will be trained using the generated data.
        if 'rf' in args.models:
            rf_model = train_random_forest(data_set, args.save_figures, args.hide_figures)
            joblib.dump(rf_model, "rf_model.pkl")
            print("Saved model to 'rf_model.pkl'\n")

        if 'gb' in args.models:
            gb_model = train_gradient_boost(data_set, args.save_figures, args.hide_figures)
            joblib.dump(gb_model, "gb_model.pkl")
            print("Saved model to 'gb_model.pkl'\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting main program...")