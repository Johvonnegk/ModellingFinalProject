from multiprocessing import Pool, cpu_count
from tune import run_monte_carlo_simulation, run_success_based_sampling_simulation
import pandas as pd
import time
from train_models import train_random_forest, train_gradient_boost
import joblib
import argparse
from tabulate import tabulate

# This function is responsible for generating the data using parallel processing.
# It will create a pool of workers and run the simulation for each worker.
# Each worker will run the simulation for a specified number of successful trials.

def print_elapsed_time(start_time, end_time):
    elapsed = end_time - start_time
    if elapsed < 60:
        print(f"Elapsed time: {elapsed:.2f} seconds")
    else:
        mins, secs = divmod(elapsed, 60)
        print(f"Elapsed time: {int(mins)}m {int(secs)}s")
        
def generate_data(mode="success-based", workers=1, trials=100):
    try:
        print("Generating data with parallel processing...")
        with Pool(workers) as pool:
            args = [(trials, i+1) for i in range(workers)]
            
            if mode == "success-based":
                print(f"Running Success-Based sampling simulation with {workers} workers")
                results = pool.starmap(run_success_based_sampling_simulation, args)
            elif mode == "monte-carlo":
                print(f"Running Monte Carlo simulation with {workers} workers")
                results = pool.starmap(run_monte_carlo_simulation, args)
            else:
                raise ValueError("Invalid mode. Choose 'success' or 'monte_carlo'.")
            
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
    parser.add_argument("--count", type=int, default=100, 
                        help="Trials per worker (or successes per worker)") # Specifies the number of trials each worker will run
    parser.add_argument("--mode", choices=["success-based", "monte-carlo"], default="success-based", help="Data generation mode")
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
        ["Mode", args.mode],
        ["Workers", args.workers],
        ["Trials/Successes per worker", args.count],
        ["Total number of data points", args.workers * args.count],
        ["Models to train", ", ".join(args.models)],
        ["Output filename", f"{args.output}.csv"],
        ["Save figures", args.save_figures],
        ["Hide figures", not args.hide_figures]
    ], headers=["Parameter", "Value"], tablefmt="fancy_grid"))

    
    try:
        data_sets = []
        output_file = f"{args.mode}_{args.output}.csv"
        # Timer to measure the time taken for data generation.
        start_time = time.time()
        
        # Generates the data using the specified number of workers and successes. 
        data_set = generate_data(workers=args.workers, trials=args.count, mode=args.mode)
        data_set.to_csv(output_file, index=False)
        data_sets.append(data_set)
        print(f"Time spent generating data for mode '{args.mode}': {print_elapsed_time(start_time, time.time())}") # Prints elapsed time
        print(f"Data saved to '{output_file}'")
        print("Training models...\n")

        # Train the models based on the specified arguments.
        # The models will be trained using the generated data.
        for set in data_sets:
            capatalized_str = "-".join(word.capitalize() for word in args.mode.split("-"))
            if 'rf' in args.models:
                rf_model = train_random_forest(set, capatalized_str, args.save_figures, args.hide_figures)
                if rf_model:
                    joblib.dump(rf_model, "rf_model.pkl")
                    print("Saved model to 'rf_model.pkl'\n")

            if 'gb' in args.models:
                gb_model = train_gradient_boost(set, capatalized_str, args.save_figures, args.hide_figures)
                if gb_model:
                    joblib.dump(gb_model, "gb_model.pkl")
                    print("Saved model to 'gb_model.pkl'\n")
        
        print("Model training completed.")

    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting main program...")