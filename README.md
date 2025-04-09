# ModellingFinalroject

To run the program:

1. Simply install the required libraries via -> pip install -r requirements.txt
2. Run driver.py -> python driver.py
3. Optional flags -> python driver.py --workers <number_of_workers> --successes <number_of_sample_successes> --models <training_model> --output <output_filename> --save-figures --hide-figures
4. If no flag values are passed the program will run with its default values.
5. Example run: python driver.py --workers 5 --successes 40 --output generated_points --save-figures --hide-figures

**Please note that the length of the data set is -> dataset_length = workers \* successes**

See the help messge for more details:
usage: driver.py [-h] [--workers WORKERS] [--successes SUCCESSES] [--models {rf,gb} [{rf,gb} ...]] [--output OUTPUT]
[--save-figures] [---hide-figures]

Spring Constant Predictor

options:
-h, --help show this help message and exit
--workers WORKERS Number of parallel workers
--successes SUCCESSES
Target successes per worker
--models {rf,gb} [{rf,gb} ...]
Which model(s) to train: rf, gb, or both (e.g. --models rf gb). Will run both by default.
--output OUTPUT Output CSV filename, you do not have to include the file extension.
--save-figures Save all figures to disk
---hide-figures Hide all figures
