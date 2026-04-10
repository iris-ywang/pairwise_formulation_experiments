import logging

import pandas as pd
import numpy as np
import os
import warnings

from pairwise_formulation.pa_basics.import_data import dataset, kfold_splits
from run_experiments.run_utils import run
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

warnings.filterwarnings("ignore")

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

if __name__ == '__main__':
    root_dir = os.getcwd()
    chembl_info = pd.read_csv(
        root_dir + "/data/boolean_chembl_datasets_info.csv"
    ).sort_values(by=["N(sample)"], ascending=False)

    output_dir = root_dir + "/output/boolean_chembl/"
    results_filename = "boolean_chembl_rf_trueskil_plz_delete.npy"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        existing_results = np.load(output_dir + results_filename)
        existing_count = len(existing_results)
        all_metrics = list(existing_results)
    except FileNotFoundError:
        existing_results = None
        existing_count = 0
        all_metrics = []

    count = 0

    tally = {
        50: 0,
        100: 0,
        200: 0,
        300: 0,
        400: 0,
    }
    for file in range(len(chembl_info)):
        count += 1
        if count <= existing_count:
            continue

        filename = chembl_info.iloc[file]["File name"]
        logging.info(f"On Dataset No. {count}, filename: {filename}")
        data_folder = os.getcwd() + "/data/qsar_data_unsorted/"
        train_test = dataset(data_folder + filename, shuffle_state=1)

        if len(np.unique(train_test[:, 0])) == 1:
            logging.warning("Cannot build model with only one target value for Dataset " + filename)
            logging.warning(f"Skip Dataset {filename}")
            continue
        if len(train_test) < 50:
            logging.warning("Dataset " + filename + " is not watched in tally." )
            logging.warning(f"Skip Dataset {filename}")
            continue
        if chembl_info.iloc[file]["Repetition Rate"] > 0.15:
            logging.warning("Dataset " + filename + " has too high repetition rate." )
            logging.warning(f"Skip Dataset {filename}")
            continue
        if len(str(len(train_test))) == 3:
            tally_key = int(str(len(train_test))[0] + "00")
        elif len(str(len(train_test))) == 2:
            tally_key = 50
        else:
            continue
        if tally_key not in tally.keys():
            logging.warning("Dataset " + filename + " size not in tally keys." )
            logging.warning(f"Skip Dataset {filename}")
            continue
        if tally_key < 300:
            if tally[tally_key] >= 40:
                logging.warning("Already have 20 datasets of size " + str(len(train_test)) )
                logging.warning(f"Skip Dataset {filename}")
                continue
        else:
            if tally[tally_key] >= 10:
                logging.warning("Already have 5 datasets of size " + str(len(train_test)) )
                logging.warning(f"Skip Dataset {filename}")
                continue
        logging.info(f"Dataset " + filename + f" is added to the tally at tally key of {tally_key}." )
        tally[tally_key] += 1

        train_test_splits_dict = kfold_splits(train_test=train_test, fold=5)

        metrics_per_dataset = run(
            train_test_splits_dict=train_test_splits_dict,
            ML_cls=RandomForestClassifier(random_state=1, n_jobs=-1),
            ML_reg=RandomForestRegressor(random_state=1, n_jobs=-1),
            percentage_of_top_samples=0.1,  # top-performing as in top 10%
        )
        all_metrics.append(metrics_per_dataset)
        np.save(output_dir + results_filename, np.array(all_metrics))
