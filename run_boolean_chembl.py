import logging

import pandas as pd
import numpy as np
import os
import warnings

from sklearn.svm import SVR

from pairwise_formulation.pa_basics.import_data import dataset, kfold_splits
from run_experiments.run_utils import run
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    root_dir = os.getcwd()
    chembl_info = pd.read_csv(
        root_dir + "/data/boolean_chembl_datasets_info.csv"
    ).sort_values(by=["N(sample)"])

    output_dir = root_dir + "/output/boolean_chembl/"
    results_filename = "boolean_chembl_svm_sbbr1.npy"

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
        if chembl_info.iloc[file]["Repetition Rate"] > 0.15:
            logging.warning("Dataset " + filename + " has too high repetition rate." )
            logging.warning(f"Skip Dataset {filename}")
            continue
        if chembl_info.iloc[file]["N(sample)"] < 50 or chembl_info.iloc[file]["N(sample)"] > 500:
            logging.warning("Dataset " + filename + " too big or small." )
            logging.warning(f"Skip Dataset {filename}")
            continue


        train_test_splits_dict = kfold_splits(train_test=train_test, fold=5)

        metrics_per_dataset = run(
            train_test_splits_dict=train_test_splits_dict,
            ML_cls=None,
            ML_reg=SVR(),
            percentage_of_top_samples=0.1,  # top-performing as in top 10%
        )
        all_metrics.append(metrics_per_dataset)
        np.save(output_dir + results_filename, np.array(all_metrics))
