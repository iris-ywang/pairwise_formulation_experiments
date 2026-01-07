import time

import numpy as np

from pairwise_formulation.pairwise_data import PairwiseDataInfo
from pairwise_formulation.pairwise_model import PairwiseModel, build_ml_model
from pairwise_formulation.pa_basics.rating import rating_elo, rating_sbbr, rating_trueskill
from pairwise_formulation.evaluations.extrapolation_evaluation import ExtrapolationEvaluation
from pairwise_formulation.evaluations.stock_return_evaluation import calculate_returns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scipy.stats import spearmanr


def estimate_y_from_averaging(Y_pa_c2, c2_test_pair_ids, test_ids, y_true, Y_weighted=None):
    """
    Estimate activity values from C2-type test pairs via arithmetic mean or weighted average, It is calculated by
    estimating y_test from [Y_(test, train)_pred + y_train_true] and [ - Y_(train, test)_pred + y_train_true]

    :param Y_pa_c2: np.array of (predicted) differences in activities for C2-type test pairsc
    :param c2_test_pair_ids: list of tuples, each specifying samples IDs for a c2-type pair.
            * Y_pa_c2 and c2_test_pair_ids should match in position; their length should be the same.
    :param test_ids: list of int for test sample IDs
    :param y_true: np.array of true activity values of all samples
    :param Y_weighted: np.array of weighting of each Y_pred (for example, from model prediction probability)
    :return: np.array of estimated activity values for test set
    """
    if y_true is None:
        y_true = y_true
    if Y_weighted is None:  # linear arithmetic
        Y_weighted = np.ones((len(Y_pa_c2)))

    records = np.zeros((len(y_true)))
    weights = np.zeros((len(y_true)))

    for pair in range(len(Y_pa_c2)):
        ida, idb = c2_test_pair_ids[pair]
        delta_ab = Y_pa_c2[pair]
        weight = Y_weighted[pair]

        if ida in test_ids:
            # (test, train)
            weighted_estimate = (y_true[idb] + delta_ab) * weight
            records[ida] += weighted_estimate
            weights[ida] += weight

        elif idb in test_ids:
            # (train, test)
            weighted_estimate = (y_true[ida] - delta_ab) * weight
            records[idb] += weighted_estimate
            weights[idb] += weight

    return np.divide(records[test_ids], weights[test_ids])


def metrics_evaluation(y_true, y_predict):
    rho = spearmanr(y_true, y_predict, nan_policy="omit")[0]
    mse = mean_squared_error(y_true, y_predict)
    mae = mean_absolute_error(y_true, y_predict)
    r2 = r2_score(y_true, y_predict)
    return [rho, mse, mae, r2, np.nan, np.nan]


def results_of_pairwise_combinations(
    pairwise_model: PairwiseModel,
    if_rank_with_dist: bool,
    rank_method=rating_trueskill,
    percentage_of_top_samples=0.1,
):
    # Extrapolation performance evaluation:
    y_ranking_c2 = pairwise_model.predict(
        ranking_method=rank_method,
        ranking_input_type="c2",
        if_sbbr_dist=if_rank_with_dist,
    )

    metrics_c2 = ExtrapolationEvaluation(
        percentage_of_top_samples=percentage_of_top_samples,
        y_train_with_predicted_test=y_ranking_c2,
        pairwise_data_info=pairwise_model.pairwise_data_info,
    ).run_extrapolation_evaluation()

    y_ranking_c2_c3 = pairwise_model.predict(
        ranking_method=rank_method,
        ranking_input_type="c2_c3",
        if_sbbr_dist=if_rank_with_dist,
    )

    metrics_c2_c3 = ExtrapolationEvaluation(
        percentage_of_top_samples=percentage_of_top_samples,
        y_train_with_predicted_test=y_ranking_c2_c3,
        pairwise_data_info=pairwise_model.pairwise_data_info,
    ).run_extrapolation_evaluation()

    y_ranking_c1_c2_c3 = pairwise_model.predict(
        ranking_method=rank_method,
        ranking_input_type="c1_c2_c3",
        if_sbbr_dist=if_rank_with_dist,
    )

    metrics_c1_c2_c3 = ExtrapolationEvaluation(
        percentage_of_top_samples=percentage_of_top_samples,
        y_train_with_predicted_test=y_ranking_c1_c2_c3,
        pairwise_data_info=pairwise_model.pairwise_data_info,
    ).run_extrapolation_evaluation()

    # Regressive prediction performance evaluation:
    if not if_rank_with_dist:
        metrics_est = [np.nan for _ in range (6)]
    else:
        y_est = estimate_y_from_averaging(
            pairwise_model.Y_values.Y_pa_c2_nume,
            pairwise_model.pairwise_data_info.c2_test_pair_ids,
            pairwise_model.pairwise_data_info.test_ids,
            pairwise_model.pairwise_data_info.y_true_all,
        )

        metrics_est = metrics_evaluation(
            pairwise_model.pairwise_data_info.test_ary[:, 0],
            y_est
        )
    return [metrics_c2, metrics_c2_c3, metrics_c1_c2_c3], metrics_est


def run_per_dataset(
        foldwise_data: dict,
        ML_cls,
        ML_reg,
        percentage_of_top_samples=0.1,
        target_value_col_name='y'
):

    train_set = foldwise_data['train_set']
    test_set = foldwise_data['test_set']

    # pairwise approach
    pairwise_data = PairwiseDataInfo(
        train_set, test_set, target_value_col_name=target_value_col_name
    )

    pa_start_time = time.time()
    pairwise_model = PairwiseModel(
        pairwise_data_info=pairwise_data,
        ML_cls=ML_cls,
        ML_reg=ML_reg,
    ).fit()
    pa_training_time = time.time() - pa_start_time

    pa_eval_start_time = time.time()
    metrics_pa_v1, metrics_est_pa_v1 = results_of_pairwise_combinations(
        pairwise_model=pairwise_model,
        if_rank_with_dist=False,
        rank_method=rating_trueskill,
        percentage_of_top_samples=percentage_of_top_samples,
    )

    metrics_pa_v2, metrics_est_pa_v2 = results_of_pairwise_combinations(
        pairwise_model=pairwise_model,
        if_rank_with_dist=True,
        rank_method=rating_sbbr,
        percentage_of_top_samples=percentage_of_top_samples,
    )
    pa_eval_time = time.time() - pa_eval_start_time

    # standard approach
    sa_start_time = time.time()
    _, y_sa_pred = build_ml_model(
        model=ML_reg,
        train_data=pairwise_data.train_ary,
        test_data=pairwise_data.test_ary
    )
    sa_training_time = time.time() - sa_start_time

    sa_eval_start_time = time.time()
    y_sa_pred_w_train = np.array(pairwise_data.y_true_all)
    y_sa_pred_w_train[pairwise_data.test_ids] = y_sa_pred

    metrics_sa = ExtrapolationEvaluation(
        percentage_of_top_samples=percentage_of_top_samples,
        y_train_with_predicted_test=y_sa_pred_w_train,
        pairwise_data_info=pairwise_model.pairwise_data_info,
    ).run_extrapolation_evaluation()
    metrics_est_sa = metrics_evaluation(
        pairwise_model.pairwise_data_info.test_ary[:, 0],
        y_sa_pred
    )
    sa_eval_time = time.time() - sa_eval_start_time

    metrics_per_fold = (
        [metrics_sa] + metrics_pa_v1 + metrics_pa_v2 +
        [metrics_est_sa] + [metrics_est_pa_v2]
    )

    training_time_log = "training_time_log2.txt"
    with open(training_time_log, "a") as time_log_file:
        #training_size,pa_training_time,pa_eval_time,sa_training_time,sa_eval_time
        time_log_file.write(
            f"{len(train_set)},{pa_training_time},{pa_eval_time},{sa_training_time},{sa_eval_time}\n"
        )
        time_log_file.flush()

    return metrics_per_fold


def run(
    train_test_splits_dict: dict, ML_cls, ML_reg,
    percentage_of_top_samples=0.1, target_value_col_name='y', n_jobs=None
):
    metrics_per_dataset = []
    if n_jobs is None:
        for fold_id, foldwise_data in train_test_splits_dict.items():
            metrics_per_fold = run_per_dataset(
                foldwise_data=foldwise_data,
                ML_cls=ML_cls,
                ML_reg=ML_reg,
                percentage_of_top_samples=percentage_of_top_samples,
                target_value_col_name=target_value_col_name,
            )
            metrics_per_dataset.append(metrics_per_fold)
        return metrics_per_dataset


def run_per_stock_dataset(
        foldwise_data: dict,
        ML_cls,
        ML_reg,
        n_portofolio,
        target_value_col_name="annual_pc_price_change",
):

    train_set = foldwise_data['train_set']
    test_set = foldwise_data['test_set']

    pred_true_return_list = []

    # pairwise approach
    pairwise_data = PairwiseDataInfo(
        train_set, test_set, target_value_col_name=target_value_col_name
    )
    pairwise_model = PairwiseModel(
        pairwise_data_info=pairwise_data,
        ML_cls=ML_cls,
        ML_reg=ML_reg,
    ).fit()

    # standard approach
    _, y_sa_pred = build_ml_model(
        model=ML_reg,
        train_data=pairwise_data.train_ary,
        test_data=pairwise_data.test_ary
    )

    pred_return_sa, true_return = calculate_returns(
        y_test_pred=y_sa_pred[pairwise_model.pairwise_data_info.test_ids],
        y_test_true=pairwise_model.pairwise_data_info.test_ary[:,0],
        n_portofolio=n_portofolio,
    )
    pred_true_return_list.append(true_return)
    pred_true_return_list.append(pred_return_sa)

    y_ranking_c2 = pairwise_model.predict(
        ranking_method=rating_trueskill,
        ranking_input_type="c2",
        if_sbbr_dist=False,
    )

    pred_return_c2, _ = calculate_returns(
        y_test_pred=y_ranking_c2[pairwise_model.pairwise_data_info.test_ids],
        y_test_true=pairwise_model.pairwise_data_info.test_ary[:,0],
        n_portofolio=n_portofolio,
    )
    pred_true_return_list.append(pred_return_c2)

    y_ranking_c2_c3 = pairwise_model.predict(
        ranking_method=rating_trueskill,
        ranking_input_type="c2_c3",
        if_sbbr_dist=False,
    )

    pred_return_c2_c3, _ = calculate_returns(
        y_test_pred=y_ranking_c2_c3[pairwise_model.pairwise_data_info.test_ids],
        y_test_true=pairwise_model.pairwise_data_info.test_ary[:,0],
        n_portofolio=n_portofolio,
    )
    pred_true_return_list.append(pred_return_c2_c3)

    y_ranking_c1_c2_c3 = pairwise_model.predict(
        ranking_method=rating_trueskill,
        ranking_input_type="c1_c2_c3",
        if_sbbr_dist=False,
    )

    pred_return_c1_c2_c3, _ = calculate_returns(
        y_test_pred=y_ranking_c1_c2_c3[pairwise_model.pairwise_data_info.test_ids],
        y_test_true=pairwise_model.pairwise_data_info.test_ary[:,0],
        n_portofolio=n_portofolio,
    )
    pred_true_return_list.append(pred_return_c1_c2_c3)

    return pred_true_return_list
