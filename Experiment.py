from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score,  balanced_accuracy_score, roc_auc_score

from imblearn.over_sampling import RandomOverSampler
from smote_variants import SMOTE, Borderline_SMOTE2, ADASYN, NoSMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import os
import argparse
import torch
from matplotlib import pyplot as plt
import random

from sklearn.base import clone
from _multilayer_perceptron import MLPClassifier, base_mlp_classifier
from objective_functions import OVACrossEntropyLoss, MSELoss, CrossEntropyLoss, FocalLoss, OVAFocalLoss, \
    DEOHyperbolicTangentRelaxation, OneClassMSELoss, BinaryCrossEntropyLoss
from _ensembles import LossPropEnsemble, TrainedEnsemble, SuppMajorityEnsemble, HardMajorityEnsemble
from ParetoMTL import ParetoMTLMethod
from MGDA import MGDAMethod
from DatasetsCollection import load

import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger("Maciek")

# datasets = ['adult', 'MiniBooNE_PID', 'ForestTypes1vsrest', 'page-blocks0']
# datasets = ['page-blocks0', 'adult']
datasets = ['compas_labeled']
# datasets = ['adult', 'page-blocks0', 'bank_additional', 'compas_labeled', 'MiniBooNE_PID']
metrics = [balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]

# file_list = [
#              # "page-blocks-1-3_vs_4", "yeast-0-5-6-7-9_vs_4", "yeast-1-2-8-9_vs_7",
#              # "yeast-1-4-5-8_vs_7", "yeast-1_vs_7",
#              #  "yeast-2_vs_4", "yeast-2_vs_8", "yeast4", "yeast5", "yeast6",
#              # "ecoli-0-1-4-7_vs_2-3-5-6",
#              #  "ecoli-0-1_vs_2-3-5", "ecoli-0-2-6-7_vs_3-5",
#              # "ecoli-0-6-7_vs_3-5", "ecoli-0-6-7_vs_5",
#              # "yeast-0-2-5-6_vs_3-7-8-9", "yeast-0-3-5-9_vs_7-8",
#              # "abalone-17_vs_7-8-9-10", "abalone-19_vs_10-11-12-13",
#              # "abalone-20_vs_8-9-10", "flare-F", "kr-vs-k-zero_vs_eight",
#              "poker-8-9_vs_5", "poker-8-9_vs_6", "poker-8_vs_6",
#              "winequality-red-4",
#               "winequality-white-3-9_vs_5", "winequality-white-3_vs_7",
#              "ecoli1", "ecoli2", "ecoli3", "glass0", "glass1", "haberman",
#              "pima", "yeast3"
# ]
file_list = ['ecoli1', 'ecoli3', 'glass0', 'glass1', 'haberman', 'pima', 'yeast-0-2-5-6_vs_3-7-8-9', 'yeast-0-3-5-9_vs_7-8',
            'yeast-0-5-6-7-9_vs_4', 'yeast3', 'yeast4']


def load_dataset(dataset):
    try:
        df = pd.read_csv(os.path.join("datasets", f"{dataset}.csv"), index_col=0)
    except Exception as e:
        print("Wrong dataset")
        raise e
    if "class" in df.columns:
        class_label = "class"
    elif "Class" in df.columns:
        class_label = "Class"
    elif "label" in df.columns:
        class_label = "label"
    else:
        raise Exception("Class label not known")

    y = df[class_label].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    df.drop(columns=class_label, inplace=True)
    X = df.values
    X = StandardScaler().fit_transform(X)
    return X, y


def experiment_single_fold(X, y, objective_functions, fold, dataset, diversity=False, path='results', full_data=True):
    path_fold = os.path.join(path, str(fold))


    try:
        os.mkdir(path_fold)
    except:
        pass
    if full_data:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=75)
        splits = list(sss.split(X,y))
        X_train = X[splits[fold//2][fold % 2]]
        y_train = y[splits[fold//2][fold % 2]]
        X_test = X[splits[fold//2][(fold+1) % 2]]
        y_test = y[splits[fold//2][(fold+1) % 2]]
    else:
        X_train = X[0]
        y_train = y[0]
        X_test = X[1]
        y_test = y[1]

    # if full_data:
    #     model = MLPClassifier(objectives=objective_functions, random_state=966, path=path_fold, alpha=0.5, max_iter=100,
    #                           n_test_rays=25)
    # else:
    #     model = MLPClassifier(objectives=objective_functions, random_state=966, path=path_fold, alpha=0.5, max_iter=150,
    #                           n_test_rays=25, batch_size='auto')
    # model.fit(X_train, y_train)
    # if len(model.objectives) > 3:
    #     if os.path.exists(os.path.join(path, "test_rays.npy")):
    #         test_rays = np.load(os.path.join(path, "test_rays.npy"))
    #     else:
    #         test_rays = np.random.rand(model.n_test_rays, len(model.objectives))
    #         test_rays /= test_rays.sum(axis=1).reshape(model.n_test_rays, 1)
    #         np.save(os.path.join(path, "test_rays.npy"), test_rays)
    # else:
    #     test_rays = None
    # if not diversity:
    #     proba_predictions_train = model.get_logits(X_train,test_rays=test_rays)
    #     # draw_pareto_front(proba_predictions_train, y_train, dataset, fold)
    # predictions_train = model.predict(X_train, test_rays=test_rays)
    # proba_predictions = model.get_logits(X_test, test_rays=test_rays)
    # predictions = model.predict(X_test, test_rays=test_rays)
    # # metrics_array = np.zeros((len(predictions), len(metrics)))
    # # for i_p, prediction in enumerate(predictions):
    # #     for i_m, metric in enumerate(metrics):
    # #         try:
    # #             metrics_array[i_p, i_m] = metric(y_test, prediction)
    # #         except:
    # #             try:
    # #                 metrics_array[i_p, i_m] = metric(y_test, prediction, average="macro")
    # #             except:
    # #                 pass
    # # df_metric = pd.DataFrame(columns=[m.__name__ for m in metrics], data=metrics_array)
    # # df_metric.to_csv(f"wyniki_{dataset}_{fold}.csv")
    # # proba_predictions.save(f"proba_predictions_{dataset}.npy")
    # np.save(os.path.join(path_fold, f"predictions_test_{dataset}.npy"), predictions)
    # np.save(os.path.join(path_fold, f"predictions_proba_test_{dataset}.npy"), proba_predictions)
    # np.save(os.path.join(path_fold, f"predictions_proba_train_{dataset}.npy"), proba_predictions_train)
    # np.save(os.path.join(path_fold, f"predictions_train_{dataset}.npy"), predictions_train)
    # #
    # names = ['MLP_focal_loss_0', 'MLP_focal_loss_2', 'MLP_bce', 'Cosmos_ensemble_loss', 'Cosmos_ensemble_train_fl',
    #          'Cosmos_ensemble_train_bce', 'Cosmos_ensemble_majority_soft', 'Cosmos_ensemble_majority_hard']
    # mlp1 = base_mlp_classifier(loss_function=FocalLoss(0), output_dim=2, max_iter=150, batch_size='auto')
    # mlp2 = base_mlp_classifier(loss_function=FocalLoss(2), output_dim=2, max_iter=150, batch_size='auto')
    #
    # _, counts = np.unique(y_train, return_counts=True)
    # if counts[1] < counts[0]:
    #     minc, maxc = counts[1], counts[0]
    # else:
    #     minc, maxc = counts[0], counts[1]
    #
    # mlp3 = base_mlp_classifier(loss_function=BinaryCrossEntropyLoss(maxc/minc), max_iter=150, batch_size='auto')
    #
    # ensemble1 = LossPropEnsemble(objectives=objective_functions, model=model)
    # ensemble2 = TrainedEnsemble(objective='focalloss', model=model)
    # ensemble3 = TrainedEnsemble(objective='binarycrossentropy', model=model)
    # ensemble4 = SuppMajorityEnsemble(objectives=objective_functions, model=model)
    # ensemble5 = HardMajorityEnsemble(objectives=objective_functions, model=model)
    #
    # classifiers = [mlp1, mlp2, mlp3, ensemble1, ensemble2, ensemble3, ensemble4, ensemble5]
    # # classifiers = [ensemble3]
    # for ic, classifier in enumerate(classifiers):
    #     classifier.fit(X_train, y_train)
    #     predictions_train = classifier.predict(X_train, test_rays=test_rays)
    #     predictions = classifier.predict(X_test, test_rays=test_rays)
    #     np.save(os.path.join(path_fold, f"predictions_test_{names[ic]}_{dataset}.npy"), predictions)
    #     np.save(os.path.join(path_fold, f"predictions_train_{names[ic]}_{dataset}.npy"), predictions_train)
    predictions_test = np.zeros((25, len(y_test)))
    predictions_train = np.zeros((25, len(y_train)))
    predictions_proba_test = np.zeros((25, len(y_test), 2))
    predictions_proba_train = np.zeros((25, len(y_train), 2))
    # model = ParetoMTLMethod(objective_functions)
    random.seed(666)
    for i in range(25):
        model = MGDAMethod(objective_functions, random_state=random.randint(0, 10000), max_iter=250, batch_size='auto')
        model.fit(X_train, y_train)
        predictions_train[i, :] = model.predict(X_train)
        predictions_test[i, :] = model.predict(X_test)
        predictions_proba_train[i, :, :] = model.get_logits(X_train)
        predictions_proba_test[i, :, :] = model.get_logits(X_test)

    np.save(os.path.join(path_fold, f"predictions_test_{dataset}_mgda.npy"), predictions_test)
    np.save(os.path.join(path_fold, f"predictions_proba_test_{dataset}_mgda.npy"), predictions_proba_test)
    np.save(os.path.join(path_fold, f"predictions_proba_train_{dataset}_mgda.npy"), predictions_proba_train)
    np.save(os.path.join(path_fold, f"predictions_train_{dataset}_mgda.npy"), predictions_train)




def conduct_experiment(path='results',dataset='all', fold=10, diversity=False, gamma=2):
    if dataset == 'adults':
        X = np.load(os.path.join("datasets", "adult_X_race_dummies.npy"))
        y = np.load(os.path.join("datasets", "adult_y_race_dummies.npy"))
    elif dataset == 'all':
        data = [load_dataset(dataset) for dataset in datasets]
    elif  dataset == 'small_data':
        data = [load(dataset) for dataset in file_list]
    else:
        X, y = load_dataset(dataset)


    # X, y = load_dataset(dataset)
    try:
        os.mkdir(path)
    except:
        pass

    logger.info(f"Starting experiment on dataset {dataset} on {fold} fold")



    gamma = [2, 5]
    # objective_functions = [OVACrossEntropyLoss(c) for c in np.unique(y)]
    # objective_functions = [CrossEntropyLoss(), MSELoss()]
    # objective_functions = [CrossEntropyLoss(c, len(np.unique(y))) for c in np.unique(y)]
    # objective_functions = [CrossEntropyLoss(0), CrossEntropyLoss(1)]
    # objective_functions = [FocalLoss(gamma=g) for g in gamma]
    # objective_functions = [OVAFocalLoss(cls=c, gamma=gamma) for c in np.unique(y)]
    # objective_functions = [OneClassMSELoss(c) for c in np.unique(y)]
    # objective_functions = [DEOHyperbolicTangentRelaxation(), CrossEntropyLoss()]

    # classifier = MLPClassifier(objectives=objective_functions, random_state=966)
    if fold == 10:
        if dataset == 'all':
            for j, (X, y) in enumerate(data):
                try:
                    os.mkdir(os.path.join(path, datasets[j]))
                except:
                    pass
                _, counts = np.unique(y, return_counts=True)
                if counts[1] < counts[0]:
                    minc, maxc = counts[1], counts[0]
                else:
                    minc, maxc = counts[0], counts[1]
                objective_functions = [BinaryCrossEntropyLoss(minc/maxc), BinaryCrossEntropyLoss(maxc/minc)]
                # objective_functions = [CrossEntropyLoss(0), CrossEntropyLoss(1)]
                for i in range(10):
                    experiment_single_fold(X, y, objective_functions, i, datasets[j], diversity,
                                           os.path.join(path, datasets[j]))
        elif dataset == 'small_data':
            for j, d in enumerate(data):
                try:
                    os.mkdir(os.path.join(path, file_list[j]))
                except:
                    pass
                _, counts = np.unique(d[0][0][1], return_counts=True)
                if counts[1] < counts[0]:
                    minc, maxc = counts[1], counts[0]
                else:
                    minc, maxc = counts[0], counts[1]
                # objective_functions = [BinaryCrossEntropyLoss(minc/maxc), BinaryCrossEntropyLoss(maxc/minc)]
                objective_functions = [CrossEntropyLoss(0), CrossEntropyLoss(1)]
                for i in range(10):
                    experiment_single_fold((d[i][0][0], d[i][1][0]), (d[i][0][1], d[i][1][1]), objective_functions, i, file_list[j], diversity,
                                           os.path.join(path, file_list[j]), full_data=False)
        else:
            for i in range(10):
                experiment_single_fold(X, y, objective_functions, i, dataset, diversity, os.path.join(path, dataset))
    else:
        _, counts = np.unique(y, return_counts=True)
        if counts[1] < counts[0]:
            minc, maxc = counts[1], counts[0]
        else:
            minc, maxc = counts[0], counts[1]
        # objective_functions = [BinaryCrossEntropyLoss(minc / maxc), BinaryCrossEntropyLoss(maxc / minc)]
        try:
            os.mkdir(os.path.join(path, dataset))
        except:
            pass
        objective_functions = [CrossEntropyLoss(0), CrossEntropyLoss(1)]
        experiment_single_fold(X, y, objective_functions, fold, dataset, diversity, os.path.join(path, dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("fold", type=int)
    parser.add_argument("--diversity", dest='diversity', action='store_true')
    parser.add_argument("--gamma", type=float, default=2)
    parser.set_defaults(history=False)
    args = parser.parse_args()
    logger.info(f"I got arguments!")
    conduct_experiment(args.path, args.dataset, args.fold, args.diversity, args.gamma)
