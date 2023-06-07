from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score,  balanced_accuracy_score, roc_auc_score

from imblearn.over_sampling import RandomOverSampler
from smote_variants import SMOTE, Borderline_SMOTE2, ADASYN, NoSMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import os
import argparse
import torch
from matplotlib import pyplot as plt

from sklearn.base import clone
from _multilayer_perceptron import MLPClassifier
from objective_functions import OVACrossEntropyLoss, MSELoss, CrossEntropyLoss, FocalLoss, OVAFocalLoss, DEOHyperbolicTangentRelaxation, OneClassMSELoss

import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger("Maciek")

metrics = [balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]


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
    return X, y


def draw_pareto_front(predictions, target, dataset, fold):
    classes = np.unique(target)
    objectives = np.zeros((len(predictions), 2))
    # weights = [np.array([0.1, 0.9]), np.array([0.9, 0.1])]
    gamma = [1, 5]
    # for j in range(len(predictions)):
    #     for i in range(2):
    #         with torch.no_grad():
    #             objectives[j,i] = torch.nn.functional.cross_entropy(torch.from_numpy(predictions[j][:,[(i+1)%2,i]]),
    #                                                                 torch.from_numpy(target == classes[i]).long(), reduction='mean').item()
    # for j in range(len(predictions)):
    #     with torch.no_grad():
    #         ohe = OneHotEncoder(sparse=False)
    #         objectives[j, 0] = torch.nn.functional.mse_loss(torch.from_numpy(predictions[j]),
    #                                                         torch.from_numpy(ohe.fit_transform(target.reshape(-1,1)).astype("float32")).long(),reduction='mean').item()
    #         objectives[j, 1] = torch.nn.functional.cross_entropy(torch.from_numpy(predictions[j]), torch.from_numpy(target).long(), reduction='mean').item()
    # for j in range(len(predictions)):
    #     # for i in range(len(classes)):
    #     # for i in range(2):
    #     for i,w in enumerate(gamma):
    #         # w = np.zeros((len(classes)))
    #         # w[i] = 1.0
    #         with torch.no_grad():
    #             objectives[j, i] = torch.nn.functional.cross_entropy(torch.from_numpy(predictions[j]),
    #                                                                  torch.from_numpy(target).long(),
    #                                                                  weight=torch.from_numpy(w.astype("float32")),
    #                                                                  reduction='mean').item()

    for j in range(len(predictions)):
        # for i in range(len(classes)):
        # for i in range(2):
        for i, w in enumerate(gamma):
            # w = np.zeros((len(classes)))
            # w[i] = 1.0
            fl = FocalLoss(gamma=w)
            with torch.no_grad():
                objectives[j, i] = fl.forward(torch.from_numpy(predictions[j]), torch.from_numpy(target).long(),)

    fig = plt.figure(figsize=(6, 6), num=1, clear=True)
    ax = fig.add_subplot(111)
    # ax.set_facecolor('antiquewhite')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)
    ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.7)
    ax.set_ylabel("cross_entropy0")
    ax.set_xlabel("cross_entropy1")

    # ax.set_xlim([0, 1.1])
    # ax.set_ylim([0, 1.1])

    ax.scatter(objectives[:, 0], objectives[:, 1], c="darkorchid", marker="o", zorder=10)

    plt.tight_layout()
    plt.savefig(os.path.join(f"pareto_front_{dataset}_{fold}.png"))
    plt.clf()


def get_objectives(predictions, y, if_loss):
    objectives = np.zeros((len(predictions), 2))
    if if_loss:
        # weights = [np.array([0.1, 0.9]), np.array([0.9, 0.1])]
        gamma = [1, 5]
        # for j in range(len(predictions)):
        #     for i in range(2):
        #         with torch.no_grad():
        #             objectives[j,i] = torch.nn.functional.cross_entropy(torch.from_numpy(predictions[j][:,[(i+1)%2,i]]),
        #                                                                 torch.from_numpy(target == classes[i]).long(), reduction='mean').item()
        # for j in range(len(predictions)):
        #     with torch.no_grad():
        #         ohe = OneHotEncoder(sparse=False)
        #         objectives[j, 0] = torch.nn.functional.mse_loss(torch.from_numpy(predictions[j]),
        #                                                         torch.from_numpy(ohe.fit_transform(target.reshape(-1,1)).astype("float32")).long(),reduction='mean').item()
        #         objectives[j, 1] = torch.nn.functional.cross_entropy(torch.from_numpy(predictions[j]), torch.from_numpy(target).long(), reduction='mean').item()
        # for j in range(len(predictions)):
        #     # for i in range(len(classes)):
        #     # for i in range(2):
        #     for i,w in enumerate(gamma):
        #         # w = np.zeros((len(classes)))
        #         # w[i] = 1.0
        #         with torch.no_grad():
        #             objectives[j, i] = torch.nn.functional.cross_entropy(torch.from_numpy(predictions[j]),
        #                                                                  torch.from_numpy(target).long(),
        #                                                                  weight=torch.from_numpy(w.astype("float32")),
        #                                                                  reduction='mean').item()

        for j in range(len(predictions)):
            # for i in range(len(classes)):
            # for i in range(2):
            for i, w in enumerate(gamma):
                # w = np.zeros((len(classes)))
                # w[i] = 1.0
                fl = FocalLoss(gamma=w)
                with torch.no_grad():
                    objectives[j, i] = fl.forward(torch.from_numpy(predictions[j]), torch.from_numpy(y).long(), )
    else:
        for j in range(len(predictions)):
            objectives[j, 0] = recall_score(y, predictions[j])
            objectives[j, 1] = precision_score(y, predictions[j])


def experiment_single_fold(X, y, objective_functions, fold, dataset, diversity=False, path='results'):
    path_fold = os.path.join(path, str(fold))
    try:
        os.mkdir(path_fold)
    except:
        pass
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=75)
    splits = list(sss.split(X,y))
    X_train = X[splits[fold//2][fold % 2]]
    y_train = y[splits[fold//2][fold % 2]]
    X_test = X[splits[fold//2][(fold+1) % 2]]
    y_test = y[splits[fold//2][(fold+1) % 2]]

    model = MLPClassifier(objectives=objective_functions, random_state=966, path=path_fold, alpha=0.5)
    model.fit(X_train, y_train)
    if len(model.objectives) > 3:
        if os.path.exists(os.path.join(path, "test_rays.npy")):
            test_rays = np.load(os.path.join(path, "test_rays.npy"))
        else:
            test_rays = np.random.rand(model.n_test_rays, len(model.objectives))
            test_rays /= test_rays.sum(axis=1).reshape(model.n_test_rays, 1)
            np.save(os.path.join(path, "test_rays.npy"), test_rays)
    else:
        test_rays = None
    if not diversity:
        proba_predictions_train = model.predict_proba(X_train,test_rays=test_rays)
        # draw_pareto_front(proba_predictions_train, y_train, dataset, fold)
    predictions_train = model.predict(X_train, test_rays=test_rays)
    proba_predictions = model.predict_proba(X_test, test_rays=test_rays)
    predictions = model.predict(X_test, test_rays=test_rays)
    metrics_array = np.zeros((len(predictions), len(metrics)))
    for i_p, prediction in enumerate(predictions):
        for i_m, metric in enumerate(metrics):
            try:
                metrics_array[i_p, i_m] = metric(y_test, prediction)
            except:
                try:
                    metrics_array[i_p, i_m] = metric(y_test, prediction, average="macro")
                except:
                    pass
    df_metric = pd.DataFrame(columns=[m.__name__ for m in metrics], data=metrics_array)
    # df_metric.to_csv(f"wyniki_{dataset}_{fold}.csv")
    # proba_predictions.save(f"proba_predictions_{dataset}.npy")
    np.save(os.path.join(path_fold,f"predictions_test_{dataset}.npy"), predictions)
    np.save(os.path.join(path_fold,f"predictions_proba_test_{dataset}.npy"), proba_predictions)
    np.save(os.path.join(path_fold, f"predictions_proba_train_{dataset}.npy"), proba_predictions_train)
    np.save(os.path.join(path_fold, f"predictions_train_{dataset}.npy"), predictions_train)


def conduct_experiment(path='results',dataset='all', fold=10, diversity=False, gamma=2):
    X, y = load_dataset(dataset)
    try:
        os.mkdir(path)
    except:
        pass

    logger.info(f"Starting experiment on dataset {dataset} on {fold} fold")

    try:
        os.mkdir(os.path.join(path, dataset))
    except:
        pass

    # gamma = [0.2, 5]
    # objective_functions = [OVACrossEntropyLoss(c) for c in np.unique(y)]
    # objective_functions = [CrossEntropyLoss(), MSELoss()]
    # objective_functions = [CrossEntropyLoss(c, len(np.unique(y))) for c in np.unique(y)]
    # objective_functions = [CrossEntropyLoss(np.array([0.1, 0.9])), CrossEntropyLoss(np.array([0.9, 0.1]))]
    # objective_functions = [FocalLoss(gamma=g) for g in gamma]
    # objective_functions = [OVAFocalLoss(cls=c, gamma=gamma) for c in np.unique(y)]
    # objective_functions = [OneClassMSELoss(c) for c in np.unique(y)]
    objective_functions = [DEOHyperbolicTangentRelaxation(9), CrossEntropyLoss()]

    classifier = MLPClassifier(objectives=objective_functions, random_state=966)
    if fold == 10:
        for i in range(10):
            experiment_single_fold(X, y, objective_functions, i, dataset, diversity, os.path.join(path, dataset))
    else:
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
