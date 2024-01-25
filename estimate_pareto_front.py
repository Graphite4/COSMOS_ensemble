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


radom_state = [966, 4, 666, 13, 1001, 25, 589, 17098, 9087, 453]

datasets = ['adult', 'page-blocks0', 'bank_additional', 'compas_labeled', 'MiniBooNE_PID']
file_list = ['ecoli1', 'ecoli3', 'glass0', 'glass1', 'haberman', 'pima', 'yeast-0-2-5-6_vs_3-7-8-9', 'yeast-0-3-5-9_vs_7-8',
            'yeast-0-5-6-7-9_vs_4', 'yeast3', 'yeast4']

def get_not_dominated_solutions(objectives, if_min=True):

    potentialy_non_dominated = [0]

    for i in range(1, len(objectives)):
        if_dominated = False
        for i_o in potentialy_non_dominated:
            if if_min:
                if all(objectives[i, :] < objectives[i_o, :]):
                    potentialy_non_dominated.remove(i_o)
                elif all(objectives[i, :] >= objectives[i_o, :]):
                    if_dominated = True
                    break
            else:
                if all(objectives[i, :] > objectives[i_o, :]):
                    potentialy_non_dominated.remove(i_o)
                elif all(objectives[i, :] <= objectives[i_o, :]):
                    if_dominated = True
                    break
        if not if_dominated:
            potentialy_non_dominated.append(i)

    return objectives[potentialy_non_dominated]

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


def draw_plot(objectives, pareto, path):
    fig = plt.figure(figsize=(8, 8), num=1, clear=True)
    ax = fig.add_subplot(111)
    # ax.set_facecolor('antiquewhite')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)
    ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

    # ax.set_xlim([0, 1.1])
    # ax.set_ylim([0, 1.1])

    ax.scatter(objectives[:, 0], objectives[:, 1], c="deeppink", marker=".", zorder=10)
    ax.scatter(pareto[:, 0], pareto[:, 1], c="yellow", marker="*", zorder=10)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()


def experiment_single_fold(X, y, objective_functions, fold, path='results', full_data=True):
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

    objectives_gradient = []
    objectives_quality = []
    for i in range(10):
        if full_data:
            model = MLPClassifier(objectives=objective_functions, random_state=radom_state[i], path=path_fold, alpha=0.5, max_iter=100,
                                  n_test_rays=25)
        else:
            model = MLPClassifier(objectives=objective_functions, random_state=radom_state[i], path=path_fold, alpha=0.5, max_iter=150,
                                  n_test_rays=25, batch_size='auto')
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

        proba_predictions = model.get_logits(X_test, test_rays=test_rays)
        predictions = model.predict(X_test, test_rays=test_rays)
        gradient_metrics_array = np.zeros((len(predictions), len(objective_functions)))
        for i_p, prediction in enumerate(proba_predictions):
            for i_m, metric in enumerate(objective_functions):
                gradient_metrics_array[i_p, i_m] = metric(torch.from_numpy(prediction.astype('float32')),
                                                          torch.from_numpy(y_test).long(), reduction='mean')

        quality_metrics_array = np.zeros((len(predictions), len(objective_functions)))
        for i_p, prediction in enumerate(predictions):
            quality_metrics_array[i_p, 0] = recall_score(y_test, prediction)
            quality_metrics_array[i_p, 1] = recall_score(y_test, prediction, pos_label=0)

        objectives_gradient.append(gradient_metrics_array)
        objectives_quality.append(quality_metrics_array)

        gradient_metrics_array = np.zeros((40, len(objective_functions)))
        quality_metrics_array = np.zeros((40, len(objective_functions)))


        # model = ParetoMTLMethod(objective_functions)
        random.seed(666)
        for i in range(40):
            model = MGDAMethod(objective_functions, random_state=random.randint(0, 10000), max_iter=250)
            model.fit(X_train, y_train)
            prediction_test = model.predict(X_test)
            prediction_proba_test = model.get_logits(X_test)
            gradient_metrics_array[i, 0] = objective_functions[0](torch.from_numpy(prediction_proba_test.astype('float32')),
                                                          torch.from_numpy(y_test).long(), reduction='mean')
            gradient_metrics_array[i, 1] = objective_functions[1](torch.from_numpy(prediction_proba_test.astype('float32')),
                                                                  torch.from_numpy(y_test).long(), reduction='mean')
            quality_metrics_array[i, 0] = recall_score(y_test, prediction_test)
            quality_metrics_array[i, 1] = recall_score(y_test, prediction_test, pos_label=0)

        objectives_gradient.append(gradient_metrics_array)
        objectives_quality.append(quality_metrics_array)

        gradient_metrics_array = np.zeros((25, len(objective_functions)))
        quality_metrics_array = np.zeros((25, len(objective_functions)))

        model = ParetoMTLMethod(objective_functions)
        random.seed(666)
        for i in range(25):
            model.fit(X_train, y_train)
            prediction_test = model.predict(X_test)
            prediction_proba_test = model.get_logits(X_test)
            gradient_metrics_array[i, 0] = objective_functions[0](torch.from_numpy(prediction_proba_test.astype('float32')),
                                                                  torch.from_numpy(y_test).long(), reduction='mean')
            gradient_metrics_array[i, 1] = objective_functions[1](torch.from_numpy(prediction_proba_test.astype('float32')),
                                                                  torch.from_numpy(y_test).long(), reduction='mean')
            quality_metrics_array[i, 0] = recall_score(y_test, prediction_test)
            quality_metrics_array[i, 1] = recall_score(y_test, prediction_test, pos_label=0)

        objectives_gradient.append(gradient_metrics_array)
        objectives_quality.append(quality_metrics_array)

    objectives_gradient = np.concatenate(objectives_gradient, axis=0)
    objectives_quality = np.concatenate(objectives_quality, axis=0)

    pareto_gradient = get_not_dominated_solutions(objectives_gradient, if_min=True)
    pareto_quality = get_not_dominated_solutions(objectives_quality, if_min=False)

    draw_plot(objectives_gradient, pareto_gradient, os.path.join(path_fold, "pareto_gradient_plot.png"))
    draw_plot(objectives_quality, pareto_quality, os.path.join(path_fold, "pareto_quality_plot.png"))
    np.save(os.path.join(path_fold, f"pareto_gradient.npy"), pareto_gradient)
    np.save(os.path.join(path_fold, f"pareto_quality.npy"), pareto_quality)


def conduct_experiment(path='results',dataset='all', fold=10):
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
                # objective_functions = [BinaryCrossEntropyLoss(minc/maxc), BinaryCrossEntropyLoss(maxc/minc)]
                objective_functions = [CrossEntropyLoss(0), CrossEntropyLoss(1)]
                for i in range(10):
                    experiment_single_fold(X, y, objective_functions, i, os.path.join(path, datasets[j]))
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
                objective_functions = [BinaryCrossEntropyLoss(minc/maxc), BinaryCrossEntropyLoss(maxc/minc)]
                # objective_functions = [CrossEntropyLoss(0), CrossEntropyLoss(1)]
                for i in range(10):
                    experiment_single_fold((d[i][0][0], d[i][1][0]), (d[i][0][1], d[i][1][1]), objective_functions, i,
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
        experiment_single_fold(X, y, objective_functions, fold, os.path.join(path, dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("fold", type=int)
    parser.set_defaults(history=False)
    args = parser.parse_args()
    conduct_experiment(args.path, args.dataset, args.fold)




