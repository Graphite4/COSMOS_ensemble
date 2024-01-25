# code from https://github.com/intel-isl/MultiObjectiveOptimization/blob/master/multi_task/train_multi_task.py
# and adapted

import torch
from torch.autograd import Variable

from _cosmos import MinNormSolver, gradient_normalizers

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.base import BaseEstimator, ClassifierMixin
from _cosmos import MinNormSolver
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from _cosmos import calc_gradients, circle_points,  reset_weights


class MGDAMethod(BaseEstimator, ClassifierMixin):

    def __init__(self, objectives, approximate_norm_solution=False, normalization_type='loss+',max_iter=100, random_state=None,
                 learning_rate=0.001, weight_decay=0.0001, device="auto", batch_size=256, **kwargs) -> None:
        super().__init__()

        self.objectives = objectives
        self.approximate_norm_solution = approximate_norm_solution
        self.normalization_type = normalization_type

        self.random_state = random_state
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.max_iter = max_iter
        self.batch_size = batch_size

        self.model = None


    def _initialize_model(self, n_features: int, n_classes: int) -> nn.modules.Module:
        layers = []

        # for i in range(len(self.hidden_layer_size) + 1):
        #     if i == 0:
        #         # input_dim = n_features + len(self.objectives) - 1
        #         input_dim = n_features + len(self.objectives)
        #     else:
        #         input_dim = self.hidden_layer_size[i - 1]
        #
        #     if i == len(self.hidden_layer_size):
        #         output_dim = n_classes
        #         # output_dim = 1
        #     else:
        #         output_dim = self.hidden_layer_size[i]
        #
        #     layers.append(nn.Linear(input_dim, output_dim))
        #
        #     if i < len(self.hidden_layer_size):
        #         layers.append(self.activation())

        model = nn.Sequential(
            nn.Linear(n_features, 60),
            nn.ReLU(),
            nn.Linear(60, 25),
            nn.ReLU(),
            nn.Linear(25, n_classes),
        )

        # return nn.Sequential(*layers)
        return model

    def _get_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = self.device

        return device

    def _get_batch_size(self, n_samples: int, default: int = 200) -> int:
        if self.batch_size == "auto":
            batch_size = min(default, n_samples)
        else:
            batch_size = self.batch_size

        return batch_size

    def new_epoch(self, e):
        self.model.train()

    def step(self, batch):

        # Scaling the loss functions based on the algorithm choice
        # loss_data = {}
        # grads = {}
        # scale = {}
        # mask = None
        # masks = {}

        # Will use our MGDA_UB if approximate_norm_solution is True. Otherwise, will use MGDA
        if self.approximate_norm_solution:
            self.model.zero_grad()
            # First compute representations (z)
            with torch.no_grad():
                # images_volatile = Variable(images.data, volatile=True)
                # rep, mask = model['rep'](images_volatile, mask)
                rep = self.model.forward_feature_extraction(batch)

            # As an approximate solution we only need gradients for input
            # if isinstance(rep, list):
            #     # This is a hack to handle psp-net
            #     rep = rep[0]
            #     rep_variable = [Variable(rep.data.clone(), requires_grad=True)]
            #     list_rep = True
            # else:
            #     rep_variable = Variable(rep.data.clone(), requires_grad=True)
            #     list_rep = False

            # Compute gradients of each loss function wrt z

            gradients = []
            obj_values = []
            for i, objective in enumerate(self.objectives):
                # zero grad
                self.model.zero_grad()

                logits = self.model.forward_linear(rep, i)
                batch.update(logits)

                output = objective(**batch)
                output.backward()

                obj_values.append(output.item())
                gradients.append({})

                private_params = self.model.private_params() if hasattr(self.model, 'private_params') else []
                for name, param in self.model.named_parameters():
                    not_private = all([p not in name for p in private_params])
                    if not_private and param.requires_grad and param.grad is not None:
                        gradients[i][name] = param.grad.data.detach().clone()
                        param.grad = None
                self.model.zero_grad()

            grads = gradients

            # for t in tasks:
            #     self.model.zero_grad()
            #     out_t, masks[t] = model[t](rep_variable, None)
            #     loss = loss_fn[t](out_t, labels[t])
            #     loss_data[t] = loss.data[0]
            #     loss.backward()
            #     grads[t] = []
            #     if list_rep:
            #         grads[t].append(Variable(rep_variable[0].grad.data.clone(), requires_grad=False))
            #         rep_variable[0].grad.data.zero_()
            #     else:
            #         grads[t].append(Variable(rep_variable.grad.data.clone(), requires_grad=False))
            #         rep_variable.grad.data.zero_()

        else:
            # This is MGDA
            grads, obj_values = calc_gradients(batch, self.model, self.objectives)

            # for t in tasks:
            #     # Comptue gradients of each loss function wrt parameters
            #     self.model.zero_grad()
            #     rep, mask = model['rep'](images, mask)
            #     out_t, masks[t] = model[t](rep, None)
            #     loss = loss_fn[t](out_t, labels[t])
            #     loss_data[t] = loss.data[0]
            #     loss.backward()
            #     grads[t] = []
            #     for param in self.model['rep'].parameters():
            #         if param.grad is not None:
            #             grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))

        # Normalize all gradients, this is optional and not included in the paper.

        gn = gradient_normalizers(grads, obj_values, self.normalization_type)
        for t in range(len(self.objectives)):
            for gr_i in grads[t]:
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        # Frank-Wolfe iteration to compute scales.
        grads = [[v for v in d.values()] for d in grads]
        sol, min_norm = MinNormSolver.find_min_norm_element(grads)
        # for i, t in enumerate(range(len(self.objectives))):
        #     scale[t] = float(sol[i])

        # Scaled back-propagation
        self.model.zero_grad()
        logits = self.model(batch)
        batch.update(logits)
        loss_total = None
        for a, objective in zip(sol, self.objectives):
            task_loss = objective(**batch)
            loss_total = a * task_loss if not loss_total else loss_total + a * task_loss

        loss_total.backward()
        return loss_total.item(), 0

        # rep, _ = model['rep'](images, mask)
        # for i, t in enumerate(tasks):
        #     out_t, _ = model[t](rep, masks[t])
        #     loss_t = loss_fn[t](out_t, labels[t])
        #     loss_data[t] = loss_t.data[0]
        #     if i > 0:
        #         loss = loss + scale[t]*loss_t
        #     else:
        #         loss = scale[t]*loss_t
        # loss.backward()

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        objective_history = []

        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        n_samples = X.shape[0]

        device = self._get_device()

        self.model = self._initialize_model(n_features, n_classes)
        self.model.to(device)
        self.model.train()

        train_dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y).long())
        train_loader = DataLoader(
            train_dataset, batch_size=self._get_batch_size(n_samples), shuffle=True
        )

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        for epoch in range(self.max_iter):
            self.model.train()

            # if self.verbose:
            #     iterator = tqdm(train_loader)
            # else:
            #     iterator = train_loader

            for batch in train_loader:
                if self.approximate_norm_solution:
                    self.model.zero_grad()
                    # First compute representations (z)
                    with torch.no_grad():
                        # images_volatile = Variable(images.data, volatile=True)
                        # rep, mask = model['rep'](images_volatile, mask)
                        rep = self.model.forward_feature_extraction(batch)

                    # As an approximate solution we only need gradients for input
                    # if isinstance(rep, list):
                    #     # This is a hack to handle psp-net
                    #     rep = rep[0]
                    #     rep_variable = [Variable(rep.data.clone(), requires_grad=True)]
                    #     list_rep = True
                    # else:
                    #     rep_variable = Variable(rep.data.clone(), requires_grad=True)
                    #     list_rep = False

                    # Compute gradients of each loss function wrt z

                    gradients = []
                    obj_values = []
                    for i, objective in enumerate(self.objectives):
                        # zero grad
                        self.model.zero_grad()

                        logits = self.model.forward_linear(rep, i)
                        batch.update(logits)

                        output = objective(**batch)
                        output.backward()

                        obj_values.append(output.item())
                        gradients.append({})

                        private_params = self.model.private_params() if hasattr(self.model, 'private_params') else []
                        for name, param in self.model.named_parameters():
                            not_private = all([p not in name for p in private_params])
                            if not_private and param.requires_grad and param.grad is not None:
                                gradients[i][name] = param.grad.data.detach().clone()
                                param.grad = None
                        self.model.zero_grad()

                    grads = gradients

                    # for t in tasks:
                    #     self.model.zero_grad()
                    #     out_t, masks[t] = model[t](rep_variable, None)
                    #     loss = loss_fn[t](out_t, labels[t])
                    #     loss_data[t] = loss.data[0]
                    #     loss.backward()
                    #     grads[t] = []
                    #     if list_rep:
                    #         grads[t].append(Variable(rep_variable[0].grad.data.clone(), requires_grad=False))
                    #         rep_variable[0].grad.data.zero_()
                    #     else:
                    #         grads[t].append(Variable(rep_variable.grad.data.clone(), requires_grad=False))
                    #         rep_variable.grad.data.zero_()

                else:
                    # This is MGDA
                    grads, obj_values = calc_gradients(batch, self.model, self.objectives)

                    # for t in tasks:
                    #     # Comptue gradients of each loss function wrt parameters
                    #     self.model.zero_grad()
                    #     rep, mask = model['rep'](images, mask)
                    #     out_t, masks[t] = model[t](rep, None)
                    #     loss = loss_fn[t](out_t, labels[t])
                    #     loss_data[t] = loss.data[0]
                    #     loss.backward()
                    #     grads[t] = []
                    #     for param in self.model['rep'].parameters():
                    #         if param.grad is not None:
                    #             grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))

                # Normalize all gradients, this is optional and not included in the paper.

                gn = gradient_normalizers(grads, obj_values, self.normalization_type)
                for t in range(len(self.objectives)):
                    for gr_i in grads[t]:
                        grads[t][gr_i] = grads[t][gr_i] / gn[t]

                # Frank-Wolfe iteration to compute scales.
                grads = [[v for v in d.values()] for d in grads]
                sol, min_norm = MinNormSolver.find_min_norm_element(grads)
                # for i, t in enumerate(range(len(self.objectives))):
                #     scale[t] = float(sol[i])

                # Scaled back-propagation
                self.model.zero_grad()

                logits = self.model(batch[0])
                target = batch[1]
                loss_total = None
                for a, objective in zip(sol, self.objectives):
                    task_loss = objective(logits, target)
                    loss_total = a * task_loss if not loss_total else loss_total + a * task_loss

                loss_total.backward()
                optimizer.step()

        # np.save(os.path.join(self.path, 'objective_history.npy'), objective_history)
        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        device = self._get_device()

        self.model.eval()

        # dataset = TensorDataset(torch.Tensor(X[:, :-1]))
        dataset = TensorDataset(torch.Tensor(X))
        loader = DataLoader(dataset, batch_size=self._get_batch_size(n_samples))

        predictions = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device)

                # outputs = torch.sigmoid(self.model(inputs))
                outputs = F.softmax(self.model(inputs), dim=1)

                for output in outputs:
                    predictions.append(output.cpu().detach().numpy())

        return np.array(predictions)

    def get_logits(self, X):
        n_samples = X.shape[0]
        device = self._get_device()

        self.model.eval()

        # dataset = TensorDataset(torch.Tensor(X[:, :-1]))
        dataset = TensorDataset(torch.Tensor(X))
        loader = DataLoader(dataset, batch_size=self._get_batch_size(n_samples))

        predictions = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device)

                outputs = self.model(inputs)

                for output in outputs:
                    predictions.append(output.cpu().detach().numpy())

        return np.array(predictions)

    def predict(self, X):
        probas = self.predict_proba(X)

        return np.argmax(probas, axis=1)
        # return (probas > 0.5).astype(int)