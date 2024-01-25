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


def get_d_paretomtl_init(grads, losses, preference_vectors, pref_idx):
    """
    calculate the gradient direction for ParetoMTL initialization

    Args:
        grads: flattened gradients for each task
        losses: values of the losses for each task
        preference_vectors: all preference vectors u
        pref_idx: which index of u we are currently using

    Returns:
        flag: is a feasible initial solution found?
        weight:
    """

    flag = False
    nobj = losses.shape

    # check active constraints, Equation 7
    current_pref = preference_vectors[pref_idx]  # u_k
    w = preference_vectors - current_pref  # (u_j - u_k) \forall j = 1, ..., K
    gx = torch.matmul(w, losses / torch.norm(losses))  # In the paper they do not normalize the loss
    idx = gx > 0  # I(\theta), i.e the indexes of the active constraints

    active_constraints = w[idx]  # constrains which are violated, i.e. gx > 0

    # calculate the descent direction
    if torch.sum(idx) <= 0:
        flag = True
        return flag, torch.zeros(nobj)
    if torch.sum(idx) == 1:
        sol = torch.ones(1).float()
    else:
        # Equation 9
        # w[idx] = set of active constraints, i.e. where the solution is closer to another preference vector than the one desired.
        gx_gradient = torch.matmul(active_constraints,
                                   grads)  # We need to take the derivatives of G_j which is w.dot(grads)
        sol, nd = MinNormSolver.find_min_norm_element([[gx_gradient[t]] for t in range(len(gx_gradient))])
        sol = torch.Tensor(sol)

    # from MinNormSolver we get the weights (alpha) for each gradient. But we need the weights for the losses?
    weight = torch.matmul(sol, active_constraints)

    return flag, weight


def get_d_paretomtl(grads, losses, preference_vectors, pref_idx):
    """
    calculate the gradient direction for ParetoMTL

    Args:
        grads: flattened gradients for each task
        losses: values of the losses for each task
        preference_vectors: all preference vectors u
        pref_idx: which index of u we are currently using
    """

    # check active constraints
    current_weight = preference_vectors[pref_idx]
    rest_weights = preference_vectors
    w = rest_weights - current_weight

    gx = torch.matmul(w, losses / torch.norm(losses))
    idx = gx > 0

    # calculate the descent direction
    if torch.sum(idx) <= 0:
        # here there are no active constrains in gx
        sol, nd = MinNormSolver.find_min_norm_element_FW([[grads[t]] for t in range(len(grads))])
        return torch.tensor(sol).float()
    else:
        # we have active constraints, i.e. we have move too far away from out preference vector
        # print('optim idx', idx)
        vec = torch.cat((grads, torch.matmul(w[idx], grads)))
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])
        sol = torch.Tensor(sol)

        # FIX: handle more than just 2 objectives
        n = preference_vectors.shape[1]
        weights = []
        for i in range(n):
            weight_i = sol[i] + torch.sum(
                torch.stack([sol[j] * w[idx][j - n, i] for j in torch.arange(n, n + torch.sum(idx))]))
            weights.append(weight_i)
        # weight0 =  sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,0] for j in torch.arange(2, 2 + torch.sum(idx))]))
        # weight1 =  sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,1] for j in torch.arange(2, 2 + torch.sum(idx))]))
        # weight = torch.stack([weight0,weight1])

        weight = torch.stack(weights)

        return weight


class ParetoMTLMethod(BaseEstimator, ClassifierMixin):

    def __init__(self, objectives, num_starts=25, max_iter=50, random_state=None, learning_rate=0.001, weight_decay=0.0001, device="auto", batch_size=256, **kwargs):
        assert len(objectives) <= 2
        self.objectives = objectives
        self.num_pareto_points = num_starts
        self.init_solution_found = False
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.max_iter = max_iter
        self.batch_size = batch_size

        self.model = None
        self.pref_idx = -1
        # the original ref_vec can be obtained by circle_points(self.num_pareto_points, min_angle=0.0, max_angle=0.5 * np.pi)
        # we use the same min angle / max angle as for the other methods for comparison.
        self.ref_vec = torch.Tensor(circle_points(self.num_pareto_points)).float()

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

    def _find_initial_solution(self, batch):

        grads = {}
        losses_vec = []
        device = self._get_device()

        # obtain and store the gradient value
        for i in range(len(self.objectives)):
            self.model.zero_grad()
            # batch.update(self.model(batch))
            input, targets = batch[0].to(device), batch[1].to(device)
            task_loss = self.objectives[i](self.model(input), targets)
            losses_vec.append(task_loss.data)

            task_loss.backward()

            grads[i] = []

            # can use scalable method proposed in the MOO-MTL paper for large scale problem
            # but we keep use the gradient of all parameters in this experiment
            private_params = self.model.private_params() if hasattr(self.model, 'private_params') else []
            for name, param in self.model.named_parameters():
                if name not in private_params and param.grad is not None:
                    grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

        grads_list = [torch.cat([g for g in grads[i]]) for i in range(len(grads))]
        grads = torch.stack(grads_list)

        # calculate the weights
        losses_vec = torch.stack(losses_vec)
        self.init_solution_found, weight_vec = get_d_paretomtl_init(grads, losses_vec, self.ref_vec, self.pref_idx)

        if self.init_solution_found:
            print("Initial solution found")

        # optimization step
        self.model.zero_grad()
        for i in range(len(self.objectives)):
            # batch.update(self.model(batch))
            input, targets = batch[0].to(device), batch[1].to(device)
            task_loss = self.objectives[i](self.model(input), targets)
            if i == 0:
                loss_total = weight_vec[i] * task_loss
            else:
                loss_total = loss_total + weight_vec[i] * task_loss

        loss_total.backward()

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
            if epoch == 0:
                # we're restarting
                self.pref_idx += 1
                reset_weights(self.model)
                self.init_solution_found = False

            self.model.train()

            # if self.verbose:
            #     iterator = tqdm(train_loader)
            # else:
            #     iterator = train_loader

            for batch in train_loader:
                if epoch < 2 and not self.init_solution_found:
                    # run at most 2 epochs to find the initial solution
                    # stop early once a feasible solution is found
                    # usually can be found with a few steps
                    self._find_initial_solution(batch)
                else:
                    # run normal update
                    gradients, obj_values = calc_gradients(batch, self.model, self.objectives)

                    grads = [torch.cat([torch.flatten(v) for k, v in sorted(grads.items())]) for grads in gradients]
                    grads = torch.stack(grads)

                    # calculate the weights
                    losses_vec = torch.Tensor(obj_values)
                    weight_vec = get_d_paretomtl(grads, losses_vec, self.ref_vec, self.pref_idx)

                    normalize_coeff = len(self.objectives) / torch.sum(torch.abs(weight_vec))
                    weight_vec = weight_vec * normalize_coeff

                    # optimization step
                    loss_total = None
                    for a, objective in zip(weight_vec, self.objectives):
                        logits = self.model(batch[0])
                        targets = batch[1]
                        # batch.update(logits)
                        task_loss = objective(logits, targets)

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