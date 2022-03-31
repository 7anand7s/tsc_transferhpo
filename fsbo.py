###
# Code by Sebastian Pineda Arango
#
###

from email.policy import strict

import matplotlib.pyplot
import pandas as pd
from torch import nn
import torch
import random
import gpytorch
import copy
import numpy as np
import torch.nn.functional as F

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
np.random.seed(0)


class SmallConvNet(nn.Module):

    def __init__(self, in_channels=2, length=11, out_size=5, n_filters=4, kernel_size=3):
        super(SmallConvNet, self).__init__()

        self.length = length
        self.out_size = out_size
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=n_filters, kernel_size=kernel_size, stride=1,
                              padding=1)
        self.linear = nn.Linear(n_filters * (length - 2), out_size)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.max_pool1d(x, 3, stride=1)
        x = x.flatten(start_dim=1)
        x = F.relu(self.linear(x))

        return x


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers,
                 n_output=None,
                 batch_norm=False,
                 dropout_rate=0.0,
                 use_cnn=False,
                 **kwargs):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.out_features = n_hidden

        self.hidden = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.dropout = nn.ModuleList()

        self.n_output = n_hidden if n_output is None else n_output

        if use_cnn:
            self.small_cnn = SmallConvNet(length=kwargs["seq_len"])
            self.n_input = self.n_input + self.small_cnn.out_size

        self.hidden.append(nn.Linear(self.n_input, n_hidden))
        for i in range(1, n_layers - 1):
            self.hidden.append(nn.Linear(n_hidden, n_hidden))

            if batch_norm:
                self.bn.append(nn.BatchNorm1d(n_hidden))

            if dropout_rate > 0.0:
                self.dropout.append(nn.Dropout(dropout_rate))

        self.hidden.append(nn.Linear(n_hidden, self.n_output))

        self.relu = nn.ReLU()

    def forward(self, x, w=None):

        if w is not None:
            w = self.small_cnn(w)
            x = torch.cat((x, w), axis=1)

        x = self.hidden[0](x)
        for i in range(1, self.n_layers - 1):

            if self.batch_norm:
                x = self.bn[i - 1](x)
            x = self.relu(x)
            if self.dropout_rate > 0.0:
                x = self.dropout[i - 1](x)
            x = self.hidden[i](x)
        out = self.hidden[-1](self.relu(x))

        return out


class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, config, dims):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        ## RBF kernel
        if (config["kernel"] == 'rbf' or config["kernel"] == 'RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=dims if config["ard"] else None))
        elif (config["kernel"] == '52'):
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=config["nu"], ard_num_dims=dims if config["ard"] else None))
        ## Spectral kernel
        else:
            raise ValueError("[ERROR] the kernel '" + str(
                config["kernel"]) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, 2))
        self.out_features = 2


class FSBO(nn.Module):

    def __init__(self, train_data, validation_data, conf, feature_extractor):

        super(FSBO, self).__init__()
        self.train_data = train_data
        self.validation_data = validation_data
        self.feature_extractor = feature_extractor
        self.conf = conf
        self.context_size = conf.get("context_size", 5)
        self.device = conf.get("device", "cpu")
        self.lr = conf.get("lr", 0.0001)
        self.model_path = conf.get("model_path", "model.pt")
        self.scheduler_fn = lambda x, y: torch.optim.lr_scheduler.CosineAnnealingLR(x, y, eta_min=1e-7)
        self.get_model_likelihood_mll(self.context_size)
        self.training_tasks = list(train_data.keys())
        self.validation_tasks = list(validation_data.keys())
        self.use_perf_hist = conf.get("use_perf_hist", False)

    def get_train_batch(self):

        tasks = list(self.train_data.keys())
        task = np.random.choice(tasks, 1).item()

        shape = len(self.train_data[task]["X"])
        idx = np.random.randint(0, shape, self.context_size)
        x = torch.FloatTensor(self.train_data[task]["X"])[idx].to(self.device)
        y = torch.FloatTensor(self.train_data[task]["y_val"])[idx].to(self.device)

        if self.use_perf_hist:
            w = torch.FloatTensor(self.train_data[task]["perf_hist"])[idx].transpose(1, 2).to(self.device)
        else:
            w = None

        return x, y, w

    def get_val_batch(self, task):

        tasks = list(self.validation_data.keys())
        task = np.random.choice(tasks, 1).item()

        shape = len(self.validation_data[task]["X"])
        idx_spt = np.random.randint(0, shape, self.context_size)
        idx_qry = np.random.randint(0, shape, self.context_size)

        x_spt = torch.FloatTensor(self.validation_data[task]["X"])[idx_spt].to(self.device)
        y_spt = torch.FloatTensor(self.validation_data[task]["y_val"])[idx_spt].to(self.device)

        x_qry = torch.FloatTensor(self.validation_data[task]["X"])[idx_qry].to(self.device)
        y_qry = torch.FloatTensor(self.validation_data[task]["y_val"])[idx_qry].to(self.device)

        if self.use_perf_hist:
            w_spt = torch.FloatTensor(self.validation_data[task]["perf_hist"])[idx_spt].transpose(1, 2).to(self.device)
            w_qry = torch.FloatTensor(self.validation_data[task]["perf_hist"])[idx_qry].transpose(1, 2).to(self.device)
        else:
            w_spt = None
            w_qry = None

        return x_spt, x_qry, y_spt, y_qry, w_spt, w_qry

    def get_model_likelihood_mll(self, train_size):
        train_x = torch.ones(train_size, self.feature_extractor.out_features).to(self.device)
        train_y = torch.ones(train_size).to(self.device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, config=self.conf,
                             dims=self.feature_extractor.out_features)

        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device)

    def train(self, epochs=10, n_batches=20):

        best_loss = np.inf

        val_losses = []
        train_losses = []


        for epoch in range(epochs):
            # c2 = 0
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = self.scheduler_fn(optimizer, n_batches)
            temp_train_loss = 0
            for batch in range(n_batches):
                try:
                    optimizer.zero_grad()
                    x, y, w = self.get_train_batch()
                    z = self.feature_extractor(x, w)
                    self.model.set_train_data(inputs=z, targets=y)
                    predictions = self.model(z)
                    loss = -self.mll(predictions, self.model.train_targets)
                    temp_train_loss += loss.detach().cpu().numpy().item()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                except Exception as e:
                    print(e)
                # finally:
                #     c2 += 1
            train_losses.append(temp_train_loss/n_batches)
            print("Train :", epoch, train_losses[-1])
            temp_val_loss = 0

            count = 0
            for task in self.validation_tasks:
                val_batch = self.get_val_batch(task)
                done, val = self.test(val_batch)
                temp_val_loss += val.detach().cpu().numpy().item()

                if done:
                    count += 1

            if count > 0:
                val_losses.append(temp_val_loss / count)
                print("Valid :", epoch, val_losses[-1])
                if best_loss > val_losses[-1]:
                    best_loss = val_losses[-1]
                    self.save_checkpoint(self.model_path)

        matplotlib.pyplot.plot(np.arange(len(train_losses)), train_losses, 'r')
        matplotlib.pyplot.plot(np.arange(len(val_losses)), val_losses, 'g')
        matplotlib.pyplot.show()

    def test(self, val_batch):

        x_spt, x_qry, y_spt, y_qry, w_spt, w_qry = val_batch
        done = False
        try:
            z_spt = self.feature_extractor(x_spt, w_spt).detach()
            self.model.set_train_data(inputs=z_spt, targets=y_spt, strict=False)

            self.model.eval()
            self.feature_extractor.eval()
            self.likelihood.eval()

            with torch.no_grad():
                z_qry = self.feature_extractor(x_qry, w_qry).detach()
                pred = self.likelihood(self.model(z_qry))
                loss = -self.mll(pred, y_qry)

            done = True
        except Exception as e:
            print(e)

        self.model.train()
        self.feature_extractor.train()
        self.likelihood.train()

        return done, loss

    def save_checkpoint(self, checkpoint_path):
        # save state
        gp_state_dict = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict = self.feature_extractor.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'net': nn_state_dict}, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['net'])

    def finetuning(self, x, y, w, epochs=10, patience=10, finetuning_lr=0.01, tol=0.0001):

        best_loss = np.inf
        patience_counter = 0
        self.load_checkpoint(self.model_path)
        weights = copy.deepcopy(self.state_dict())

        self.model.train()
        self.feature_extractor.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=finetuning_lr)
        losses = [np.inf]

        for epoch in range(epochs):

            try:
                optimizer.zero_grad()
                z = self.feature_extractor(x, w)
                self.model.set_train_data(inputs=z, targets=y, strict=False)
                predictions = self.model(z)
                loss = -self.mll(predictions, self.model.train_targets)
                loss.backward()
                optimizer.step()

                losses.append(loss.detach().cpu().item())
                if best_loss > losses[-1]:
                    best_loss = losses[-1]
                    weights = copy.deepcopy(self.state_dict())

                if np.allclose(losses[-1], losses[-2], atol=self.conf["loss_tol"]):
                    patience_counter += 1
                else:
                    patience_counter = 0
                if patience_counter > patience:
                    break


            except Exception as ada:
                print(f"Exception {ada}")
                break

        self.load_state_dict(weights)
        return losses

    def predict(self, x_spt, y_spt, x_qry, w_spt, w_qry):

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        z_spt = self.feature_extractor(x_spt, w_spt)
        self.model.set_train_data(inputs=z_spt, targets=y_spt, strict=False)

        with torch.no_grad():
            z_qry = self.feature_extractor(x_qry, w_qry)
            pred = self.likelihood(self.model(z_qry))

        mu = pred.mean.detach().to("cpu").numpy().reshape(-1, )
        stddev = pred.stddev.detach().to("cpu").numpy().reshape(-1, )

        return mu, stddev


def retreive_table(dataset_name):
    df = pd.read_json(
        'E:/thesis_work/tsc_transferhpo/benchmark/benchmark/' + dataset_name + '/Running_' + dataset_name + '.json',
        lines=True)
    df['use_residual'] = df.use_residual == 'True'
    df['use_residual'] = df['use_residual'].astype(int)
    df['use_bottleneck'] = df['use_bottleneck'] == 'True'
    df['use_bottleneck'] = df['use_bottleneck'].astype(int)
    df['depth'] = df['depth'].astype(int)
    df['nb_filters'] = df['nb_filters'].astype(int)
    df['kernel_size'] = df['kernel_size'].astype(int)
    df['batch_size'] = df['batch_size'].astype(int)
    acc = df['acc'].to_numpy()
    df = df.drop(columns=['dataset', 'acc', 'precision', 'recall', 'budget_ran', 'duration'])
    hps = df.to_numpy()

    return hps, acc


if __name__ == '__main__':

    train_data = {}
    t_data = ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
              'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee',
              'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
              'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
              'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour',
              'FacesUCR', 'FISH', 'FordA', 'FordB', 'Gun_Point', 'Ham', 'HandOutlines',
              'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand',
              'LargeKitchenAppliances', 'Lighting2', 'Lighting7']
    for each_data in t_data:
        try:
            batch, batch_labels = retreive_table(each_data)
            batch_labels = batch_labels
            train_data[each_data] = {}
            train_data[each_data]["X"] = batch
            train_data[each_data]["y_val"] = batch_labels
            print("train_data : ", each_data)
        except:
            pass

    v_data = ['MALLAT', 'Meat', 'MedicalImages',
              'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
              'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'OliveOil']

    validation_data = {}
    for each_data in t_data:
        try:
            batch, batch_labels = retreive_table(each_data)
            batch_labels = batch_labels
            validation_data[each_data] = {}
            validation_data[each_data]["X"] = batch
            validation_data[each_data]["y_val"] = batch_labels
            print("val_data : ", each_data)
        except:
            pass

    testing_data = ['OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
                 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
                 'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface',
                 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols',
                 'synthetic_control', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
                 'Two_Patterns', 'UWaveGestureLibraryAll', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
                 'uWaveGestureLibrary_Z', 'wafer', 'Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'yoga']

    test_data = {}
    tc = 0
    for each_data in testing_data:
        try:
            batch, batch_labels = retreive_table(each_data)
            batch_labels = batch_labels
            test_data[each_data] = {}
            test_data[each_data]["X"] = batch
            test_data[each_data]["y_val"] = batch_labels
            print("test_data : ", each_data)
            tc+=1
        except:
            pass

    data_dim = train_data[np.random.choice(list(train_data.keys()), 1).item()]["X"].shape[1]
    print(data_dim)
    feature_extractor = MLP(data_dim, n_hidden=32, n_layers=3, n_output=None,
                            batch_norm=False,
                            dropout_rate=0.0,
                            use_cnn=False)
    #n outpuit = out dim
    #n hidden = number of neurons
    conf = {
        "context_size": 10,
        "device": "cpu",
        "lr": 0.001,
        "model_path": 'E:/thesis_work/tsc_transferhpo/benchmark/model.pt',
        "use_perf_hist": False,
        "loss_tol": 0.0001,
        "kernel": "rbf",
        "ard": None
    }

    fsbo = FSBO(train_data, validation_data, conf, feature_extractor)
    fsbo.train(epochs=2500)

    for x in range(tc//2):

        tasks = list(test_data.keys())
        task = test_data.pop(random.choice(tasks))

        # task = test_data.pop(np.random.choice(test_data.keys()))

        shape = len(task["X"])
        idx_spt = np.random.randint(0, shape, 5)
        idx_qry = np.random.randint(0, shape, 5)

        x_qry = torch.FloatTensor(task["X"])[idx_qry].to("cpu")
        y_qry = torch.FloatTensor(task["y_val"])[idx_qry].to("cpu")

        x_spt = torch.FloatTensor(task["X"])[idx_spt].to("cpu")
        y_spt = torch.FloatTensor(task["y_val"])[idx_spt].to("cpu")

        fsbo.finetuning(x_spt, y_spt, w=None, epochs=100)

        y_predicted, _ = fsbo.predict(x_spt, y_spt, x_qry, w_spt=None, w_qry=None)
        # print("original", y_qry, "predicted", y_predicted)

        # loss = -fsbo.mll(y_predicted, y_qry)

        loss = np.abs(np.subtract(y_predicted, y_qry))
        print(loss)

        # print(np.subtract(y_predicted, y_qry))