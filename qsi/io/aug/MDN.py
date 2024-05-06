import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random

class MDN(nn.Module):
    def __init__(self, n_hidden, n_gaussians):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            # nn.Linear(n_hidden, n_hidden),
            # nn.ReLU(),
            # nn.Linear(n_hidden, n_hidden),
            # nn.ReLU(),
            # nn.Linear(n_hidden, n_hidden),
            # nn.ReLU(),
            # nn.Linear(n_hidden, n_hidden),
            # nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
#             nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)

    def forward(self, x):
        z_h = self.z_h(x)
        pi = F.softmax(self.z_pi(z_h), -1)
        mu = self.z_mu(z_h)
        sigma = torch.exp(self.z_sigma(z_h))
        return pi, mu, sigma


model = MDN(n_hidden=80, n_gaussians=5)
optimizer = torch.optim.Adam(model.parameters())

def mdn_loss_fn(y, mu, sigma, pi):
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = torch.exp(m.log_prob(y))
    loss = torch.sum(loss * pi, dim=1)
    loss = -torch.log(loss)
    return torch.mean(loss)

def add_one(x):
    if isinstance(x, (int, float)):
        return x + random.uniform(0,1)
    else:
        return x

def expand_dataset(X, y, nobs, n_gaussians=5, epochs=1000, n_hidden=80, verbose = False):

    final_data = pd.DataFrame()
    for label in set(y):
        print('Generating data for label',label)
        data = pd.concat([X, y], axis=1)
        data = data[data[y.name] == label]
        data = data.applymap(add_one)
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data.iloc[:, :-1])
        shift = (data.columns)[:-1]
        cls = list(data.iloc[:, -1])
        data = pd.DataFrame(normalized_data)
        data.columns = shift
        data[y.name] = cls

        input_x = data.columns[:-1].astype(float).to_list() * data.shape[0]
        input_y = []
        for i in range(data.shape[0]):
            input_y.append(list(data.iloc[i, :-1]))

        x_data_ = torch.tensor(input_x)
        x_data = x_data_.reshape(-1, 1)
        y_data_ = torch.tensor(input_y)
        y_data = y_data_.reshape(-1, 1)

        model = MDN(n_hidden=n_hidden, n_gaussians=n_gaussians)

        if verbose: # show model structure
            
            from torchviz import make_dot
            import IPython.display
            
            input_vec = torch.zeros(n_hidden, 1, dtype=torch.float, requires_grad=False) #.to('cuda')
            IPython.display.display(make_dot(model(input_vec)))

        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(epochs):
            pi, mu, sigma = model(x_data)
            loss = mdn_loss_fn(y_data, mu, sigma, pi)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print('loss of epoch', epoch, ':', loss.data.tolist())

        pi, mu, sigma = model(x_data[:(data.shape[1]-1)])
        k = torch.multinomial(pi, 1).view(-1)
        sam = []
        for i in range(nobs):
            y_pred = torch.normal(mu, sigma)[np.arange(2000), k].data
            original_data = scaler.inverse_transform(y_pred.reshape(1, -1))
            sam.append(original_data[0])

        MDN_sam = pd.DataFrame(sam)
        MDN_sam.columns = data.columns[:-1]
        MDN_sam[y.name] = [label] * len(MDN_sam)

        final_data = pd.concat([final_data, MDN_sam], axis=0)
    final_data.reset_index(drop=True, inplace=True)
    return final_data.iloc[:, :-1], final_data.iloc[:, -1]