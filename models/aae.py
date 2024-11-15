import torch.nn as nn
import torchbnn as bnn

# Discriminator
class D_net_gauss(nn.Module):
    def __init__(self, N, z_dim):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
        self.nonlinear = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.lin1(x))
        x = self.nonlinear(x)
        x = self.dropout(self.lin2(x))
        x = self.nonlinear(x)
        return self.sigmoid(self.lin3(x))

#Encoder
class Q_net(nn.Module):
    def __init__(self, X_dim, N, z_dim):
        super(Q_net, self).__init__()
        self.lin1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=X_dim, out_features=N)
        self.lin2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=N, out_features=N)
        self.lin3gauss = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=N, out_features=z_dim)
        self.nonlinear = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, x):
        x = self.dropout(self.lin1(x))
        x = self.nonlinear(x)
        x = self.dropout(self.lin2(x))
        x = self.nonlinear(x)
        xgauss = self.lin3gauss(x)
        return xgauss


# Decoder
class P_net(nn.Module):
    def __init__(self, X_dim, N, z_dim):
        super(P_net, self).__init__()
        self.lin1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=z_dim, out_features=N)
        self.lin2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=N, out_features=N)
        self.lin3 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=N, out_features=X_dim)
        self.nonlinear = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, x):
        x = self.dropout(self.lin1(x))
        x = self.nonlinear(x)
        x = self.dropout(self.lin2(x))
        x = self.lin3(x)
        return x