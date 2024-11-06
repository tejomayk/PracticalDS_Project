import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import RegressorMixin, BaseEstimator


class FFN(nn.Module):
    def __init__(self, input_dim):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, epochs=200, lr=0.01):
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def fit(self, X, y):
        input_dim = X.shape[1]
        self.model = FFN(input_dim=input_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        if hasattr(X, "toarray"):
            X = X.toarray()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32).view(-1, 1)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            if hasattr(X, "toarray"):
                X = X.toarray()
            X_tensor = torch.tensor(X, dtype=torch.float32)
            predictions = self.model(X_tensor)
        return predictions.numpy().flatten()
