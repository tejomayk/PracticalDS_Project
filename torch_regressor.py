import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import RegressorMixin, BaseEstimator


class FFN(nn.Module):
    def __init__(self, input_dim):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.nn.functional.softplus(self.fc4(x))
        return x


class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * input_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.softplus(self.fc(x))
        return x


class TorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, epochs=1000, lr=0.01):
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
