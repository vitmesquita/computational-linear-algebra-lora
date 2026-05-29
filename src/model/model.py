import torch
import torch.nn as nn
import numpy as np

class SimpleModel(nn.Module):
    def __init__(self,weights_old, rank,adalora=False):
        super().__init__()
        self.old_weights = weights_old
        self.old_weights_size = weights_old.shape
        self.linear = nn.Linear(self.old_weights_size[1], self.old_weights_size[0], bias=False)
        self.activation = nn.Sigmoid()
        self.r = rank

        if adalora == False:
            self.A = nn.Parameter(torch.rand(self.r, self.old_weights_size[1]))
            self.B = nn.Parameter(torch.zeros(self.old_weights_size[0], self.r))
        elif adalora == True:
            pass
            #P,Lambda, Q = svd_decomposition()

        #self.linear.weight = 0
        self.linear.weight.requires_grad = False # garante congelamento
    
    def forward(self,x):
        deltaW = self.B @ self.A
        x = self.linear(x) + x @ deltaW.T
        x = self.activation(x)
        return x
    
    def merge(self):
        with torch.no_grad():
            self.linear.weight +=  self.B @ self.A

    # def train(self, data):
    #     pass

    # def merge(self, weights_old):
    #     pass

    # def predict(self, X):
    #     pass

if __name__ == "__main__":
    weights = np.matrix([[1,1,1,2],[4,4,4,5],[5,5,5,2],[6,6,6,3]])
    print('Defining variables')
    model = SimpleModel(weights_old=weights, rank=2)

    X = torch.randn(10, 4)   # 10 amostras, 4 features
    y = torch.randint(0, 2, (10, 4)).float()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    loss_fn = nn.BCELoss()

    for epoch in range(100):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        print("Epoch: ", epoch,"\n", f"Loss:{loss.item():.4f}")
    print(f"Loss final: {loss.item():.4f}")