import torch
import torch.nn as nn
import numpy as np

class SimpleModel(nn.Module):
    def __init__(self,weights_old, rank):
        super().__init__()
        self.old_weights = torch.tensor(weights_old,dtype=torch.float32)
        self.old_weights_size = self.old_weights.shape
        self.linear = nn.Linear(self.old_weights_size[1], self.old_weights_size[0], bias=False)
        self.activation = nn.Sigmoid()
        self.r = rank

        elif self.adalora == True:
            self.P = nn.Parameter(torch.rand(self.old_weights_size[1],self.r))
            self.Q = nn.Parameter(torch.rand(self.r,self.old_weights_size[0]))
            self.Lambda = nn.Parameter(torch.zeros(self.r,1))

        self.linear.weight.data.copy_(self.old_weights)
        self.linear.weight.requires_grad = False # garante congelamento
    
    def forward(self,x):
        if self.adalora == False:
            deltaW = self.B @ self.A
            x = self.linear(x) + x @ deltaW.T
        elif self.adalora == True:
            deltaW = self.P @ (self.Lambda * self.Q)
            x = self.linear(x) + x @ deltaW.T
        x = self.activation(x)
        return x
    
    def merge(self):
        with torch.no_grad():
            if self.adalora==False:
                self.linear.weights +=  self.B @ self.A
            elif self.adalora==True:
                self.linear.weights += self.P @ (self.Lambda * self.Q)

    # def train(self, data):
    #     pass

    # def predict(self, X):
    #     pass

class ScoreCalculator():
    def __init__(self, weight_old):
        self.old_weight = weight_old
        self.score_matrix = torch.zeros(self.old_weights.shape[0],self.old_weights.shape[1])
        self.I_bar = torch.zeros_like(weight_old)
        self.U_bar = torch.zeros_like(weight_old)

    def calculate_sensibility(self, param):
        with torch.no_grad():
            grad = param.grad
            I = (param * grad).abs()
            return I

    def scoring(self, param, beta_1, beta_2):
        with torch.no_grad():
            I = self.calculate_sensibility(param)
            self.I_bar = beta_1*self.I_bar + (1-beta_1)*I
            self.U_bar = beta_2*self.U_bar + (1-beta_2)*(I-self.I_bar).abs()
            score_matrix = self.I_bar @ self.U_bar
        return score_matrix
    
class BudgetAllocation(ScoreCalculator):
    def __init__(self,P,Lambda,Q, beta_1,beta_2,budget):
        self.d = P.shape[0]
        self.k = Q.shape[1]
        self.P_score = self.scoring(P,beta_1,beta_2)
        self.Lambda_score = self.scoring(Lambda,beta_1,beta_2)
        self.Q_score = self.scoring(Q,beta_1,beta_2)
        self.budget = budget
        self.ipt = 

    def calculate_ipt():
        with torch.no_grad():
            self.ipt = self.Lambda_score
            self.ipt +=  (1/self.d)*(self.P_score.sum(dim=0))
            self.ipt += (1/self.k)*(self.Q_score.sum(dim=1))
    
    def create_mask():



if __name__ == "__main__":
    weights = np.array([[1,1,1,2],[4,4,4,5],[5,5,5,2],[6,6,6,3]],dtype=np.float32)
    reg_weight = 1
    adalora = True
    rank=2
    print('Defining variables')
    model = SimpleModel(weights_old=weights, rank=rank)

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

        if adalora == True:
            for name,p in model.named_parameters():
                if name == "P":
                    reg_P = p.T @ p
                if name == "Q":
                    reg_Q = p @ p.T
            reg = torch.norm(reg_P - torch.eye(reg_P.shape[0]), p="fro") + torch.norm(reg_Q - torch.eye(reg_Q.shape[0]), p="fro")
                
            loss += reg_weight*reg

        loss.backward()
        optimizer.step()

        if adalora == True:
            for name,p in model.named_parameters():
                if name == "Lambda":
                    Lambda_ = p.T @ p

        print("Epoch: ", epoch,"\n", f"Loss:{loss.item():.4f}")
    print(f"Loss final: {loss.item():.4f}")