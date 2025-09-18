import torch.nn as nn
import torch

class LinearRegressionModel(nn.Module):
    def __init__(self, in_params, out_params):
       super().__init__()
       
       self.linear_layer = nn.Linear(in_params, out_params)
       
    def forward(self, x):
        return self.linear_layer(x)
    

D_in, D_out = 1, 1
N = 10
epochs = 2500

X = torch.randn(N, D_in)

model = LinearRegressionModel(D_in, D_out)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())


# True parameters (what the model should learn)
true_weight = torch.tensor([[2.0]])  # shape (D_in, D_out)
true_bias = torch.tensor([1.0])      # shape (D_out)

y_true = X @ true_weight + true_bias
y_true += 0.1 * torch.randn(N, D_out)

for epoch in range(epochs):
    y_hat = model(X)

    loss = loss_fn(y_hat, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch%10 == 0:
        print(f"Epoch {epoch}/{epochs}: Loss: {loss.item():.4f}")
print(f"Final Loss: {loss.item():.4f}")