import torch

N = 10

D_in = 1
D_out = 1

X = torch.randn(N, D_in)

epochs = 250
learning_rate = 0.01

# SETUP

true_W = torch.tensor([[2.0]])
true_b = torch.tensor(1.0)
y_true = X @ true_W + true_b + torch.randn(N, D_out) * 0.1

print(y_true)

# REGRESSION

W = torch.randn(D_in, D_out, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print(f"Initial W: {W} \nInitial b: {b}")

# y = XW + b

for epoch in range(epochs):
    y_hat = X @ W + b

    error = y_hat - y_true
    squared_error = error ** 2
    loss = squared_error.mean()
    loss.backward()

    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad

    W.grad.zero_()
    b.grad.zero_()
    
    if epoch%10 == 0:
        print(f"Epoch {epoch}/{epochs}: Loss: {loss.item():.4f} W: {W.item():.3f} b: {b.item():.3f}")
        
    
print(f"Final Params: W: {W.item():.3f} b: {b.item():.3f}")
print("Actual params: W: 2.000, b: 1.000")
