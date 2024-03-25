import torch

x = torch.randn(1, 2)
print(x)

lower = torch.Tensor([[-0.65, -0.65]])
upper = torch.Tensor([[0.65, 0.65]])

clamped_x = torch.clip(x, lower, upper)
print(clamped_x)