import torch

a = torch.tensor([[1., 2., 3.],
                  [2., 2., 4.],
                  [4., 5., 1.],
                  [2., 1., 1.]])

b = torch.tensor([[2., 2., 1.],
                  [3., 2., 1.],
                  [3., 5., 1.],
                  [8., 4., 2.]])

print(a.shape)
print(b.shape)

c1 = torch.bmm(a.view(a.shape[0], 1, a.shape[1]),
               b.view(b.shape[0], b.shape[1], 1)).squeeze(-1)

print(c1.shape, c1)

# divide length
l = ( torch.tensor([2, 1, 3, 2]) * torch.tensor([2, 2, 1, 3]) ).unsqueeze(-1)
print(l.shape, l)

l1 = torch.tensor([2, 1, 3, 2]) * torch.tensor([2, 2, 1, 3])
print("##### l1", l1)

# c = c1 / l
# print(c.shape, c)
downward_xy = a[:, :2]
command_xy = b[:, :2]
print("*****", ( torch.norm(downward_xy, dim=1) * torch.norm(command_xy, dim=1) ).unsqueeze(-1))
cosine_facing = c1 / ( torch.norm(downward_xy, dim=1) * torch.norm(command_xy, dim=1) ).unsqueeze(-1)
print(cosine_facing.shape, cosine_facing)

print("#####", torch.norm(downward_xy, dim=1).unsqueeze(-1))
downward_projection = torch.norm(downward_xy, dim=1).unsqueeze(-1) * cosine_facing
print(downward_projection)


# e = b * c
# print(e.shape, e)