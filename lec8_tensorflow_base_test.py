import torch
import numpy as np

r0 = torch.tensor(1.0).float()
# print(type(r0))
# print(r0.type)
# print(r0.shape)
# print(r0.data)
# print(r0)

r1 = torch.tensor(np.array([1,2,3,4,5])).float()
# print(type(r1))
# print(r1.type)
# print(r1.shape)
# print(r1.data)
# print(r1)


r2_np = np.array([[1,5,6], [4,3,2]])
# print(r2_np.shape)
r2 = torch.tensor(r2_np).float()
# print(type(r2))
# print(r2.type)
# print(r2.shape)
# print(r2.data)
# print(r2)

torch.manual_seed(123)
# r3 = torch.randn((3,2,2))
# print(type(r3))
# print(r3.type)
# print(r3.shape)
# print(r3.data)
# print(r3)

x = torch.Tensor([[1,2],[3,4]])
# print("1.", x)
# print("2.", x.numpy())

y = torch.from_numpy(np.array([[1,2],[3,4]]))
# print("3.", y)
# print(y.to('cuda'))    # device='cuda:0'
# print(y.to('cpu').numpy())
#
# print(x.dim())
# print(x.size())

temp = torch.FloatTensor([1,2,3,4,5,6,7])
# print(temp[0], temp[1], temp[-1])
# print(temp[2:5], temp[4:-1])   ##temp[4:-1]'s output : 5., 6.

v = torch.tensor([1,2,3])
w = torch.tensor([3,4,6])
# print(v + w)
# print(v - w)

temp_2 = torch.tensor([[1,2], [3,4]])
# print(temp_2.shape)
# print("1.", temp_2.view(4,1))   # 4 by 1 형태로 재구성함. (차원 수는 2차원으로 기존과 동일함.)
# print("2.", temp_2.view(-1))    #차원 수 하나 줄임. 즉, 1차원으로 바뀜
# print("3.", temp_2.view(1, -1))    # 1(고정) by outo 형태로 재구성함.(shape : 1 by 4)
# print("4.", temp_2.view(-1, 1))    # outo by 1(고정) 형태로 재구성함.(shape : 4 by 1)

ft = torch.Tensor([[0], [1], [2]])   #2d의 3by 1행렬 구성
print(ft)
print(ft.shape)
print(ft.squeeze())   #squeeze : 1인 차원 제거
print(ft.squeeze().shape)

x_2 = torch.FloatTensor([[1,2],[3,4]])
y_2 = torch.FloatTensor([[5,6],[7,8]])
print(x_2)
print(y_2)
print(torch.cat([x,y], dim=0))   #dim = 0 세로로 붙임(수직으로). row를 늘린다고 생각하면 이해하기 쉬움.
print(torch.cat([x,y], dim=1))   #dim = 1 가로로 붙임(수평으로). column을 늘린다고 생각하면 이해하기 쉬움