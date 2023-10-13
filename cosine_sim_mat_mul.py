import torch

def cosine_distance_torch(x1, x2, eps=1e-8):
    assert len(x2.size())==2
    w1 = x1.norm(p=2, dim=-1, keepdim=True)
    w2 = x2.transpose(0,1).norm(p=2, dim=-1, keepdim=True)
    # print('w1', w1.size()) # 1,2,1
    # print('w2', w2.size()) # 4,1

    mm=torch.matmul(x1, x2)
    norm=torch.matmul(w1, w2.transpose(0,1)).clamp(min=eps)

    # print('mm:', mm.size()) # 1,2,4
    # print('norm:', norm.size()) # 1,2,4
    return torch.div(mm, norm)
    
#torch.mm(x1, x2) / torch.mm(w1, w2) #.clamp(min=eps)


a=torch.rand(1,2,3) -0.5
b=torch.rand(3,4) -0.5

cosine_sim=cosine_distance_torch(a,b)
print('a:',a)
print('b:',b)
# print('axb:', torch.matmul(a,b))
print('a.b:', cosine_sim)
print(torch.softmax(cosine_sim,dim=-1))

for i in range(4):
    bb=b.transpose(0,1)[i]
    print(torch.cosine_similarity(a,bb, dim=-1))