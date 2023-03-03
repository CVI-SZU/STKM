import torch
import collections
from model.resnet import resnet50

model_path = './saved_models/res_encoder.pth'
model = torch.load(model_path)

print(type(model))
for key, _ in model.items():
    print(key)

new_dict = collections.OrderedDict()
for key, v in model.items():
    if key.split('.')[0] == 'resnet':
        new_dict[key.split('resnet.')[-1]] = model[key]
        print(key.split('resnet.')[-1])


torch.save(new_dict, './res.pth')

model_res = resnet50()
model_res.load_state_dict(torch.load('./res.pth'))
print(model_res)
print("yes")
