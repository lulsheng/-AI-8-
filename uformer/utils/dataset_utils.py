import paddle

def rot90(input, k, dims):
    l = len(input.shape)
    new_dims = list(range(l))
    new_dims[dims[0]] = dims[1]
    new_dims[dims[1]] = dims[0]
    flip_dim = min(dims)
    for i in range(k):
        input = paddle.transpose(input, new_dims)
        input = paddle.flip(input, [flip_dim])
    return input

## rotate and flip
class Augment_RGB_paddle:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor
    def transform1(self, torch_tensor):
        torch_tensor = rot90(torch_tensor, k=1, dims=[2,1])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = rot90(torch_tensor, k=2, dims=[2,1])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = rot90(torch_tensor, k=3, dims=[2,1])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (rot90(torch_tensor, k=1, dims=[2,1])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (rot90(torch_tensor, k=2, dims=[2,1])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (rot90(torch_tensor, k=3, dims=[2,1])).flip(-2)
        return torch_tensor