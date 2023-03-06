from math import sqrt
import torch
import traceback

class NewTensor:
    def __init__(self, value: torch.Tensor):
        self.value = value  # x坐标

    def __matmul__(self, other: "NewTensor") -> torch.Tensor:
        return -torch.matmul(self.value, other.value)

    def __add__(self, other: "NewTensor") -> torch.Tensor:
        return -torch.add(self.value, other.value)


class mul_model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: NewTensor, y: NewTensor):
        return x @ y


class add_model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: NewTensor, y: NewTensor):
        return x + y
   


HANDLED_FUNCTIONS = {}

import functools
def implements(torch_function):
    """Register a torch function override for ScalarTensor"""
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator
class ScalarTensor(object):
    def __init__(self, N, value):
        self._N = N
        self._value = value
        self.HANDLED_FUNCTIONS = {}

    def __repr__(self):
        return "DiagonalTensor(N={}, value={})".format(self._N, self._value)

    def tensor(self):
        return self._value * torch.eye(self._N)

    @torch.jit.unused
    @classmethod
    def __torch_function__(cls, func, types, args=()):

        return HANDLED_FUNCTIONS[func](*args)
    
    @implements(torch.matmul)
    def __matmul__(input:"ScalarTensor", other:"ScalarTensor"):
        return float(input._value* other._value)
    
d = ScalarTensor(5, 2)
d2 = ScalarTensor(4, 3)
print(d@d2)

# @implements(torch.mean)
# def mean(input):
#     return float(input._value) / input._N



# @implements(torch.matmul)
# def matmul(input: ScalarTensor, other: ScalarTensor)->torch.Tensor:
#     return torch.eye(input._value) @ torch.eye(other._value)


scripted_model = torch.jit.script(d)



# class mul_model:
#     def __init__(self) -> None:
#         pass

#     def forward(self, x: ScalarTensor, y: ScalarTensor):
#         return x @ y

# model_mul = mul_model()
# try:
#     a_s = torch.jit.script(model_mul)
#     print("mul model success")
# except:
#     print("mul model fail")
#     traceback.print_exc()
# model_add = add_model()
# try:
#     a_s = torch.jit.script(model_add)
#     print("add model success")
# except Exception as e:
#     print("add model fail")
#     traceback.print_exc()

# model_mul = mul_model()
# a_s = torch.jit.script(model_mul)
# print("mul model success")
