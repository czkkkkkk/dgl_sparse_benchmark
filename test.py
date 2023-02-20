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


class mul_model:
    def __init__(self) -> None:
        pass

    def forward(self, x: NewTensor, y: NewTensor):
        return x @ y


class add_model:
    def __init__(self) -> None:
        pass

    def forward(self, x: NewTensor, y: NewTensor):
        return x + y


model_add = add_model()
try:
    a_s = torch.jit.script(model_add)
    print("add model success")
except Exception as e:
    print("add model fail")
    traceback.print_exc()

model_mul = mul_model()
try:
    a_s = torch.jit.script(model_mul)
    print("mul model success")
except:
    print("mul model fail")
    traceback.print_exc()