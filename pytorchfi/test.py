import torch
import random
from pytorchfi.core import fault_injection as pfi_core
from pytorchfi.errormodels import (
    random_weight_inj,
    zeroFunc_rand_weight,
)
from LENET import  train_eval_lenet
from Data_load import load_mnist
train_loader,test_loader = load_mnist()
BATCH_SIZE = 4
img_size = 32
data,label = iter(test_loader).next()
model,acc = train_eval_lenet(training=False)
model.eval()
with torch.no_grad():
    output = model(data)
    p = pfi_core(model,img_size,img_size,BATCH_SIZE)

def random_weight_inj(pfi_model, corrupt_conv=-1, min_val=-1, max_val=1):
    return pfi_model.declare_weight_fi(
        conv_num=4, h=0, w=0, value=10)
def random_neuron_inj(pfi_model, min_val=-1, max_val=1):
    b = 1
    err_val = 10

    return pfi_model.declare_neuron_fi(
        batch=b, conv_num=4, w=4, value=err_val
    )

def test_random_weight_inj():
    inj_model = random_weight_inj(p, min_val=10000, max_val=20000)
    inj_model.eval()
    with torch.no_grad():
        corrupted_output_1 = inj_model(data)
    assert not torch.all(corrupted_output_1.eq(output))

def test_random_neuron_inj():
        inj_model = random_neuron_inj(p, min_val=10000, max_val=20000)

        inj_model.eval()
        with torch.no_grad():
            corrupted_output_1 = inj_model(data)

        assert not torch.all(corrupted_output_1.eq(output))


if __name__ == '__main__':
    test_random_neuron_inj()