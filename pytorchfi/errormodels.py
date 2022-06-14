"""
pytorchfi.errormodels provides different error models out-of-the-box for use.
"""

import random
import logging
import torch
from pytorchfi import core
from fxpmath import Fxp

"""
helper functions
"""


def random_batch_element(pfi_model):
    return random.randint(0, pfi_model.get_total_batches() - 1)


def random_neuron_location(pfi_model, conv=-1):
    if conv == -1:
        conv = random.randint(0, pfi_model.get_total_conv() - 1)

    c = random.randint(0, pfi_model.get_fmaps_num(conv) - 1)
    h = random.randint(0, pfi_model.get_fmaps_H(conv) - 1)
    w = random.randint(0, pfi_model.get_fmaps_W(conv) - 1)

    return (conv, c, h, w)


def random_weight_location(pfi_model, conv=-1):
    loc = list()

    if conv == -1:
        corrupt_layer = random.randint(0, pfi_model.get_total_conv() - 1)
    else:
        corrupt_layer = conv
    loc.append(corrupt_layer)

    curr_layer = 0
    for name, param in pfi_model.get_original_model().named_parameters():
        if "features" in name and "weight" in name:
            if curr_layer == corrupt_layer:
                for dim in param.size():
                    loc.append(random.randint(0, dim - 1))
            curr_layer += 1
        if "classifier" in name and "weight" in name:
            if curr_layer == corrupt_layer:
                for dim in param.size():
                    loc.append(random.randint(0, dim - 1))
            curr_layer += 1
    # assert curr_layer == pfi_model.get_total_conv()
    # assert len(loc) == 5

    return tuple(loc)


def random_value(min_val=-1, max_val=1):
    return random.uniform(min_val, max_val)


"""
Neuron Perturbation Models
"""


# single random neuron error in single batch element
def random_neuron_inj(pfi_model, min_val=-1, max_val=1):
    b = random_batch_element(pfi_model)
    (conv, C, H, W) = random_neuron_location(pfi_model)
    err_val = random_value(min_val=min_val, max_val=max_val)

    return pfi_model.declare_neuron_fi(
        batch=b, conv_num=conv, c=C, h=H, w=W, value=err_val
    )


# single random neuron error in each batch element.
def random_neuron_inj_batched(
    pfi_model, min_val=-1, max_val=1, randLoc=True, randVal=True
):
    batch, conv_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    if not randLoc:
        (conv, C, H, W) = random_neuron_location(pfi_model)
    if not randVal:
        err_val = random_value(min_val=min_val, max_val=max_val)

    for i in range(pfi_model.get_total_batches()):
        if randLoc:
            (conv, C, H, W) = random_neuron_location(pfi_model)
        if randVal:
            err_val = random_value(min_val=min_val, max_val=max_val)

        batch.append(i)
        conv_num.append(conv)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)
        value.append(err_val)

    return pfi_model.declare_neuron_fi(
        batch=batch, conv_num=conv_num, c=c_rand, h=h_rand, w=w_rand, value=value
    )


# one random neuron error per layer in single batch element
def random_inj_per_layer(pfi_model, min_val=-1, max_val=1):
    batch, conv_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    b = random_batch_element(pfi_model)
    for i in range(pfi_model.get_total_conv()):
        (conv, C, H, W) = random_neuron_location(pfi_model, conv=i)
        batch.append(b)
        conv_num.append(conv)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)
        value.append(random_value(min_val=min_val, max_val=max_val))

    return pfi_model.declare_neuron_fi(
        batch=batch, conv_num=conv_num, c=c_rand, h=h_rand, w=w_rand, value=value
    )


# one random neuron error per layer in each batch element
def random_inj_per_layer_batched(
    pfi_model, min_val=-1, max_val=1, randLoc=True, randVal=True
):
    batch, conv_num, c_rand, h_rand, w_rand, value = ([] for i in range(6))

    for i in range(pfi_model.get_total_conv()):
        if not randLoc:
            (conv, C, H, W) = random_neuron_location(pfi_model, conv=i)
        if not randVal:
            err_val = random_value(min_val=min_val, max_val=max_val)

        for b in range(pfi_model.get_total_batches()):
            if randLoc:
                (conv, C, H, W) = random_neuron_location(pfi_model, conv=i)
            if randVal:
                err_val = random_value(min_val=min_val, max_val=max_val)

            batch.append(b)
            conv_num.append(conv)
            c_rand.append(C)
            h_rand.append(H)
            w_rand.append(W)
            value.append(err_val)

    return pfi_model.declare_neuron_fi(
        batch=batch, conv_num=conv_num, c=c_rand, h=h_rand, w=w_rand, value=value
    )


class single_bit_flip_func(core.fault_injection):
    def __init__(self, model, h, w, batch_size, **kwargs):
        super().__init__(model, h, w, batch_size, **kwargs)
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")
        self.bits = kwargs.get("bits", 8)
        self.LayerRanges = []

    def set_conv_max(self, data):
        self.LayerRanges = data

    def reset_conv_max(self, data):
        self.LayerRanges = []

    def get_conv_max(self, layer):
        return self.LayerRanges[layer]

    def _twos_comp_shifted(self, val, nbits):
        if val < 0:
            val = (1 << nbits) + val
        else:
            val = self._twos_comp(val, nbits)
        return val

    def _twos_comp(self, val, bits):
        # compute the 2's complement of int value val
        if (val & (1 << (bits - 1))) != 0:  # if sign bit is set e.g., 8bit: 128-255
            val = val - (1 << bits)  # compute negative value
        return val  # return positive value as is

    def _flip_bit_signed(self, orig_value, max_value, bit_pos):
        # quantum value
        save_type = orig_value.dtype
        total_bits = self.bits
        logging.info("orig value:", orig_value)

        quantum = int((orig_value / max_value) * ((2.0 ** (total_bits - 1))))
        twos_comple = self._twos_comp_shifted(quantum, total_bits)  # signed
        logging.info("quantum:", quantum)
        logging.info("twos_comple:", twos_comple)

        # binary representation
        bits = bin(twos_comple)[2:]
        logging.info("bits:", bits)

        # sign extend 0's
        temp = "0" * (total_bits - len(bits))
        bits = temp + bits
        assert len(bits) == total_bits
        logging.info("sign extend bits", bits)

        # flip a bit
        # use MSB -> LSB indexing
        assert bit_pos < total_bits

        bits_new = list(bits)
        bit_loc = total_bits - bit_pos - 1
        if bits_new[bit_loc] == "0":
            bits_new[bit_loc] = "1"
        else:
            bits_new[bit_loc] = "0"
        bits_str_new = "".join(bits_new)
        logging.info("bits", bits_str_new)

        # GPU contention causes a weird bug...
        if not bits_str_new.isdigit():
            logging.info("Error: Not all the bits are digits (0/1)")

        # convert to quantum
        assert bits_str_new.isdigit()
        new_quantum = int(bits_str_new, 2)
        out = self._twos_comp(new_quantum, total_bits)
        logging.info("out", out)

        # get FP equivalent from quantum
        new_value = out * ((2.0 ** (-1 * (total_bits - 1))) * max_value)
        logging.info("new_value", new_value)

        return torch.tensor(new_value, dtype=save_type)


    def single_bit_flip_signed_across_batch(self, module, input, output):
        corrupt_conv_set = self.get_corrupt_conv()
        range_max = self.get_conv_max(self.get_curr_conv())
        logging.info("curr_conv", self.get_curr_conv())
        logging.info("range_max", range_max)

        if type(corrupt_conv_set) == list:
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.get_curr_conv(),
                    range(len(corrupt_conv_set)),
                )
            )
            for i in inj_list:
                self.assert_inj_bounds(index=i)
                prev_value = output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][
                    self.CORRUPT_H[i]
                ][self.CORRUPT_W[i]]

                rand_bit = random.randint(0, self.bits - 1)
                logging.info("rand_bit", rand_bit)
                new_value = self._flip_bit_signed(prev_value, range_max, rand_bit)

                output[self.CORRUPT_BATCH[i]][self.CORRUPT_C[i]][self.CORRUPT_H[i]][
                    self.CORRUPT_W[i]
                ] = new_value

        else:
            self.assert_inj_bounds()
            if self.get_curr_conv() == corrupt_conv_set:
                prev_value = output[self.CORRUPT_BATCH][self.CORRUPT_C][self.CORRUPT_H][
                    self.CORRUPT_W
                ]

                rand_bit = random.randint(0, self.bits - 1)
                logging.info("rand_bit", rand_bit)
                new_value = self._flip_bit_signed(prev_value, range_max, rand_bit)

                output[self.CORRUPT_BATCH][self.CORRUPT_C][self.CORRUPT_H][
                    self.CORRUPT_W
                ] = new_value

        self.updateConv()
        if self.get_curr_conv() >= self.get_total_conv():
            self.reset_curr_conv()


def random_neuron_single_bit_inj_batched(pfi_model, layer_ranges, randLoc=True):
    pfi_model.set_conv_max(layer_ranges)
    batch, conv_num, c_rand, h_rand, w_rand = ([] for i in range(5))

    if not randLoc:
        (conv, C, H, W) = random_neuron_location(pfi_model)

    for i in range(pfi_model.get_total_batches()):
        if randLoc:
            (conv, C, H, W) = random_neuron_location(pfi_model)

        batch.append(i)
        conv_num.append(conv)
        c_rand.append(C)
        h_rand.append(H)
        w_rand.append(W)

    return pfi_model.declare_neuron_fi(
        batch=batch,
        conv_num=conv_num,
        c=c_rand,
        h=h_rand,
        w=w_rand,
        function=pfi_model.single_bit_flip_signed_across_batch,
    )


def random_neuron_single_bit_inj(pfi_model, layer_ranges):
    pfi_model.set_conv_max(layer_ranges)

    batch = random_batch_element(pfi_model)
    (conv, C, H, W) = random_neuron_location(pfi_model)

    return pfi_model.declare_neuron_fi(
        batch=batch,
        conv_num=conv,
        c=C,
        h=H,
        w=W,
        function=pfi_model.single_bit_flip_signed_across_batch,
    )


"""
Weight Perturbation Models
"""


def twos_comp_shifted(val, nbits):
    if val < 0:
        val = (1 << nbits) + val
    else:
        val = twos_comp(val, nbits)
    return val
def twos_comp(val, bits):
    # compute the 2's complement of int value val
    if (val & (1 << (bits - 1))) != 0:  # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)  # compute negative value
    return val  # return positive value as is
def bit_flip_weights(orig_value,max_value, bit_pos):
        # quantum value
        save_type = orig_value.dtype
        total_bits = 8
        logging.info("orig value:", orig_value)

        quantum = int((orig_value/max_value) * ((2.0 ** (total_bits - 1))))
        twos_comple = twos_comp_shifted(quantum, total_bits)  # signed
        logging.info("quantum:", quantum)
        logging.info("twos_comple:", twos_comple)

        # binary representation
        bits = bin(twos_comple)[2:]
        logging.info("bits:", bits)

        # sign extend 0's
        temp = "0" * (total_bits - len(bits))
        bits = temp + bits
        # print(bits)
        assert len(bits) == total_bits
        logging.info("sign extend bits", bits)

        # flip a bit
        # use MSB -> LSB indexing
        assert bit_pos < total_bits

        bits_new = list(bits)
        bit_loc = total_bits - bit_pos - 1
        if bits_new[bit_loc] == "0":
            bits_new[bit_loc] = "1"
        else:
            bits_new[bit_loc] = "0"
        bits_str_new = "".join(bits_new)
        logging.info("bits", bits_str_new)

        # GPU contention causes a weird bug...
        if not bits_str_new.isdigit():
            logging.info("Error: Not all the bits are digits (0/1)")

        # convert to quantum
        assert bits_str_new.isdigit()
        new_quantum = int(bits_str_new, 2)
        out = twos_comp(new_quantum, total_bits)
        logging.info("out", out)

        # get FP equivalent from quantum
        new_value = out * ((2.0 ** (-1 * (total_bits - 1))))
        logging.info("new_value", new_value)

        return torch.tensor(new_value, dtype=save_type)

def bit_flip_weight_IEEE(orig_value,bit_pos):
    save_type = orig_value.dtype
    total_bits=31
    # binary representation
    bits=floatingPoint(orig_value)

    bits_new = list(bits)
    bit_loc = total_bits - bit_pos - 1
    if bits_new[bit_loc] == "0":
        bits_new[bit_loc] = "1"
    else:
        bits_new[bit_loc] = "0"
    bits_str_new = "".join(bits_new)
    new_value = convert_to_real(bits_str_new)

    return torch.tensor(new_value, dtype=save_type)
def bit_flip_weights_FXP(orig_value, bit_pos):
    # quantum value
    save_type = orig_value.dtype
    total_bits = 8
    x = Fxp(orig_value, True, 6, 2)
    bits = x.bin()
    assert bit_pos < total_bits
    bits_new = list(bits)
    bit_loc = total_bits - bit_pos - 1
    if bits_new[bit_loc] == "0":
        bits_new[bit_loc] = "1"
    else:
        bits_new[bit_loc] = "0"
    bits_str_new = "".join(bits_new)

    # GPU contention causes a weird bug...
    if not bits_str_new.isdigit():
        logging.info("Error: Not all the bits are digits (0/1)")

    # convert to quantum
    assert bits_str_new.isdigit()
    FXP_value = x("0b"+bits_str_new)
    new_value = FXP_value.astype(float)

    return torch.tensor(new_value, dtype=save_type)
def random_weight_inj(pfi_model, corrupt_conv=-1, min_val=-1, max_val=1):
    (conv, k, c_in, kH, kW) = random_weight_location(pfi_model, corrupt_conv)
    faulty_val = random_value(min_val=min_val, max_val=max_val)

    return pfi_model.declare_weight_fi(
        conv_num=conv, k=k, c=c_in, h=kH, w=kW, value=faulty_val
    )


def zeroFunc_rand_weight(pfi_model):
    (conv, k, c_in, kH, kW) = random_weight_location(pfi_model)
    return pfi_model.declare_weight_fi(
        function=_zero_rand_weight, conv_num=conv, k=k, c=c_in, h=kH, w=kW
    )


def _zero_rand_weight(data, location):
    newData = data[location] * 0
    return newData







def binaryOfFraction(fraction):
    binary = str()
    while (fraction):
        fraction *= 2
        if (fraction >= 1):
            int_part = 1
            fraction -= 1
        else:
            int_part = 0
        binary += str(int_part)
    return binary
def floatingPoint(real_no):
    sign_bit = 0
    if (real_no < 0):
        sign_bit = 1
    real_no = abs(real_no)
    int_str = bin(int(real_no))[2:]
    fraction_str = binaryOfFraction(real_no - int(real_no))
    if int(real_no)==0:
        ind = int_str.index('0')
    else:
        ind = int_str.index('1')
    exp_str = bin((len(int_str) - ind - 1) + 127)[2:]
    mant_str = int_str[ind + 1:] + fraction_str
    mant_str = mant_str + ('0' * (23 - len(mant_str)))
    ieee_32 = str(sign_bit) + exp_str  + mant_str
    return ieee_32
def convertToInt(mantissa_str):
    power_count = -1
    mantissa_int = 0
    for i in mantissa_str:
        mantissa_int += (int(i) * pow(2, power_count))
        power_count -= 1
    return (mantissa_int + 1)
def convert_to_real(ieee_32):
    sign_bit = int(ieee_32[0])
    exponent_bias = int(ieee_32[1: 9], 2)
    exponent_unbias = exponent_bias - 127
    mantissa_str = ieee_32[9:]
    mantissa_int = convertToInt(mantissa_str)
    real_no = pow(-1, sign_bit) * mantissa_int * pow(2, exponent_unbias)
    return real_no

if __name__ == '__main__':
    # import torch
    # value = torch.tensor(50,dtype=torch.float32)
    # bit =7
    # value = bit_flip_weights(value,bit)
    # print(value)
    print(bit_flip_weight_IEEE(torch.tensor(-2.250000),29))
    binary = floatingPoint(-2.250000)
    print(binary)
    real = convert_to_real("11000000000100000000000000000000")
    print(real)