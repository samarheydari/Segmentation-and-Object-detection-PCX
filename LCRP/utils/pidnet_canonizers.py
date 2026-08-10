from copy import deepcopy
from LCRP.models.pidnet import PIDNet
from LCRP.utils.base_canonizers import (
    CorrectSequentialMergeBatchNorm,
    ThreshReLUMergeBatchNorm,
)
import torch
from copy import deepcopy
from zennit.canonizers import Canonizer
import zennit.canonizers as zcanon
from zennit.layer import Sum
import torch
import torch.nn.functional as F
import torch.nn as nn

import torch
from zennit.composites import EpsilonPlusFlat, LAYER_MAP_BASE
from zennit.layer import Sum
from zennit.rules import Epsilon, Norm, Pass, Flat
from zennit.core import Hook, BasicHook


algc = False

# Disable TensorFloat-32 (TF32) for higher precision
import torch

# 1. Handle Matrix Multiplication TF32 (introduced in PyTorch 1.7)
if hasattr(torch.backends.cuda, "matmul"):
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False

# 2. Handle cuDNN TF32 (also introduced in PyTorch 1.7)
if hasattr(torch.backends.cudnn, "allow_tf32"):
    torch.backends.cudnn.allow_tf32 = False


class SigmoidWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)


class Mult(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, weight, signal):
        if weight.shape != signal.shape:
            weight = weight.expand_as(signal)
        return torch.mul(weight, signal)


class InterpolateWrapper(nn.Module):
    def __init__(
        self, size=None, scale_factor=None, mode="bilinear", align_corners=False
    ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
    
def disable_inplace(model: nn.Module):
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    return model

# Canonizer for PIDNet
class PIDNetBaseCanonizer(zcanon.AttributeCanonizer):
    def __init__(self, attribute_map=None, recursive=True):
        if attribute_map is None:
            attribute_map = self._attribute_map
        super().__init__(attribute_map)
        self.recursive = recursive

    def copy(self):
        return PIDNetBaseCanonizer()

    def apply(self, module):
        instances = []
        if self.recursive:
            instances = super().apply(module)
        else:
        # Apply only to root module.
        # PIDNetCanonizer handles submodules for our specific PIDNet arch.
        # The roadblock for a PIDNet-general canonizer is module canonizers for modules not included in our module
        # And also a way to account for the forward function of the root PIDNet module
            attributes = self.attribute_map(module.__class__.__name__, module)
            if attributes is not None:
                instance = self.copy()
                instance.register(module, attributes)
                instances = [instance]
        return instances
    
    @classmethod
    def _attribute_map(cls, name, module):
        # PIDNet
        if module.__class__.__name__ == "PIDNet":
            return {
                "forward": cls.forward_pidnet.__get__(module),
                "sum1": Sum(),
                "sum2": Sum(),
                "interp1": InterpolateWrapper(
                    size=[90, 160], mode="bilinear", align_corners=False
                ),

                "interp2": InterpolateWrapper(
                    size=[90, 160], mode="bilinear", align_corners=False
                ),
                "interp3": InterpolateWrapper(
                    size=[90, 160], mode="bilinear", align_corners=False
                )
            }
        # BasicBlock
        if module.__class__.__name__ == "BasicBlock":
            return {
                "forward": cls.forward_basicblock.__get__(module),
                "sum1": Sum(),
            }

        # Bottleneck
        if module.__class__.__name__ == "Bottleneck":
            return {
                "forward": cls.forward_bottleneck.__get__(module),
                "sum1": Sum(),
            }

        # PAPPM
        if module.__class__.__name__ == "PAPPM":
            return {
                "forward": cls.forward_pappm.__get__(module),
                "sum1": Sum(),
                "sum2": Sum(),
                "sum3": Sum(),
                "sum4": Sum(),
                "sum5": Sum(),
                "orig_scale_process_params": cls.get_conv_layer_params(module.scale_process[2]),
                "scale_process": cls.convert_grouped_conv_to_regular(
                    module.scale_process
                ),
                "interp1": InterpolateWrapper(
                    size=[12, 20], mode="bilinear", align_corners=False
                ),
                "interp2": InterpolateWrapper(
                    size=[12, 20], mode="bilinear", align_corners=False
                ),
                "interp3": InterpolateWrapper(
                    size=[12, 20], mode="bilinear", align_corners=False
                ),
                "interp4": InterpolateWrapper(
                    size=[12, 20], mode="bilinear", align_corners=False
                ),
            }

        # Light_Bag
        if module.__class__.__name__ == "Light_Bag":
            return {
                "forward": cls.forward_lightbag.__get__(module),
                "sum1": Sum(),
                "sum2": Sum(),
                "sum3": Sum(),
                "sigmoid": SigmoidWrapper(),
                "mult1": Mult(),
                "mult2": Mult(),
            }
        # segmenthead
        if module.__class__.__name__ == "segmenthead":
            return {
                "forward": cls.forward_segmenthead.__get__(module),
                "sequential": cls.get_segmenthead_sequential(module),
            }

        # PagFM
        if module.__class__.__name__ == "PagFM":
            return {
                "forward": cls.forward_pagfm.__get__(module),
                "sum1": Sum(),
                "sum2": Sum(),
                "sigmoid": SigmoidWrapper(),
                "mult1": Mult(),
                "mult2": Mult(),
                "mult3": Mult(),
                "interp1": InterpolateWrapper(
                    size=[90, 160], mode="bilinear", align_corners=False
                ),
                "interp2": InterpolateWrapper(
                    size=[90, 160], mode="bilinear", align_corners=False
                ),
            }
        return None

    @staticmethod
    def get_segmenthead_sequential(segmenthead):
        seq = torch.nn.Sequential()
        seq.add_module("bn1", deepcopy(segmenthead.bn1))
        seq.add_module("relu1", torch.nn.ReLU(inplace=False))
        seq.add_module("conv1", deepcopy(segmenthead.conv1))
        seq.add_module("bn2", deepcopy(segmenthead.bn2))
        seq.add_module("relu2", torch.nn.ReLU(inplace=False))
        seq.add_module("conv2", deepcopy(segmenthead.conv2))
        return seq

    @staticmethod
    def get_conv_layer_params(conv_g):
        return {
            "init": {
                "in_channels": conv_g.in_channels,
                "out_channels": conv_g.out_channels,
                "kernel_size": conv_g.kernel_size,
                "stride": conv_g.stride,
                "padding": conv_g.padding,
                "dilation": conv_g.dilation,
                "bias": (conv_g.bias is not None),
                "groups": conv_g.groups,
            },
            "params": {
                "weight": conv_g.weight.data.detach(),
                "bias": conv_g.bias.data.detach() if conv_g.bias is not None else None,
            },
        }

    def remove(self):
        if "orig_scale_process_params" in self.attribute_keys:
            mdl = nn.Conv2d(**self.module.orig_scale_process_params["init"])
            mdl.weight.data = self.module.orig_scale_process_params["params"]["weight"]
            if self.module.orig_scale_process_params["params"]["bias"] is not None:
                mdl.bias.data = self.module.orig_scale_process_params["params"]["bias"]
            self.module.scale_process[2] = mdl
        for key in self.attribute_keys:
            if key != "scale_process" and hasattr(self.module, key):
                delattr(self.module, key)

    @staticmethod
    def convert_grouped_conv_to_regular(seq):
        new_seq = deepcopy(seq)
        conv_g = seq[2]
        G = conv_g.groups
        Cin_per_group = conv_g.in_channels // G
        Cout_per_group = conv_g.out_channels // G

        # Create equivalent regular conv
        conv_regular = nn.Conv2d(
            in_channels=conv_g.in_channels,
            out_channels=conv_g.out_channels,
            kernel_size=conv_g.kernel_size,
            stride=conv_g.stride,
            padding=conv_g.padding,
            dilation=conv_g.dilation,
            bias=(conv_g.bias is not None),
            groups=1,
            device=conv_g.weight.device,
            dtype=conv_g.weight.dtype,
        )

        # Zero all weights first
        with torch.no_grad():
            conv_regular.weight.zero_()

            # Copy group weights into corresponding block
            for g in range(G):
                out_start = g * Cout_per_group
                in_start = g * Cin_per_group

                conv_regular.weight[
                    out_start : out_start + Cout_per_group,
                    in_start : in_start + Cin_per_group,
                ] = conv_g.weight[out_start : out_start + Cout_per_group]

            # Copy biases
            if conv_g.bias is not None:
                conv_regular.bias.copy_(conv_g.bias)
        new_seq[2] = conv_regular
        return new_seq

    @staticmethod
    def forward_pidnet(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))

        term = self.interp1(self.diff3(x))
        x_d = self.sum1(torch.stack([x_d, term], dim=-1))

        if self.augment:
            temp_p = x_

        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))

        x_ = self.pag4(x_, self.compression4(x))

        term = self.interp2(self.diff4(x))
        x_d = self.sum2(torch.stack([x_d, term], dim=-1))

        if self.augment:
            temp_d = x_d

        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))

        x = self.interp3(self.spp(self.layer5(x)))
        x_ = self.final_layer(self.dfm(x_, x, x_d))

        if self.augment:
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return [x_extra_p, x_, x_extra_d]
        else:
            return x_

    @staticmethod
    def forward_segmenthead(self, x):
        out = self.sequential(x)
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            # self.interp1 = InterpolateWrapper(
            #     size=[height, width], mode="bilinear", align_corners=algc
            # ) # Note: this should not be run in practice , scale_factor is supposed to be None
            out = self.interp1(out)

        return out

    @staticmethod
    def forward_pagfm(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = self.interp1(y_q)
        x_k = self.f_x(x)
        # if we are using equal distribution, order does not matter.
        # if we are using signal takes all, then sim_map will take 0 relevance anyways.
        # so order of Mult call does not matter here.
        term = self.mult1(x_k, y_q)
        if self.with_channel:
            sim_map = self.sigmoid(self.up(term))
        else:
            term=term.permute(0,2,3,1)
            sim_map = self.sigmoid(
                self.sum1(term).unsqueeze(-1)
            )
            sim_map=sim_map.permute(0,3,1,2)
        y = self.interp2(y)
        term1 = self.mult2(1 - sim_map, x)
        term2 = self.mult3(sim_map, y)
        x = self.sum2(torch.stack([term1, term2],dim=-1)) 
        return x

    @staticmethod
    def forward_basicblock(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.sum1(torch.stack([out, residual], dim=-1))

        if self.no_relu:
            return out
        else:
            return self.relu(out)

    @staticmethod
    def forward_bottleneck(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.sum1(torch.stack([out, residual], dim=-1))
        if self.no_relu:
            return out
        else:
            return self.relu(out)

    @staticmethod
    def forward_pappm(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale_list = []

        x_ = self.scale0(x)

        s1 = self.interp1(self.scale1(x))
        s2 = self.interp2(self.scale2(x))
        s3 = self.interp3(self.scale3(x))
        s4 = self.interp4(self.scale4(x))

        scale_list.append(self.sum1(torch.stack([s1, x_], dim=-1)))
        scale_list.append(self.sum2(torch.stack([s2, x_], dim=-1)))
        scale_list.append(self.sum3(torch.stack([s3, x_], dim=-1)))
        scale_list.append(self.sum4(torch.stack([s4, x_], dim=-1)))

        scale_out = self.scale_process(torch.cat(scale_list, 1))
        compression_out = self.compression(torch.cat([x_, scale_out], 1))
        shortcut_out = self.shortcut(x)

        out = self.sum5(torch.stack([compression_out, shortcut_out], dim=-1))
        return out

    @staticmethod
    def forward_lightbag(self, p, i, d):
        # Detaching branches P and D here
        edge_att = self.sigmoid(d)
        term1 = self.mult1(1 - edge_att, i)
        term2 = self.mult2(edge_att, p)

        p_add = self.sum1(torch.stack([term1, p], dim=-1))
        p_add = self.conv_p(p_add)
        i_add = self.sum2(torch.stack([term2, i], dim=-1))
        i_add = self.conv_i(i_add)
        return self.sum3(torch.stack([p_add, i_add], dim=-1))


class PIDNetCanonizer(Canonizer):
    # New Module that computes the same functions as model, including layer wrappers and canonizations

    def __init__(self):
        self.handles = []
        super(PIDNetCanonizer).__init__()

    def canonize(self, layer, additional_canonizers=None, submodule_names=None):
        self.handles += PIDNetBaseCanonizer(recursive=False).apply(layer)
        if not isinstance(additional_canonizers, list):
            if not isinstance(submodule_names, list):
                additional_canonizers = [additional_canonizers]
                submodule_names = [submodule_names]
            else:
                additional_canonizers = [additional_canonizers] * len(submodule_names)
        for i, canonizer in enumerate(additional_canonizers):
            obj = layer
            if submodule_names[i] is not None:
                obj = getattr(obj, submodule_names[i], None)
            if canonizer is not None and obj is not None:
                h2 = canonizer.apply(obj)
                self.handles += h2

    def register(self, model):
        pass
        self.canonize(model)
        # I Branch
        i_branch = ["conv1", "layer1", "layer2", "layer3", "layer4", "layer5"]

        # P Branch
        p_branch = ["compression3", "compression4", "layer3_", "layer4_", "layer5_"]
        model.pag3.first=True
        model.pag4.first=False
        self.canonize(
            model.pag3,
            CorrectSequentialMergeBatchNorm(),
            ["f_x", "f_y"] + (["up"] if model.pag3.with_channel else []),
        )
        self.canonize(
            model.pag4,
            CorrectSequentialMergeBatchNorm(),
            ["f_x", "f_y"] + (["up"] if model.pag4.with_channel else []),
        )

        # D Branch
        d_branch = ["layer3_d", "layer4_d", "diff3", "diff4", "layer5_d"]
        for layer in i_branch + p_branch + d_branch:
           self.canonize(
               getattr(model, layer), CorrectSequentialMergeBatchNorm()
           )

        TReLU_modules = [
           "scale1",
           "scale2",
           "scale3",
           "scale4",
           "scale0",
           "scale_process",
           "compression",
           "shortcut",
        ]
        self.canonize(model.spp, ThreshReLUMergeBatchNorm(), TReLU_modules)
        self.canonize(
            model.dfm, CorrectSequentialMergeBatchNorm(), ["conv_p", "conv_i"]
        )
        # Prediction Head 
        segheads = ["final_layer"] + (["seghead_p", "seghead_d"] if model.augment else [])
        for sh_layer in segheads:
            self.canonize(
                getattr(model, sh_layer),
                [CorrectSequentialMergeBatchNorm(), ThreshReLUMergeBatchNorm()],
                ["sequential", "sequential"],
            )

    def remove(self):
        self.handles.reverse()
        for i, h in enumerate(self.handles):
            h.remove()
            pass

    def apply(self, module):
        if isinstance(module, PIDNet):
            disable_inplace(module)
            if module.seghead_p.scale_factor is not None or module.seghead_d.scale_factor is not None:
                raise Exception("canonizer only works for segmenthead.scale_factor=None currently")
            instance = self.copy()
            instance.register(module)
            return [instance]
        else:
            return []

def unbroadcast_like(R: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Sum-reduce R so it matches ref.shape (inverse of broadcasting)."""
    out = R
    # If ref has fewer dims, sum leading dims
    while out.dim() > ref.dim():
        out = out.sum(dim=0)
    # Now same number of dims: sum dims where ref was broadcast (size 1)
    for d, (os, rs) in enumerate(zip(out.shape, ref.shape)):
        if rs == 1 and os != 1:
            out = out.sum(dim=d, keepdim=True)
        elif os != rs:
            # Should not happen for valid broadcast; keep as hard fail
            raise RuntimeError(f"Cannot unbroadcast: out {out.shape} -> ref {ref.shape}")
    return out

class SignalTakesAllMul(Hook):
    def backward(self, module, grad_input, grad_output):
        R = grad_output[0]
        a = grad_input[0]  # weight
        b = grad_input[1]  # signal

        Ra = torch.zeros_like(a) if a is not None else None
        Rb = unbroadcast_like(R, b) if b is not None else None
        assert torch.isclose(Rb.sum(), torch.tensor([g.sum() for g in grad_output]).sum())
        return (Ra, Rb)


class FlatMul(Hook):
    def backward(self, module, grad_input, grad_output):
        R = grad_output[0]
        a = grad_input[0]
        b = grad_input[1]

        Ra = unbroadcast_like(0.5 * R, a) if a is not None else None
        Rb = unbroadcast_like(0.5 * R, b) if b is not None else None
        return (Ra, Rb)

from zennit.types import Convolution, Linear, AvgPool, Activation
from zennit.types import Activation, AvgPool
from zennit.core import Composite

class TestComposite(Composite):
    def __init__(self, canonizers=None):
        self.layer_map = [
                (Activation, Pass()),
                (Sum, Norm()),
                (AvgPool, Norm()),
                (Convolution, Epsilon()),
                (torch.nn.Linear, Epsilon()),
                (InterpolateWrapper, Epsilon()),
                (SigmoidWrapper, Pass()),
                (torch.nn.BatchNorm2d, Pass()),
                (Mult, SignalTakesAllMul())
        ]
        
        super().__init__(self.mapping, canonizers)
    

    def mapping(self, ctx, name, module):
        '''Get the appropriate hook given a mapping from module types to hooks.

        Parameters
        ----------
        ctx: dict
            A context dictionary to keep track of previously registered hooks.
        name: str
            Name of the module.
        module: obj:`torch.nn.Module`
            Instance of the module to find a hook for.

        Returns
        -------
        obj:`Hook` or None
            The hook found with the module type in the given layer map, or None if no applicable hook was found.
        '''
        return next((hook for types, hook in self.layer_map if isinstance(module, types)), None)

class EpsilonPlusFlatforPIDNet(EpsilonPlusFlat):
    def __init__(self, canonizers=None):
        super().__init__(canonizers=canonizers)
        self.layer_map += LAYER_MAP_BASE + [
            (InterpolateWrapper, Epsilon()),
            (SigmoidWrapper, Pass()),
            (torch.nn.BatchNorm2d, Pass()),
            (Mult, SignalTakesAllMul())
        ]