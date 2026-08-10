
import os
import sys

sys.path.append(os.getcwd())
import torch
from zennit.canonizers import Canonizer, CompositeCanonizer, MergeBatchNorm, SequentialMergeBatchNorm, AttributeCanonizer
from zennit.core import collect_leaves
from zennit.types import ConvolutionTranspose, BatchNorm
from torch.nn.modules.activation import ReLU
from torch.nn import AdaptiveAvgPool2d


class CorrectCompositeCanonizer(CompositeCanonizer):
    # Zennit canonizer returns handles in the order they are applied.
    # We reverse the list so we can detach correctly after attach two different canonizers for the same parameter
    def apply(self, root_module):
        ret = super(CorrectCompositeCanonizer, self).apply(root_module)
        ret.reverse()
        return ret


class CorrectSequentialMergeBatchNorm(SequentialMergeBatchNorm):
    # Zennit does not set bn.epsilon to 0, resulting in an incorrect canonization. We solve this issue.
    def __init__(self):
        super(CorrectSequentialMergeBatchNorm, self).__init__()

    def apply(self, root_module):
        '''Finds a batch norm following right after a linear layer, and creates a copy of this instance to merge
        them by fusing the batch norm parameters into the linear layer and reducing the batch norm to the identity.

        Parameters
        ----------
        root_module: obj:`torch.nn.Module`
            A module of which the leaves will be searched and if a batch norm is found right after a linear layer, will
            be merged.

        Returns
        -------
        instances: list
            A list of instances of this class which modified the appropriate leaves.
        '''
        instances = []
        last_leaf = None
        for leaf in collect_leaves(root_module):
            if isinstance(last_leaf, self.linear_type) and isinstance(leaf, self.batch_norm_type):
                if last_leaf.weight.shape[0] == leaf.weight.shape[0]:
                    instance = self.copy()
                    instance.register((last_leaf,), leaf)
                    instances.append(instance)
            last_leaf = leaf

        return instances

    def merge_batch_norm(self, modules, batch_norm):
        # print(f"(Correctly) Merging BN")
        self.bn_fwd_handles = BNForwardSuppressCanonizer().apply(batch_norm)
        self.batch_norm_eps = batch_norm.eps
        super(CorrectSequentialMergeBatchNorm, self).merge_batch_norm(modules, batch_norm)
        batch_norm.eps = 0.
        batch_norm.eps = 1e-05 # Torch cuda 12.8 does not allow 0 eps in eval mode


    def remove(self):
        '''Undo the merge by reverting the parameters of both the linear and the batch norm modules to the state before
        the merge.
        '''
        for h in self.bn_fwd_handles:
            h.remove()
        super(CorrectSequentialMergeBatchNorm, self).remove()
        self.batch_norm.eps = self.batch_norm_eps

class BNForwardSuppressCanonizer(AttributeCanonizer):
    def __init__(self, attribute_map=None, recursive=True):
        if attribute_map is None:
            attribute_map = self._attribute_map
        super().__init__(attribute_map)
        self.recursive = recursive

    def copy(self):
        return BNForwardSuppressCanonizer()

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, BatchNorm):
            return {
                "forward": cls.id_forward.__get__(module),
            }
  
    @staticmethod
    def id_forward(self, x):
        return x


# Canonizer to canonize BN->Linear or BN->Conv modules.
class SequentialMergeBatchNormtoRight(MergeBatchNorm):
    @staticmethod
    def convhook(module, x, y):
        # Add the feature map bias to the output of canonized conv layer with padding
        x = x[0]
        bias_kernel = module.canonization_params["bias_kernel"]
        pad1, pad2 = module.padding
        # ASSUMING module.kernel_size IS ALWAYS STRICTLY GREATER THAN module.padding
        # Upscale bias kernel to full feature map size
        if pad1 > 0:
            left_margin = bias_kernel[:, :, 0:pad1, :]
            right_margin = bias_kernel[:, :, pad1 + 1:, :]
            middle = bias_kernel[:, :, pad1:pad1 + 1, :].expand(1, bias_kernel.shape[1],
                                                                x.shape[2] - module.weight.shape[2] + 1,
                                                                bias_kernel.shape[-1])
            bias_kernel = torch.cat((left_margin, middle, right_margin), dim=2)

        if pad2 > 0:
            left_margin = bias_kernel[:, :, :, 0:pad2]
            right_margin = bias_kernel[:, :, :, pad2 + 1:]
            middle = bias_kernel[:, :, :, pad2:pad2 + 1].expand(1, bias_kernel.shape[1], bias_kernel.shape[-2],
                                                                x.shape[3] - module.weight.shape[3] + 1)
            bias_kernel = torch.cat((left_margin, middle, right_margin), dim=3)

        # account for stride by dropping some of the tensor
        if module.stride[0] > 1 or module.stride[1] > 1:
            indices1 = [i for i in range(0, bias_kernel.shape[2]) if i % module.stride[0] == 0]
            indices2 = [i for i in range(0, bias_kernel.shape[3]) if i % module.stride[1] == 0]
            bias_kernel = bias_kernel[:, :, indices1, :]
            bias_kernel = bias_kernel[:, :, :, indices2]
        ynew = y + bias_kernel
        return ynew

    def __init__(self):
        super().__init__()
        self.handles = []

    def apply(self, root_module):
        instances = []
        last_leaf = None
        for leaf in collect_leaves(root_module):
            if isinstance(last_leaf, self.batch_norm_type) and isinstance(leaf, self.linear_type):
                instance = self.copy()
                instance.register((leaf,), last_leaf)
                instances.append(instance)
            last_leaf = leaf

        return instances

    def register(self, linears, batch_norm):
        '''Store the parameters of the linear modules and the batch norm module and apply the merge.

        Parameters
        ----------
        linear: list of obj:`torch.nn.Module`
            List of linear layer with mandatory attributes `weight` and `bias`.
        batch_norm: obj:`torch.nn.Module`
            Batch Normalization module with mandatory attributes
            `running_mean`, `running_var`, `weight`, `bias` and `eps`
        '''
        self.linears = linears
        self.batch_norm = batch_norm
        self.linear_params = [(linear.weight.data, getattr(linear.bias, 'data', None)) for linear in linears]

        self.batch_norm_params = {
            key: getattr(self.batch_norm, key).data for key in ('weight', 'bias', 'running_mean', 'running_var')
        }
        returned_handles = self.merge_batch_norm(self.linears, self.batch_norm)
        self.handles = returned_handles
        self.bn_fwd_handles = BNForwardSuppressCanonizer().apply(batch_norm)


    def remove(self):
        '''Undo the merge by reverting the parameters of both the linear and the batch norm modules to the state before
        the merge.
        '''
        for h in self.bn_fwd_handles:
            h.remove()
        super(SequentialMergeBatchNormtoRight, self).remove()
        self.batch_norm.eps = self.batch_norm_eps
        for h in self.handles:
            h.remove()
        for module in self.linears:
            if isinstance(module, torch.nn.Conv2d):
                if module.padding != (0, 0):
                    delattr(module, "canonization_params")

    def merge_batch_norm(self, modules, batch_norm):
        self.batch_norm_eps = batch_norm.eps
        return_handles = []
        denominator = (batch_norm.running_var + batch_norm.eps) ** .5
        scale = (batch_norm.weight / denominator)  # Weight of the batch norm layer when seen as a linear layer
        shift = batch_norm.bias - batch_norm.running_mean * scale  # bias of the batch norm layer when seen as a linear layer

        for module in modules:
            original_weight = module.weight.data
            if module.bias is None:
                module.bias = torch.nn.Parameter(
                    torch.zeros(module.out_channels, device=original_weight.device, dtype=original_weight.dtype)
                )
            original_bias = module.bias.data

            if isinstance(module, ConvolutionTranspose):
                index = (slice(None), *((None,) * (original_weight.ndim - 1)))
            else:
                index = (None, slice(None), *((None,) * (original_weight.ndim - 2)))

            # merge batch_norm into linear layer to the right
            scale_device = scale[index].device
            original_weight = original_weight.to(scale_device)
            original_bias = original_bias.to(scale_device)
            module.weight.data = (original_weight * scale[index])

            # module.bias.data = original_bias
            if isinstance(module, torch.nn.Conv2d):
                if module.padding == (0, 0):
                    module.bias.data = (original_weight * shift[index]).sum(dim=[1, 2, 3]) + original_bias
                else:
                    bias_kernel = shift[index].expand(*(shift[index].shape[0:-2] + original_weight.shape[-2:]))
                    temp_module = torch.nn.Conv2d(in_channels=module.in_channels, out_channels=module.out_channels,
                                                  kernel_size=module.kernel_size, padding=module.padding,padding_mode=module.padding_mode, bias=False)
                    temp_module.weight.data = original_weight
                    bias_kernel = temp_module(bias_kernel).detach()

                    module.canonization_params = {}
                    module.canonization_params["bias_kernel"] = bias_kernel
                    return_handles.append(module.register_forward_hook(SequentialMergeBatchNormtoRight.convhook))
            elif isinstance(module, torch.nn.Linear):
                module.bias.data = (original_weight * shift).sum(dim=1) + original_bias

        # change batch_norm parameters to produce identity
        batch_norm.running_mean.data = torch.zeros_like(batch_norm.running_mean.data)
        batch_norm.running_var.data = torch.ones_like(batch_norm.running_var.data)
        batch_norm.bias.data = torch.zeros_like(batch_norm.bias.data)
        batch_norm.weight.data = torch.ones_like(batch_norm.weight.data)
        batch_norm.eps = 0.
        batch_norm.eps = 1e-05 # Torch cuda 12.8 does not allow 0 eps in eval mode
        return return_handles


class ThreshReLUMergeBatchNorm(SequentialMergeBatchNormtoRight):
    # Hook functions for ReLU_thresh
    @staticmethod
    def prehook(module, x):
        module.canonization_params["original_x"] = x[0].clone()

    @staticmethod
    def fwdhook(module, x, y):
        x = module.canonization_params["original_x"]
        index = (None, slice(None), *((None,) * (module.canonization_params['weights'].ndim + 1)))
        y = module.canonization_params['weights'][index] * x + module.canonization_params['biases'][index]
        baseline_vals = -1. * (module.canonization_params['biases'] / module.canonization_params['weights'])[index]
        return torch.where(y > 0, x, baseline_vals)

    def __init__(self):
        super().__init__()
        self.relu = None

    @torch.no_grad()    # Need to force no_grad, as this will otherwise cause issues with CRP visiulization, trying to backward thorugh the graph multiple times
    def apply(self, root_module):
        instances = []
        oldest_leaf = None
        old_leaf = None
        mid_leaf = None
        counter = 0
        merged_adaptive_pooling = False
        for leaf in collect_leaves(root_module):
            # if isinstance(mid_leaf, AdaptiveAvgPool2d):
            #     print("mid_leaf is Adaptive!")
            #     print(".")
            if isinstance(old_leaf, self.batch_norm_type) and isinstance(mid_leaf, ReLU) and isinstance(leaf,
                                                                                                        self.linear_type):
                if not (torch.all(old_leaf.weight==1.) and torch.all(old_leaf.bias==0.)):
                    instance = self.copy()
                    counter += 1
                    # print(f"[{counter}] Registering ThreshRelu: {leaf}/{old_leaf}")
                    instance.register((leaf,), old_leaf, mid_leaf)
                    instances.append(instance)
            elif isinstance(oldest_leaf, self.batch_norm_type) and isinstance(old_leaf, ReLU) and isinstance(mid_leaf,
                                                                                                             AdaptiveAvgPool2d) and isinstance(
                leaf, self.linear_type):
                print("TRYING TO CANONIZE THROUGH AVGPOOL!")
                instance = self.copy()
                merged_adaptive_pooling = True
                instance.register((leaf,), oldest_leaf, old_leaf)
                instances.append(instance)
            oldest_leaf = old_leaf
            old_leaf = mid_leaf
            mid_leaf = leaf
        #assert merged_adaptive_pooling, "Did not find AdaptiveAvgPool2d!"
        return instances

    def register(self, linears, batch_norm, relu):
        '''Store the parameters of the linear modules and the batch norm module and apply the merge.

        Parameters
        ----------
        linear: list of obj:`torch.nn.Module`
            List of linear layer with mandatory attributes `weight` and `bias`.
        batch_norm: obj:`torch.nn.Module`
            Batch Normalization module with mandatory attributes
            `running_mean`, `running_var`, `weight`, `bias` and `eps`
        '''
        # print(f"Registering ThreshRELU")
        self.relu = relu

        denominator = (batch_norm.running_var + batch_norm.eps) ** .5
        scale = (batch_norm.weight / denominator)  # Weight of the batch norm layer when seen as a linear layer
        shift = batch_norm.bias - batch_norm.running_mean * scale  # bias of the batch norm layer when seen as a linear layer
        self.relu.canonization_params = {}
        self.relu.canonization_params['weights'] = scale
        self.relu.canonization_params['biases'] = shift
        self.relu.inplace=False

        super().register(linears,batch_norm)
        self.handles.append(self.relu.register_forward_pre_hook(ThreshReLUMergeBatchNorm.prehook))
        self.handles.append(self.relu.register_forward_hook(ThreshReLUMergeBatchNorm.fwdhook))

    def remove(self):
        '''Undo the merge by reverting the parameters of both the linear and the batch norm modules to the state before
        the merge.
        '''
        super().remove()
        delattr(self.relu, "canonization_params")

class SequentialThreshCanonizer(CorrectCompositeCanonizer):
    def __init__(self):
        super().__init__((
            CorrectSequentialMergeBatchNorm(),
            ThreshReLUMergeBatchNorm(),
        ))