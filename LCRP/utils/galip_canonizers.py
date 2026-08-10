from zennit.core import collect_leaves,stabilize
import zennit.canonizers as zcanon
from zennit.layer import Sum
from zennit.image import imgify
import torch

class SequentialMergeBatchNorm(zcanon.SequentialMergeBatchNorm):
    # Zennit does not set bn.epsilon to 0, resulting in an incorrect canonization. We solve this issue.
    def __init__(self):
        super(SequentialMergeBatchNorm, self).__init__()

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
        self.batch_norm_eps = batch_norm.eps
        super(SequentialMergeBatchNorm, self).merge_batch_norm(modules, batch_norm)
        # batch_norm.eps = 0.

    def remove(self):
        '''Undo the merge by reverting the parameters of both the linear and the batch norm modules to the state before
        the merge.
        '''
        super(SequentialMergeBatchNorm, self).remove()
        self.batch_norm.eps = self.batch_norm_eps

#Copied from LCRP.utils.zennit_canonizers, but uses our the fixed zennit canonizers implemented above
class YoloCanonizer(zcanon.AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        # print(module.__class__.__name__)
        if module.__class__.__name__ == "Bottleneck": # for Yolov5
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        if module.__class__.__name__ == "RepVGGBlock": # for Yolov5-6
            attributes = {
                'forward': cls.forward_.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes

        return None

    @staticmethod
    def forward(self, x):
        if not self.add:
            return self.cv2(self.cv1(x))
        else:
            x = torch.stack([x, self.cv2(self.cv1(x))], dim=-1)
            x = self.canonizer_sum(x)
            return x

    @staticmethod
    def forward_(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        x = torch.stack([self.rbr_dense(inputs), self.rbr_1x1(inputs), id_out]
                        if self.rbr_identity is not None else [self.rbr_dense(inputs), self.rbr_1x1(inputs)], dim=-1)
        x = self.canonizer_sum(x)
        return self.nonlinearity(self.se(x))

class YoloV6Canonizer(zcanon.CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''

    def __init__(self):
        super().__init__((
            SequentialMergeBatchNorm(),
            YoloCanonizer(),
        ))