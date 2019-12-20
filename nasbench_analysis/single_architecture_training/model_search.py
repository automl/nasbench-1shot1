from optimizers.darts.model_search import Network, MixedOp, ChoiceBlock, Cell
from optimizers.darts.operations import *


class MixedOpIndependentTraining(MixedOp):
    def __init__(self, *args, **kwargs):
        super(MixedOpIndependentTraining, self).__init__(*args, **kwargs)

    def forward(self, x, weights):
        cpu_weights = weights.tolist()
        clist = []
        for j, cpu_weight in enumerate(cpu_weights):
            if abs(cpu_weight) > 1e-10:
                clist.append(weights[j] * self._ops[j](x))
        assert len(clist) > 0, 'invalid length : {:}'.format(cpu_weights)
        return sum(clist)


class ChoiceBlockIndependent(ChoiceBlock):
    """
    Adapted to match Figure 3 in:
    Bender, Gabriel, et al. "Understanding and simplifying one-shot architecture search."
    International Conference on Machine Learning. 2018.
    """

    def __init__(self, C_in):
        super(ChoiceBlockIndependent, self).__init__(C_in)
        # Use the GDAS Mixed Op instead of the DARTS Mixed op
        self.mixed_op = MixedOpIndependentTraining(C_in, stride=1)


class CellIndependent(Cell):

    def __init__(self, steps, C_prev, C, layer, search_space):
        super(CellIndependent, self).__init__(steps, C_prev, C, layer, search_space)
        # Create the choice block.
        self._choice_blocks = nn.ModuleList()
        for i in range(self._steps):
            # Use the GDAS cell instead of the DARTS cell
            choice_block = ChoiceBlockIndependent(C_in=C)
            self._choice_blocks.append(choice_block)


class NetworkIndependent(Network):

    def __init__(self, C, num_classes, layers, criterion, output_weights, search_space, steps=4):
        super(NetworkIndependent, self).__init__(C, num_classes, layers, criterion, output_weights, search_space,
                                                 steps=steps)

        # Override the cells module list of DARTS with GDAS variants
        self.cells = nn.ModuleList()
        C_curr = C
        C_prev = C_curr
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                # Double the number of channels after each down-sampling step
                # Down-sample in forward method
                C_curr *= 2

            cell = CellIndependent(steps, C_prev, C_curr, layer=i, search_space=search_space)
            self.cells += [cell]
            C_prev = C_curr
