import torch.nn.functional as F

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

    def __init__(self, num_linear_layers, C, num_classes, layers, criterion, output_weights, search_space, steps=4):
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

        # Add dense linear layers
        self.dense_linear_layers = nn.ModuleList()
        for i in range(num_linear_layers - 1):
            self.dense_linear_layers += [nn.Linear(C_prev, C_prev)]

    def forward(self, input, discrete=False, normalize=False):
        # NASBench only has one input to each cell
        s0 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if i in [self._layers // 3, 2 * self._layers // 3]:
                # Perform down-sampling by factor 1/2
                # Equivalent to https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L68
                s0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)(s0)

            # Normalize mixed_op weights for the choice blocks in the graph
            mixed_op_weights = self._preprocess_op(self._arch_parameters[0], discrete=discrete, normalize=False)

            # Normalize the output weights
            output_weights = self._preprocess_op(self._arch_parameters[1], discrete,
                                                 normalize) if self._output_weights else None

            # Normalize the input weights for the nodes in the cell
            input_weights = [self._preprocess_op(alpha, discrete, normalize) for alpha in self._arch_parameters[2:]]
            s0 = cell(s0, mixed_op_weights, output_weights, input_weights)

        # Include one more preprocessing step here
        s0 = self.postprocess(s0)  # [N, C_max * (steps + 1), w, h] -> [N, C_max, w, h]

        # Global Average Pooling by averaging over last two remaining spatial dimensions
        # https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L92
        out = s0.view(*s0.shape[:2], -1).mean(-1)

        # Detach to prevent the parameters from being updated
        out = out.view(out.size(0), -1).detach()

        # Apply additional linear layers if necessary followed by relu activation function
        for dense_linear_layer in self.dense_linear_layers:
            out = dense_linear_layer(out)
            out = F.relu(out)

        logits = self.classifier(out)
        return logits
