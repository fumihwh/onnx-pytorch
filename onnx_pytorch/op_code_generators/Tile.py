import onnx
import onnx.numpy_helper
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class TileOpCodeGenerator(OpCodeGenerator):

    def __init__(self,
                 onnx_ver=onnx.defs.onnx_opset_version(),
                 torch_ver=torch.__version__):
        super(TileOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

    def gen(self, node, value_infos, initializers):
        # attr_value_dict = self.get_attr_value_dict(node)
        inputs_str, outputs_str = self.gen_input_output_string(
            node, initializers, self.rename_helper, self.tensor_inplace)

        init_str, forward_str = [], []
        repeats_np = onnx.numpy_helper.to_array(initializers[node.input[1]])
        repeats = tuple(i for i in repeats_np)
        params_str = self.gen_params_str(dims=repeats)
        temp_str = f"{', '.join(outputs_str)} = torch.tile({inputs_str[0]}, **{{{params_str}}})"
        forward_str.append(temp_str)
        # forward_str.append(f"{outputs_str[0]} = torch.tile({inputs_str[0]}, {repeats})")

        # forward_str.append(f"{outputs_str[0]} = torch.tile({', '.join(inputs_str)})")
        # forward_str.append(f"{outputs_str[0]} = torch.normal(**{{{params_str}}})")
        return {"init": init_str, "forward": forward_str}

        # shape = attr_value_dict['shape']
        # mean = attr_value_dict['mean']
        # std = attr_value_dict['scale']
        # seed = attr_value_dict['seed']
        # dtype = attr_value_dict['dtype']
