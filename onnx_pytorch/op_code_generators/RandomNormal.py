import onnx
import onnx.numpy_helper
import torch
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from onnx_pytorch.op_code_generators import OpCodeGenerator


class RandomNormalOpCodeGenerator(OpCodeGenerator):

    def __init__(self,
                 onnx_ver=onnx.defs.onnx_opset_version(),
                 torch_ver=torch.__version__):
        super(RandomNormalOpCodeGenerator, self).__init__(onnx_ver, torch_ver)


    def gen(self, node, value_infos, initializers):
        attr_value_dict = self.get_attr_value_dict(node)
        gen_seed=f'torch.manual_seed({attr_value_dict["seed"]})'
        params_str = self.gen_params_str(
            mean=attr_value_dict['mean'],
            std=attr_value_dict['scale'],
            size=attr_value_dict['shape'],
            dtype=f"torch.{TENSOR_TYPE_TO_NP_TYPE[attr_value_dict['dtype']]}",
            generator=gen_seed)
        inputs_str, outputs_str = self.gen_input_output_string(
            node, initializers, self.rename_helper, self.tensor_inplace)
        init_str, forward_str = [], []
        forward_str.append(f"{outputs_str[0]} = torch.normal(**{{{params_str}}})")
        # forward_str.append(f"{outputs_str[0]} = torch.randn(**{{{params_str}}})")
        return {"init": init_str, "forward": forward_str}

        # shape = attr_value_dict['shape']
        # mean = attr_value_dict['mean']
        # std = attr_value_dict['scale']
        # seed = attr_value_dict['seed']
        # dtype = attr_value_dict['dtype']