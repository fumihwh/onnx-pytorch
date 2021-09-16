import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class PadOpCodeGenerator(OpCodeGenerator):

    def __init__(self,
                 onnx_ver=onnx.defs.onnx_opset_version(),
                 torch_ver=torch.__version__):
        super(PadOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

    def gen(self, node, value_infos, initializers):
        attr_value_dict = self.get_attr_value_dict(node)
        inputs_str, outputs_str = self.gen_input_output_string(
            node, initializers, self.rename_helper, self.tensor_inplace)
        init_str, forward_str = [], []
        mode = attr_value_dict.get("mode", b"constant")
        value = 0.
        if mode == b"constant":
            if len(node.input) == 3:   # get padding value
                value = onnx.numpy_helper.to_array(initializers[node.input[2]])[0]
        d = len(value_infos[node.input[0]].type.tensor_type.shape.dim) - 2
        if len(node.input) > 1:  # inputs num >= 1  actully input num == 2
            pads = initializers.get(node.input[1], None)
            assert pads is not None, "Currently PadOpCodeGenerator only support all of [pads] is in initializers."
            pads = onnx.numpy_helper.to_array(pads)
        else:
            pads = attr_value_dict["pads"]
        # this method is more general
        dd = len(value_infos[node.input[0]].type.tensor_type.shape.dim)
        pt_pads = [0, 0] * dd
        for i in range(dd):
            pt_pads[2*dd-1-i*2] = pads[i]
            pt_pads[2*dd-2-i*2] = pads[i+dd]
        forward_str.append(
            f"{outputs_str[0]} = F.pad({inputs_str[0]}, {pt_pads.__repr__()}, \"{mode.decode()}\", {value})"
        )
        # original pad code ::: below
        # pt_pads = [0, 0] * d
        # for i in range(d):
        #     pt_pads[2 * (d - i - 1)] = pads[2 + i]
        #     pt_pads[2 * (d - i - 1) + 1] = pads[d + 2 + 2 + i]
        # forward_str.append(
        #     f"{outputs_str[0]} = F.pad({inputs_str[0]}, {pt_pads.__repr__()}, \"{mode.decode()}\", {value})"
        # )
        return {"init": init_str, "forward": forward_str}
