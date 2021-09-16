import onnx
import onnx.numpy_helper
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class ConvTransposeOpCodeGenerator(OpCodeGenerator):

    def __init__(self,
                 onnx_ver=onnx.defs.onnx_opset_version(),
                 torch_ver=torch.__version__):
        super(ConvTransposeOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

    def gen(self, node, value_infos, initializers):
        flag_from_layer = False
        attr_value_dict = self.get_attr_value_dict(node)
        inputs_str, outputs_str = self.gen_input_output_string(
            node, initializers, self.rename_helper, self.tensor_inplace)

        d = len(value_infos[node.input[0]].type.tensor_type.shape.dim) - 2
        input_size = [
                         d.dim_value
                         for d in value_infos[node.input[0]].type.tensor_type.shape.dim
                     ][2:]
        assert (d in (1, 2, 3))

        weightsProto = initializers.get(node.input[1], None)
        if weightsProto is None:  # can't get weight arrary frome init weight
            flag_from_layer = True
            weightsProto = value_infos.get(inputs_str[1], None)
            if weightsProto is None:
                raise NotImplementedError
            shape_contain = value_infos.get(inputs_str[1], None).type.tensor_type.shape.dim
            shape = []
            for dim_info in shape_contain:
                shape.append(dim_info.dim_value)
        else:
            weights_np = onnx.numpy_helper.to_array(weightsProto)
            shape = weights_np.shape

        padding = [0] * d
        output_padding = [0] * d
        stride = attr_value_dict.get("strides", [1] * d)
        kernel_shape = shape[2:]
        dilation = attr_value_dict.get("dilations", [1] * d)
        if "pads" in attr_value_dict:
            padding = [attr_value_dict["pads"][i] for i in range(d)]
        if "output_padding" in attr_value_dict:
            output_padding = [attr_value_dict["output_padding"][i] for i in range(d)]
        if "output_shape" in attr_value_dict:
            output_shape = attr_value_dict["output_shape"]
            total_padding = [0] * d

            # total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
            # If (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
            # Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).

            for i in range(d):
                total_padding[i] = stride[i] * (
                        input_size[i] - 1) + output_padding[i] + (
                                           (kernel_shape[i] - 1) * dilation[i] + 1) - output_shape[i]
                assert total_padding[
                           i] % 2 == 0, "Padding for ConvTranspose should be even."
                padding[i] = total_padding[i] // 2

        if flag_from_layer is not True:
            params_str = self.gen_params_str(groups=attr_value_dict["group"],
                                             dilation=dilation,
                                             out_channels=shape[1],
                                             padding=padding,
                                             output_padding=output_padding,
                                             kernel_size=shape[2:],
                                             stride=stride,
                                             in_channels=shape[0],
                                             bias=len(node.input) > 2)

            nn_name = f"ConvTranspose{d}d"
            node_name = self.rename_helper.get_node_name(node.name, node.op_type)
            init_str, forward_str = [], []
            init_str.append(f"self.{node_name} = nn.{nn_name}(**{{{params_str}}})")
            init_str.append(f"self.{node_name}.weight.data = {inputs_str[1]}")
            if len(node.input) > 2:
                init_str.append(f"self.{node_name}.bias.data = {inputs_str[2]}")
            forward_str.append(f"{outputs_str[0]} = self.{node_name}({inputs_str[0]})")
        else:
            params_str = self.gen_params_str(groups=attr_value_dict["group"],
                                             dilation=dilation,
                                             # out_channels=shape[1],
                                             padding=padding,
                                             output_padding=output_padding,
                                             # kernel_size=shape[2:],
                                             stride=stride,
                                             # in_channels=shape[0],
                                             # bias=len(node.input) > 2
                                             )
            init_str, forward_str = [], []
            forward_str.append(f"{outputs_str[0]} = F.conv_transpose2d({inputs_str[0]},{inputs_str[1]},**{{{params_str}}})")


        return {"init": init_str, "forward": forward_str}
