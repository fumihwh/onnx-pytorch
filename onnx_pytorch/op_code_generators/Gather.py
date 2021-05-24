import onnx
import torch
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

from onnx_pytorch.op_code_generators import OpCodeGenerator


class GatherOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(GatherOpCodeGenerator, self).__init__(onnx_ver, torch_ver)
    self.embedding_conf = None

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    init_str, forward_str = [], []
    if self.embedding_conf is not None and node.name in self.embedding_conf:
      conf = self.embedding_conf[node.name]
      node_name = self.rename_helper.get_node_name(node.name, node.op_type)
      params_str = self.gen_params_str(num_embeddings=conf.num_embeddings,
                                       embedding_dim=conf.embedding_dim)
      init_str.append(f"self.{node_name} = nn.Embedding(**{{{params_str}}})")
      dtype = "int"
      if node.input[1] in value_infos:
        np_type = TENSOR_TYPE_TO_NP_TYPE[value_infos[
            node.input[1]].type.tensor_type.elem_type]
        if np_type.name == "int32":
          pass
        elif np_type.name == "int64":
          dtype = "long"
      forward_str.append(
          f"{outputs_str[0]} = self.{node_name}({inputs_str[1]}.{dtype}())")
      if conf.initializer is not None:
        class_name = conf.initializer["class_name"]
        init_conf = conf.initializer["config"]
        if class_name == "RandomNormal":
          init_str.append(
              f"nn.init.normal_(self.{node_name}.weight, mean={init_conf['mean']}, std=math.sqrt({init_conf['stddev']}))"
          )
        elif class_name == "Zeros":
          init_str.append(f"nn.init.constant_(self.{node_name}.weight, 0.0)")
      if conf.regularizer is not None:
        reg_conf = conf.regularizer["config"]
        init_str.append(
            f"self._regularizer_params.append((self.{node_name}.weight, {reg_conf.get('l1', 0.0)}, {reg_conf.get('l2', 0.0)}))"
        )
    else:
      axis = attr_value_dict.get("axis", 0)

      # Simple solution
      # forward_str.append(
      #     f"{outputs_str[0]} = {inputs_str[0]}.__getitem__([slice(None) for _ in range({axis})] + [{inputs_str[1]}.to(device={inputs_str[0]}.device, dtype=torch.int64)])"
      # )
      forward_str.append(
          f'''{outputs_str[0]} = self.gather({inputs_str[0]}, {axis}, {inputs_str[1]})'''
      )
    return {"init": init_str, "forward": forward_str}

  @staticmethod
  def gen_method():
    return '''def gather(self, input, dim, indices, **kwargs):
    shape_l, shape_r = list(input.shape), list(indices.shape)
    indices = indices.flatten().to(device=indices.device, dtype=torch.int64)
    for r in range(0, dim):
      indices = indices.unsqueeze(0)
    for r in range(dim, len(shape_l) - 1):
      indices = indices.unsqueeze(-1)
    indices = indices.expand(*(shape_l[:dim] + [int(np.prod(shape_r))] + shape_l[dim + 1:]))
    indices = torch.where(indices >= 0, indices, indices + shape_l[dim])
    output = torch.gather(input, dim, indices)
    output = torch.reshape(output, shape_l[:dim] + shape_r + shape_l[dim + 1:])
    return output
'''
