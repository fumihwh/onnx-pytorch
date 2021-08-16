import onnx
import torch

from onnx_pytorch.op_code_generators import ReduceOpCodeGenerator


class NonMaxSuppressionOpCodeGenerator(ReduceOpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(NonMaxSuppressionOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)
    init_str, forward_str = [], []
    iou_threshold = inputs_str[3] if len(inputs_str) > 3 else "0.0"
    score_threshold = inputs_str[4] if len(inputs_str) > 4 else "0.0"
    max_output_boxes_per_class = inputs_str[2] if len(inputs_str) > 2 else "0"
    forward_str.append(
        f"{outputs_str[0]} = self.nms({inputs_str[0]}, {inputs_str[1]}, {max_output_boxes_per_class}, {iou_threshold}, {score_threshold}, center_point_box={attr_value_dict.get('center_point_box', 0)})"
    )
    return {"init": init_str, "forward": forward_str}

  @staticmethod
  def gen_method():
    return f'''def nms(self, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box=0, **kwargs):
    nms_rs_list = []
    for i in range(boxes.shape[0]):
      for j in range(scores.shape[1]):
        for k in range(boxes.shape[1]):
          if center_point_box == 1:
            boxes[i][k] = torchvision.ops.box_convert(boxes[i][k], "cxcywh", "xyxy")
          else:
            x1, y1, x2, y2 = boxes[i][k]
            if x1 < x2 and y1 < y2:
              continue
            indices = [0, 1, 2, 3]
            if x1 > x2:
              indices = [indices[l] for l in (2, 1, 0, 3)]
            if y1 > y2:
              indices = [indices[l] for l in (0, 3, 2, 1)]
            boxes[i][k] = boxes[i][k].gather(0, torch.tensor(indices))
        mask = scores[i][j] >= score_threshold
        nms_rs = torchvision.ops.nms(boxes[i], scores[i][j], float(iou_threshold))[:max_output_boxes_per_class]
        nms_rs_masked = nms_rs[:mask[nms_rs].nonzero(as_tuple=False).flatten().shape[0]]
        batch_index = torch.full((nms_rs_masked.shape[0], 1), i)
        class_index = torch.full((nms_rs_masked.shape[0], 1), j)
        nms_rs_list.append(torch.cat((batch_index, class_index, nms_rs_masked.unsqueeze(1)), dim=1))
    return torch.cat(nms_rs_list, dim=0)
'''
