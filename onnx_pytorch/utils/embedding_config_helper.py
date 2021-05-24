import argparse
import re

import onnx
from onnx.numpy_helper import to_array
import yaml


class EmbeddingParam:

  def __init__(self,
               name,
               num_embeddings,
               embedding_dim,
               padding_idx=None,
               max_norm=None,
               norm_type=2.0,
               scale_grad_by_freq=False,
               sparse=False,
               embeddings_initializer=None,
               embeddings_regularizer=None):
    self.name = name
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.padding_idx = padding_idx
    self.max_norm = max_norm
    self.norm_type = norm_type
    self.scale_grad_by_freq = scale_grad_by_freq
    self.sparse = sparse
    self.initializer = embeddings_initializer
    self.regularizer = embeddings_regularizer


def gen_embedding_config(onnx_model_path, embedding_conf_file):
  model = onnx.load(onnx_model_path)
  initializers = {i.name: i for i in model.graph.initializer}
  inputs = {i.name: i for i in model.graph.input if i.name not in initializers}
  gathers = [
      n for n in model.graph.node
      if n.op_type == "Gather" and len(n.input) > 1 and n.input[1] in inputs
  ]
  embeddings = [
      EmbeddingParam(name=n.name,
                     num_embeddings=to_array(initializers[n.input[0]]).shape[0],
                     embedding_dim=to_array(initializers[n.input[0]]).shape[1])
      for n in gathers
  ]
  with open(embedding_conf_file, "w") as f:
    f.write(
        yaml.dump([{
            "name": e.name,
            "num_embeddings": e.num_embeddings,
            "embedding_dim": e.embedding_dim
        } for e in embeddings],
                  sort_keys=False))


def load_embedding_config(embedding_conf_file):
  with open(embedding_conf_file, "r") as f:
    embeddings = yaml.load(f)
  embeddings = {
      f'{re.sub("[:/.]", "_", e["name"])}': EmbeddingParam(**e)
      for e in embeddings
  }
  return embeddings


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--onnx_model_path",
                      default=None,
                      type=str,
                      help="The onnx model path.")
  parser.add_argument("--embedding_conf_file",
                      type=str,
                      help="Embedding config file path.")
  args = parser.parse_args()

  gen_embedding_config(onnx_model_path=args.onnx_model_path,
                       embedding_conf_file=args.embedding_conf_file)


if __name__ == '__main__':
  main()
