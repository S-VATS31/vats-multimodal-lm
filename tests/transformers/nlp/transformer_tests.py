from configs.setup_env import device, dtype

import torch

from src.transformers.nlp.model import AutoregressiveTextTransformer
from configs.transformers.nlp.model_args.model_args_xsmall import ModelArgs
from configs.transformers.nlp.training_args import TrainingArgs

model_args, training_args = ModelArgs(), TrainingArgs()

def setup():
    pass

def test_logits_shape():
    pass

def test_input_ids_dtype():
    pass

def test_aux_loss_dtype():
    pass

def test_padding_shape():
    pass

if __name__ == "__main__":
    test_logits_shape()
    test_input_ids_dtype()
    test_input_ids_dtype()
    test_padding_shape()
    