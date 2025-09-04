from transformers import AutoTokenizer
from model_info import models, vocab_sizes
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
args = parser.parse_args()

model_ckpt = models[args.model]
tokenizer1 = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code=True)
print("The vocabulary size of", args.model, "is", tokenizer1.vocab_size)
