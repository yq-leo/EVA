from transformers import AutoModelForCausalLM
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cpu")
embedding_matrix = model.get_input_embeddings().weight
print("Embedding matrix shape of", args.model, ":", embedding_matrix.shape)
