from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "meta-llama/Meta-Llama-3-8B-Instruct"  # or your repo
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, device_map="auto")

# 1) Grab the output head module safely
head = model.get_output_embeddings()          # nn.Linear or similar
emb  = model.get_input_embeddings()           # nn.Embedding

# Shapes
print("vocab size x hidden:", head.weight.shape)

# 2) If you really need the parameter *name* in state_dict:
name_of_head_weight = None
target_id = head.weight.data_ptr()
for n, p in model.named_parameters():
    if p.data_ptr() == target_id:
        name_of_head_weight = n
        break
print("head weight param name:", name_of_head_weight)

# 3) Check if weights are tied (same underlying storage)
print("tied with input embeddings?",
      emb.weight.data_ptr() == head.weight.data_ptr())
