python identity.py --model1 llama2 --model2 internlm
python hidden.py --model llama2
python hidden.py --model internlm
sed -i '1s/^/32000 4096\n/' map_file/llama2.emb
sed -i '1s/^/103168 4096\n/' map_file/internlm.emb
