main_model=llama2
aux_model=internlm

# Get the files required for Vecmap input
python identity.py --model1 ${aux_model} --model2 ${main_model}
python hidden.py --model ${aux_model}
python hidden.py --model ${main_model}

# Vocabulary projection
mkdir -vp map_file/${aux_model}_${main_model}
python vecmap/map_embeddings.py --supervised map_file/${aux_model}-${main_model}.dict map_file/${aux_model}.emb map_file/${main_model}.emb map_file/${aux_model}_${main_model}/${aux_model}_mapped_sup.emb map_file/${aux_model}_${main_model}/${main_model}_mapped_sup.emb

# Get similarity matrix
mkdir -vp sparse_matrix_filter
python vecmap/eval_translation_scipy_matrix.py map_file/${aux_model}_${main_model}/${aux_model}_mapped_sup.emb map_file/${aux_model}_${main_model}/${main_model}_mapped_sup.emb -d map_file/${aux_model}-${main_model}-test.dict --retrieval csls --cuda --neighborhood 1 --precision fp32
