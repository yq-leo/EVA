model1=InternLM7b
model2=OpenChat

export CUDA_VISIBLE_DEVICES=4
python vecmap/eval_translation_scipy_matrix.py map_file/${model1}_${model2}/${model1}_mapped_sup.emb map_file/${model1}_${model2}/${model2}_mapped_sup.emb -d map_file/${model1}-${model2}-test.dict --retrieval csls --cuda --neighborhood 1 --precision fp32
