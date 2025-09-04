model1=llama2
model2=internlm

python vecmap/eval_translation_scipy_matrix.py map_file/${model1}_${model2}/${model1}_mapped_sup.emb map_file/${model1}_${model2}/${model2}_mapped_sup.emb -d map_file/${model1}-${model2}-test.dict --retrieval csls --cuda --neighborhood 1 --precision fp32