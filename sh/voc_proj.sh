model1=InternLM7b
model2=OpenChat

mkdir -vp map_file/${model1}_${model2}
python vecmap/map_embeddings.py --supervised map_file/${model1}-${model2}.dict map_file/${model1}.emb map_file/${model2}.emb map_file/${model1}_${model2}/${model1}_mapped_sup.emb map_file/${model1}_${model2}/${model2}_mapped_sup.emb
