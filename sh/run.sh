export CUDA_VISIBLE_DEVICES=0,2

main_model=OpenChat
aux_model1=InternLM7b

python ensemble/eva_multi.py --task nq --model ${main_model} --aux_models ${aux_model1} --aux_method linear_sum --matrix_name filter --topk 320 --drop 3