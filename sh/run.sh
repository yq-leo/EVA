main_model=internlm
aux_model1=llama2

python ensemble/eva_multi.py --task nq --model ${main_model} --aux_models ${aux_model1} --aux_method drop --matrix_name filter --topk 320 --drop 3