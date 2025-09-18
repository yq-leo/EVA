export CUDA_VISIBLE_DEVICES=2,3

task=TriviaQA
run_mode=test
main_model=OpenChat
aux_model1=InternLM7b
ensemble_method=tas3+mas2

log_path=./log/${task}/${run_mode}/${main_model}+${aux_model1}/${ensemble_method}
mkdir -vp ${log_path}

nohup python ensemble/eva_multi.py --task ${task} --run_mode ${run_mode} --model ${main_model} --aux_models ${aux_model1} --aux_method drop --ensemble_method ${ensemble_method} --matrix_name filter --topk 320 --drop 3 > ${log_path}/run.log 2>&1 &
