task=TriviaQA
run_mode=dev
main_model=OpenChat
aux_model1=InternLM7b

python utils/evaluate/EM_dir_test.py ensemble/results/${task}/${run_mode}/${main_model}-${aux_model1}-drop3-top320-filter/vanilla
# python utils/evaluate/EM_dir_test.py res/${task}/${run_mode}/${main_model}/tas
# python utils/evaluate/EM_dir_test.py res/${task}/${run_mode}/${main_model}/tas2
# python utils/evaluate/EM_dir_test.py ensemble/results/${task}/${run_mode}/${main_model}-${aux_model1}-drop3-top320-filter/tas2+mas2
python utils/evaluate/EM_dir_test.py ensemble/results/${task}/${run_mode}/${main_model}-${aux_model1}-drop3-top320-filter/tas3+mas2
