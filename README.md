# AMultiESG

Implementation code of paper "An Interactive Multi-Task ESG Classification Method For Chinese Financial Texts".

AMultiESG model introduces an interactive multi-task learning network ,which combines sentiment dictionary expansion and financial text classification, with financial text classification as the primary task and sentiment dictionary expansion as the secondary task. This network can promote the collaboration between the two tasks, so as to improve the performance of the two
tasks. Only a few data sets are disclosed for commercial reasons.

## Configuration

A suitable gpu is needed in the experiment

The bert and ltp models are used in the experiment. The version we use is the Chinese Bert of Harbin Institute of Technology. There are also models of test results. You can replace them yourself or download them through Baidu Netdisk:

ltp: https://pan.baidu.com/s/1VnWQDU9_m_t34pLsUu3BTQ
	|  Extraction code：72iu

bert:https://pan.baidu.com/s/15p2ebPAilgRYTv13VgEZSg 
	|  Extraction code：hw6q

model: https://pan.baidu.com/s/1UeCSetnk-jsXsNsjTEN7Qg 
	|  Extraction code：3mxa

## Quick Start

### Reqests

This code is based on pytorch_ crf, pyltp, transformers and other modules. All detailed modules and versions are included in the requirements.txt file.

### Train

If you want to train the model from scratch, you can run main.py. The important parameters are train_data_dir, test_data_dir, clause_len, clause_num_oneart, max_len. They represent training set, test set, sentence length. More parameters and descriptions can be found in main.py.


### Test

If there are trained bilstm, main and sub model parameter files,you can directly load the model parameter file through the predict.py file to make predictions. The important parameters are test_file, bilstm_model_data, main_model_data, sub_model_data, max_len , clause_len ,clause_num_oneart. They represent test set, model parameter files and sentence length.

Such as running 'python predict.py --test_file ./data/emo_test.tsv --bilstm_model_data ./model/bilstm_model.pkl   --main_model_data ./model/main_model.pkl  --sub_model_data ./model/sub_model.pkl  --max_len 512  --clause_len 30 --clause_num_oneart 16'

### Other File

jinrong_model.py includes the bilstm, main model and sub model in the paper. 

tool.py includes functions such as data processing.

