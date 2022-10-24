import argparse
import json
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel

import jinrong_model
import tool


def parse_args():
    parser = argparse.ArgumentParser()
    # test set
    parser.add_argument('--test_file', type=str, required=True)
    # trained params of bilstm
    parser.add_argument('--bilstm_model_data', type=str, required=True)
    # trained params of main task
    parser.add_argument('--main_model_data', type=str, required=True)
    # trained params of sub task
    parser.add_argument('--sub_model_data', type=str, required=True)
    #Maximum length in one article
    parser.add_argument('--max_len', type=int, required=True)
    # clause length
    parser.add_argument('--clause_len', type=int, required=True)
    # Number of sentences in an article
    parser.add_argument('--clause_num_oneart', type=int, required=True)


    # Confusion matrix
    parser.add_argument('--out_file', type=str, required=False,default='./pred.txt')
    parser.add_argument('--bert_mdoel', type=str, required=False, default='./bert-base-chinese')
    # vocab file
    parser.add_argument('--vocab_dir', type=str, required=False, default='./vocab.txt')
    parser.add_argument('--bert_feat', type=int, required=False,default=768)
    parser.add_argument('--emo_dict_dir', type=str, default='./data', required=False)
    parser.add_argument('--cws_dir', type=str, default='./ltp_data_v3.4.0/cws.model', required=False)
    parser.add_argument('--test_batch_size', type=int, default=128, required=False)
    parser.add_argument('--epoch_num', type=int, default=1, required=False)
    # number of categories in sub task
    parser.add_argument('--class_num', type=int, default=6, required=False)
    parser.add_argument('--totol_feat_hidden_size', type=int, required=False,default=1000)
    parser.add_argument('--transformer_head_num', type=int, required=False,default=8)
    parser.add_argument('--padding_str', type=str, required=False,default='[PAD]')
    #hiden size in main and sub task
    parser.add_argument('--main_linear_out_size', type=int, required=False,default=300)
    parser.add_argument('--sub_linear_out_size', type=int, required=False,default=300)
    # number of categories in sub task
    parser.add_argument('--emo_class_num', type=int, required=False,default=12)

    args = parser.parse_args()
    return args






if __name__ == '__main__':
    args=parse_args()
    test_file=args.test_file
    bilstm_model_data=args.bilstm_model_data
    main_model_data=args.main_model_data
    sub_model_data=args.sub_model_data
    max_len=args.max_len
    clause_len=args.clause_len
    clause_num_oneart=args.clause_num_oneart
    bert_mdoel=args.bert_mdoel
    vocab_dir=args.vocab_dir
    bert_feat=args.bert_feat
    emo_dict_dir=args.emo_dict_dir
    cws_dir=args.cws_dir
    test_batch_size=args.test_batch_size
    epoch_num=args.epoch_num
    class_num=args.class_num
    totol_feat_hidden_size=args.totol_feat_hidden_size
    transformer_head_num=args.transformer_head_num
    padding_str=args.padding_str
    main_linear_out_size=args.main_linear_out_size
    sub_linear_out_size=args.sub_linear_out_size
    out_pre=args.out_file
    emo_class_num=args.emo_class_num
    f = open(bert_mdoel+"/config.json", 'r', encoding="utf-8")
    file = json.load(f)
    vocab_size = file["vocab_size"]
    f.close()
    z_size = clause_num_oneart + sub_linear_out_size + main_linear_out_size

    test_data = tool.jinrong_dataset(test_file,load_type='test')
    test_loader = DataLoader(dataset=test_data, batch_size=test_batch_size, shuffle=False, drop_last=False)

    bilstm_model = jinrong_model.MyBiLstm(embedding_dim=bert_feat, hidden_size=totol_feat_hidden_size,
                                     num_layers=1, bidirectional=True,
                                     word2idx="bert_vocab", vocab_len=vocab_size,
                                     padding_id=tool.get_padding(padding_str))

    my_text_model = jinrong_model.My_text_cla(in_feat_size=totol_feat_hidden_size * 2, out_feat_size=main_linear_out_size,
                                              head_num=transformer_head_num, class_num=class_num,
                                              clause_num_oneart=clause_num_oneart,
                                                z_size = z_size,bert_feat=bert_feat
                                              )
    my_emo_model =jinrong_model.My_emo_word_exam(
        in_feat_size=totol_feat_hidden_size*2,out_feat_size=sub_linear_out_size,
        p_cla_size=(main_linear_out_size+clause_num_oneart)
        ,emo_class_num=emo_class_num
    )
    # eval
    bilstm_model.eval()
    my_emo_model.eval()
    my_text_model.eval()
    bilstm_model.load_state_dict(torch.load(bilstm_model_data))
    my_text_model.load_state_dict(torch.load(main_model_data))
    my_emo_model.load_state_dict(torch.load(sub_model_data))
    bilstm_model = bilstm_model.to(0)
    my_text_model = my_text_model.to(0)
    my_emo_model = my_emo_model.to(0)
    bert_model = BertModel.from_pretrained(bert_mdoel)

    best_target = {"pre": 0, "recall": 0, "f1_score": 0}
    end_epoch = False
    now_eval_interval = 0
    with torch.no_grad():
        for epoch in range(epoch_num):
            if epoch==(epoch_num-1):
                end_epoch=True
            now_eval_interval += 1



            test_loss = 0

            real_list=[]
            pre_list=[]

            print("*" * 20, " test start, epoch : %d " % (epoch), "*" * 20)
            tem_dict = {"y_cat": list([]), "y": list([])}

            total_test_num = 0
            rig_num = 0


            for step_test, (batch_x_test, batch_y_test) in enumerate(test_loader):
                end_batch = False
                if len(batch_x_test) < test_batch_size:
                    real_num = len(batch_x_test)
                    end_batch = True
                    lack_num = test_batch_size - len(batch_x_test)
                    batch_x_test = tuple([i for i in batch_x_test] + [batch_x_test[0]] * lack_num)
                    batch_y_test = tuple([i for i in batch_y_test] + [batch_y_test[0]] * lack_num)


                x_test, y_test, clause_list_test, label_ids_test, emo_tensor_test = tool.make_data(
                    padding_str=padding_str, max_len=max_len,
                    clause_num_oneart=clause_num_oneart,
                    x=batch_x_test, y=batch_y_test, class_num=class_num,
                    clause_len=clause_len,
                    batch_emo_y=None,mkdype='test')

                one_art_clause = []
                #  batch × hidden_size
                for i in clause_list_test:
                    one_art_tensor = bert_model(i)[1]
                    one_art_clause.append(one_art_tensor)
                one_art_clause = torch.tensor(
                    np.array([item.cpu().detach().numpy() for item in one_art_clause])).cuda()

                h_label =(one_art_clause.to(0))
                z_t = torch.ones(len(batch_x_test), max_len,
                                 z_size)

                x_test = x_test.to(0)
                y_test = y_test.to(0)
                z_t = z_t.to(0)


                a = bilstm_model(x_test)

                p_cla_size = main_linear_out_size + clause_num_oneart
                p_cla = torch.zeros((test_batch_size, max_len, p_cla_size))
                p_cla = p_cla.to(0)
                emo_z_dec = my_emo_model.predict(a['out'], p_cla)['z_dec']
                outs = my_text_model(a['out'], emo_z_dec, h_label)

                pre_labels = nn.functional.softmax(outs['out'], -1)

                if end_batch:
                    pre_labels = pre_labels[:real_num]
                    y_test = y_test[:real_num]
                    total_test_num += real_num
                else:
                    total_test_num += len(batch_x_test)

                pre_labels = torch.argmax(pre_labels, -1)
                y_test = torch.argmax(y_test, dim=-1)

                for i in y_test.tolist():
                    tem_dict['y'].append(i)
                for j in pre_labels.tolist():
                    tem_dict['y_cat'].append(j)

                real_list+=y_test.tolist()
                pre_list+=pre_labels.tolist()

                batch_real = (y_test == pre_labels).sum().item()
                rig_num += batch_real
                print("test batch %d ,finished %f %%" % (step_test, (step_test / len(test_loader) * 100)))
            epoch_target = tool.get_target(tem_dict['y_cat'], tem_dict['y'], class_num,save_dir=out_pre)


            print("#" * 40)
            print(" test end, epoch : %d ,test-acc: %f %% ,loss : %f , pre:  %f %%,recall: %f %%,f1_score: %f"
                  % (epoch, (rig_num / total_test_num * 100), test_loss,
                     epoch_target['pre'],epoch_target['recall'],
                     epoch_target['f1_score']))
            print("#" * 40)
            f=open('./predicts.txt','w',encoding='utf-8')
            f.write("real : ")
            f.write(str(real_list))
            f.write('\n')
            f.write("pred : ")
            f.write(str(pre_list))
            f.close()
