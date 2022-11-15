# This is a sample Python script.



# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import json
import math
import os
import math
import warnings
from math import e as e

import torch
import torch.nn as nn
from transformers import BertModel
import numpy as np
import jinrong_model
import tool
from torch.utils.data import DataLoader
import time



def parse_args():
    parser = argparse.ArgumentParser()

    # train set in the experiment
    parser.add_argument('--train_data_dir', type=str, required=True)
    # test set in the experiment
    parser.add_argument('--test_data_dir', type=str, required=True)
    # clause length
    parser.add_argument('--clause_len', type=int, required=True)
    # Number of sentences in an article
    parser.add_argument('--clause_num_oneart', type=int, required=True)
    #  input sentence length of Bilstm model
    parser.add_argument('--max_len', type=int, required=True)

    # vocab file
    parser.add_argument('--vocab_dir', type=str, required=False,default='./vocab.txt')
    # load pre-processed data
    # you can find them under the dir ./workname/time/  if the tool.pretreat_data function has been run
    # if these two parameters is available, data will not be loaded from train_data_dir,test_data_dir
    parser.add_argument('--pre_train_data_dir', type=str, required=False,default='')
    parser.add_argument('--pre_test_data_dir', type=str, required=False,default='')
    #Word Segmentation and Emotional Dictionary
    parser.add_argument('--emo_dict_dir', type=str,default='./data', required=False)
    parser.add_argument('--cws_dir', type=str,default='./ltp_data_v3.4.0/cws.model', required=False)
    parser.add_argument('--lr', type=float, default=0.0015, required=False)
    parser.add_argument('--train_batch_size', type=int, default=128, required=False)
    parser.add_argument('--test_batch_size', type=int, default=128, required=False)
    parser.add_argument('--epoch_num', type=int, default=15, required=False)
    # number of categories
    parser.add_argument('--class_num', type=int, default=6, required=False)
    # config file of bert
    parser.add_argument('--bert_config', type=str, default='./bert-base-chinese/config.json', required=False)

    parser.add_argument('--work_name', type=str, required=False,default='zyz')
    parser.add_argument('--loss_lambda', type=int, required=False,default=0.001)
    #Hidden parameters of bilstm
    parser.add_argument('--totol_feat_hidden_size', type=int, required=False,default=1000)
    parser.add_argument('--transformer_head_num', type=int, required=False,default=8)
    parser.add_argument('--padding_str', type=str, required=False,default='[PAD]')
    #hiden size in main and sub task
    parser.add_argument('--main_linear_out_size', type=int, required=False,default=300)
    parser.add_argument('--sub_linear_out_size', type=int, required=False,default=300)
    # number of categories in sub task
    parser.add_argument('--emo_class_num', type=int, required=False,default=12)
    # Load parameters from a trained model
    parser.add_argument('--has_predict', type=bool, required=False, default=False)
    # trained params of bilstm
    parser.add_argument('--bilstm_model_data', type=str, required=False, default='')
    # trained params of main task
    parser.add_argument('--main_model_data', type=str, required=False, default='')
    # trained params of sub task
    parser.add_argument('--sub_model_data', type=str, required=False, default='')
    parser.add_argument('--eval_interval', type=int, required=False, default=100)
    args = parser.parse_args()
    return args


def make_work_dir(work_dir):
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    if not os.path.exists(z_dec_dir):
        os.mkdir(z_dec_dir)
    ms=os.path.join(z_dec_dir, 'model')
    if not os.path.exists(ms):
        os.mkdir(ms)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    warnings.warn("Note that the path should use / as the path separator!")
    args = parse_args()
    max_len = args.max_len
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir
    clause_len = args.clause_len
    clause_num_oneart = args.clause_num_oneart
    cws_dir = args.cws_dir
    lr = args.lr
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    epoch_num = args.epoch_num
    class_num = args.class_num
    bert_config = args.bert_config
    work_name = args.work_name
    loss_lambda = args.loss_lambda
    totol_feat_hidden_size = args.totol_feat_hidden_size
    transformer_head_num = args.transformer_head_num
    padding_str = args.padding_str
    main_linear_out_size = args.main_linear_out_size
    sub_linear_out_size = args.sub_linear_out_size
    emo_class_num = args.emo_class_num
    has_predict = args.has_predict
    pre_train_data_dir = args.pre_train_data_dir
    pre_test_data_dir = args.pre_test_data_dir
    bilstm_model_data = args.bilstm_model_data
    main_model_data = args.main_model_data
    sub_model_data = args.sub_model_data
    eval_interval = args.eval_interval
    vocab_dir = args.vocab_dir
    emo_dict_dir = args.emo_dict_dir
    time_dir = int(time.time())
    z_dec_dir = os.path.join(work_name, str(time_dir))
    f = open(bert_config, 'r', encoding="utf-8")
    file = json.load(f)
    vocab_size = file["vocab_size"]
    bert_feat = file['hidden_size']
    f.close()

    # working directory
    work_dir = './' + work_name
    make_work_dir(work_dir)
    print("*" * 20, 'Start data preprocessing', '*' * 20)

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    import random
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



    # Data preprocessing, removing blank space, messing up, and adding emotional words
    if (pre_test_data_dir!=''  and  os.path.exists(pre_test_data_dir) ) \
            and  (pre_train_data_dir != '' and  os.path.exists(pre_train_data_dir)) :
        print("*" * 20, 'Load the specified preprocessed text and do not preprocess any more', '*' * 20)
        end_dir=pre_train_data_dir
        end_test_dir=pre_test_data_dir
    else:
        print("*" * 20, 'Data preprocessing in progress', '*' * 20)

        train_data=tool.pretreat_data(train_data_dir)
        save_dir = os.path.join(train_data_dir.rsplit('/', 1)[0], 'shuffle_train.tsv')
        tool.save_shuffle_data(train_data, save_dir)
        end_dir = os.path.join(z_dec_dir, 'emo_train.tsv')
        print("train_data start add emo")
        tool.add_emo_dict(save_dir, end_dir, emo_dict_dir, cws_dir)
        print("\ntrain_data finish add emo")

        test_data=tool.pretreat_data(test_data_dir,shuffle=False)
        save_test_dir=os.path.join(test_data_dir.rsplit('/',1)[0],'respace_test.tsv')
        tool.save_shuffle_data(test_data,save_test_dir)
        end_test_dir=os.path.join(z_dec_dir,'emo_test.tsv')
        print("test_data start add emo")
        tool.add_emo_dict(save_test_dir,end_test_dir,emo_dict_dir,cws_dir)
        print("\ntest_data finish add emo")
        print("*" * 20, 'Data preprocessing completed', '*' * 20)
    # File name of file z_dec
    z_t_last = os.path.join(train_data_dir, 'z_t')
    z_size = clause_num_oneart + sub_linear_out_size + main_linear_out_size



    # Load Train Collection

    train_data=tool.jinrong_dataset(end_dir)
    train_loader=DataLoader(dataset=train_data,batch_size=train_batch_size,shuffle=False,drop_last=True)
    if eval_interval<=0:
        eval_interval= len(train_loader)
    # Load Test Collection
    test_data = tool.jinrong_dataset(end_test_dir,load_type='test')
    test_loader=DataLoader(dataset=test_data,batch_size=test_batch_size,shuffle=False,drop_last=False)



    # Definition of two models
    # Joint learning potential representation bilstm
    bilstm_model = jinrong_model.MyBiLstm(embedding_dim=bert_feat, hidden_size=totol_feat_hidden_size,
                                     num_layers=1, bidirectional=True,
                                     word2idx="bert_vocab", vocab_len=vocab_size,
                                     padding_id=tool.get_padding(padding_str))
    # Main task text classification
    # Because bilstm is bidirectional, *2
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
    if has_predict:
        bilstm_model.load_state_dict(torch.load(bilstm_model_data))
        my_emo_model.load_state_dict(torch.load(sub_model_data))
        my_text_model.load_state_dict(torch.load(main_model_data))
        print(" loading model finished ")
    else:
        print("train from zero")

    # Put the model into the gpu
    bilstm_model = bilstm_model.to(0)
    my_text_model = my_text_model.to(0)
    my_emo_model = my_emo_model.to(0)
    '''
    weights = torch.tensor(bili, dtype=torch.float32)
    weights = torch.tensor([max(weights) / x for x in weights])
    weights=weights.to(0)
    criterion=nn.CrossEntropyLoss(weight=weights)
    '''
    criterion=nn.CrossEntropyLoss()
    # Optimizer, joint learning
    optimizer1 = torch.optim.Adam(bilstm_model.parameters(), lr=math.pow(10,-3))
    optimizer2 = torch.optim.Adam(my_text_model.parameters(), lr=lr,weight_decay=0.01)
    optimizer3 = torch.optim.Adam(my_emo_model.parameters(),lr=math.pow(10,-5))

    #Change the learning rate
    warmup_lr=jinrong_model.my_jinrong_warmup(optimizer=optimizer2,lr=lr,warmup_step_rate=0.1,total_iters=len(train_loader)*epoch_num,end_iters_rate=0.5)

    # The pre trained bert is used to extract the characteristics of clauses
    bert_model = BertModel.from_pretrained('./bert-base-chinese')
    best_test_loss=9999
    best_target={"pre":0,"recall":0,"f1_score":0}

    end_epoch=False
    now_eval_interval=0

    train_num={0:0,1:0,2:0,3:0,4:0,5:0}



    for epoch in range(epoch_num):
        if epoch==(epoch_num-1):
            end_epoch=True

        # Start training the model

        print( "train batch_total_num : %d" % (int(len(train_loader))))
        print(("-"*20+"  epoch : %d train start "+"-"*20) % epoch)

        for step,(batch_x,batch_y,batch_emo_y) in enumerate(train_loader):
            #  test method will disable dropout and normalization
            z_dec_file=os.path.join(z_dec_dir,'z_dec'+str(step)+'.pt')
            bilstm_model.train()
            my_text_model.train()
            my_emo_model.train()
            # Load z_ dec
            if epoch == 0:
                z_t = torch.ones(len(batch_x), max_len,
                                 z_size)
            else:
                z_t = torch.load(z_dec_file)
            # Data processing
            # clause_list represents the matrix of clauses
            x,y,clause_list,label_ids,emo_tensor=tool.make_data(padding_str=padding_str,max_len=max_len,
                                                                clause_num_oneart=clause_num_oneart,
                                                                x=batch_x,y=batch_y,class_num=class_num,
                                                                clause_len=clause_len,
                                                                batch_emo_y=batch_emo_y)

            #Get clause characteristics through bert
            with torch.no_grad():
                one_art_clause = []
                #  batch × hidden_size
                one_art_label = bert_model(label_ids)[1]
                for i in clause_list:
                    #for one art   i : Number of clauses × clause length
                    one_art_tensor  = bert_model(i)[1]
                    one_art_clause.append(one_art_tensor)
                one_art_clause= torch.tensor(np.array([item.cpu().detach().numpy() for item in one_art_clause])).cuda()

                h_label=torch.concat((one_art_clause.to(0),torch.unsqueeze(one_art_label, -2).to(0)),dim=-2  )


            x=x.to(0)
            y=y.to(0)
            z_t=z_t.to(0)
            emo_tensor=emo_tensor.to(0)

            #Zero gradient
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            a = bilstm_model(x)

            outs=my_text_model(a['out'],z_t,h_label)

            sub_loss=my_emo_model(a['out'],outs['p_cla'],emo_tensor,z_dec_file)

            loss_main = criterion(outs['out'], y)


            for tem_ooo in torch.argmax(y,dim=-1).tolist():
                train_num[tem_ooo]+=1

            loss_totoal=(loss_lambda/emo_class_num)*sub_loss+loss_main

            loss_totoal.backward()

            warmup_lr.step()

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()




            print("train   batch %d : loss %f ,totoal_batch :%d, finished : %f %% "
                  % (step,loss_main,len(train_loader),(step/len(train_loader)*100) ))

            now_eval_interval+=1
            if (now_eval_interval % eval_interval ==0)  or (end_epoch and (step==(len(train_loader)-1) )):
                # Enter Assessment
                bilstm_model.eval()
                my_emo_model.eval()
                my_text_model.eval()
                test_loss = 0


                print("*" * 20, " test start, epoch : %d " % (epoch), "*" * 20)
                tem_dict = {"y_cat": list([]), "y": list([])}
                with torch.no_grad():
                    total_test_num = 0
                    rig_num = 0
                    # Output the number of training tags

                    print(train_num, 'Number of text labels trained before this round of update')
                    fs=open(os.path.join(z_dec_dir,'train_num.txt'),'a',encoding='utf-8')
                    fs.write(str(train_num))
                    fs.close()
                    for key ,val in train_num.items():
                        train_num[key]=0

                    for step_test, (batch_x_test, batch_y_test) in enumerate(test_loader):
                        end_batch = False
                        if len(batch_x_test) < test_batch_size:
                            real_num = len(batch_x_test)
                            end_batch = True
                            lack_num = test_batch_size - len(batch_x_test)
                            batch_x_test = tuple([i for i in batch_x_test] + [batch_x_test[0]] * lack_num)
                            batch_y_test = tuple([i for i in batch_y_test] + [batch_y_test[0]] * lack_num)


                        x_test, y_test, clause_list_test, label_ids_test, _ = tool.make_data(
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
                        # The right part of the transformer input
                        h_label=(one_art_clause.to(0))


                        z_t = torch.ones(len(batch_x_test), max_len,
                                         z_size)

                        x_test = x_test.to(0)
                        y_test = y_test.to(0)
                        z_t = z_t.to(0)

                        a = bilstm_model(x_test)
                        p_cla_size=main_linear_out_size+clause_num_oneart
                        p_cla=torch.zeros((test_batch_size,max_len,p_cla_size))
                        p_cla=p_cla.to(0)
                        emo_z_dec = my_emo_model.predict(a['out'], p_cla)['z_dec']
                        outs = my_text_model(a['out'], emo_z_dec, h_label)

                        pre_labels = nn.functional.softmax(outs['out'], -1)
                        test_loss += criterion(pre_labels, y_test)

                        # Whether it is the last batch or not, and whether it cannot be retrieved exactly
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

                        batch_real = (y_test == pre_labels).sum().item()
                        rig_num += batch_real
                        print("test batch %d ,finished %f %%" % (step_test, (step_test / len(test_loader) * 100)))
                    epoch_target = tool.get_target(tem_dict['y_cat'], tem_dict['y'], class_num,save_dir=os.path.join(z_dec_dir,'log2.txt'))

                    if test_loss < best_test_loss:
                        best_test_loss = test_loss

                        f = open(os.path.join(z_dec_dir, 'log.txt'), 'a', encoding='utf-8')
                        f.write("best_epoch: " + str(epoch) + " best_loss:" + str(best_test_loss))
                        f.write("best_target: " + str(epoch_target))
                        f.write('\n')
                        f.close()
                        print("save model",epoch)
     
                    else:
                        print("no berter model  ",epoch)
                    torch.save(bilstm_model.state_dict(),
                               os.path.join(z_dec_dir, 'model/bilstm_model_' + str(epoch) + '_.pkl'))
                    torch.save(my_emo_model.state_dict(),
                               os.path.join(z_dec_dir, 'model/sub_model_' + str(epoch) + '_.pkl'))
                    torch.save(my_text_model.state_dict(),
                               os.path.join(z_dec_dir, 'model/main_model_' + str(epoch) + '_.pkl'))

                    
                    print("#" * 40)
                    print(" test end, epoch : %d ,test-acc: %f %% , loss : %f ,pre : %f %%,recall: %f %%,f1_score: %f"
                          % (epoch, (rig_num / total_test_num * 100),
                             test_loss, epoch_target['pre'],
                             epoch_target['recall'],
                             epoch_target['f1_score']))
                    print("#" * 40)



        print(("-" * 10 + "  epoch : %d train end " + "-" * 10) %
              (epoch))




