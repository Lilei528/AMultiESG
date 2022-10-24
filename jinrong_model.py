#coding=utf-8

from torch import nn
import torch
import torchcrf as crf
class MyBiLstm(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers, bidirectional,vocab_len,padding_id,word2idx="bert_vocab"):
        super(MyBiLstm, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.num_layers = num_layers
        # bidirect是否双向
        self.bidirectional = bidirectional
        if word2idx!="bert_vocab":
            print("use %s，not bert_vocab, Please change the dictionary"%word2idx)
            exit(-1)


        if self.bidirectional:
            self.feat_dim=hidden_size * 2
        else:
            self.feat_dim=hidden_size
        # embedding_dim is the vector of each word

        self.embedding=nn.Embedding(num_embeddings=vocab_len,embedding_dim=embedding_dim,
                                    padding_idx=padding_id)
        self.total_feat=nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, batch_first=True,
                    num_layers=self.num_layers, bidirectional=self.bidirectional)
        self.drop_out=nn.Dropout(p=0.3)

    def forward(self, x):
        batch_size, seq_len = x.shape
        # Initialize an h0, that is, c0. In RNN, the ht and Ct output by a cell are the same,
        # while the ht and Ct output by a cell of LSTM are different
        # 维度[layers, batch, hidden_len]

        x = self.embedding(x)

        out, (_, _) = self.total_feat(x)

        out=self.drop_out(out)
        return {'out': out}
class My_text_cla(nn.Module):
    def __init__(self, class_num,in_feat_size,out_feat_size,
                 bert_feat,clause_num_oneart,head_num,z_size,activation="ReLU",tran_feed_size=1024):
        super(My_text_cla, self).__init__()
        self.in_feat_size = in_feat_size
        self.class_num=class_num
        self.clause_num_oneart=clause_num_oneart
        self.z_size=z_size
        self.z_linear=nn.Linear(z_size,bert_feat)
        self.out_feat_size = out_feat_size
        self.text_cla_linear=nn.Linear(in_features=in_feat_size,out_features=out_feat_size)
        self.activation=activation
        self.head_num=head_num

        if activation!="ReLU":
            print("The activation function is not ReLU, please modify it")
            exit(-1)
        self.text_cla_linear_act=nn.ReLU()
        self.att_linear=nn.Linear(in_features=out_feat_size,out_features=out_feat_size)
        self.att_drop=nn.Dropout(p=0.5)
        self.transformer = nn.TransformerEncoderLayer(clause_num_oneart, head_num, dim_feedforward=tran_feed_size,batch_first=True)

        self.classifier=nn.Sequential(
            nn.Linear((clause_num_oneart + out_feat_size), class_num)

        )
    def forward(self, x,z_dec,clauses):
        out=self.text_cla_linear_act(self.text_cla_linear(x))
        out_att= self.att_linear(out)
        att_matrix= torch.matmul(out_att,torch.transpose(out,1,2) )
        att_matrix=torch.nn.functional.softmax(att_matrix,dim=-1)
        out=torch.matmul(att_matrix,out)
        #out=self.att_drop(out)
        clauses=torch.permute(clauses, (0, 2, 1))

        z_dec=self.z_linear(z_dec)
        clauses= torch.bmm(z_dec,clauses)

        trans_out=self.transformer(clauses)
        p_cla=torch.concat((out,trans_out),dim=-1)

        out=p_cla[:,-1,:]

        out=self.classifier(out)
        return {'out': out,'p_cla':p_cla}
class My_emo_word_exam(nn.Module):
    def __init__(self,in_feat_size,out_feat_size,emo_class_num,p_cla_size,activation="ReLU"):
        super(My_emo_word_exam, self).__init__()
        self.p_cla_size=p_cla_size
        self.in_feat_size = in_feat_size
        self.out_feat_size = out_feat_size
        self.emo_linear = nn.Linear(in_features=in_feat_size, out_features=out_feat_size)
        if activation!="ReLU":
            print("The activation function is not ReLU, please modify it yourself")
            exit(-1)
        self.emo_linear_act=nn.ReLU()
        self.att_linear = nn.Linear(in_features=out_feat_size, out_features=out_feat_size)

        #emo_class_num+1
        #because label O is not considered
        self.hidden_linear=nn.Linear(in_features=p_cla_size+out_feat_size,out_features=emo_class_num+1)
        self.crf=crf.CRF(emo_class_num+1)

    def predict(self,x,p_cla):
        out = self.emo_linear_act(self.emo_linear(x))
        out_att = self.att_linear(out)
        att_matrix = torch.matmul(out_att, torch.transpose(out, 1, 2))
        att_matrix = torch.nn.functional.softmax(att_matrix, dim=-1)
        out = torch.matmul(att_matrix, out)
        z_dec=  torch.concat((p_cla,out),dim=-1 )
        out = self.hidden_linear(z_dec)
        out = self.crf.decode(out)
        return {'out':out,'z_dec':z_dec }

    def forward(self,x,p_cla,emo_list,save_dir):
        out = self.emo_linear_act(self.emo_linear(x))
        out_att = self.att_linear(out)
        att_matrix = torch.matmul(out_att, torch.transpose(out, 1, 2))
        att_matrix = torch.nn.functional.softmax(att_matrix, dim=-1)
        out = torch.matmul(att_matrix, out)
        z_dec=  torch.concat((p_cla,out),dim=-1 )
        torch.save(z_dec,save_dir)

        out = self.hidden_linear(z_dec)

        out = -self.crf(out,emo_list)
        return out



class my_jinrong_warmup:
    def __init__(self, optimizer, lr, warmup_step_rate, total_iters,end_iters_rate):
        self.optimizer = optimizer
        self.lr = lr
        self.end_iters_rate=end_iters_rate
        self.warmup_step_rate = warmup_step_rate
        self.total_iters = total_iters
        self.current_iter = 0

    def get_lr(self):
        if self.current_iter < (self.warmup_step_rate*self.total_iters):
            lr = self.lr * (self.current_iter + 1) / (self.warmup_step_rate*self.total_iters)
        elif self.current_iter > (self.end_iters_rate*self.total_iters):
            lr = self.lr * ( 0.6 +
                             (self.total_iters-self.current_iter)
                             /((1-self.end_iters_rate)*self.total_iters)*0.4
                             )
        else:
            lr=self.lr
        return lr

    def step(self):
        lr = self.get_lr()
        for param in self.optimizer.param_groups:
            param['lr'] = lr
        self.current_iter += 1