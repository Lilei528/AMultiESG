import os.path
import random
import sys
import warnings

from pyltp import Segmentor
from torch.utils.data import Dataset
import torch

from transformers import BertModel, BertTokenizer,BertConfig

space_in_html={' ',' ',' ','​','—',"　",' ','\n'}

label_dict={'0':"负面环境",'1':"正面环境",'2':"负面管理"
            ,'3':"正面管理",'4':"负面社会",'5':"正面社会"
            }
emo_label_dict={'0':"E-",'1':"E-",'2':"G-"
            ,'3':"G-",'4':"S-",'5':"S-"
            }
emo2index={'O':0,"BE-NEG":1,'BE-POS':2,'IE-NEG':3,'IE-POS':4,
"BG-NEG":5,'BG-POS':6,'IG-NEG':7,'IG-POS':8,
"BS-NEG":9,'BS-POS':10,'IS-NEG':11,'IS-POS':12

}
class jinrong_dataset(Dataset):
    def __init__(self,data_dir,load_type='train',padding_str='[PAD]'):
        self.data_dir=data_dir
        self.padding_str=padding_str
        f=open(data_dir,"r",encoding='utf-8')
        self.len=len(f.readlines())
        self.load_type=load_type
        f.close()
    def __getitem__(self,index):
        if self.load_type=='train':
            f = open(self.data_dir, "r", encoding='utf-8')
            sentence = f.readlines()[index]
            f.close()
            y=sentence.split("\t")[0]
            x=sentence.split("\t")[1].replace(self.padding_str,'')
            for spad in space_in_html:
                x=x.replace(spad,'')
            emo_y='\t'.join(sentence.split('\t')[2:])
            return (x,y,emo_y)
        else :
            f = open(self.data_dir, "r", encoding='utf-8')
            sentence = f.readlines()[index]
            f.close()
            y = sentence.split("\t")[0]
            x = sentence.split("\t")[1].replace(self.padding_str,'')
            for spad in space_in_html:
                x = x.replace(spad, '')

            return (x, y)

    def __len__(self):
        return self.len

def get_padding(padding_str='[PAD]'):
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    sep_id = tokenizer.encode(padding_str, add_special_tokens=False)
    return sep_id[0]
def make_data(max_len,x,y,class_num,clause_len,batch_emo_y,clause_num_oneart,padding_str='[PAD]',
              mkdype='train'
              ):

    if mkdype=='train':
        clause_num_oneart-=1
    f=open('./data/lack.txt','a',encoding='utf-8')


    max_len-=2
    clause_len-=2
    new_x=[]
    for i in x:
        tem=''
        if len(i)<max_len:
            tem=i+padding_str*(max_len-len(i))
        if len(i)>max_len:
            tem=i[0:max_len]

        new_x.append(tem)
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    #The clause is converted into an id and put into a list
    clauses=[]
    for i in x:
        # i is an article
        tem_clause=i.split("。")
        # The number of clauses divided into an article should be fixed
        # max_+ 1 is because the other is a label
        while (len(tem_clause[:]) < (clause_num_oneart)):
            tem_clause.append("[PAD]"*(clause_len) )
        if (len(tem_clause[:]) > (clause_num_oneart)):
            tem_clause = tem_clause[:clause_num_oneart]

        for index,j in enumerate(tem_clause):
            pass_end=False


            # J is a clause
            if index==(len(tem_clause)-1) and str(j).replace(" ",'').replace("\n","")==""  :
                pass_end=True
                tem = padding_str*clause_len

            if not pass_end:

                if len(j) < clause_len:
                    add_len = clause_len - len(j)
                    tem = ' '.join([i for i in j])
                    tem = tem + padding_str * (add_len)

                elif len(j) > clause_len:
                    tem = j[0:clause_len]
                    tem = ' '.join([i for i in tem])
                else:
                    if j.count('[PAD]')!=0:
                        j=j.replace(padding_str,' ')

                    tem = ' '.join([i for i in j])
                    tem.replace(' ','[PAD]')

            tem_clause[index]=tokenizer.encode(tem, add_special_tokens=True)

            if len(tem_clause[index])<(clause_len+2):
                add_len2 = (clause_len+2) - len(tem_clause[index])
                add_content = tokenizer.encode(
                    ((padding_str + " ") * add_len2), add_special_tokens=False)
                tem_clause[index]+=add_content
                print('Contains special characters')
                print("sentence：",j)


            if None in tem_clause[index]:
                for tem_i,tem_j in enumerate(tem_clause[index]):
                    if tem_j==None:
                        tem_clause[index][tem_i]=get_padding()
                        f.write(j[tem_i-1]+"\t")


        clauses.append( torch.LongTensor(tem_clause) )



    # 一篇文章转成id
    ret_x = []
    for one_art in new_x:

        if one_art.count(padding_str)!=0:
            one_art=one_art.replace(padding_str,' ')

        input_ids = tokenizer.encode((' '.join([i for i in one_art])).replace(' ',padding_str)
                                     , add_special_tokens=True)

        if None in input_ids:
            for tem_i, tem_j in enumerate(input_ids):
                if tem_j == None:
                    input_ids[tem_i] = get_padding()
                    f.write(one_art[tem_i - 1] + "\t")

        if len(input_ids) < (max_len + 2):
            add_len2=(max_len + 2)-len(input_ids)
            add_content=tokenizer.encode(
                ((padding_str+" ")*add_len2), add_special_tokens=False)
            input_ids+=add_content
            print('Contains special characters')
            print("sentence：", one_art)


        ret_x.append(input_ids)
    #Convert tags to one pot
    tensor_y=  [    [  0  for i in range(class_num)] for j in y  ]
    for index,val in enumerate(tensor_y):
        tensor_y[index][int(y[index])]=1

    # Convert label to ID
    label_list=[]
    for val in y:
        label_cont=label_dict[val]
        label_list.append(tokenizer.encode(label_cont, add_special_tokens=True))
    f.close()
    #Data set of subtasks


    if mkdype=='train':
        ret_emo_list = []
        for one_batch in batch_emo_y:
            one_emo=one_batch.split('\t')
            if '\n' in str(one_emo[-1]):
                one_emo[-1]=one_emo[-1].replace('\n','')

            if len(one_emo)>max_len:
                one_emo=one_emo[:max_len]
            elif len(one_emo)<max_len:
                one_emo=one_emo+["O"]*(max_len-len(one_emo))
            one_emo.append('O')
            one_emo.insert(0, "O")
            try:
                ret_emo_list.append([emo2index[i] for i in one_emo ])
            except:
                print(one_emo)
        return torch.LongTensor(ret_x),\
           torch.FloatTensor(tensor_y),\
           clauses,\
           torch.LongTensor(label_list),\
           torch.LongTensor(ret_emo_list)

    return torch.LongTensor(ret_x),\
           torch.FloatTensor(tensor_y),\
           clauses,\
           torch.LongTensor(label_list),\
           None



def shuff_data(my_data):
    random.shuffle(my_data)


def remove_space_data(my_data,padding_str="[PAD]"):

    for index,val in enumerate(my_data):

        label,content=val.split('\t')
        val_tem=str(content).replace(padding_str, '')
        for spad in space_in_html:
            val_tem=val_tem.replace(spad,'')
        my_data[index]=label+'\t'+val_tem

def pretreat_data(data_dir,shuffle=True,padding_str='[PAD]'):
    f=open(data_dir,'r',encoding='utf-8')
    data=f.readlines()
    f.close()
    remove_space_data(data,padding_str=padding_str)
    if shuffle:
        shuff_data(data)
    return data

def remove_space(da):
    return da.replace('\n','')

def add_emo_dict(data_dir,mk_data_dir,dict_dir,cws_dir):
    batch=32

    sr=SynonymsReplacer(cws_dir)
    f = open(data_dir, 'r', encoding='utf-8')
    data = f.readlines()
    f.close()

    totoal_num= len(data)
    finish_num=0
    f = open(os.path.join(dict_dir,"degrade.txt"), 'r', encoding='utf-8')
    degrade_dict_data = f.readlines()
    degrade_dict_data=list(map(remove_space,degrade_dict_data))
    f.close()
    f = open(os.path.join(dict_dir, "praise.txt"), 'r', encoding='utf-8')
    praise_dict_data = f.readlines()
    praise_dict_data=list(map(remove_space,praise_dict_data))
    f.close()
    f = open(os.path.join(mk_data_dir), 'w', encoding='utf-8')
    one_write = ''
    for index,val in enumerate(data):

        label,content=val.split("\t")
        content=content.replace('\n','')
        ret=sr.segment(content)
        sen_emo = ['O\t' for i in range(len(content))]
        now_index=0

        for index2,word in enumerate(ret):

            indict=False
            if word in degrade_dict_data:
                la='NEG'
                indict=True
            elif word in praise_dict_data:
                la='POS'
                indict=True
            if indict:
                emo = emo_label_dict[label]
                sen_emo[now_index] = 'B' + emo +la+ '\t'
                try :
                    for tem in range(1,len(word) ):
                        sen_emo[now_index + tem] = 'I' + emo +la + '\t'
                except:
                    print('An error occurred while adding emotional word tags to the data')
                    exit(-1)
            now_index+=len(word)

        finish_num += 1
        sen_emo[-1]=sen_emo[-1].replace('\t','')
        one_write+=val.replace('\n','')+'\t'+''.join(sen_emo)+'\n'
        if one_write.count('\n')==batch or index==len(data)-1:
            f.write(one_write)
            one_write=''
            print(end='\r speed of progress %d / %d' % (finish_num, totoal_num))
    f.close()

def save_shuffle_data(data,dir):
    f=open(dir,'w',encoding='utf-8')
    for i in data:
        f.write(i)
        f.write('\n')
    f.close()
def get_target(y_cat,y,class_num,save_dir=None):


    matx=[]
    for i in range(class_num):
        matx.append([0 for j in range(class_num)])

    for index,val in enumerate(y):
        if val==y_cat[index]:
            matx[int(val)][int(val)]+=1
        else:
            matx[int(val)][y_cat[index]]+=1

    if save_dir!=None:
        f=open(save_dir,'a',encoding='utf-8')
        f.write('*'*30)
        f.write('\n')
        f.write(str(matx))
        f.write('\n')
        f.write('*'*30)
        f.close()

    return get_target_by_matx(matx)


def get_target_by_matx(matx):

    n=len(matx)
    tem_dict={"pre":0,"recall":0,'f1_score':0}
    total_len=0

    for i in range(n):
        rowsum, colsum = sum(matx[i]), sum(matx[r][i] for r in range(n))
        if rowsum==0 or colsum==0:
            print('The sum of labels with index %d is zero, skipping'%i)
            continue
        total_len+=1
        f1 = 0
        pre = 0
        recall = 0
        try:

            pre = (matx[i][i] / float(colsum))
            recall = (matx[i][i] / float(rowsum))
            f1 = 2 * pre * recall / (pre + recall)
        except(ZeroDivisionError):
            print("pre or recall is zero!!")
            print('pre:',pre)
            print('recall:',recall)
        try:
            tem_dict['pre']+=pre
            tem_dict['recall']+=recall
            tem_dict['f1_score']+=f1
        except ZeroDivisionError:
            print('precision: %s' % 0, 'recall: %s' % 0)
    if total_len==0:
        warnings.warn("The confusion matrix is 0")
    for key,val in tem_dict.items():
        tem_dict[key]=val/total_len
    return tem_dict

class SynonymsReplacer:
    def __init__(self,  cws_model_path):

        self.segmentor = self.load_segmentor(cws_model_path)

    def __del__(self):
        """The Pyltp word segmentation model should be released when the object is destroyed"""
        self.segmentor.release()

    def load_segmentor(self, cws_model_path):
        """
        Load ltp word segmentation model
        """
        segmentor = Segmentor(model_path=cws_model_path)
        return segmentor

    def segment(self, sentence):
        """Call the word segmentation method of Pyltp to segment str sentences and return them in list form"""
        return list(self.segmentor.segment(sentence))



