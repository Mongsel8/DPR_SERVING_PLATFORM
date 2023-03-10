import torch.nn
from transformers import DPRQuestionEncoder
from torch import Tensor as T
import torch.nn.functional as F
from transformers import XLNetTokenizer, BertModel
import torch
import subprocess
import logging
import os
tokenizer = XLNetTokenizer.from_pretrained('skt/kobert-base-v1')
model = BertModel.from_pretrained('skt/kobert-base-v1')
def unzip_model_file():
    current_path = os.getcwd()
    folder_path = current_path+'\\model_data\\'
    if 'dpr_biencoder.13' in os.listdir(folder_path):
        return 'success'
    else :
        try :
            unzip_command = '"C:\\Program Files\\7-Zip\\7z.exe" e "' + folder_path + '"'
            subprocess.run(unzip_command, shell=True, cwd=folder_path)
        except:
            logging.log(-1,'"C:\\Program Files\\7-Zip\\7z.exe"이 설치되어있는 지 확인해주세요.')

    li = os.listdir('./model_data/')
    if 'dpr_biencoder.13' in li:
        return 'success'
    else :
        return 'failed'

unzip_model_file()
model.load_state_dict(torch.load('./model_data/dpr_biencoder.13',map_location=torch.device('cpu')),strict=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def get_idx(*args):
    idxs = []
    for _ in args:
        idxs.append(tokenizer(_,return_tensors="pt")["input_ids"])
    return idxs
def get_pooleroutput(List):
    embeddings = []
    for _ in List:
        embeddings.append(model(_).pooler_output)
    return embeddings

def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)
def get_total_scores(q_vector: T, ctx_vectors: T):
    cos_score=cosine_scores(q_vector, ctx_vectors)
    dot_pdt_score=dot_product_scores(q_vector, ctx_vectors)
    total_score = (cos_score*0.5)+(dot_pdt_score*0.5)
    return round(total_score.item(),4)
def get_title_dpr(item):
    _ = tokenizer(item, return_tensors="pt")["input_ids"]
    input_data = _.to(device)
    res = model(input_data).pooler_output.detach().numpy().tolist()
    return res

def get_content_dpr(item):
    _ = tokenizer(item, return_tensors="pt",truncation=True,max_length=512)["input_ids"]
    input_data = _.to(device)
    res = model(input_data).pooler_output.detach().numpy().tolist()
    return res