# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:41:13 2020

@author: JianjinL
"""
import sys
sys.path.append('..')
import requests
import json
from bert import tokenization

tokenizer = tokenization.FullTokenizer(
        vocab_file='../chinese_L-12_H-768_A-12/vocab.txt', 
        do_lower_case=True)

def text2ids(textList):
    '''
    将输入的待分类文本编码为模型所需要的编码形式
    :parma textList: 输入的文本列表，形如 ['今天天气真好','我爱你']
    :return input_ids_list: 句子的向量表示形式
    :return input_mask_list: 只有一个句子，所以目前为固定格式
    '''
    input_ids_list = []
    input_mask_list = []
    for text in textList:
        if len(text) >= 128:
            text = text[:128]
        tokens_a = tokenizer.tokenize(text)
        ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
        tokens = []
        segment_ids = []
        tokens.append(0)
        segment_ids.append(0)
        for token in ids_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(0)
        segment_ids.append(0)
        input_ids = [tokens + [0]*(128-len(tokens))]
        input_mask = [[0]*128]
        input_ids_list.append(input_ids)
        input_mask_list.append(input_mask)
    return input_ids_list, input_mask_list

def bert_class(textList):
    '''
    调用tfserving服务的接口，对外提供服务
    :parma textList: 输入的文本列表，形如 ['今天天气真好','我爱你']
    :return result: 结果
    '''
    input_ids_list, input_mask_list = text2ids(textList)
    url = 'http://127.0.0.1:8501/v1/models/versions:predict'
    data = json.dumps(
            {
                    "name": 'deeplab',
                    "signature_name":'result',
                    "inputs":{
                            'input_ids': input_ids_list,
                            'input_mask': input_mask_list}})
    result = requests.post(url,data=data).json()
    return result

if __name__ == '__main__':
    textList = ['''感谢您百忙之中接听电话，海伦春天新楼王x号楼xx-xxx㎡阔景洋房应势加推，全城热抢，，更多惊喜等你来，销售代表—陈静^_^''']
    result = bert_class(textList)
    print(result)