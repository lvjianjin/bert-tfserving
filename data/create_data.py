# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:48:43 2020

@author: JianjinL
"""

import csv
import random
import mysql.connector

#数据库连接
conn = mysql.connector.connect(
  host="127.0.0.1",
  port=3306,
  user="root",
  passwd="123456",
  db='cat')
#创建游标对象
cur = conn.cursor()
#查询
sql = '''select content,label from s_email_classification limit 10000'''
cur.execute(sql)
#获取所有结果
data = cur.fetchall()
#关闭游标
cur.close()
#关闭连接
conn.close()
#将数据集打乱顺序
random.shuffle(data)
#分位数
train_point = int(len(data)*0.90)
dev_point = int(len(data)*0.95)
#写入csv
with open('train.csv','w',newline='',encoding='utf8') as file:
    writer = csv.writer(file)
    for i in range(train_point):
        writer.writerow(data[i])
        
with open('dev.csv','w',newline='',encoding='utf8') as file:
    writer = csv.writer(file)
    for i in range(train_point, dev_point):
        writer.writerow(data[i])
        
with open('test.csv','w',newline='',encoding='utf8') as file:
    writer = csv.writer(file)
    for i in range(dev_point, len(data)):
        writer.writerow(data[i])