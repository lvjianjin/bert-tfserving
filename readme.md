# bert-tfserving

**\*\*\*\*\* BERT模型从模型训练到服务部署\*\*\*\*\***
```
项目详细介绍可参见：https://www.jianshu.com/p/383129b2bf7f
凡对本项目有任何疑惑可加QQ群交流：1081332609
```
## 目录结构：

```
bert-tfserving--|--bert(geogle发布的bert项目)
                |--chinese_L-12_H-768_A-12（下载的预训练文件）
                |--data（训练数据）
                |--output（保存的结果）
                |--client(客户端)
                |--readme.md
```
## 数据准备

在data/中准备形如data.csv,dev.csv,test.csv的文件。

## 修改标签

请修改run_classifier.py中EmailProcessor类的get_labels方法，改为你的训练标签。

## 训练

在项目路径下运行：
(若使用显卡，在速度上会有质的提升，但是若显卡的显存不够可调小train_batch_size参数)
```
python3 ./bert/run_classifier.py \
        --data_dir=./data/ \
        --task_name=email \
        --vocab_file=./chinese_L-12_H-768_A-12/vocab.txt \
        --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json \
        --output_dir=./output/ \
        --do_train=true \
        --do_eval=true \
        --init_checkpoint=./chinese_L-12_H-768_A-12/bert_model.ckpt \
        --max_seq_length=128 \
        --train_batch_size=32 \
        --learning_rate=5e-5 \
        --num_train_epochs=2.0
```
训练过程中会输出当前模型损失函数的值，模型训练完成会显示整体准确率，训练结果位于output/文件夹中。

## 服务部署

首先，运行如下语句生产模型启动所需version文件。

```
python3 ./bert/save_model.py
```
完成后，即可于output/文件夹下看到一个versions文件夹。

生产环境一般以tensorflow-servng对模型进行部署。部署前需先拉取其docker镜像

```
docker pull tensorflow/serving
```

拉取完成后，以如下语句启动服务端,其中source后接的是刚才模型生成的versions文件夹对应路径。

```
docker run --name tfserving-bert \
        --hostname tfserving-bert \
        -tid \
        --restart=on-failure:10 \
        -v  /etc/timezone:/etc/timezone \
        -v  /etc/localtime:/etc/localtime \
        -p 8501:8501 \
        -p 8502:8502 \
        --mount type=bind,source=/home/python-project/bert-tfserving/output/versions,target=/models/versions \
        -e MODEL_NAME=versions \
        -t tensorflow/serving &
```
## 客户端

请求方式可参照 ./bert/client.py 文件