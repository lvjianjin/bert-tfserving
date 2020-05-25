#!/usr/bin/env python3
#coding:utf-8
#author:JianjinL
'''
BERT模型ckpt文件转为部署tensorflow-serving所需文件

'''
import json
import os
from enum import Enum
from termcolor import colored
import sys
import modeling
import logging
import tensorflow as tf
import argparse
import pickle

tf.app.flags.DEFINE_string('export_model_dir', "./output/versions", 'Directory where the model exported files should be placed.')
tf.app.flags.DEFINE_integer('model_version', 56248, 'Models version number.')
FLAGS = tf.app.flags.FLAGS

def create_classification_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels):
    """

    :param bert_config:
    :param is_training:
    :param input_ids:
    :param input_mask:
    :param segment_ids:
    :param labels:
    :param num_labels:
    :param use_one_hot_embedding:
    :return:
    """

    #import tensorflow as tf
    #import modeling

    # 通过传入的训练数据，进行representation
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
    )

    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        if labels is not None:
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
        else:
            loss, per_example_loss = None, None
    return (loss, per_example_loss, logits, probabilities)

def main(max_seq_len, model_dir, num_labels):

    with tf.Session() as sess:
        #输入占位符
        input_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_ids')
        input_mask = tf.placeholder(tf.int32, (None, max_seq_len), 'input_mask')
        #模型前向传播
        bert_config = modeling.BertConfig.from_json_file('./chinese_L-12_H-768_A-12/bert_config.json')
        loss, per_example_loss, logits, probabilities = create_classification_model(bert_config=bert_config, is_training=False,
            input_ids=input_ids, input_mask=input_mask, segment_ids=None, labels=None, num_labels=num_labels)
        #转换结果格式
        logits = tf.argmax(logits, 1)
        probabilities = tf.identity(probabilities, 'pred_prob')
        #模型保存的对象
        saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess,latest_checkpoint )
        # Create SavedModelBuilder class
        # defines where the model will be exported
        export_path_base = FLAGS.export_model_dir
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(FLAGS.model_version)))
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        # Creates the TensorInfo protobuf objects that encapsulates the input/output tensors
        input_ids_tensor = tf.saved_model.utils.build_tensor_info(input_ids)
        input_mask_tensor = tf.saved_model.utils.build_tensor_info(input_mask)
        # output tensor info
        logits_output = tf.saved_model.utils.build_tensor_info(logits)
        probabilities_output = tf.saved_model.utils.build_tensor_info(probabilities)

        # Defines the DeepLab signatures, uses the TF Predict API
        # It receives an image and its dimensions and output the segmentation mask
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'input_ids': input_ids_tensor, 'input_mask': input_mask_tensor},
                outputs={'pred_label': logits_output, 'score':probabilities_output},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'result':
                    prediction_signature,
            })
        # export the model
        builder.save(as_text=True)
        print('Done exporting!')

if __name__ == '__main__':
    max_seq_len = 128
    num_labels = 2
    model_dir = './output/'
    main(max_seq_len, model_dir, num_labels)
