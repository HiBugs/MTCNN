#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/11/28
@Description:
"""
import tensorflow as tf
import keras.backend as K
from keras.utils import to_categorical
# def cls_loss(y_true, y_pred):
#     print("cls", y_true.shape, y_pred.shape)
#
#     label = y_true[:,0]
#
#     zeros = tf.zeros_like(label)
#     #label=-1 --> label=0net_factory
#
#     #pos -> 1, neg -> 0, others -> 0
#     label_filter_invalid = tf.where(tf.less(label,0), zeros, label)
#
#     return K.categorical_crossentropy(label_filter_invalid, y_pred)
#
# def bbox_loss(y_true, y_pred):
#     '''
#
#     :param bbox_pred:
#     :param bbox_target:
#     :param label: class label
#     :return: mean euclidean loss for all the pos and part examples
#     '''
#     print("bbox", y_true.shape, y_pred.shape)
#     bbox_pred = y_pred
#     label, bbox_target = y_true[:,0], y_true[:,1]
#     zeros_index = tf.zeros_like(label, dtype=tf.float32)
#     ones_index = tf.ones_like(label,dtype=tf.float32)
#     # keep pos and part examples
#     valid_inds = tf.where(tf.equal(tf.abs(label), 1),ones_index,zeros_index)
#     square_error = tf.square(bbox_pred - bbox_target)
#     square_error = tf.reduce_sum(square_error, axis=1)
#     square_error = square_error * valid_inds
#     return tf.reduce_sum(square_error)


num_keep_radio = 0.7


def cls_ohem(label, cls_prob):
    '''计算类别损失
    参数：
      cls_prob：预测类别，是否有人
      label：真实值
    返回值：
      损失
    '''

    print(label)
    print(cls_prob)
    zeros = tf.zeros_like(label)
    # 只把pos的label设定为1,其余都为0
    label_filter_invalid = tf.where(tf.less(label, 0), zeros, label)
    # 类别size[2*batch]
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshpae = tf.reshape(cls_prob, [num_cls_prob, -1])
    label_int = tf.cast(label_filter_invalid, tf.int32)
    # 获取batch数
    num_row = tf.to_int32(tf.shape(cls_prob)[0])
    # 对应某一batch而言，batch*2为非人类别概率，batch*2+1为人概率类别,indices为对应 cls_prob_reshpae
    # 应该的真实值，后续用交叉熵计算损失
    row = tf.range(num_row) * 2
    indices_ = row + label_int
    # 真实标签对应的概率
    label_prob = tf.squeeze(tf.gather(cls_prob_reshpae, indices_))
    loss = -tf.log(label_prob + 1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob, dtype=tf.float32)
    # 统计neg和pos的数量
    valid_inds = tf.where(label < zeros, zeros, ones)
    num_valid = tf.reduce_sum(valid_inds)
    # 选取70%的数据
    keep_num = tf.cast(num_valid * num_keep_radio, dtype=tf.int32)
    # 只选取neg，pos的70%损失
    loss = loss * valid_inds
    loss, _ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)


# In[4]:


def box_ohem(label, bbox_target, bbox_pred):
    '''计算box的损失'''
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label, dtype=tf.float32)
    # 保留pos和part的数据
    valid_inds = tf.where(tf.equal(tf.abs(label), 1), ones_index, zeros_index)
    # 计算平方差损失
    square_error = tf.square(bbox_pred - bbox_target)
    square_error = tf.reduce_sum(square_error, axis=1)
    # 保留的数据的个数
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    # 保留pos和part部分的损失
    square_error = square_error * valid_inds
    square_error, _ = tf.nn.top_k(square_error, k=keep_num)
    return tf.reduce_mean(square_error)


def landmark_ohem(label, landmark_target, landmark_pred):
    '''计算关键点损失'''
    ones=tf.ones_like(label,dtype=tf.float32)
    zeros=tf.zeros_like(label,dtype=tf.float32)
    #只保留landmark数据
    valid_inds=tf.where(tf.equal(label,-2),ones,zeros)
    #计算平方差损失
    square_error=tf.square(landmark_pred-landmark_target)
    square_error=tf.reduce_sum(square_error,axis=1)
    #保留数据个数
    num_valid=tf.reduce_sum(valid_inds)
    keep_num=tf.cast(num_valid,dtype=tf.int32)
    #保留landmark部分数据损失
    square_error=square_error*valid_inds
    square_error,_=tf.nn.top_k(square_error,k=keep_num)
    return tf.reduce_mean(square_error)


def mt_loss(y_true, y_pred):
    labels_true = y_true[:, 0]
    bbox_true = y_true[:, 1:5]

    labels_pred = y_pred[:, :2]
    bbox_pred = y_pred[:, 2:6]

    label_loss = cls_ohem(labels_true, labels_pred)
    bbox_loss = box_ohem(labels_true, bbox_true, bbox_pred)

    return label_loss + bbox_loss * 0.5


def mt_lmk_loss(y_true, y_pred):
    labels_true = y_true[:, 0]
    bbox_true = y_true[:, 1:5]
    landmark_true = y_true[:, 5:15]

    labels_pred = y_pred[:, :2]
    bbox_pred = y_pred[:, 2:6]
    landmark_pred = y_pred[:, 6:16]

    label_loss = cls_ohem(labels_true, labels_pred)
    bbox_loss = box_ohem(labels_true, bbox_true, bbox_pred)
    landmark_loss = landmark_ohem(labels_true, landmark_true, landmark_pred)
    return label_loss + bbox_loss * 0.5 + landmark_loss*0.5


def cal_accuracy(y_true, y_pred):
    """计算分类准确率"""
    cls_prob = y_pred[:, :2]
    label = y_true[:, 0]
    # 预测最大概率的类别，0代表无人，1代表有人
    pred = tf.argmax(cls_prob,axis=1)
    label_int=tf.cast(label,tf.int64)
    # 保留label>=0的数据，即pos和neg的数据
    cond=tf.where(tf.greater_equal(label_int,0))
    picked=tf.squeeze(cond)
    # 获取pos和neg的label值
    label_picked=tf.gather(label_int,picked)
    pred_picked=tf.gather(pred,picked)
    # 计算准确率
    accuracy_op=tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return accuracy_op

# def cal_mask(label_true, _type='label'):
#     def true_func():
#         return 0
#
#     def false_func():
#         return 1
#
#     label_true_int32 = tf.cast(label_true, dtype=tf.int32)
#     if _type == 'label':
#         label_filtered = tf.map_fn(lambda x: tf.cond(tf.equal(x[0], x[1]), true_func, false_func), label_true_int32)
#     elif _type == 'bbox':
#         label_filtered = tf.map_fn(lambda x: tf.cond(tf.equal(x[0], 1), true_func, false_func), label_true_int32)
#     elif _type == 'landmark':
#         label_filtered = tf.map_fn(lambda x: tf.cond(tf.logical_and(tf.equal(x[0], 1), tf.equal(x[1], 1)),
#                                                      false_func, true_func), label_true_int32)
#     else:
#         raise ValueError('Unknown type of: {} while calculate mask'.format(_type))
#
#     mask = tf.cast(label_filtered, dtype=tf.int32)
#     return mask
#
#
# def label_ohem(label_true, label_pred):
#     label_int = cal_mask(label_true, 'label')
#
#     num_cls_prob = tf.size(label_pred)
#     print('num_cls_prob: ', num_cls_prob)
#     cls_prob_reshape = tf.reshape(label_pred, [num_cls_prob, -1])
#     print('label_pred shape: ', tf.shape(label_pred))
#     num_row = tf.shape(label_pred)[0]
#     num_row = tf.to_int32(num_row)
#     row = tf.range(num_row) * 2
#     indices_ = row + label_int
#     label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
#     loss = -tf.log(label_prob + 1e-10)
#
#     valid_inds = cal_mask(label_true, 'label')
#     num_valid = tf.reduce_sum(valid_inds)
#
#     keep_num = tf.cast(tf.cast(num_valid, dtype=tf.float32) * num_keep_radio, dtype=tf.int32)
#     # set 0 to invalid sample
#     loss = loss * tf.cast(valid_inds, dtype=tf.float32)
#     loss, _ = tf.nn.top_k(loss, k=keep_num)
#     return tf.reduce_mean(loss)
#
#
# def bbox_ohem(label_true, bbox_true, bbox_pred):
#     mask = cal_mask(label_true, 'bbox')
#     num = tf.reduce_sum(mask)
#     keep_num = tf.cast(num, dtype=tf.int32)
#
#     bbox_true1 = tf.boolean_mask(bbox_true, mask, axis=0)
#     bbox_pred1 = tf.boolean_mask(bbox_pred, mask, axis=0)
#
#     square_error = tf.square(bbox_pred1 - bbox_true1)
#     square_error = tf.reduce_sum(square_error, axis=1)
#
#     _, k_index = tf.nn.top_k(square_error, k=keep_num)
#     square_error = tf.gather(square_error, k_index)
#
#     return tf.reduce_mean(square_error)
#
#
# def landmark_ohem(label_true, landmark_true, landmark_pred):
#     mask = cal_mask(label_true, 'landmark')
#     num = tf.reduce_sum(mask)
#     keep_num = tf.cast(num, dtype=tf.int32)
#
#     landmark_true1 = tf.boolean_mask(landmark_true, mask)
#     landmark_pred1 = tf.boolean_mask(landmark_pred, mask)
#
#     square_error = tf.square(landmark_pred1 - landmark_true1)
#     square_error = tf.reduce_sum(square_error, axis=1)
#
#     _, k_index = tf.nn.top_k(square_error, k=keep_num)
#     square_error = tf.gather(square_error, k_index)
#
#     return tf.reduce_mean(square_error)


if __name__ == '__main__':
    # x = tf.cast([1.,0.,0.], dtype=tf.float32)
    # y = tf.cast([[0.,1.],[1.,0.],[1.,0.]], dtype=tf.float32)
    # l = cls_ohem(x, y)

    # x = tf.cast([[0, 1.], [1., 0.], [1, 0.]], dtype=tf.float32)
    # y = tf.cast([[0.,1.],[1.,0.],[1.,0.]], dtype=tf.float32)
    # l = label_ohem(x, y)
    t = tf.cast([1,1,1], dtype=tf.float32)
    x = tf.cast([[0, 1., 1., 1.], [0, 1., 1., 1.], [0, 1., 1., 1.]], dtype=tf.float32)
    y = tf.cast([[0, 1., 1., 1.], [0, 1., 1., 1.], [0, 1., 1., 1.]], dtype=tf.float32)
    bl = box_ohem(t, x, y)

    with tf.Session() as sess:
        print(bl.eval())



