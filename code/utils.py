# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
author:satoshi tsutsui
utilities
'''

import tensorflow as tf
import sys
# sys.path.insert(0, './')
import numpy as np
import math
import cv2
import os
from code.box import BoundBox, box_iou, prob_compare
from code.box import prob_compare2, box_intersection
import argparse

def expit(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def postprocess(meta, net_out, imgcv, annotate=False):
    '''
    citation: https://github.com/thtrieu/darkflow/blob/99f9a95468f9bd858d610530524f83612bf635eb/net/yolov2/test.py
    meta: meta data
    net_out:output from the CNN
    imgcv: original image array
    annotate: anntoate bounding box to the image or not
    '''

    # meta
    meta = meta
    H, W, _ = meta['out_size']
    threshold = meta['thresh']
    C, B = meta['classes'], meta['num']
    anchors = meta['anchors']
    net_out = net_out.reshape([H, W, B, -1])

    boxes = list()
    for row in range(H):
        for col in range(W):
            for b in range(B):
                bx = BoundBox(C)
                bx.x, bx.y, bx.w, bx.h, bx.c = net_out[row, col, b, :5]
                bx.c = expit(bx.c)
                bx.x = (col + expit(bx.x)) / W
                bx.y = (row + expit(bx.y)) / H
                bx.w = math.exp(bx.w) * anchors[2 * b + 0] / W
                bx.h = math.exp(bx.h) * anchors[2 * b + 1] / H
                classes = net_out[row, col, b, 5:]
                bx.probs = _softmax(classes) * bx.c
                bx.probs *= bx.probs > threshold
                boxes.append(bx)

    # non max suppress boxes
    for c in range(C):
        for i in range(len(boxes)):
            boxes[i].class_num = c
        boxes = sorted(boxes, key = prob_compare, reverse = True)
        for i in range(len(boxes)):
            boxi = boxes[i]
            if boxi.probs[c] == 0: continue
            for j in range(i + 1, len(boxes)):
                boxj = boxes[j]
                if box_iou(boxi, boxj) >= .4:
                    boxes[j].probs[c] = 0.

    colors = meta['colors']
    labels = meta['labels']
    h, w, _ = imgcv.shape
    
    outboxes=[]
    for b in boxes:
        max_indx = np.argmax(b.probs)
        max_prob = b.probs[max_indx]
        label = labels[max_indx]
        # print(max_prob)
        if max_prob > threshold:
            left  = int(round((b.x - b.w/2.) * w))
            right = int(round((b.x + b.w/2.) * w))
            top   = int(round((b.y - b.h/2.) * h))
            bot   = int(round((b.y + b.h/2.) * h))
            if left  < 0    :  left = 0
            if right > w - 1: right = w - 1
            if top   < 0    :   top = 0
            if bot   > h - 1:   bot = h - 1

            x_box=left
            y_box=top
            w_box=right-left
            h_box=bot-top
            #print({"x":x_box,"y":y_box,"w":w_box,"h":h_box,"conf":max_prob})
            outboxes.append({"x":x_box,"y":y_box,"w":w_box,"h":h_box,"conf":float(max_prob)})

            if annotate:
                thick = int((h+w)/300)
                mess = '%03.3f'%max_prob
                cv2.rectangle(imgcv,(left, top), (right, bot),colors[max_indx], thick)
                cv2.putText(imgcv, mess, (left+thick*4, top +thick*6),0, 1e-3 * h, colors[max_indx],thick//3)

    return outboxes,imgcv

def preprocess(img_path,w=544,h=544):
    imgcv = cv2.imread(img_path)
    if imgcv is None:
        return None,None,None
    imgcv_resized = cv2.resize(imgcv, (w, h))
    img_input = imgcv_resized / 255.
    img_input = img_input[:,:,::-1]
    img_input = np.expand_dims(img_input, axis=0)
    return imgcv,imgcv_resized,img_input

def load_graph(frozen_graph_filename):
    #citation: code is taken from https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc#.137byfk9k
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,name="")
    return graph

#here is my simple implementation
if __name__ == '__main__':
    #test
    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, help=u"input image",required=True)
    parser.add_argument("--out",default="predicted.png", type=str, help=u"output image file path")
    parser.add_argument("--model",default="./data/figure-sepration-model-submitted-544.pb", type=str, help=u"model pb file")
    parser.add_argument("--thresh",default=0.5, type=float, help=u"detection threshold")
    args = parser.parse_args()
    graph=load_graph(args.model)

    # for op in graph.get_operations():
    #     print(op.name)

    imgcv,imgcv_resized,img_input=preprocess(args.img)

    with tf.Session(graph=graph) as sess:
        detections = sess.run('output:0', feed_dict={'input:0': img_input})

    meta={'object_scale': 5, 'classes': 1, 'out_size': [17, 17, 30],  'colors': [(0, 0, 254)], 'thresh': args.thresh, 'anchors': [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52],'num': 5,'labels': ['figure']}

    outboxes,detected=postprocess(meta,detections,imgcv)
    cv2.imwrite(args.out,detected)

    print("Detected %d figures"%len(outboxes))
    print("Saved to %s"%args.out)