# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
author:satoshi tsutsui
Bulk figure extractor
'''

import tensorflow as tf
from code.utils import postprocess,preprocess,load_graph
import os
import cv2
import argparse
import json

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--images", type=str, help=u"the directory that has figures",required=True)
parser.add_argument("--model",default="./data/figure-sepration-model-submitted-544.pb", type=str, help=u"model pb file. Default is ./data/figure-sepration-model-submitted-544.pb")
parser.add_argument("--thresh",default=0.5, type=float, help=u"sub-figuere detection threshold. Default is 0.5")
parser.add_argument("--output",default="./results", type=str, help=u"output directory ./results")
parser.add_argument("--annotate",default=0, type=int, help=u"save annotation to the image or not. 1 is yes, 0 is no. Default is 0.")
args = parser.parse_args()

#network settings
meta={'object_scale': 5, 'classes': 1, 'out_size': [17, 17, 30],  'colors': [(0, 0, 254)], 'thresh': args.thresh, 'anchors': [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52],'num': 5,'labels': ['figure']}

#annotation settings
annotate = False
if args.annotate==0:
    pass
elif args.annotate==1:
    annotate = True
else:
    print("Warining! This might be invalid annotation config")

#load graph
graph=load_graph(args.model)

#list image files
images=os.listdir(args.images)

sub_figures=[]
with tf.Session(graph=graph) as sess:
    print("---------------")
    print("Input diractory: %s"%args.images)
    print("The diractory has %s images"%len(images))
    print("Extraction started")
    for img_file in images:
            #load image
            imgcv,imgcv_resized,img_input = preprocess(args.images+"/"+img_file)

            #check if it is really a image or not
            if imgcv is None:
                print("%s is skipped because it is currupted or not a image file."%img_file)
                continue
            else:
                print(img_file)

            #detect it!
            detections = sess.run('output:0', feed_dict={'input:0': img_input})

            #post process it
            sub_figures,annotated_image=postprocess(meta,detections,imgcv,annotate)

            #if annotation is enabled, save the annotated image
            if annotate:
                annotated_image_name=args.output+"/"+img_file+".annotated.png"
                cv2.imwrite(annotated_image_name,annotated_image)

            #save output json
            json_name=args.output+"/"+img_file+".json"
            with open(json_name, 'w') as f:
                json.dump(sub_figures, f, sort_keys=True, indent=4)