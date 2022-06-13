#!/usr/bin/python
#
# Cityscapes labels
#

from collections import namedtuple

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# label and all information

Label = namedtuple('Label',['name','Id','trainId','back_foreground','color'])


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------


cs_labels = [
    #       name                       Id   trainId  back_foreground          color
    Label(  'unlabeled'            ,    0 ,     255 ,              0 , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,    1 ,     255 ,              0 , (  0,  0,  0) ),
    Label(  'rectification border' ,    2 ,     255 ,              0 , (  0,  0,  0) ),
    Label(  'out of roi'           ,    3 ,     255 ,              0 , (  0,  0,  0) ),
    Label(  'static'               ,    4 ,     255 ,              0 , (  0,  0,  0) ),
    Label(  'dynamic'              ,    5 ,     255 ,              0 , (111, 74,  0) ),
    Label(  'ground'               ,    6 ,     255 ,              0 , ( 81,  0, 81) ),
    Label(  'road'                 ,    7 ,       0 ,              0 , (128, 64,128) ),
    Label(  'sidewalk'             ,    8 ,       1 ,              0 , (244, 35,232) ),
    Label(  'parking'              ,    9 ,     255 ,              0 , (250,170,160) ),
    Label(  'rail track'           ,   10 ,     255 ,              0 , (230,150,140) ),
    Label(  'building'             ,   11 ,       2 ,              0 , ( 70, 70, 70) ),
    Label(  'wall'                 ,   12 ,       3 ,              0 , (102,102,156) ),
    Label(  'fence'                ,   13 ,       4 ,              0 , (190,153,153) ),
    Label(  'guard rail'           ,   14 ,     255 ,              0 , (180,165,180) ),
    Label(  'bridge'               ,   15 ,     255 ,              0 , (150,100,100) ),
    Label(  'tunnel'               ,   16 ,     255 ,              0 , (150,120, 90) ),
    Label(  'pole'                 ,   17 ,       5 ,              0 , (153,153,153) ),
    Label(  'polegroup'            ,   18 ,     255 ,              0 , (153,153,153) ),
    Label(  'traffic light'        ,   19 ,       6 ,              0 , (250,170, 30) ),
    Label(  'traffic sign'         ,   20 ,       7 ,              0 , (220,220,  0) ),
    Label(  'vegetation'           ,   21 ,       8 ,              0 , (107,142, 35) ),
    Label(  'terrain'              ,   22 ,       9 ,              0 , (152,251,152) ),
    Label(  'sky'                  ,   23 ,      10 ,              0 , ( 70,130,180) ),
    Label(  'person'               ,   24 ,      11 ,              1 , (220, 20, 60) ),
    Label(  'rider'                ,   25 ,      12 ,              1 , (255,  0,  0) ),
    Label(  'car'                  ,   26 ,      13 ,              1 , (  0,  0,142) ),
    Label(  'truck'                ,   27 ,      14 ,              1 , (  0,  0, 70) ),
    Label(  'bus'                  ,   28 ,      15 ,              1 , (  0, 60,100) ),
    Label(  'caravan'              ,   29 ,     255 ,              0 , (  0,  0, 90) ),
    Label(  'trailer'              ,   30 ,     255 ,              0 , (  0,  0,110) ),
    Label(  'train'                ,   31 ,      16 ,              1 , (  0, 80,100) ),
    Label(  'motorcycle'           ,   32 ,      17 ,              1 , (  0,  0,230) ),
    Label(  'bicycle'              ,   33 ,      18 ,              1 , (119, 11, 32) ),
    Label(  'license plate'        ,   -1 ,      -1 ,              0 , (  0,  0,142) ),
    Label(  'ood'                  ,   -1 ,     254 ,              1 , (255,102,  0) ),
]

