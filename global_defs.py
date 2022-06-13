#!/usr/bin/env python3
'''
script including
class object with global settings
'''

class CONFIG:
  
    #---------------------#
    # set necessary paths #
    #---------------------#
  
    io_path   = '/home/user/'   # directory with inputs and outputs, i.e. saving and loading data
  
    #----------------------------#
    # paths for data preparation #
    #----------------------------#
    
    IMG_DIR            = io_path + 'images/' 
    GT_DIR             = io_path + 'gt/' 
    DEPTH_DIR          = io_path + 'depth/' 
    PRED_SEM_SEG_DIR   = io_path + 'sem_seg/' 
    PRED_DEPTH_SEG_DIR = io_path + 'depth_seg/' 
    
    #------------------#
    # select or define #
    #------------------#
  
    datasets          = [ 'cityscapes', 'lost_and_found', 'a2d2', 'idd' ] 
    depth_model_names = [ 'bts', 'monodepth2'] 
    seg_model_names   = [ 'DeepLabV3Plus_WideResNet38', 'DualSeg_ResNet50' ] 
    
    DATASET           = datasets[0]    
    DEPTH_MODEL_NAME  = depth_model_names[0]
    SEG_MODEL_NAME    = seg_model_names[0]

    #--------------------------------------------------------------------#
    # select tasks to be executed by setting boolean variable True/False #
    #--------------------------------------------------------------------#

    COMPUTE_METRICS = False
    META_CLASSIF    = False
    VISUALIZE_PRED  = False
    COMPUTE_EVAL    = False
    VISUALIZE_EVAL  = False

    #-----------#
    # optionals #
    #-----------#
    
    NUM_CORES = 1
    # 0: trained on dataset, 1: only on cityscapes, 2: cityscapes + percent_model % of dataset
    SAVED_MODEL = 1
    if SAVED_MODEL == 2:
        percent_model = [20,40,60,80]
        SAVED_MODEL = 2 + percent_model[0]
    
    META_MODEL_DIR = io_path + 'meta_model/' + DEPTH_MODEL_NAME + '/' + SEG_MODEL_NAME + '/'
    METRICS_DIR    = io_path + 'metrics/'    + DATASET + '/' + DEPTH_MODEL_NAME + '/' + SEG_MODEL_NAME + '/'
    VIS_PRED_DIR   = io_path + 'vis_pred/'   + DATASET + '/' + DEPTH_MODEL_NAME + '/' + SEG_MODEL_NAME + '/'
    EVAL_PRED_DIR  = io_path + 'eval_pred/'  + DATASET + '/' + DEPTH_MODEL_NAME + '/' + SEG_MODEL_NAME + '/'


'''
In case of problems, feel free to contact
  Kira Maag, kira.maag@rub.de
'''
