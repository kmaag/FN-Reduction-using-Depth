#!/usr/bin/env python3
"""
script including functions for easy usage in main scripts
"""

import os
import sys
import pickle
import numpy as np

from global_defs import CONFIG


def concat_metrics(loader):
    print('concat metrics')

    if os.path.isfile(CONFIG.METRICS_DIR+'metrics_all.p'):
        metrics =  pickle.load( open( CONFIG.METRICS_DIR+'metrics_all.p', "rb" ) )
    else:
        metrics =  pickle.load( open( CONFIG.METRICS_DIR+loader[0][5]+'_metrics.p', "rb" ) )
        for item,i in zip(loader,range(len(loader))):
            if i == 0:
                continue
            sys.stdout.write("\t concatenated file number {} / {}\r".format(i,len(loader)))
            m = pickle.load( open( CONFIG.METRICS_DIR+item[5]+'_metrics.p', "rb" ) )
            for j in metrics:
                metrics[j] += m[j]
        pickle.dump( metrics, open( CONFIG.METRICS_DIR+'metrics_all.p', "wb" ) )
    print(" ")
    print("components (combination):", len(metrics['iou0']) )
    print("connected components:", np.sum( np.asarray(metrics['S']) != 0) )
    print("non-empty connected components:", np.sum( np.asarray(metrics['S_in']) != 0) )
    print("IoU = 0:", np.sum( np.asarray(metrics['iou0']) == 1) )
    print("IoU > 0:", np.sum( np.asarray(metrics['iou0']) == 0) )

    if os.path.isfile(CONFIG.METRICS_DIR+'metrics_seg_all.p'):
        metrics_seg =  pickle.load( open( CONFIG.METRICS_DIR+'metrics_seg_all.p', "rb" ) )
    else:
        metrics_seg =  pickle.load( open( CONFIG.METRICS_DIR+loader[0][5]+'_metrics_seg.p', "rb" ) )
        for item,i in zip(loader,range(len(loader))):
            if i == 0:
                continue
            sys.stdout.write("\t concatenated file number {} / {}\r".format(i,len(loader)))
            m = pickle.load( open( CONFIG.METRICS_DIR+item[5]+'_metrics_seg.p', "rb" ) )
            for j in metrics_seg:
                metrics_seg[j] += m[j]
        pickle.dump( metrics_seg, open( CONFIG.METRICS_DIR+'metrics_seg_all.p', "wb" ) )
    print(" ")
    print("components (semantic segmentation):", len(metrics_seg['iou0']) )
    print("connected components:", np.sum( np.asarray(metrics_seg['S']) != 0) )
    print("non-empty connected components:", np.sum( np.asarray(metrics_seg['S_in']) != 0) )
    print("IoU = 0:", np.sum( np.asarray(metrics_seg['iou0']) == 1) )
    print("IoU > 0:", np.sum( np.asarray(metrics_seg['iou0']) == 0) )
    return metrics, metrics_seg


def metrics_to_nparray( metrics, names, normalize=False, non_empty=False ):

    I = range(len(metrics['S_in']))
    if non_empty == True:
        I = np.asarray(metrics['S_in']) > 0
    M = np.asarray( [ np.asarray(metrics[ m ])[I] for m in names ] )
    MM = M.copy()
    if normalize == True:
        for i in range(M.shape[0]):
            if names[i] != "class":
                M[i] = ( np.asarray(M[i]) - np.mean(MM[i], axis=-1 ) ) / ( np.std(MM[i], axis=-1 ) + 1e-10 )
    M = np.squeeze(M.T)
    return M


def label_as_onehot(label, num_classes, shift_range=0):
    y = np.zeros((num_classes, label.shape[0], label.shape[1]))
    for c in range(shift_range,num_classes+shift_range):
        y[c-shift_range][label==c] = 1
    y = np.transpose(y,(1,2,0)) # shape is (height, width, num_classes)
    return y.astype('uint8')


def classes_to_categorical( classes, nc = None ):
    classes = np.squeeze( np.asarray(classes) )
    if nc == None:
        nc      = np.max(classes)
    classes = label_as_onehot( classes.reshape( (classes.shape[0],1) ), nc ).reshape( (classes.shape[0], nc) )
    names   = [ "C_"+str(i) for i in range(nc) ]
    return classes, names


def metrics_to_dataset( metrics, non_empty=False, probs=True ):
  
    X_names = sorted([ m for m in metrics if m not in ["class","iou0","index"] and "cprob" not in m ])
  
    class_names = []
    for m in metrics:
        if "cprob" in m:
            class_names.append(m)
    num_classes = len(class_names)
    if not probs:
        class_names = ["class"]

    Xa      = metrics_to_nparray( metrics, X_names    , normalize=True , non_empty=non_empty )
    classes = metrics_to_nparray( metrics, class_names, normalize=False, non_empty=non_empty )
    y0a     = metrics_to_nparray( metrics, ["iou0"]   , normalize=False, non_empty=non_empty )
    
    if not probs:
        classes, class_names = classes_to_categorical( classes, num_classes )
  
    Xa = np.concatenate( (Xa,classes), axis=-1 )
    X_names += class_names
    return Xa, np.squeeze(y0a), X_names  


def save_perform_metrics_all(path, name, loader):
    
    print('concatenate performance metrics')
    
    foregrounds = [11,12,13,14,15,16,17,18]
    
    metrics = { "seg": np.zeros((2,3)), "combi": np.zeros((2,3)), "seg_segm": np.zeros((1,3)), "combi_segm": np.zeros((1,3)) }
    
    if CONFIG.DATASET != 'lost_and_found':
        
        for m in ['seg_class','combi_class','seg_segm_class','combi_segm_class']:
            metrics[m] = np.zeros((len(foregrounds),3))

    for i in loader:
        if os.path.isfile(os.path.join(path, str(i[5]) + '.p')):
            metrics_i =  pickle.load( open( os.path.join(path, str(i[5]) + '.p'), "rb" ) )
            for m in metrics:
                metrics[m] += metrics_i[m]
        else:
            print('file', i[5], 'is missing')
    pickle.dump( metrics, open( os.path.join(path, name), "wb" ) )  


def fill_data(path, method='combi_segm', idx=0):
    # tp, fp, fn
    data = np.zeros((101,3))
    for i in range(0,101,1):
        if os.path.isfile(os.path.join(path, 'performance_all' + str(i) + '.p')):
            metrics_mc =  pickle.load( open( os.path.join(path, 'performance_all' + str(i) + '.p'), "rb" ) )
            data[i,:] = metrics_mc[method][idx]
        else:
            data[i,:] = (np.nan, np.nan, np.nan)
    return data


def compute_best_th( path_in ):

    metrics = pickle.load( open( os.path.join(path_in, str(CONFIG.SAVED_MODEL), 'performance_all100.p'), "rb" ) )

    data = fill_data(os.path.join(path_in, str(CONFIG.SAVED_MODEL)))
    data[np.logical_and(data[:,0]==0,data[:,1]==0),0] = 1

    prec_seg = metrics['seg_segm'][0,0] / (metrics['seg_segm'][0,0]+metrics['seg_segm'][0,1])
    rec_seg = metrics['seg_segm'][0,0] / (metrics['seg_segm'][0,0]+metrics['seg_segm'][0,2])
    
    prec = data[:,0] / (data[:,0]+data[:,1])
    rec = data[:,0] / (data[:,0]+data[:,2])

    idx_th = np.argmin(np.abs(prec-prec_seg))

    return idx_th, rec_seg, rec[idx_th]

