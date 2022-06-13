#!/usr/bin/env python3
"""
script including functions that do calculations
"""

import os
import time
import pickle
import numpy as np
import xgboost as xgb
from scipy.stats import entropy
from skimage import measure as ms
from sklearn.metrics import roc_curve, auc
from skimage.segmentation import find_boundaries

from global_defs import CONFIG
from helper      import fill_data


def comp_entropy( probs ):
        return entropy(probs, axis=0)
  

def variation_ratio( probs ):
    output = np.ones((probs.shape[1],probs.shape[2]))
    return output - np.sort(probs, axis=0)[-1,:,:]
  

def probdist( probs ):
    output = np.ones((probs.shape[1],probs.shape[2]))
    return output - np.sort(probs, axis=0)[-1,:,:] + np.sort(probs, axis=0)[-2,:,:]


def calculate_metrics_i(idx, foregrounds, components_pos, components_in_bd, components_class, pred_seg, pred_depth, gt_01, name):

    heatmaps = { "E": comp_entropy( pred_seg ), "V": variation_ratio( pred_seg ), "M": probdist( pred_seg ), "F": comp_entropy( pred_depth ) }  
    metrics = { "index": list([]), "iou0": list([]), "class": list([]), "mean_x": list([]), "mean_y": list([]) } 
  
    for m in list(heatmaps)+["S"]:
        metrics[m          ] = list([])
        metrics[m+"_in"    ] = list([])
        metrics[m+"_bd"    ] = list([])
        metrics[m+"_rel"   ] = list([])
        metrics[m+"_rel_in"] = list([])
    for c in foregrounds:
        metrics['cprob'+str(c)] = list([])
    
    for i in range(1,len(np.unique(components_pos))):

        for m in metrics:
            metrics[m].append( 0 )
    
        metrics["index"][-1] = idx
    
        metrics["S_in"][-1] = np.sum(components_in_bd==i)
        metrics["S_bd"][-1] = np.sum(components_in_bd==-i)
        metrics["S"][-1] = metrics["S_in"][-1] + metrics["S_bd"][-1]
        metrics["S_rel"][-1] = metrics["S"][-1] / metrics["S_bd"][-1]
        metrics["S_rel_in"][-1] = metrics["S_in"][-1] / metrics["S_bd"][-1]

        for m in heatmaps:

            metrics[m+"_in"][-1] = np.sum(heatmaps[m][components_in_bd==i])
            metrics[m+"_bd"][-1] = np.sum(heatmaps[m][components_in_bd==-i])
            metrics[m][-1] = (metrics[m+"_in"][-1] + metrics[m+"_bd"][-1]) / metrics["S"][-1]
            if metrics["S_in"][-1] > 0:
                metrics[m+"_in"][-1] /= metrics["S_in"][-1]
            metrics[m+"_bd"][-1] /= metrics["S_bd"][-1]
            metrics[m+"_rel"][-1] = metrics[m][-1] * metrics["S_rel"][-1]
            metrics[m+"_rel_in"][-1] = metrics[m+"_in"][-1] * metrics["S_rel_in"][-1]
        
        metrics["class"][-1] = components_class[components_pos==i].max()

        for c in foregrounds:
            metrics["cprob"+str(c)][-1] = np.sum( np.asarray(pred_seg[c,components_pos==i],dtype=np.float32) ) / metrics["S"][-1]

        x_y_indices = np.sum( np.asarray( np.where(components_pos==i) ),axis=1 )
        metrics["mean_x"][-1] = x_y_indices[1] / metrics["S"][-1]
        metrics["mean_y"][-1] = x_y_indices[0] / metrics["S"][-1]

        if np.sum( np.logical_and(components_pos==i, gt_01==1) ) == 0:
            metrics["iou0"][-1] = 1

    pickle.dump( metrics, open( CONFIG.METRICS_DIR+name, "wb" ) )
    

def comp_metrics_i(item, idx):

    start = time.time()

    imx = item[1].shape[0]
    imy = item[1].shape[1]
    foregrounds = [11,12,13,14,15,16,17,18]
    num_classes = 19
    
    gt_01 = np.zeros((imx, imy))
    if CONFIG.DATASET == 'lost_and_found':
        gt_01[item[1]==254] = 1
    else:
        for c in foregrounds:
            gt_01[item[1]==c] = 1

    pred_seg = np.argmax(item[3], axis=0)
    pred_seg[item[1]==255] = 255

    pred_depth = np.argmax(item[4], axis=0)
    pred_depth[item[1]==255] = 255

    combi = np.zeros((imx, imy))
    combi[pred_depth==1] = -1
    for c in foregrounds:
        combi[pred_seg==c] = c
    combi[item[1]==255] = 0
            
    for x in range(imx):
        for y in range(imy):
            if combi[x,y] == -1:
                sort_list = np.argsort(-item[3][:,x,y])
                for c in range(num_classes):
                    if sort_list[c] in foregrounds:
                        combi[x,y] = sort_list[c]
                        break
    np.save(CONFIG.METRICS_DIR+item[5]+'_combi.npy', combi)

    combi_pos = ms.label(combi, background=0) 
    combi_in_bd_tmp = find_boundaries(combi_pos, connectivity=combi_pos.ndim, mode='inner')
    combi_in_bd = combi_pos.copy()
    combi_in_bd[combi_in_bd_tmp==1] *= -1
    np.save(CONFIG.METRICS_DIR+item[5]+'_combi_seg.npy', combi_pos)

    calculate_metrics_i(idx, foregrounds, combi_pos, combi_in_bd, combi, item[3], item[4], gt_01, item[5]+'_metrics.p')

    seg_tmp = np.zeros((imx, imy))
    for c in foregrounds:
        seg_tmp[pred_seg==c] = c
    seg_pos = ms.label(seg_tmp, background=0)
    seg_in_bd_tmp = find_boundaries(seg_pos, connectivity=seg_pos.ndim, mode='inner')
    seg_in_bd = seg_pos.copy()
    seg_in_bd[seg_in_bd_tmp==1] *= -1
    np.save(CONFIG.METRICS_DIR+item[5]+'_seg_seg.npy', seg_pos)

    calculate_metrics_i(idx, foregrounds, seg_pos, seg_in_bd, pred_seg, item[3], item[4], gt_01, item[5]+'_metrics_seg.p')
    print('image', item[5], 'processed in {}s\r'.format( round(time.time()-start,4) ) )


def classification_fit_and_predict( X_train, y_train, X_test, y_test=[] ):

    model = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, subsample=0.5, reg_alpha=0.5, reg_lambda=0.5) 
    model.fit( X_train, y_train )

    y_train_pred = model.predict_proba(X_train)
    fpr, tpr, _ = roc_curve(y_train.astype(int),y_train_pred[:,1])
    print("model train auroc score:", auc(fpr, tpr) )
    y_test_pred = model.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test.astype(int),y_test_pred[:,1])
    print("model test auroc score:", auc(fpr, tpr) )
    print(" ")
    return y_test_pred, y_train_pred, model


def classification_retrain( model, X_train, y_train, X_test, y_test=[] ):

    model = model.fit(X_train, y_train, xgb_model=model.get_booster())
        
    y_train_pred = model.predict_proba(X_train)
    fpr, tpr, _ = roc_curve(y_train.astype(int),y_train_pred[:,1])
    print("model train auroc score:", auc(fpr, tpr) )
    y_test_pred = model.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test.astype(int),y_test_pred[:,1])
    print("model test auroc score:", auc(fpr, tpr) )
    print(" ")
    return y_test_pred, y_train_pred, model


def comp_tp_fp_fn(pred,gt,c):
    tp = np.sum(np.logical_and(pred==c,gt==c))
    fp = np.sum(np.logical_and(pred==c,gt!=c))
    fn = np.sum(np.logical_and(pred!=c,gt==c))
    return tp, fp, fn


def compute_perform_i(item, y0a_pred_i_in, y0a_pred_seg_i_in, mc_th):

    imx = item[1].shape[0]
    imy = item[1].shape[1]
    foregrounds = [11,12,13,14,15,16,17,18]

    y0a_pred_i = [1 if y0a_pred_i_in[i,1]>mc_th else 0 for i in range(y0a_pred_i_in.shape[0])]
    y0a_pred_seg_i = [1 if y0a_pred_seg_i_in[i,1]>mc_th else 0 for i in range(y0a_pred_seg_i_in.shape[0])]

    metrics = { "seg": np.zeros((2,3)), "combi": np.zeros((2,3)), "seg_segm": np.zeros((1,3)), "combi_segm": np.zeros((1,3)) }
    
    if CONFIG.DATASET == 'lost_and_found':
        gt_01 = item[1].copy()
        gt_01[item[1]==254] = 1

    else:
        for m in ['seg_class','combi_class','seg_segm_class','combi_segm_class']:
            metrics[m] = np.zeros((len(foregrounds),3))
        
        gt_01 = np.zeros((imx, imy))
        for c in foregrounds:
            gt_01[item[1]==c] = 1
        gt_01[item[1]==255] = 255
    
    ## pixelwise
    # semantic segmentation prediction with meta classification
    seg = np.argmax(item[3], axis=0)
    seg[item[1]==255] = 255
    seg_seg = np.load(CONFIG.METRICS_DIR+item[5]+'_seg_seg.npy')
    seg_meta01 = np.zeros((imx, imy))
    seg_meta = np.zeros((imx, imy))
    for c in np.unique(seg_seg)[1:]:
        if y0a_pred_seg_i[c-1] == 0:
            seg_meta01[seg_seg==c] = 1
            seg_meta[seg_seg==c] = seg[seg_seg==c].max()
    seg_meta01[item[1]==255] = 255
    seg_meta[item[1]==255] = 255
    
    for c in range(2):
        metrics['seg'][c,:] = comp_tp_fp_fn(seg_meta01, gt_01, c)
    
    if CONFIG.DATASET != 'lost_and_found':
        for c, cx in zip(foregrounds, range(len(foregrounds))):
            metrics['seg_class'][cx,:] = comp_tp_fp_fn(seg_meta, item[1], c)
    
    # combined prediction with meta classification
    combi = np.load(CONFIG.METRICS_DIR+item[5]+'_combi.npy')
    combi[item[1]==255] = 255
    combi_seg = np.load(CONFIG.METRICS_DIR+item[5]+'_combi_seg.npy')
    combi_meta01 = np.zeros((imx, imy))
    combi_meta = np.zeros((imx, imy))
    for c in np.unique(combi_seg)[1:]:
        if y0a_pred_i[c-1] == 0:
            combi_meta01[combi_seg==c] = 1
            combi_meta[combi_seg==c] = combi[combi_seg==c].max()
    combi_meta01[item[1]==255] = 255
    combi_meta[item[1]==255] = 255

    for c in range(2):
        metrics['combi'][c,:] = comp_tp_fp_fn(combi_meta01, gt_01, c)
    
    if CONFIG.DATASET != 'lost_and_found':
        for c, cx in zip(foregrounds, range(len(foregrounds))):
            metrics['combi_class'][cx,:] = comp_tp_fp_fn(combi_meta, item[1], c)
    
    ## segmentwise
    # semantic segmentation prediction & combi with meta classification
    if CONFIG.DATASET == 'lost_and_found':
        gt_seg = ms.label(gt_01==1, background=0)
    else:
        gt_tmp = np.zeros((imx, imy))
        for c in foregrounds:
            gt_tmp[item[1]==c] = c
        gt_seg = ms.label(gt_tmp, background=0)

    seg_tmp = np.zeros((imx, imy))
    combi_tmp = np.zeros((imx, imy))
    for c in foregrounds:
        seg_tmp[seg_meta==c] = c
        combi_tmp[combi_meta==c] = c
    seg_segm = ms.label(seg_tmp, background=0)
    combi_segm = ms.label(combi_tmp, background=0)
    
    for s in np.unique(seg_segm)[1:]:
        if np.sum(np.logical_and(seg_segm==s,gt_seg>0)) == 0:
            metrics['seg_segm'][0,1] += 1 
    
    for s in np.unique(combi_segm)[1:]:
        if np.sum(np.logical_and(combi_segm==s,gt_seg>0)) == 0:
            metrics['combi_segm'][0,1] += 1 

    for s in np.unique(gt_seg)[1:]:
        if np.sum(np.logical_and(gt_seg==s,seg_segm>0)) == 0:
            metrics['seg_segm'][0,2] += 1 
        else:
            metrics['seg_segm'][0,0] += 1 

        if np.sum(np.logical_and(gt_seg==s,combi_segm>0)) == 0:
            metrics['combi_segm'][0,2] += 1
        else:
            metrics['combi_segm'][0,0] += 1 

    if CONFIG.DATASET != 'lost_and_found':
        
        for c, cx in zip(foregrounds, range(len(foregrounds))):

            gt_seg = ms.label(item[1]==c, background=0)
            seg_segm = ms.label(seg_meta==c, background=0)
            combi_segm = ms.label(combi_meta==c, background=0)

            for s in np.unique(seg_segm)[1:]:
                if np.sum(np.logical_and(seg_segm==s,gt_seg>0)) == 0:
                    metrics['seg_segm_class'][cx,1] += 1 
            
            for s in np.unique(combi_segm)[1:]:
                if np.sum(np.logical_and(combi_segm==s,gt_seg>0)) == 0:
                    metrics['combi_segm_class'][cx,1] += 1 

            for s in np.unique(gt_seg)[1:]:
                if np.sum(np.logical_and(gt_seg==s,seg_segm>0)) == 0:
                    metrics['seg_segm_class'][cx,2] += 1 
                else:
                    metrics['seg_segm_class'][cx,0] += 1 

                if np.sum(np.logical_and(gt_seg==s,combi_segm>0)) == 0:
                    metrics['combi_segm_class'][cx,2] += 1 
                else:
                    metrics['combi_segm_class'][cx,0] += 1 

    pickle.dump( metrics, open( os.path.join(CONFIG.EVAL_PRED_DIR, str(CONFIG.SAVED_MODEL), item[5]+'.p'), "wb" ) )


def print_perform_metrics(mc_th):

    foregrounds = [11,12,13,14,15,16,17,18]

    metrics =  pickle.load( open( os.path.join(CONFIG.EVAL_PRED_DIR, str(CONFIG.SAVED_MODEL), 'performance_all' + str(mc_th) + '.p'), "rb" ) )
        
    result_path = os.path.join( CONFIG.EVAL_PRED_DIR, str(CONFIG.SAVED_MODEL), 'results' + str(mc_th) + '.txt')
    with open(result_path, 'wt') as fi:
            
        print('----------pixelwise----------', file=fi)
        print(' ', file=fi)
        
        for m in ['seg','combi']:

            print('---', m, '---', file=fi)
            print(' ', file=fi)
            
            print('TP / FP / FN class background:', metrics[m][0,:],file=fi)
            print('TP / FP / FN class foreground:', metrics[m][1,:],file=fi)
            print(' ', file=fi)

            print('Accuracy:', np.sum(metrics[m][:,0])/(np.sum(metrics[m][:,0])+np.sum(metrics[m][:,2])),file=fi)
            print(' ', file=fi)
            
            print('IoU class background:', metrics[m][0,0]/(metrics[m][0,0]+metrics[m][0,1]+metrics[m][0,2]), file=fi)
            print('IoU class foreground:', metrics[m][1,0]/(metrics[m][1,0]+metrics[m][1,1]+metrics[m][1,2]), file=fi)
            print(' ', file=fi)
            
            print('precision class background:',  metrics[m][0,0]/(metrics[m][0,0]+metrics[m][0,1]), file=fi)
            print('recall class background:',  metrics[m][0,0]/(metrics[m][0,0]+metrics[m][0,2]), file=fi)
            print('precision class foreground:',  metrics[m][1,0]/(metrics[m][1,0]+metrics[m][1,1]), file=fi)
            print('recall class foreground:',  metrics[m][1,0]/(metrics[m][1,0]+metrics[m][1,2]), file=fi)
            print(' ', file=fi)
        
        if CONFIG.DATASET != 'lost_and_found':
            print('----------pixelwise per class----------', file=fi)
            print(' ', file=fi)
            
            for m in ['seg_class','combi_class']:
                
                print('---', m, '---', file=fi)
                print(' ', file=fi)
                
                for c, cx in zip(foregrounds, range(len(foregrounds))):
                    print('IoU class', c, ':', metrics[m][cx,0]/(metrics[m][cx,0]+metrics[m][cx,1]+metrics[m][cx,2]), file=fi)
                print(' ', file=fi)
                
                for c, cx in zip(foregrounds, range(len(foregrounds))):
                    if metrics[m][cx,0]+metrics[m][cx,1] > 0:
                        print('precision class', c, ':', metrics[m][cx,0]/(metrics[m][cx,0]+metrics[m][cx,1]), file=fi)
                    if metrics[m][cx,0]+metrics[m][cx,2] > 0:
                        print('recall class', c, ':', metrics[m][cx,0]/(metrics[m][cx,0]+metrics[m][cx,2]), file=fi)
                print(' ', file=fi)

        print('----------segmentwise----------', file=fi)
        print(' ', file=fi)
        
        for m in ['seg_segm','combi_segm']:

            print('---', m, '---', file=fi)
            print(' ', file=fi)
            
            print('TP / FP / FN class foreground:', metrics[m][0,:],file=fi)
            print(' ', file=fi)
            
            print('precision class foreground:',  metrics[m][0,0]/(metrics[m][0,0]+metrics[m][0,1]), file=fi)
            print('recall class foreground:',  metrics[m][0,0]/(metrics[m][0,0]+metrics[m][0,2]), file=fi)
            print(' ', file=fi)
        
        if CONFIG.DATASET != 'lost_and_found':
            print('----------segmentwise per class----------', file=fi)
            print(' ', file=fi)
            
            for m in ['seg_segm_class','combi_segm_class']:
                
                print('---', m, '---', file=fi)
                print(' ', file=fi)
                
                for c, cx in zip(foregrounds, range(len(foregrounds))):
                    if metrics[m][cx,0]+metrics[m][cx,1] > 0:
                        print('precision class', c, ':', metrics[m][cx,0]/(metrics[m][cx,0]+metrics[m][cx,1]), file=fi)
                    if metrics[m][cx,0]+metrics[m][cx,2] > 0:
                        print('recall class', c, ':', metrics[m][cx,0]/(metrics[m][cx,0]+metrics[m][cx,2]), file=fi)
                print(' ', file=fi)


def compute_f1_i( method_seg='seg_segm', method_combi='combi_segm', idx=0, name='' ):

    result_path = os.path.join( CONFIG.EVAL_PRED_DIR, str(CONFIG.SAVED_MODEL), 'results_f1.txt')
    with open(result_path, 'a') as fi:

        print(name, file=fi)
        print('mc threshold, averaged f_1 score, best f_1 score, recall at 80 precision', 'rec90', file=fi)

        # num th - tp, fp, fn
        data_seg = fill_data(os.path.join(CONFIG.EVAL_PRED_DIR, str(CONFIG.SAVED_MODEL)), method_seg, idx)
        data_seg[np.logical_and(data_seg[:,0]==0,data_seg[:,1]==0),0] = 1
        data_combi = fill_data(os.path.join(CONFIG.EVAL_PRED_DIR, str(CONFIG.SAVED_MODEL)), method_combi, idx)
        data_combi[np.logical_and(data_combi[:,0]==0,data_combi[:,1]==0),0] = 1

        prec80 = np.zeros((101)) + 0.8
        prec90 = np.zeros((101)) + 0.9

        prec_seg = data_seg[:,0] / (data_seg[:,0]+data_seg[:,1])
        prec_seg[prec_seg==0] = 1e-6
        rec_seg = data_seg[:,0] / (data_seg[:,0]+data_seg[:,2])
        f1_seg = 2 * (prec_seg*rec_seg) / (prec_seg+rec_seg)
        idx_th_seg80 = np.argmin(np.abs(prec_seg-prec80))
        idx_th_seg90 = np.argmin(np.abs(prec_seg-prec90))
        print('seg:  ', np.argmax(f1_seg), np.round(np.sum(f1_seg)/data_seg.shape[0]*100,2), np.round(np.max(f1_seg)*100,2), np.round(rec_seg[idx_th_seg80]*100,2), np.round(rec_seg[idx_th_seg90]*100,2), file=fi)
        
        prec = data_combi[:,0] / (data_combi[:,0]+data_combi[:,1])
        prec[prec==0] = 1e-6
        rec = data_combi[:,0] / (data_combi[:,0]+data_combi[:,2])
        f1 = 2 * (prec*rec) / (prec+rec)
        idx_th80 = np.argmin(np.abs(prec-prec80))
        idx_th90 = np.argmin(np.abs(prec-prec90))
        print('combi:', np.argmax(f1), np.round(np.sum(f1)/data_seg.shape[0]*100,2), np.round(np.max(f1)*100,2), np.round(rec[idx_th80]*100,2), np.round(rec[idx_th90]*100,2), file=fi)

        print('seg/combi f_1 without meta classification:', np.round(f1_seg[-1]*100,2), np.round(f1[-1]*100,2), file=fi)
        print(' ', file=fi)


def compute_f1( classwise=False ):

    print('compute best f1 score')

    compute_f1_i()

    if classwise and CONFIG.DATASET != 'lost_and_found':
        foregrounds = [11,12,13,14,15,16,17,18]
    
        for c, cx in zip(foregrounds, range(len(foregrounds))):
            compute_f1_i('seg_segm_class', 'combi_segm_class', cx, 'class'+str(c))               


def calculate_miou( item, mc_th, y0a_pred_i_in ):
    
    print('calculate mIoU values', item[5])
    num_classes = 19  
    data_seg = np.zeros((num_classes,3))
    data = np.zeros((num_classes,3))

    # semantic segmentation
    pred_seg = np.argmax(item[3], axis=0)
    pred_seg[item[1]==255] = 255
    for c in range(num_classes):
        data_seg[c,:] = comp_tp_fp_fn(pred_seg, item[1], c)
    
    # combination
    y0a_pred_i = [1 if y0a_pred_i_in[i,1]>mc_th else 0 for i in range(y0a_pred_i_in.shape[0])]

    combi = np.load(CONFIG.METRICS_DIR+item[5]+'_combi.npy')
    combi_seg = np.load(CONFIG.METRICS_DIR+item[5]+'_combi_seg.npy')
    for c in np.unique(combi_seg)[1:]:
        if y0a_pred_i[c-1] == 1:
            combi[combi_seg==c] = 0
    
    foregrounds = [11,12,13,14,15,16,17,18]
    seg = item[3].copy()
    for c in foregrounds:
        seg[c,:,:] = 0
    
    combi_all = np.argmax(seg, axis=0)
    combi_all[combi>0] = 0
    combi_all = combi_all + combi
    combi_all[item[1]==255] = 255
    
    for c in range(num_classes):
        data[c,:] = comp_tp_fp_fn(combi_all, item[1], c)
 
    return data, data_seg

