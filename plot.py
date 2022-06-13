#!/usr/bin/env python3
"""
script including functions for visualizations
"""

import os
import numpy as np 
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from global_defs import CONFIG
from helper      import fill_data
import labels as labels


trainId2label = { label.trainId : label for label in reversed(labels.cs_labels) }
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


def visualize_segments(comp, metric):

    R = np.asarray(metric)
    R = 1-0.5*R
    G = np.asarray(metric)
    B = 0.3+0.35*np.asarray(metric)

    R = np.concatenate((R, np.asarray([0, 1])))
    G = np.concatenate((G, np.asarray([0, 1])))
    B = np.concatenate((B, np.asarray([0, 1])))

    components = np.asarray(comp.copy(), dtype='int16')
    components[components < 0] = len(R)-1
    components[components == 0] = len(R)

    img = np.zeros(components.shape+(3,))

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img[x, y, 0] = R[components[x, y]-1]
            img[x, y, 1] = G[components[x, y]-1]
            img[x, y, 2] = B[components[x, y]-1]

    img = np.asarray(255*img).astype('uint8')

    return img


def vis_pred_i(item, y0a_pred_i):

    seg = np.argmax(item[3], axis=0)
    seg[item[1]==255] = 255
    depth_seg = np.argmax(item[4], axis=0)
    depth_seg[item[1]==255] = 255
    combi = np.load(CONFIG.METRICS_DIR+item[5]+'_combi.npy')
    combi[item[1]==255] = 0
    combi_seg = np.load(CONFIG.METRICS_DIR+item[5]+'_combi_seg.npy')
    
    combi_bd_tmp = find_boundaries(combi_seg, connectivity=combi_seg.ndim, mode='inner')
    img_pred = visualize_segments(combi_seg, y0a_pred_i[:, 0])
    img_pred[combi_bd_tmp==1,:] = (0,0,0)
    
    I1 = item[0].copy()
    I2 = item[0].copy()
    I3 = item[0].copy()
    I4 = item[0].copy()

    for x in range(item[0].shape[0]):
        for y in range(item[0].shape[1]):
            # if trainId2label[seg[x,y]].back_foreground == 1:
            if trainId2label[item[1][x,y]].back_foreground == 1 or trainId2label[seg[x,y]].back_foreground == 1:
                I1[x,y,:] = I1[x,y,:] * 0.2 + np.asarray(trainId2label[seg[x,y]].color) * 0.8
            # elif depth_seg[x,y] == 1:
            #     I1[x,y,:] = I1[x,y,:] * 0.2 + np.asarray((0,245,255)) * 0.8
            if combi[x,y] > 1:
                I2[x,y,:] = I2[x,y,:] * 0.2 + np.asarray(trainId2label[combi[x,y]].color) * 0.8
            if combi[x,y] > 1:
                I3[x,y,:] = I3[x,y,:] * 0.2 + img_pred[x,y,:] * 0.8
            if trainId2label[item[1][x,y]].back_foreground == 1:
                I4[x,y,:] = I4[x,y,:] * 0.2 + np.asarray(trainId2label[item[1][x,y]].color) * 0.8

    img12   = np.concatenate( (I1,I2), axis=1 )
    img34  = np.concatenate( (I3,I4), axis=1 )
    img   = np.concatenate( (img12,img34), axis=0 )
    image = Image.fromarray(img.astype('uint8'), 'RGB')
    image = image.resize((int(item[0].shape[1]),int(item[0].shape[0])))
    image.save(CONFIG.VIS_PRED_DIR + str(CONFIG.SAVED_MODEL) + '_' + item[5] + '.png')
    plt.close()
    print('stored:', item[5]+'.png')


def plot_fn_vs_fp_i( method_seg='seg_segm', method_combi='combi_segm', idx=0, name='' ):
    
    data_seg = fill_data(os.path.join(CONFIG.EVAL_PRED_DIR, str(CONFIG.SAVED_MODEL)), method_seg, idx)
    data_seg[np.logical_and(data_seg[:,0]==0,data_seg[:,1]==0),0] = 1
    
    data_combi = fill_data(os.path.join(CONFIG.EVAL_PRED_DIR, str(CONFIG.SAVED_MODEL)), method_combi, idx)
    data_combi[np.logical_and(data_combi[:,0]==0,data_combi[:,1]==0),0] = 1
    
    size_text = 17 

    f1 = plt.figure(1,frameon=False) 
    plt.clf()        
    plt.plot(data_seg[:,1]/1000, data_seg[:,2]/1000, color='lightseagreen', marker='o', linewidth=2, markersize=10, label='baseline', alpha=0.5)
    plt.plot(data_combi[:,1]/1000, data_combi[:,2]/1000, color='darkviolet', marker='o', linewidth=2, markersize=10, label='ours', alpha=0.5)  
    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['legend.handlelength'] = 0
    plt.xlabel('$\#$ false positives ($\\times 10^3$)', fontsize=size_text) #labelpad=-15)
    plt.ylabel('$\#$ false negatives ($\\times 10^3$)', fontsize=size_text)
    plt.xticks(fontsize=size_text)
    plt.yticks(fontsize=size_text)
    plt.legend(fontsize=size_text)
    f1.savefig(os.path.join(CONFIG.EVAL_PRED_DIR, str(CONFIG.SAVED_MODEL), 'img_fp_fn' + name + '.png'), dpi=400, bbox_inches='tight')
    plt.close()
    
    prec_seg = data_seg[:,0] / (data_seg[:,0]+data_seg[:,1])
    rec_seg = data_seg[:,0] / (data_seg[:,0]+data_seg[:,2])
    
    prec = data_combi[:,0] / (data_combi[:,0]+data_combi[:,1])
    rec = data_combi[:,0] / (data_combi[:,0]+data_combi[:,2])
    
    ap_seg = rec_seg[0] * prec_seg[0]
    ap = rec[0] * prec[0]
    for i in range(1,len(rec)):
        ap_seg += (rec_seg[i]-rec_seg[i-1])*prec_seg[i]
        ap += (rec[i]-rec[i-1])*prec[i]
    
    f1 = plt.figure(1,frameon=False) 
    plt.clf()
    plt.plot(rec_seg, prec_seg, color='cornflowerblue', marker='o', linewidth=2, markersize=10, label='baseline {:.02f}$\%$'.format(ap_seg*100), alpha=0.5)
    plt.plot(rec, prec, color='mediumvioletred', marker='o', linewidth=2, markersize=10, label='ours {:.02f}$\%$'.format(ap*100), alpha=0.5)  
    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['legend.handlelength'] = 0
    plt.xlabel('recall', fontsize=size_text) 
    plt.ylabel('precision', fontsize=size_text)
    plt.xticks(fontsize=size_text)
    plt.yticks(fontsize=size_text)
    plt.legend(fontsize=size_text)
    f1.savefig( os.path.join(CONFIG.EVAL_PRED_DIR, str(CONFIG.SAVED_MODEL),'img_rec_prec' + name + '.png'), dpi=400, bbox_inches='tight')
    plt.close()


def plot_fn_vs_fp( classwise=False ):
    
    print('start plotting')
    
    plot_fn_vs_fp_i()
    
    if classwise and CONFIG.DATASET != 'lost_and_found':
        foregrounds = [11,12,13,14,15,16,17,18]
    
        for c, cx in zip(foregrounds, range(len(foregrounds))):
            plot_fn_vs_fp_i('seg_segm_class', 'combi_segm_class', cx, '_class'+str(c))


def plot_miou( mious, mious_seg, th_values ):
    
    print('plot mean IoUs')
    num_classes = 19 

    th = th_values[0]

    result_path = os.path.join( CONFIG.EVAL_PRED_DIR, str(CONFIG.SAVED_MODEL), 'results_miou_' + str(th) + '.txt')
    with open(result_path, 'wt') as fi:

        mious_seg_th = np.sum(mious_seg, axis=1)

        iou_all_seg = 0
        counter_iou_all = 0
        for c in range(num_classes):
            if mious_seg_th[0,c,0] + mious_seg_th[0,c,2] > 0:
                iou_class = mious_seg_th[0,c,0]/(mious_seg_th[0,c,0]+mious_seg_th[0,c,1]+mious_seg_th[0,c,2])
                iou_all_seg += iou_class
                counter_iou_all += 1
                print('IoU class' + str(c) + ':', iou_class, file=fi)
        iou_all_seg *= (1/counter_iou_all)
        print('mean IoU:', iou_all_seg, file=fi)
        
        mious_th = np.sum(mious, axis=1)
        
        iou_all = np.zeros((mious.shape[0]))
        for i in range(mious.shape[0]):
            print(' ', file=fi)
            print('MC threshold', th_values[i], file=fi)
            for c in range(num_classes):
                if mious_th[i,c,0] + mious_th[i,c,2] > 0:
                    iou_class = mious_th[i,c,0]/(mious_th[i,c,0]+mious_th[i,c,1]+mious_th[i,c,2])
                    iou_all[i] += iou_class
                    print('IoU class' + str(c) + ':', iou_class, file=fi)
            print('mean IoU:', (1/counter_iou_all)*iou_all[i], file=fi)
        iou_all *= (1/counter_iou_all)
    
    size_text = 22
    f1 = plt.figure(1,frameon=False) 
    plt.clf()        
    plt.plot(th_values, np.ones((mious.shape[0]))*iou_all_seg, color='lightcoral', marker='o', linewidth=2, markersize=10, label='sem seg', alpha=0.7)  
    plt.plot(th_values, iou_all, color='slateblue', marker='o', linewidth=2, markersize=10, label='ours', alpha=0.7)
    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['legend.handlelength'] = 0
    plt.ylabel('$\mathit{mean}$ $\mathit{IoU}$', fontsize=size_text)
    plt.xticks([])
    plt.yticks(fontsize=size_text)
    plt.legend(fontsize=size_text)
    f1.savefig(os.path.join(CONFIG.EVAL_PRED_DIR, str(CONFIG.SAVED_MODEL), 'img_miou_' + str(th) + '.png'), dpi=400, bbox_inches='tight')
    plt.close()

   
