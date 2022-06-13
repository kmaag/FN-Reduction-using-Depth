#!/usr/bin/env python3
"""
script including
class objects called in main
"""

import os
import time 
import pickle
import numpy as np
import concurrent.futures
from sklearn.metrics import roc_curve, auc

from global_defs  import CONFIG
from prepare_data import Cityscapes, Lost_and_found, A2d2, Idd 
from plot         import vis_pred_i, plot_fn_vs_fp, plot_miou
from calculate    import comp_metrics_i, classification_fit_and_predict, classification_retrain, compute_perform_i, print_perform_metrics, compute_f1, calculate_miou
from helper       import concat_metrics, metrics_to_dataset, save_perform_metrics_all, compute_best_th

#----------------------------#
class load_data(object):
#----------------------------#

    def __init__(self):
        """
        object initialization
        """

    def load_dataset(self):
        """
        load dataset
        """
        print('load dataset')

        global loader
        if CONFIG.DATASET == 'cityscapes':
            loader = Cityscapes( )
        elif CONFIG.DATASET == 'lost_and_found':
            loader = Lost_and_found( )
        elif CONFIG.DATASET == 'a2d2':
            loader = A2d2( )
        elif CONFIG.DATASET == 'idd':
            loader = Idd( )
        
        print('dataset:', CONFIG.DATASET)
        print('number of images: ', len(loader))
        print('semantic segmentation network:', CONFIG.SEG_MODEL_NAME)
        print('depth estimation network:', CONFIG.DEPTH_MODEL_NAME)


#----------------------------#
class compute_metrics(object):
#----------------------------#

    def __init__(self, num_cores=1, num_imgs=0):
        """
        object initialization
        :param num_cores: (int) number of cores used for parallelization
        :param num_imgs:  (int) number of images to be processed
        """
        self.num_cores  = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
        self.num_imgs = len(loader) if num_imgs == 0 else 0

    def comp_metrics(self):
        """
        compute metrics
        """
        print("compute combined prediction and metrics")

        if not os.path.exists( CONFIG.METRICS_DIR ):
            os.makedirs( CONFIG.METRICS_DIR )

        if self.num_cores == 1:
            for i in range(self.num_imgs):
                comp_metrics_i(loader[i], i)
        else:
            p_args = [ (loader[i], i) for i in range(self.num_imgs) ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_cores) as executor:
                executor.map(comp_metrics_i, *zip(*p_args))


#----------------------------#
class compute_meta_classif(object):
#----------------------------#

    def __init__(self, num_cores=1, num_imgs=0):
        """
        object initialization
        :param num_cores: (int) number of cores used for parallelization
        :param num_imgs:  (int) number of images to be processed
        """
        self.num_cores  = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
        self.num_imgs = len(loader) if num_imgs == 0 else 0

    def comp_meta_classif(self):
        """
        compute meta classification
        """
        print("compute meta classification")
        
        metrics, metrics_seg = concat_metrics(loader)
        self.train_meta_classif(metrics)
        self.train_meta_classif(metrics_seg, '_seg')
    
    def train_meta_classif(self, metrics, name=''):
        """
        train meta classifier
        """

        Xa, y0a, _ = metrics_to_dataset( metrics, non_empty=False )

        if CONFIG.SAVED_MODEL == 0:
            
            print('train meta model')
            runs = 5 # train/val splitting of 80/20

            y0a_pred = np.zeros((len(y0a),2))

            split = np.random.random_integers(0,runs-1,len(y0a))   
            for i in range(runs):
                print('run:', i)

                y0a_pred_i, _, _ = classification_fit_and_predict( Xa[split!=i,:], y0a[split!=i], Xa[split==i,:], y0a[split==i] )
                y0a_pred[split==i,:] = y0a_pred_i
            
            np.save(CONFIG.METRICS_DIR+'y0a_pred'+name+'_'+str(CONFIG.SAVED_MODEL)+'.npy', y0a_pred)
        
        elif CONFIG.SAVED_MODEL == 1:

            if CONFIG.DATASET == 'cityscapes':
            
                print('create cityscapes meta model')
                if not os.path.exists(CONFIG.META_MODEL_DIR):
                    os.makedirs(CONFIG.META_MODEL_DIR)

                y0a_pred, _, model = classification_fit_and_predict( Xa, y0a, Xa, y0a )
                pickle.dump(model, open(CONFIG.META_MODEL_DIR + 'cityscapes_classif'+name+'.p', 'wb'))
            
            else:
            
                print('use cityscapes meta model')
                
                model = pickle.load(open(CONFIG.META_MODEL_DIR + 'cityscapes_classif'+name+'.p', "rb"))
                y0a_pred = model.predict_proba(Xa)
                
                np.save(CONFIG.METRICS_DIR+'y0a_pred'+name+'_'+str(CONFIG.SAVED_MODEL)+'.npy', y0a_pred)
        
        else:
            
            if CONFIG.DATASET == 'cityscapes':
                print('error')
            
            else:
                print('use cityscapes meta model and retrain')
                
                model = pickle.load(open(CONFIG.META_MODEL_DIR + 'cityscapes_classif'+name+'.p', "rb"))
                
                y0a_pred = np.zeros((len(y0a),2))

                percent = CONFIG.SAVED_MODEL // 20
                split = np.random.random_integers(0,4,len(y0a))  
                for i in range(5):
                    print('run:', i)
                    split_i = split.copy()

                    for j in range(i,i+percent):
                        idx = (j+1) % 5
                        split_i[split==idx] = -1
                    
                    y0a_pred_i, _, _ = classification_retrain( model, Xa[split_i==-1,:], y0a[split_i==-1], Xa[split==i,:], y0a[split==i] )
                    y0a_pred[split==i,:] = y0a_pred_i
                
                np.save(CONFIG.METRICS_DIR+'y0a_pred'+name+'_'+str(CONFIG.SAVED_MODEL)+'.npy', y0a_pred)
        
        fpr, tpr, _ = roc_curve(y0a.astype(int),y0a_pred[:,1])
        print("model overall test auroc score:", auc(fpr, tpr) , 'for', CONFIG.DATASET, CONFIG.DEPTH_MODEL_NAME, CONFIG.SEG_MODEL_NAME, CONFIG.SAVED_MODEL, name)


#--------------------------------------#    
class visualize_pred(object):
#--------------------------------------#    
    
    def __init__(self, num_cores=1, num_imgs=0):
        """
        object initialization
        :param num_cores: (int) number of cores used for parallelization
        :param num_imgs:  (int) number of images to be processed
        """
        self.num_cores  = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
        self.num_imgs = len(loader) if num_imgs == 0 else 0
        
    def vis_pred(self):
        """
        visualize predictions
        """
        print("visualize input data and predictions")

        if not os.path.exists( CONFIG.VIS_PRED_DIR ):
            os.makedirs( CONFIG.VIS_PRED_DIR )
        
        metrics, _ = concat_metrics(loader)
        indizes = np.asarray(metrics['index'])
        if CONFIG.DATASET == 'cityscapes':
            y0a_pred = np.load(CONFIG.METRICS_DIR+'y0a_pred_0.npy')
        else:
            y0a_pred = np.load(CONFIG.METRICS_DIR+'y0a_pred_'+str(CONFIG.SAVED_MODEL)+'.npy')

        if self.num_cores == 1:
            for i in range(self.num_imgs):
                vis_pred_i(loader[i],y0a_pred[indizes==i,:])
                # exit()
        else:
            p_args = [ (loader[i],y0a_pred[indizes==i,:]) for i in range(self.num_imgs) ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_cores) as executor:
                executor.map(vis_pred_i, *zip(*p_args))



#----------------------------#
class compute_performance(object):
#----------------------------#

    def __init__(self, num_cores=1, num_imgs=0):
        """
        object initialization
        :param num_cores: (int) number of cores used for parallelization
        :param num_imgs:  (int) number of images to be processed
        # :param mc_th:     (float) meta classification threshold
        """
        self.num_cores  = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
        self.num_imgs = len(loader) if num_imgs == 0 else 0
        # self.mc_th = 50 if not hasattr(CONFIG, 'META_CLASSIF_TH') else CONFIG.META_CLASSIF_TH

    def compute_perform(self):
        """
        compute performance metrics
        """
        print("compute performance metrics")

        if not os.path.exists( CONFIG.EVAL_PRED_DIR ):
            os.makedirs( CONFIG.EVAL_PRED_DIR )
        
        if not os.path.exists( os.path.join(CONFIG.EVAL_PRED_DIR, str(CONFIG.SAVED_MODEL)) ):
            os.makedirs( os.path.join(CONFIG.EVAL_PRED_DIR, str(CONFIG.SAVED_MODEL)) )

        metrics, metrics_seg = concat_metrics(loader)
        indizes = np.asarray(metrics['index'])
        indizes_seg = np.asarray(metrics_seg['index'])
        y0a_pred = np.load(CONFIG.METRICS_DIR+'y0a_pred_'+str(CONFIG.SAVED_MODEL)+'.npy')
        y0a_pred_seg = np.load(CONFIG.METRICS_DIR+'y0a_pred_seg_'+str(CONFIG.SAVED_MODEL)+'.npy')

        for j in range(0,101,1):
            self.mc_th = j
            if not os.path.isfile( os.path.join(CONFIG.EVAL_PRED_DIR, str(CONFIG.SAVED_MODEL), 'performance_all' + str(self.mc_th) + '.p') ):

                start = time.time()
                print('creating metrics', float(self.mc_th)/100)
                    
                if self.num_cores == 1:
                    for i in range(self.num_imgs):
                        compute_perform_i(loader[i],y0a_pred[indizes==i,:],y0a_pred_seg[indizes_seg==i,:],float(self.mc_th)/100)
                else:
                    p_args = [ (loader[i],y0a_pred[indizes==i,:],y0a_pred_seg[indizes_seg==i,:],float(self.mc_th)/100) for i in range(self.num_imgs) ]
                    with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_cores) as executor:
                        executor.map(compute_perform_i, *zip(*p_args))
                    
                save_perform_metrics_all(os.path.join(CONFIG.EVAL_PRED_DIR,str(CONFIG.SAVED_MODEL)), 'performance_all' + str(self.mc_th) + '.p', loader)

                print("preparation processed in {}s\r".format( round(time.time()-start,4) ) )
                
            print_perform_metrics(self.mc_th)



#----------------------------#
class visualize_eval(object):
#----------------------------#

    def __init__(self, num_cores=1, num_imgs=0):
        """
        object initialization
        :param num_cores: (int) number of cores used for parallelization
        :param num_imgs:  (int) number of images to be processed
        # :param mc_th:     (float) meta classification threshold
        """
        self.num_cores  = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
        self.num_imgs = len(loader) if num_imgs == 0 else 0
        # self.mc_th = 50 if not hasattr(CONFIG, 'META_CLASSIF_TH') else CONFIG.META_CLASSIF_TH

    def vis_eval(self):
        """
        visualize performance metrics
        """
        print("visualize performance metrics")
        classwise = True
        
        compute_f1( classwise )
        plot_fn_vs_fp( classwise )

        if CONFIG.DATASET == 'lost_and_found':

            self.mc_th, _, _ = compute_best_th(CONFIG.EVAL_PRED_DIR)
        
        else:
            start = time.time()
            
            metrics, _ = concat_metrics(loader)
            indizes = np.asarray(metrics['index'])
            y0a_pred = np.load(CONFIG.METRICS_DIR+'y0a_pred_'+str(CONFIG.SAVED_MODEL)+'.npy')

            self.mc_th, _, _ = compute_best_th(CONFIG.EVAL_PRED_DIR)
                
            th_values = np.zeros((1))
            th_values[0] = self.mc_th
            mious = np.zeros((1,self.num_imgs, 19, 3))
            mious_seg = np.zeros((1,self.num_imgs, 19, 3))
            for i in range(self.num_imgs):
                 mious[0,i,:], mious_seg[0,i,:] = calculate_miou(loader[i], float(self.mc_th)/100, y0a_pred[indizes==i,:])
            
            plot_miou(mious, mious_seg, th_values)
            print("mIoU calculated in {}s\r".format( round(time.time()-start,4) ) )


