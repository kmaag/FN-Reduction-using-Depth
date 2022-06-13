#!/usr/bin/env python3
"""
main script executing tasks defined in global settings file
"""

from global_defs    import CONFIG
from main_functions import load_data, compute_metrics, compute_meta_classif, visualize_pred, compute_performance, visualize_eval 


def main():
  
    run = load_data()
    run.load_dataset()


    """
    From this line on, it is assumed that:
      
      - IMG_DIR defined in "global_defs.py" contains 3D input images
      - GT_DIR contains 2D arrays with semantic segmentation ground truth class 
      - DEPTH_DIR contains 2D depth heatmaps
      - PRED_SEM_SEG_DIR contains 3D semantic segmentation softmax predictions
      - PRED_DEPTH_SEG_DIR contains 2D foreground-background segmentation predictions
    """


    """
    Calculation of the combined prediction and metrics.
    """
    if CONFIG.COMPUTE_METRICS:
        run = compute_metrics()
        run.comp_metrics()


    """
    Perform meta classification.
    """
    if CONFIG.META_CLASSIF:
        run = compute_meta_classif()
        run.comp_meta_classif()
    

    """
    For visualizing the input data and predictions.
    """
    if CONFIG.VISUALIZE_PRED:
        run = visualize_pred()
        run.vis_pred()
    

    """ 
    Calculation of the performance measures.
    """
    if CONFIG.COMPUTE_EVAL:
        run = compute_performance() 
        run.compute_perform()
    
    
    
    """ 
    For visualizing the the performance measures.
    """
    if CONFIG.VISUALIZE_EVAL:
        run = visualize_eval() 
        run.vis_eval()
  


if __name__ == '__main__':
  
    print( "===== START =====" )
    main()
    print( "===== DONE! =====" )



