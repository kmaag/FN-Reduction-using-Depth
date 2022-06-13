## False Negative Reduction in Semantic Segmentation under Domain Shift using Depth Estimation

State-of-the-art deep neural networks demonstrate outstanding performance in semantic segmentation. However, their performance is tied to the domain represented by the training data. Open world scenarios cause inaccurate predictions which is hazardous in safety relevant applications like automated driving. In this work, we enhance semantic segmentation predictions using monocular depth estimation to improve segmentation by reducing the occurrence of non-detected objects in presence of domain shift. To this end, we infer a depth heatmap via a modified segmentation network which generates foreground-background masks, operating in parallel to a given semantic segmentation network. Both segmentation masks are aggregated with a focus on foreground classes (here road users) to reduce false negatives. To also reduce the occurrence of false positives, we apply a pruning based on uncertainty estimates. Our approach is modular in a sense that it post-processes the output of any semantic segmentation network. In our experiments, we observe less non-detected objects of most important classes and an enhanced generalization to other domains compared to the basic semantic segmentation prediction.

For further reading, please refer to TBA.

## Preparation:
We assume that the user is already using a neural network for semantic segmentation and a corresponding dataset. For each image from the semantic segmentation dataset, this code requires the following data:

- the RGB input image (height, width, 3) as png
- the semantic segmentation ground truth (height, width) as png
- the depth estimation heatmap (height, width) as png
- a three-dimensional numpy array (height, width, classes) that contains the softmax probabilities computed for the current image
- a two-dimensional numpy array (height, width) that contains the foreground(-background) probabilities computed for the current image

Before running this code, please edit all necessary paths stored in "global_defs.py". The code is CPU based and parts of of the code trivially parallize over the number of input images, adjust "NUM_CORES" in "global_defs.py" to make use of this. Also, in the same file, select the tasks to be executed by setting the corresponding boolean variable (True/False).

## Run Code:
```python
python main.py
```

## Author:
Kira Maag (Ruhr University Bochum)
