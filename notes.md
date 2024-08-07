What is a 4 point transform ? 

A 4 point transform is a perspective transform that uses 4 points to transform an image/document/quadrilateral if it was not captured head-on. Procedure- Identify four points in the source image, define corresponding points in the destination image, compute the perspective transform matrix, and apply the transform.


Why do we need to blur an image?

It helps in reducing the noise from the images. Noise can refer to random variations in color or brightness and noise can hinder the edge detection process by making the edges less clear. Blurring can help make the image look smooth without any rapid intensity changes. 

What is the hysteresis procedure ?

The hysteresis procedure is an important part of the Canny edge detection algorithm, which helps in accurately identifying edges in an image. 

1. After applying the gaussian blur, each pixel has a gradient magnitude and direction. 
2. Non-Maximum Suppression removes pixels that are not a part of an edge by comparing each pixel's gradient magnitude to its neighbors' magnitudes along the gradient direction.
3. Double Thresholding: This step applies 2 thresholds to the gradient magnitudes - T_Low and T_High. 
- Strong Edges - Pixels with gradient magnitudes above T_High.
- Weak Edges: Pixels with gradient magnitudes between T_low and T_high.
- Non-Edges: Pixels with gradient magnitudes below T_low. These are suppressed.

4. Edge Tracking by Hysteresis: This step ensures contuinity of edges by strating from strong edges and then continuing along the edge chain to include weak edges **ONLY** if they are connected to strong edges

What are contours?

Contours are like boundaries of an object in the image.
Finding contours in the edged image helps in identifying and isolating these boundaries.

For document scanning and object detection tasks, we only focus on extracting largest contours because these represent significant objects or areas of interest (e.g., the boundary of a document).