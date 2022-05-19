# Tutorial on Computer Vision with Fisheye Cameras
# Introduction
A fisheye camera is a camera whose field of view (FoV) is so large that the image it captures cannot be represented using the perspective projection. Instead of producing rectilinear images, fisheye cameras map straight lines in the 3D world to curved lines in the image according to some known mapping. Fisheye images appear deformed compared to standard perspective images, which we tend to regard as a natural view of the world. Nevertheless, the ability of a single camera to capture a wide FoV makes fisheye cameras extremely valuable in applications such as video surveillance, aerial photography, and autonomous vehicles.

Yet, the computer vision community has neglected to develop a methodical approach to applying the immense progress which has been made since the deep learning revolution to fisheye cameras, leading some authors and practitioners to solutions that are unnecessarily inferior, overcomplicated or outright silly.

This tutorial shows how to handle fisheye cameras much like perspective cameras, making many existing computer vision models that were developed for perspective images applicable to fisheye images directly or with minimal modifications.

We cannot truly understand fisheye cameras without digging into the way computer vision works with perspective images, and the first part of this tutorial will deal only with perspective images. The second part of the tutorial will apply similar principals to fisheye cameras.

We will focus on two computer vision tasks: 2D object detection and monocular 3D object detection (the task of detecting 3D bounding boxes around objects from a single camera, with no additional sensors).

# Perspective images and the pinhole camera model
## The perspective projection
An image is a two-dimensional array of pixels, with size \$H\times\ W\$, where each pixel is a three-dimensional array for a color image. The camera model defines the mapping between a point in 3D space,
\$\$ P=\\left[\\begin{matrix}X\\\\Y\\\\Z\\\\\\end{matrix}\\right],\$\$
and the 2D coordinate of the pixel to which it is mapped,
\$\$p=\\left[\\begin{matrix}u\\\\v\\\\\\end{matrix}\\right].\$\$

In the 3D coordinates we use, the \$x\$ axis is right, the \$y\$ axis is down, the \$z\$ axis is forward (also known as the optical axis), and the origin of axes is the camera’s focal point. In the image coordinates, \$u\$ is right and \$v\$ is down, the pixel indexes can be found by rounding to whole numbers, and the pixel \$\[0, 0\]\$ is the top left corner of the image.
The pinhole camera model is a simple model that is used very often. In this model, the 3D coordinates of a point are related to the homogeneous coordinates of a pixel,
\$\$p^\\prime=\\left[\\begin{matrix}u\\\\v\\\\1\\\\\\end{matrix}\\right],\$\$
by the following transformation:
\$\$p^\\prime Z=KP\$\$
where
\$\$K=\\left[\\begin{matrix}K_{00}&K_{01}&K_{02}\\\\K_{10}&K_{11}&K_{12}\\\\K_{20}&K_{21}&K_{22}\\\\\\end{matrix}\\right]=\\left[\\begin{matrix}f&0&u_0\\\\0&f&v_0\\\\0&0&1\\\\\\end{matrix}\\right].\$\$
is called the intrinsic matrix. The above equation can be written explicitly as
\$\$\\left[\\begin{matrix}u\\\\v\\\\1\\\\\\end{matrix}\\right]Z=\\left[\\begin{matrix}f&0&u_0\\\\0&f&v_0\\\\0&0&1\\\\\\end{matrix}\\right]\\left[\\begin{matrix}X\\\\Y\\\\Z\\\\\\end{matrix}\\right].\$\$

The intrinsic matrix is a special case of a projective transformation (homography) that applies only scaling by the focal length \$f\$ and translation by the principal point \$\\left[u_0, v_0\\right]\$. Some authors allow for different focal lengths along the \$x\$ and \$y\$ axes (i.e., \$K_{00}\neq\ K_{11})\$, but it is always possible to stretch the image to produce an image for which \$K_{00}=K_{11}\$. Some authors allow a nonzero \$K_{01}\$ element, which implies a skew, but it is always possible to de-skew the image to produce an image for which \$K_{01}=0\$. We call the transformation in the equation above the perspective projection and images that obey it perspective images, also known as rectilinear images or pinhole images.

![](img/pinhole.png)

The perspective projection is equivalent to finding the intersection of the plane \$Z=1\$ with the ray pointing from the origin of axes towards the point \$P\$, then scaling the intersection point by a factor of \$f\$ horizontally and vertically, and then shifting horizontally by \$u_0\$ and vertically by \$v_0\$.

If we know the coordinates of a 3D point \$\\left[X, Y, Z\\right]\$, or even only a vector pointing in its direction with an arbitrary magnitude, then we can find exactly which pixel that point will appear in using the perspective projection equation.

Similarly, if there is a pixel in the image that is of interest to us, then we can find a 3D vector pointing in its direction by multiplying both sides of the equation by the inverse of the intrinsic matrix:
\$\${\\left[\\begin{matrix}X\\\\Y\\\\Z\\\\\\end{matrix}\\right]=\\left[\\begin{matrix}f&0&u_0\\\\0&f&v_0\\\\0&0&1\\\\\\end{matrix}\\right]}^{-1}\\left[\\begin{matrix}u\\\\v\\\\1\\\\\\end{matrix}\\right]Z.\$\$

Straight horizontal lines in 3D space correspond to straight horizontal lines in the image, and straight vertical lines in 3D space correspond to straight vertical lines in the image. If a pinhole camera takes a picture of a checkerboard pattern perpendicular to the optical axis, it will appear as a perfect checkerboard pattern in the image.

## Pinhole camera distortion
Real cameras do not perfectly obey the pinhole camera model, but we can calibrate their deviation from the pinhole model and then undistort the image to produce an image which does obey the pinhole model. Rather than define the horizontal and vertical shift of each pixel as a lookup table, we often use some parametrized distortion model, typically decomposed into a radial polynomial and a tangential polynomial. See for example the OpenCV documentation [here](https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html) and [here](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html).

A checkerboard captured by a real camera might look like this:

![](img/checkerboard_distorted.png)

After undistorting the image, it will look like this:

![](img/checkerboard.png)

The undistorted image obeys the pinhole camera model, i.e., the mapping between 3D points and pixels is determined by the perspective projection.

## 2D Object Detection
In 2D object detection, a neural network receives an image as input, and it outputs 2D bounding boxes, often parametrized as \$\\left(u_{left},v_{top},u_{right},v_{bottom}\\right)\$ and the class (e.g., car, bicycle, pedestrian).

Convolutional neural networks (CNN) have a translation invariance property: if the image size is, say, \$1000\\times1000\$, and the convolution kernel size is \$3\times3\$, and the stride is 1, then the output of the convolution layer is features of the same spatial size \$1000\\times1000\$ where each feature has a receptive field of \$3\times3\$ with respect to the input. Each feature has no access to its spatial coordinate, i.e., it does not “know” where it is in the image. As the network becomes deeper and the feature channels become spatially smaller due to strided convolutions or max-pooling, the receptive field becomes larger, but maintains the translation invariance property. This translation invariance is a prior that provides an inductive bias and plays a significant role in the incredible success of deep learning for computer vision. A car on the left side of the image activates features the same way, using the same network weights, as a car on the right side of the image. The CNN only needs to learn to recognize something that looks like a car, no matter where it is in the image.

The translation-invariance property means that we cannot input an image into a CNN and use network output channels directly as the coordinates of the bounding boxes. Each channel is only a heatmap that activates when it recognizes patterns that look like boundaries of an object, but there is no information about the spatial coordinates in the values of the features themselves. Instead, we must search for a feature whose activation is above a certain threshold, or is a local maximum, and then read its spatial coordinates.

When an image is undistorted and obeys the pinhole camera model, it exhibits a translation invariance which is compatible with the translation invariance of CNNs. If an object facing the camera moves to the left, right, up, or down, it looks the same in an undistorted image, whereas in a distorted image it appears not only shifted but also deformed depending on where it is in the image.

Think of the distorted checkerboard pattern: in order to detect something that in the real world looks like a square, a neural network must learn to recognize many different shapes that deviate from a perfect square. As a more difficult task, a larger network capacity and larger training set are needed to achieve good predictions.

Moreover, if most of the objects in the training set happen to be in the left half of the image and most the objects in the test set happen to be in the right half of the image, we should expect some degradation in distorted images, where the shapes are different, but not in undistorted images, where objects look similar no matter where they are in the image.

For these reasons, it is always a good idea to undistort images before applying computer vision models to them.

## Monocular 3D object detection 
In monocular 3D object detection, we often parametrize the 3D bounding box as:
\$\$u, v, W, H, L, q, Z\$\$
