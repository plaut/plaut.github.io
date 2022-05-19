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
\$\$p^\prime=\left[\begin{matrix}u\\v\\1\\\end{matrix}\right],\$\$
by the following transformation:
\$\$p^\prime Z=KP\$\$
where
\$\$K=\left[\begin{matrix}K_{00}&K_{01}&K_{02}\\K_{10}&K_{11}&K_{12}\\K_{20}&K_{21}&K_{22}\\\end{matrix}\right]=\left[\begin{matrix}f&0&u_0\\0&f&v_0\\0&0&1\\\end{matrix}\right].\$\$
is called the intrinsic matrix. The above equation can be written explicitly as
\$\$\left[\begin{matrix}u\\v\\1\\\end{matrix}\right]Z=\left[\begin{matrix}f&0&u_0\\0&f&v_0\\0&0&1\\\end{matrix}\right]\left[\begin{matrix}X\\Y\\Z\\\end{matrix}\right].\$\$

The intrinsic matrix is a special case of a projective transformation (homography) that applies only scaling by the focal length \$f\$ and translation by the principal point \$\left[u_0, v_0\right]\$. Some authors allow for different focal lengths along the x and y axes (i.e., \$K_{00}\neq\ K_{11})\$, but it is always possible to stretch the image to produce an image for which \$K_{00}=K_{11}\$. Some authors allow a nonzero \$K_{01}\$ element, which implies a skew, but it is always possible to de-skew the image to produce an image for which \$K_{01}=0\$. We call the transformation in the equation above the perspective projection and images that obey it perspective images, also known as rectilinear images or pinhole images.

