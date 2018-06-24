# Image Morphing

## To Compile & Run Code

```
0. OpenCV must be installed on your system.
1. To Compile $ g++ `pkg-config --cflags opencv` main.cpp `pkg-config --libs opencv`
2. Then output will be "a.out".
3. To Run For Part1 $ ./a.out image_path
	Then Enter 3x3 space separated affine matrix and then n (No. of transitions).
	Then output images will be stored in order 0.jpg to n.jpg .
4. To Run For Morphing $ ./a.out image1_path image2_path image1.txt image2.txt
	NOTE: TXT FILE MUST HAVE SPACE SEPARATED CORRESPONDING POINTS (x y) in each line.
		  E.g.: x1 y1 ( x,y Convention same as we take in class).
		  Otherwise either segmentation fault will come or wrong  will come.
	Then enter N(no. of transitions).
	Then output images will be stored in order 0.jpg(initial) to n.jpg(final) .
	There is no inbuilt command for gif in OpenCV.
	If you want to convert in gif if imagemagick installed on your system use:
	$ convert -loop 0 -delay 100 *.jpg out.gif

```

# Report-Image Morphing

## 1. Image_Morphing (Affine-Part-1)

In this part I used my previous lab affine transformation function which take cares of **exact size**. Firstly, I find tie points(corner) for each transition image using linear approximation between final image points. Then calculated its transformation matrix. And then used my affine function.

Taken Output:
Using [0 -1 0, 1 0 0, 0 0 1]


## Output Part-1

```
[Link](https://drive.google.com/open?id=1NwXP46lk95wWGTxsUcXCcDG3zq0D9mgI)
```


## 2. Image Morphing

For this I made morphing function which takes initial & final image, alpha(transition level/Total transitions) and tie points. Then I calculated corresponding points for intermediate image using linear averaging. Then I did triangulation and find triangle points using Subdiv2D::getTriangleList on initial image. Then find corresponding triangle points in both initial and final (using same relation given as input tie points). Then find transformation matrices for both between intermediate image-initial image and intermediate-final image. And finally cross dissolving.

```
Images and tie point source: learnopencv.com
```
## Output(Image Morphing)

```
[Link](https://drive.google.com/open?id=14MD4jdsp2MP0oFap_TSixVtpq9KP6jX)
```

```
Image and tie point source: learnopencv.com
```
## Output(for different image sizes)

```
[Link](https://drive.google.com/open?id=1NnWHX6mRUxrTa6rxCo5CgOtqI4mnFS4J)
```

```
Image and tie point source: learnopencv.com
```
## All Output Images

```
[Link](https://drive.google.com/open?id=1fPcdDQe2HteHFjEx1s1x0FNZZgcyFil5)
```
