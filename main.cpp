/*
Name: Chirag Khurana
Entry No.: 2016CSB1037
---------------------How to Compile and Run----------------------
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
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace cv;
using namespace std;

float** mat_mul(float** mat1, float** mat2,int a,int b,int c){
	float** mat=new float*[a];
	for (int i = 0; i < a; ++i)
	{
		mat[i] = new float[c];
	}
	for (int i = 0; i < a; ++i)
	{
		for (int j = 0; j < c; ++j)
		{
			mat[i][j]=0;
			for (int k = 0; k < b; ++k)
			{
				mat[i][j] += mat1[i][k] * mat2[k][j];
			}
		}
	}
	return mat;
}

float area(int x1, int y1, int x2, int y2, int x3, int y3)
{
   return abs((x1*(y2-y3) + x2*(y3-y1)+ x3*(y1-y2))/2.0);
}
 
bool isInside(int x1, int y1, int x2, int y2, int x3, int y3, int x, int y)
{   
   float A = area (x1, y1, x2, y2, x3, y3);
   float A1 = area (x, y, x2, y2, x3, y3);
   float A2 = area (x1, y1, x, y, x3, y3);
   float A3 = area (x1, y1, x2, y2, x, y);
   //return (A < A1 + A2 + A3+0.01 && A > A1 + A2 + A3 - 0.01);
   return (abs(A - (A1 + A2 + A3))<0.01);
}

vector< vector<int> > triangles(vector<Point2f> points, Mat image){
	Rect rectangle(0,0,image.rows, image.cols);
	Subdiv2D subdiv(rectangle);
	for (vector<Point2f>::iterator it = points.begin(); it != points.end(); it++) subdiv.insert(*it);
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);

  vector<Point> pt(3);

	vector< vector<int> > index;
    for (int i = 0; i < triangleList.size(); ++i)
    {
      vector<int> a(3);
      int j;
      for (j = 0; j < 3; ++j)
      {
        Point2f p(triangleList[i][2*j],triangleList[i][2*j+1]);
        for (int k = 0; k < points.size(); ++k)
        {
          if(points[k]==p){
            a[j]=k;
            break;
          }
          else a[j]=-1;
        }
        if(a[j]==-1) break;
      }
      if(j==3) index.push_back(a);
    }
    return index;
}

Mat image_affine(Mat image,Mat affine, int bilinear){
	//float theta;
	//cout<<"Enter theta(anticlockwise) in degrees for rotation: ";
	//cin>>theta;
	//theta=(theta*3.1416/180); //in radians
	float** rotation=new float*[3];
	rotation[0]=new float[3]; rotation[1]=new float[3]; rotation[2]=new float[3];
	float** temp1= new float*[3];
	temp1[0]=new float[1]; temp1[1]=new float[1]; temp1[2]=new float[1];
	float** temp;

	rotation[0][0]=affine.at<float>(0,0); rotation[0][1]=affine.at<float>(0,1); rotation[0][2]=affine.at<float>(0,2);
	rotation[1][0]=affine.at<float>(1,0); rotation[1][1]=affine.at<float>(1,1); rotation[1][2]=affine.at<float>(1,2);
	rotation[2][0]=affine.at<float>(2,0); rotation[2][1]=affine.at<float>(2,1); rotation[2][2]=affine.at<float>(2,2);
	Mat affineinv = Mat(3,3,CV_32F);
	invert(affine,affineinv,DECOMP_SVD);

	int maxx,minx,maxy,miny;
	int cx=image.rows/2,cy=image.cols/2;
	temp1[0][0]=0-cx;temp1[1][0]=0-cy;temp1[2][0]=1;
	temp = mat_mul(rotation,temp1,3,3,1);
	maxx=ceil(temp[0][0]);minx=floor(temp[0][0]);maxy=ceil(temp[1][0]);miny=floor(temp[1][0]);

	temp1[0][0]=image.rows-cx;temp1[1][0]=0-cy;temp1[2][0]=1;
	temp = mat_mul(rotation,temp1,3,3,1);
	if(temp[0][0]>maxx) maxx=ceil(temp[0][0]);
	if(temp[0][0]<minx) minx=floor(temp[0][0]);
	if(temp[1][0]>maxy) maxy=ceil(temp[1][0]);
	if(temp[1][0]<miny) miny=floor(temp[1][0]);

	temp1[0][0]=image.rows-cx;temp1[1][0]=image.cols-cy;temp1[2][0]=1;
	temp = mat_mul(rotation,temp1,3,3,1);
	if(temp[0][0]>maxx) maxx=ceil(temp[0][0]);
	if(temp[0][0]<minx) minx=floor(temp[0][0]);
	if(temp[1][0]>maxy) maxy=ceil(temp[1][0]);
	if(temp[1][0]<miny) miny=floor(temp[1][0]);

	temp1[0][0]=0-cx;temp1[1][0]=image.cols-cy;temp1[2][0]=1;
	temp = mat_mul(rotation,temp1,3,3,1);
	if(temp[0][0]>maxx) maxx=ceil(temp[0][0]);
	if(temp[0][0]<minx) minx=floor(temp[0][0]);
	if(temp[1][0]>maxy) maxy=ceil(temp[1][0]);
	if(temp[1][0]<miny) miny=floor(temp[1][0]);

	int new_sizex = std::max(1.0,(double)affine.at<float>(0,0)+(double)affine.at<float>(0,1))*sqrt(image.rows*image.rows + image.cols*image.cols)+affine.at<float>(0,2);
	int new_sizey = std::max(1.0,(double)affine.at<float>(1,0)+(double)affine.at<float>(1,1))*sqrt(image.rows*image.rows + image.cols*image.cols)+affine.at<float>(1,2);	
	Mat image2 = Mat(new_sizex, new_sizey, CV_8UC3);
	for (int i = (new_sizex-image.rows)/2 ; i < (image.rows+new_sizex)/2; ++i)
	{
		for (int j = (new_sizey-image.cols)/2; j < (new_sizey+image.cols)/2; ++j)
		{
			image2.at<Vec3b>(i,j)=image.at<Vec3b>(i-(new_sizex-image.rows)/2,j-(new_sizey-image.cols)/2);
		}
	}
	image = image2;
	Mat image1 = Mat(image.rows,image.cols, CV_8UC3);

	rotation[0][0]=affineinv.at<float>(0,0); rotation[0][1]=affineinv.at<float>(0,1); rotation[0][2]=affineinv.at<float>(0,2);
	rotation[1][0]=affineinv.at<float>(1,0); rotation[1][1]=affineinv.at<float>(1,1); rotation[1][2]=affineinv.at<float>(1,2);
	rotation[2][0]=affineinv.at<float>(2,0); rotation[2][1]=affineinv.at<float>(2,1); rotation[2][2]=affineinv.at<float>(2,2);

	cx=image.rows/2,cy=image.cols/2;
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
		{
			temp1[0][0]=i-cx; temp1[1][0]=j-cy; temp1[2][0]=1;
			temp = mat_mul(rotation,temp1,3,3,1);
			if(bilinear==0){
				int x1=round(temp[0][0])+cx;
				int y1=round(temp[1][0])+cy;
				if(x1<image.rows && x1>=0 && y1<image.cols && y1>=0){
					image1.at<Vec3b>(i,j) = image.at<Vec3b>(x1,y1);
				}
			}
			else if(bilinear==1){
				int x1 = floor(temp[0][0])+cx, x2 = ceil(temp[0][0])+cx;
	    		int y1 = floor(temp[1][0])+cy, y2 = ceil(temp[1][0])+cy;
	    		if(x1<image.rows && x1>=0 && y1<image.cols && y1>=0 && x2<image.rows && x2>=0 && y2<image.cols && y2>=0){
		    		float alpha = temp[0][0]+cx-x1, beta = temp[1][0]+cy-y1;
		    		image1.at<Vec3b>(i,j) = alpha*beta*image.at<Vec3b>(x2,y2)+(1-alpha)*(1-beta)*image.at<Vec3b>(x1,y1)+alpha*(1-beta)*image.at<Vec3b>(x2,y1)+(1-alpha)*beta*image.at<Vec3b>(x1,y2);
	    		}
	    		else if (x1<image.rows && x1>=0 && y1<image.cols && y1>=0)
	    		{
	    			image1.at<Vec3b>(i,j) = image.at<Vec3b>(x1,y1);
	    		}
	    		else if(x2<image.rows && x2>=0 && y2<image.cols && y2>=0){
	    			image1.at<Vec3b>(i,j) = image.at<Vec3b>(x2,y2);
	    		}
			}
		}
	}
	delete rotation;delete temp; delete temp1;
	new_sizex = maxx-minx+affine.at<float>(0,2); new_sizey = maxy-miny+affine.at<float>(1,2);
	image2 = Mat(new_sizex,new_sizey, CV_8UC3);
	for (int i = (image1.rows-new_sizex)/2 ; i < (image1.rows+new_sizex)/2; ++i)
	{
		for (int j = (image1.cols - new_sizey )/2; j < (new_sizey+image1.cols)/2; ++j)
		{
			image2.at<Vec3b>(i-(image1.rows-new_sizex)/2,j-(image1.cols-new_sizey)/2) = image1.at<Vec3b>(i,j);
		}
	}
	return image2;	
}

void part1(Mat image, Mat affine,int n){
	int rows=image.rows,cols=image.cols;
	int b=1;
	cout<<"\nEnter 0 for N-neighbour or 1 for bilinear: ";
	cin>>b;
	if(!(b==1 || b==0)){
		cout<<"Taking bilinear as you entered wrong input"<<endl;
		b=1;
	}
	stringstream s;
	s<<"0.jpg";
	imwrite(s.str(),image);
	float x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6;
	x1=0;y1=0;x2=rows-1;y2=0;x3=rows-1;y3=cols-1;
	x4=x1*affine.at<float>(0,0)+y1*affine.at<float>(0,1)+affine.at<float>(0,2);
	y4=x1*affine.at<float>(1,0)+y1*affine.at<float>(1,1)+affine.at<float>(1,2);
	x5=x2*affine.at<float>(0,0)+y2*affine.at<float>(0,1)+affine.at<float>(0,2);
	y5=x2*affine.at<float>(1,0)+y2*affine.at<float>(1,1)+affine.at<float>(1,2);
	x6=x3*affine.at<float>(0,0)+y3*affine.at<float>(0,1)+affine.at<float>(0,2);
	y6=x3*affine.at<float>(1,0)+y3*affine.at<float>(1,1)+affine.at<float>(1,2);
	for (int i = 1; i < n; ++i)
	{
		s.str("");
	    s<<i<<".jpg";
		float X1,Y1,X2,Y2,X3,Y3;
		X1=(i*x4+(n-i)*x1)/n;X2=(i*x5+(n-i)*x2)/n;X3=(i*x6+(n-i)*x3)/n;
		Y1=(i*y4+(n-i)*y1)/n;Y2=(i*y5+(n-i)*y2)/n;Y3=(i*y6+(n-i)*y3)/n;
		Mat temp = Mat::zeros(6,6,CV_32F);
		temp.at<float>(0,0)=x1; temp.at<float>(0,1)=y1; temp.at<float>(0,2)=1;
		temp.at<float>(1,3)=x1; temp.at<float>(1,4)=y1; temp.at<float>(1,5)=1;
		temp.at<float>(2,0)=x2; temp.at<float>(2,1)=y2; temp.at<float>(2,2)=1;
		temp.at<float>(3,3)=x2; temp.at<float>(3,4)=y2; temp.at<float>(3,5)=1;
		temp.at<float>(4,0)=x3; temp.at<float>(4,1)=y3; temp.at<float>(4,2)=1;
		temp.at<float>(5,3)=x3; temp.at<float>(5,4)=y3; temp.at<float>(5,5)=1;
		Mat tempinv = Mat(6,6,CV_32F);
		invert(temp,tempinv,DECOMP_SVD);
		float** temp1 = new float*[6];
		temp1[0]=new float[6]; temp1[1]=new float[6]; temp1[2]=new float[6]; temp1[3]=new float[6];
		temp1[4]=new float[6]; temp1[5]=new float[6];
		for (int j = 0; j < 6; ++j)
		{
			for (int k = 0; k < 6; ++k)
			{
				temp1[j][k]=tempinv.at<float>(j,k);
			}
		}
		float** temp3=new float*[6];
		temp3[0]=new float[1]; temp3[1]=new float[1]; temp3[2]=new float[1]; temp3[3]=new float[1];
		temp3[4]=new float[1]; temp3[5]=new float[1];
		temp3[0][0]=X1; temp3[1][0]=Y1; temp3[2][0]=X2; temp3[3][0]=Y2;
		temp3[4][0]=X3; temp3[5][0]=Y3;
		float** temp4 = mat_mul(temp1,temp3,6,6,1);
		Mat tempaffine = Mat(3,3,CV_32F);
		tempaffine.at<float>(0,0)=temp4[0][0];tempaffine.at<float>(0,1)=temp4[1][0];tempaffine.at<float>(0,2)=temp4[2][0];
		tempaffine.at<float>(1,0)=temp4[3][0];tempaffine.at<float>(1,1)=temp4[4][0];tempaffine.at<float>(1,2)=temp4[5][0];
		tempaffine.at<float>(2,0)=0;tempaffine.at<float>(2,1)=0;tempaffine.at<float>(2,2)=1;
		delete temp4;
		delete temp3;
		delete temp1;
		imwrite(s.str(),image_affine(image,tempaffine,b));
	}
	s.str("");
	s<<n<<".jpg";
	imwrite(s.str(),image_affine(image,affine,b));
}

Mat morphing(Mat image1,Mat image2,float t,char** argv){
	vector<Point2f> point1;
	vector<Point2f> point2;
	vector<Point2f> pointm;
	ifstream fp1(argv[3]);
	ifstream fp2(argv[4]);
	float x,y;
	while(fp1>>x>>y){
		point1.push_back(Point2f(x,y));
	}
	while(fp2>>x>>y){
		point2.push_back(Point2f(x,y));
	}
	fp1.close(); fp2.close();
	if(point1.size()!=point2.size()){
		cout<<"No. of points must be equal in both .txt files\n"<<endl;
		exit (EXIT_FAILURE);
	}
	for(int i = 0; i < point1.size(); i++)
	{
	    x = (1.0 - t) * point1[i].x + t * point2[i].x;
	    y = (1.0 - t) * point1[i].y + t * point2[i].y;
	    pointm.push_back(Point2f(x,y));
	}
	int p1,p2,p3;
	vector<vector<int> > index = triangles(point1,image1);

	Mat imagem = Mat(round((1-t)*image1.rows+t*image2.rows),round((1-t)*image1.cols+t*image2.cols), CV_8UC3);
	Mat temp1 = Mat(round((1-t)*image1.rows+t*image2.rows),round((1-t)*image1.cols+t*image2.cols), CV_8UC3);
	Mat temp2 = Mat(round((1-t)*image1.rows+t*image2.rows),round((1-t)*image1.cols+t*image2.cols), CV_8UC3);
	
	for (int i = 0; i < index.size(); ++i)
	{
		p1=index[i][0]; p2=index[i][1]; p3=index[i][2];
		float maxx1,maxx2,maxxm,maxy1,maxy2,maxym;
		float minx1,minx2,minxm,miny1,miny2,minym;
		maxx1=max(point1[p1].x,max(point1[p2].x,point1[p3].x));
		minx1=min(point1[p1].x,min(point1[p2].x,point1[p3].x));
		maxx2=max(point2[p1].x,max(point2[p2].x,point2[p3].x));
		minx2=min(point2[p1].x,min(point2[p2].x,point2[p3].x));
		maxxm=max(pointm[p1].x,max(pointm[p2].x,pointm[p3].x));
		minxm=min(pointm[p1].x,min(pointm[p2].x,pointm[p3].x));

		maxy1=max(point1[p1].y,max(point1[p2].y,point1[p3].y));
		miny1=min(point1[p1].y,min(point1[p2].y,point1[p3].y));
		maxy2=max(point2[p1].y,max(point2[p2].y,point2[p3].y));
		miny2=min(point2[p1].y,min(point2[p2].y,point2[p3].y));
		maxym=max(pointm[p1].y,max(pointm[p2].y,pointm[p3].y));
		minym=min(pointm[p1].y,min(pointm[p2].y,pointm[p3].y));
		Point2f a1[3],am[3],a2[3];
		/*a1[0].x=minx1; a1[0].y=miny1; a1[1].x=maxx1; a1[1].y=miny1; a1[2].x=maxx1; a1[2].y=maxy1; a1[3].x=minx1; a1[3].y=maxy1;
		a2[0].x=minx2; a2[0].y=miny2; a2[1].x=maxx2; a2[1].y=miny2; a2[2].x=maxx2; a2[2].y=maxy2; a2[3].x=minx2; a2[3].y=maxy2;
		am[0].x=minxm; am[0].y=minym; am[1].x=maxxm; am[1].y=minym; am[2].x=maxxm; am[2].y=maxym; am[3].x=minxm; am[3].y=maxym;*/
		a1[0]=point1[p1]; a1[1]=point1[p2]; a1[2]=point1[p3];
		a2[0]=point2[p1]; a2[1]=point2[p2]; a2[2]=point2[p3];
		am[0]=pointm[p1]; am[1]=pointm[p2]; am[2]=pointm[p3];
		Mat affine1 = getAffineTransform(am,a1);
		Mat affine2 = getAffineTransform(am,a2);
		//cout<<affine1.at<double>(0,0)<<" "<<affine1.at<double>(0,1)<<" "<<affine1.at<double>(0,2)<<endl;
		for (int i = minxm; i <= maxxm; ++i)
		{
			for (int j = minym; j <= maxym; ++j)
			{
				if(isInside(pointm[p1].x,pointm[p1].y,pointm[p2].x,pointm[p2].y,pointm[p3].x,pointm[p3].y,i,j)){
					int X1=round(i*affine1.at<double>(0,0)+j*affine1.at<double>(0,1)+affine1.at<double>(0,2));
					int Y1=round(i*affine1.at<double>(1,0)+j*affine1.at<double>(1,1)+affine1.at<double>(1,2));
					int X2=round(i*affine2.at<double>(0,0)+j*affine2.at<double>(0,1)+affine2.at<double>(0,2));
					int Y2=round(i*affine2.at<double>(1,0)+j*affine2.at<double>(1,1)+affine2.at<double>(1,2));
					if(X1<image1.rows && X1>=0 && Y1<image1.cols && Y1>=0){
						temp1.at<Vec3b>(i,j) = image1.at<Vec3b>(X1,Y1);
					}
					if( X2<image2.rows && X2>=0 && Y2<image2.cols && Y2>=0){
						temp2.at<Vec3b>(i,j) = image2.at<Vec3b>(X2,Y2);
					}
				}
			}
		}
	}
	imagem = (1.0-t)*temp1 + t*temp2;
	return imagem;
}


int main( int argc, char** argv )
{
    int m;
    cout<<"\nSelect:\n"<<endl;
    cout<<"1. Part1(Affine Transformation)\n2. Image Morphing\n\nEnter: ";
    cin>>m;
    if(m==1){
    	if( argc != 2)
	    {
	     cout <<"\nPlease give path of image correctly in command line argument. Try again!" << endl;
	     return -1;
	    }
	    Mat image = imread(argv[1],1);
	    if(! image.data )                              // Check for invalid input
	    {
	        cout <<  "Could not open or find the image" << std::endl ;
	        return -1;
	    }
	    Mat affine = Mat(3,3,CV_32F);
	    cout<<"Give 3x3 affine matrix: ";
	    for (int i = 0; i < 3; ++i)
	    {
	    	for (int j = 0; j < 3; ++j)
	    	{
	    		cin>>affine.at<float>(i,j);
	    	}
	    }
	    affine.at<float>(2,0)=0;affine.at<float>(2,1)=0;affine.at<float>(2,2)=1;
	    int n;
	    cout<<"Enter no. of transitions: ";
	    cin>>n;
    	part1(image,affine,n);
    	cout<<"\nImages are stored inorder 0.jpg to "<<n<<".jpg\n"<<endl;
    }
    else if(m==2){
    	if( argc != 5)
	    {
	     cout <<"\nPlease give path of image1, image2, image1.txt and image2.txt respectively in command line argument. Try again!" << endl;
	     return -1;
	    }
	    Mat image = imread(argv[1],1);
	    Mat image1 = imread(argv[2],1);
	    if(! (image.data && image1.data))                              // Check for invalid input
	    {
	        cout <<  "\nCould not open or find the images. Try again!\n" << std::endl ;
	        return -1;
	    }
	    int n;
	    cout<<"Enter no. of transitions: ";
	    cin>>n;
	    stringstream s;
	    s<<"0.jpg";
	    imwrite(s.str(),image);
	    for (int i = 1; i < n; ++i)
	    {
	    	s.str("");
	    	s<<i<<".jpg";
	    	imwrite(s.str(),morphing(image,image1,(float)i/n,argv));
	    }
	    s.str("");
	    s<<n<<".jpg";
	    imwrite(s.str(),image1);
	    cout<<"\nImages are stored inorder 0.jpg to "<<n<<".jpg\n"<<endl;
    }
    else{
    	cout<<"Please Try again!"<<endl;
    }
    return 0;
}
