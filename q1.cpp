#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
 
using namespace cv;
using namespace std;
 
 vector<Vec6f> draw_subdiv(Mat &img, Subdiv2D& subdiv, Scalar delaunay_color)
{

  vector<Vec6f> triangleList;
  subdiv.getTriangleList(triangleList);
  vector<Point> pt(3);

  for(size_t i = 0; i < triangleList.size(); ++i)
    {
      Vec6f t = triangleList[i];

      pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
      pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
      pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

      line(img, pt[0], pt[1], delaunay_color, 1);
      line(img, pt[1], pt[2], delaunay_color, 1);
      line(img, pt[2], pt[0], delaunay_color, 1);
    }
    return triangleList;
}
int main( int argc, char** argv)
{
  vector<Point2f> points;
  Mat image = imread(argv[1]);
                                                                                                                                
 Scalar delaunay_color(255, 255, 255), point_color(0,0,255);
 Rect rect(0,0,image.cols, image.rows);

  Subdiv2D subdiv(rect);

  ifstream ifs(argv[1]);
    int x, y;
    while(ifs >> x >> y)
    {
        points.push_back(Point2f(x,y));
    }
    
         for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
    {
        subdiv.insert(*it);
     
    }
 vector<Vec6f> triangleList;
 
triangleList= draw_subdiv(image, subdiv, delaunay_color);
  vector<Point> pt(3);
  for(size_t i = 0; i < triangleList.size(); ++i)
    {
      Vec6f t = triangleList[i];

      pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
      pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
      pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
      cout<<pt[0];
       cout<<pt[1];
        cout<<pt[2];
        cout<<endl;
      
      }
 imwrite("delaunay2.jpg", image);
 waitKey(0);
    /*subdiv.getTriangleList(triangleList);
    vector< vector<int> > index;
    for (int i = 0; i < triangleList.size(); ++i)
    {
      vector<int> a(3);
      int j;
      for (j = 0; j < 3; ++j)
      {
        Point2f p(cvRound(triangleList[i][2*j]),cvRound(triangleList[i][2*j+1]));
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

    for (int i = 0; i < index.size(); ++i)
    {
      cout<<index[i][0]<<" "<<index[i][1]<<" "<<index[i][2]<<endl;  
    }*/
    return 0;
}
