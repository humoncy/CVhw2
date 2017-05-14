#include <opencv2/highgui/highgui.hpp> // For VS2015
#include <opencv2/xfeatures2d.hpp>

#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

int main() {
	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
	Mat img_1 = imread("1.jpg");
	Mat img_2 = imread("2.jpg");
	//Detect the keypoints
	vector<KeyPoint> keypoints_1, keypoints_2;
	f2d->detect(img_1, keypoints_1);
	f2d->detect(img_2, keypoints_2);
	//Calculate descriptors (feature vectors)
	Mat descriptors_1, descriptors_2;
	f2d->compute(img_1, keypoints_1, descriptors_1);
	f2d->compute(img_2, keypoints_2, descriptors_2);
	//Matching descriptor vector using BFMatcher
	BFMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);
	
	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
	imshow("match", img_matches);
	waitKey(0);
}

