#include <opencv2/highgui/highgui.hpp> // For VS2015
#include <opencv2/xfeatures2d.hpp>

#include <iostream>
#include <ctime>
#include <cstring>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace cv;

Mat img_sample, img_target;
Mat* img_puzzles;
const int K = 2;

void readImage();
void doSIFT();
void kNN(int k, Mat descriptors_1, Mat descriptors_2 , Mat knn_mat);
void findFeaturepoints(Mat descriptors_1, Mat descriptors_2, Mat knn_mat, Mat& good_pairs);
void drawKeypointsPairs(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, Mat good_pairs);

int main() {

	readImage();
	
	doSIFT();

	waitKey(0);
	return 0;
}

void doSIFT()
{
	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
	//Detect the keypoints
	vector<KeyPoint> keypoints_1, keypoints_2;
	f2d->detect(img_sample, keypoints_1);
	f2d->detect(img_target, keypoints_2);
	cout << "size of keypoints: " << keypoints_1.size() << endl;

	//Calculate descriptors (feature vectors)
	Mat descriptors_1, descriptors_2;
	f2d->compute(img_sample, keypoints_1, descriptors_1);
	f2d->compute(img_target, keypoints_2, descriptors_2);
	cout << "size of descriptors: " << descriptors_1.size() << endl;

	//Matching descriptor vector using BFMatcher
	/*BFMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);*/

	/*Mat img_matches;
	drawMatches(img_sample, keypoints_1, img_target, keypoints_2, matches, img_matches);
	imshow("match", img_matches);*/

	/*Mat feature1;
	drawKeypoints(img_sample, keypoints_1, feature1, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("result1", feature1);
	Mat feature2;
	drawKeypoints(img_target, keypoints_2, feature2, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("result2", feature2);*/

	Mat knn_mat(descriptors_1.rows, K, CV_32S, Scalar(0));
	kNN(K, descriptors_1, descriptors_2, knn_mat);

	Mat good_pairs;
	findFeaturepoints(descriptors_1, descriptors_2, knn_mat, good_pairs);
	//cout << good_pairs;
	drawKeypointsPairs(keypoints_1, keypoints_2, good_pairs);

}

void kNN(int k, Mat descriptors_1, Mat descriptors_2, Mat knn_mat)
{
	for (int i = 0; i < descriptors_1.rows; i++) {
		struct IndexandNorm {
			int index;
			double distance;
		};
		IndexandNorm* ian = new IndexandNorm[descriptors_2.rows];
		for (int j = 0; j < descriptors_2.rows; j++) {
			ian[j].index = j;
			ian[j].distance = norm(descriptors_1.row(i), descriptors_2.row(j), NORM_L2);
		}
		sort(ian, ian + descriptors_2.rows, [](IndexandNorm a, IndexandNorm b) {return a.distance < b.distance; });
		for (int p = 0; p < K; p++) {
			knn_mat.at<int>(i,p) = ian[p].index;
		}
	}
}

void findFeaturepoints(Mat descriptors_1, Mat descriptors_2, Mat knn_mat, Mat& good_pairs)
{
	int num_good = 0;
	vector<int> good_indices;
	for (int i = 0; i < knn_mat.rows; i++) {
		double dist1 = norm(descriptors_1.row(i), descriptors_2.row(knn_mat.at<int>(i, 0)), NORM_L2);
		double dist2 = norm(descriptors_1.row(i), descriptors_2.row(knn_mat.at<int>(i, 1)), NORM_L2);
		if (dist1 < 50.0 && dist2 > 200.0) {
			num_good++;
			good_indices.push_back(i);
			//cout << dist1 << " " << dist2 << endl;
			//cout << i << " " << knn_mat.at<int>(i,0) << " " << knn_mat.at<int>(i,1) << endl;
		}
	}
	good_pairs = Mat(num_good, 2, CV_32S, Scalar(0));
	for (int i = 0; i < num_good; i++) {
		good_pairs.at<int>(i, 0) = good_indices.at(i);
		good_pairs.at<int>(i, 1) = knn_mat.at<int>(good_indices.at(i), 0);
	}
	//cout << good_pairs;
}

void drawKeypointsPairs(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, Mat good_pairs)
{
	vector<KeyPoint> pair1, pair2;
	for (int i = 0; i < good_pairs.rows; i++) {
		pair1.push_back(keypoints_1.at(good_pairs.at<int>(i, 0)));
		pair2.push_back(keypoints_2.at(good_pairs.at<int>(i, 1)));
	}
	Mat pair_feature1;
	drawKeypoints(img_sample, pair1, pair_feature1, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("result1", pair_feature1);
	Mat pair_feature2;
	drawKeypoints(img_target, pair2, pair_feature2, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("result2", pair_feature2);
}

void readImage()
{
	int data;
	string name;

	while (1) {
		cout << "Which object do you want? (1:table, 2:logo, 3:other)" << endl;
		cin >> data;
		if (data == 1) {
			name = "table";
			break;
		}
		else if (data == 2) {
			name = "logo";
			break;
		}
		else if (data == 3) {
			cout << "Input your testing data name: ";
			cin >> name;
			break;
		}
		else {

		}
	}

	string path_name = "test/" + name + "/";
	img_sample = imread(path_name + "sample.bmp");
	img_target = imread(path_name + "target.bmp");

	int num_puzzles = 1;
	while (1) {
		fstream fs(path_name + "puzzle" + to_string(num_puzzles) + ".bmp");
		if (fs) {
			num_puzzles++;
		}
		else {
			num_puzzles--;
			break;
		}
	}
	cout << "number of pizzles: " << num_puzzles << endl;
	img_puzzles = new Mat[num_puzzles];
	for (int i = 0; i < num_puzzles; i++) {
		img_puzzles[i] = imread(path_name + "puzzle" + to_string(i + 1) + ".bmp");
		//imshow("puzzle" + to_string(i + 1), img_puzzles[i]);
	}
}