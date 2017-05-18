#include <opencv2/highgui/highgui.hpp> // For VS2015
#include <opencv2/opencv.hpp>
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

string name;
int num_puzzles = 1;
const int K = 2;

void readImage();
void getHomographyMatrix(Mat img_1, Mat img_2, Mat& homography_matrix, bool isTarget);
void forwardWarping(Mat img_1, Mat& img_2, Mat homography_matrix);
void backwardWarping(Mat img_1, Mat& img_2, Mat homography_matrix);
void backwardWarping(Mat img_1, Mat& img_2, Mat homography_matrix_Target, Mat homography_matrix);
void kNN(int k, Mat descriptors_1, Mat descriptors_2 , Mat knn_mat);
void findFeaturepoints(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, Mat descriptors_1, Mat descriptors_2, Mat knn_mat, Mat& homography_matrix);
void doRANSAC(Mat knn_mat, Mat good_pairs, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, Mat& homography_matrix, bool isTarget);
void drawKeypointsPairs(Mat img_1, Mat img_2, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, Mat descriptors_1, Mat descriptors_2, Mat good_pairs);
int calInliners(Mat H, Mat knn_mat, Mat good_pairs, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, bool isTarget);
void makeA(int RandIndex[], Mat& A, Mat good_pairs, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2);


Point2f origin[4];
Point2f target[4];

int main() {

	srand(time(NULL));

	readImage();
	
	Mat* homography_matrices;
	// 0 for target, others for puzzles
	homography_matrices = new Mat[num_puzzles + 1];
	
	Mat img_samplesize(img_sample.rows, img_sample.cols, CV_8UC3, Scalar(0));

	for (int i = 0; i <= num_puzzles; i++) {
		if (i == 0) {
			getHomographyMatrix(img_sample, img_target, homography_matrices[0], 1);
			cout << "H:\n" << homography_matrices[0] << endl << endl;
			//backwardWarping(img_sample, img_target, homography_matrices[0]);
		}
		else {
			// img_puzzle start from 0 to num_puzzles-1
			getHomographyMatrix(img_puzzles[i - 1], img_sample, homography_matrices[i], 0);
			cout << "H " << i << endl;
			cout << homography_matrices[i] << endl << endl;
			backwardWarping(img_puzzles[i - 1], img_samplesize, homography_matrices[i]);
			for (int j = 0; j < 4; j++) {
				cout << origin[j] << ' ' << target[j] << endl;
				cv::circle(img_puzzles[i - 1], origin[j], 10, Scalar(255 - 75 * j, 70 * j, 75 * j), 1);
				cv::circle(img_sample, target[j], 10, Scalar(255 - 75 *j, 70 * j, 75 * j), 1);
			}
			//imshow("Result" + to_string(i), img_samplesize);
			imshow("Origin", img_puzzles[i - 1]);
			imshow("Target", img_sample);

			//if (i == 2) break;
		}
		cout << "=========================================================" << endl << endl;
	}

	backwardWarping(img_samplesize, img_target, homography_matrices[0]);

	imshow("Test", img_samplesize);
	imshow("Result", img_target);
	imwrite(name + ".bmp", img_samplesize);
	//imshow("Result", img_sample);

	waitKey(0);
	return 0;
}

void getHomographyMatrix(Mat img_1, Mat img_2, Mat& homography_matrix, bool isTarget)
{
	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
	//Detect the keypoints
	vector<KeyPoint> keypoints_1, keypoints_2;
	f2d->detect(img_1, keypoints_1);
	//f2d->detect(img_puzzles[0], keypoints_1);
	f2d->detect(img_2, keypoints_2);
	cout << "size of keypoints_1: " << keypoints_1.size() << endl;
	cout << "size of keypoints_2: " << keypoints_2.size() << endl;

	//Calculate descriptors (feature vectors)
	Mat descriptors_1, descriptors_2;
	f2d->compute(img_1, keypoints_1, descriptors_1);
	f2d->compute(img_2, keypoints_2, descriptors_2);
	cout << "size of descriptors_1: " << descriptors_1.size() << endl;
	cout << "size of descriptors_2: " << descriptors_2.size() << endl;

	Mat knn_mat(descriptors_1.rows, K, CV_32S, Scalar(0));
	kNN(K, descriptors_1, descriptors_2, knn_mat);
	cout << "size of knn_mat: " << knn_mat.size() << endl;

	Mat good_pairs;
	findFeaturepoints(keypoints_1, keypoints_2, descriptors_1, descriptors_2, knn_mat, good_pairs);
	//cout << good_pairs;
	
	/*cout << "Key points:\n";
	for (int i = 0; i < keypoints_1.size(); i++) {
		cout << keypoints_1.at(i).pt << ' ' << keypoints_2.at(i).pt << endl;
	}*/

	cout << "Good Points:\n";
	for (int i = 0; i < good_pairs.rows; i++) {
		cout << keypoints_1.at(good_pairs.at<int>(i, 0)).pt << ' ' << keypoints_2.at(good_pairs.at<int>(i, 1)).pt << endl;
	}


	drawKeypointsPairs(img_1, img_2, keypoints_1, keypoints_2, descriptors_1, descriptors_2, good_pairs);
	
	doRANSAC(knn_mat, good_pairs, keypoints_1, keypoints_2, homography_matrix, isTarget);

}

bool isBlack(Vec3b color)
{
	if (color.val[0] == 0 && color.val[1] == 0 && color.val[2] == 0) {
		return true;
	}
	else return false;
}

void forwardWarping(Mat img_1, Mat & img_2, Mat homography_matrix)
{
	for (int rowIndex = 0; rowIndex < img_1.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < img_1.cols; colIndex++) {
			if ( !isBlack(img_1.at<Vec3b>(rowIndex, colIndex))) {
				Mat position = (Mat_<float>(3, 1) << rowIndex, colIndex, 1);
				Mat hx = homography_matrix * position;

				int x = (int)(hx.at<float>(0, 0) / hx.at<float>(2, 0));
				int y = (int)(hx.at<float>(1, 0) / hx.at<float>(2, 0));

				//cout << x << ' ' << y << endl;

				//tryH.at<Vec3b>(rowIndex, colIndex) = img_1.at<Vec3b>(x, y);
				img_2.at<Vec3b>(x, y) = img_1.at<Vec3b>(rowIndex, colIndex);
			}
		}
	}
}

void backwardWarping(Mat img_1, Mat & img_2, Mat homography_matrix)
{
	//Mat tryH = img_2.clone();
	//Mat tryH(img_2.rows, img_2.cols, CV_8UC3, Scalar(0, 0, 0));
	
	for (int rowIndex = 0; rowIndex < img_2.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < img_2.cols; colIndex++) {
			Mat position = (Mat_<float>(3, 1) << colIndex, rowIndex, 1);
			Mat hx = homography_matrix.inv() * position;

			int x = (int)(hx.at<float>(1, 0) / hx.at<float>(2, 0));
			int y = (int)(hx.at<float>(0, 0) / hx.at<float>(2, 0));

			if (x < 0)
				x = 0;
			if (x >= img_1.rows)
				x = img_1.rows - 1;
			if (y < 0)
				y = 0;
			if (y >= img_1.cols)
				y = img_1.cols - 1;

			//tryH.at<Vec3b>(rowIndex, colIndex) = img_1.at<Vec3b>(x, y);
			if (!isBlack(img_1.at<Vec3b>(x, y)))
				img_2.at<Vec3b>(rowIndex, colIndex) = img_1.at<Vec3b>(x, y);
		}
	}
	//imshow("try H", tryH);
}

void backwardWarping(Mat img_1, Mat & img_2, Mat homography_matrix_Target,Mat homography_matrix)
{
	for (int rowIndex = 0; rowIndex < img_2.rows; rowIndex++) {
		for (int colIndex = 0; colIndex < img_2.cols; colIndex++) {
			Mat position = (Mat_<float>(3, 1) << rowIndex, colIndex, 1);
			Mat hx = homography_matrix_Target.inv() * homography_matrix.inv() * position;

			int x = (int)(hx.at<float>(0, 0) / hx.at<float>(2, 0));
			int y = (int)(hx.at<float>(1, 0) / hx.at<float>(2, 0));

			//cout << x << ' ' << y << endl;
			if (x < 0)
				x = 0;
			if (x >= img_1.rows)
				x = img_1.rows - 1;
			if (y < 0)
				y = 0;
			if (y >= img_1.cols)
				y = img_1.cols - 1;

			if (!isBlack(img_1.at<Vec3b>(x, y)))
				img_2.at<Vec3b>(rowIndex, colIndex) = img_1.at<Vec3b>(x, y);
		}
	}
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

		delete[] ian;
	}
}

void findFeaturepoints(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, Mat descriptors_1, Mat descriptors_2, Mat knn_mat, Mat& good_pairs)
{
	vector<int> good_indices;
	int num_good = 0;
	double threashold = 5.0;
	while (num_good < 20) {
		for (int i = 0; i < knn_mat.rows; i++) {
			bool same_point = false;
			vector<int>::iterator it;
			for (it = good_indices.begin(); it != good_indices.end(); ++it) {
				if (keypoints_1.at(*it).pt == keypoints_1.at(i).pt) {
					same_point = true;
				}
			}
			if (!same_point) {
				double dist1 = norm(descriptors_1.row(i), descriptors_2.row(knn_mat.at<int>(i, 0)), NORM_L2);
				double dist2 = norm(descriptors_1.row(i), descriptors_2.row(knn_mat.at<int>(i, 1)), NORM_L2);
				if (dist2 / dist1 > threashold) {
					num_good++;
					good_indices.push_back(i);
					//cout << dist1 << " " << dist2 << endl;
					//cout << i << " " << knn_mat.at<int>(i,0) << " " << knn_mat.at<int>(i,1) << endl;
				}
			}
		}
		cout << "dist2 / dist1 > " << threashold << endl;
		threashold -= 0.25;
	}
	good_pairs = Mat(num_good, 2, CV_32S, Scalar(0));
	for (int i = 0; i < num_good; i++) {
		good_pairs.at<int>(i, 0) = good_indices.at(i);
		good_pairs.at<int>(i, 1) = knn_mat.at<int>(good_indices.at(i), 0);
	}

	/*good_pairs = Mat(knn_mat.rows, 2, CV_32S, Scalar(0));
	for (int i = 0; i < knn_mat.rows; i++) {
		good_pairs.at<int>(i, 0) = i;
		good_pairs.at<int>(i, 1) = knn_mat.at<int>(i, 0);
	}*/

	//cout << good_pairs;
}
// for fun
void doSAC(Mat knn_mat, Mat good_pairs, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, Mat& homography_matrix, bool isTarget)
{
	int N = good_pairs.rows;
	int K = 4;
	int index[4] = { 0 };
	int num_inliners = 0;
	Mat H(3, 3, CV_32F, Scalar(0));
	int indexRecorder[4] = { 0 };
	Mat debugA(8, 9, CV_32F, Scalar(0));

	std::string bitmask(K, 1); // K leading 1's
	bitmask.resize(N, 0); // N-K trailing 0's

						  // print integers and permute bitmask
	do {
		int cnt = 0;
		for (int i = 0; i < N; ++i) // [0..N-1] integers
		{
			if (bitmask[i]) {
				index[cnt] = i;
				//std::cout << " " << i;
				cnt++;
			}
			if (cnt == K) break;
		}
		/*for (int j = 0; j < 4; j++) {
			cout << index[j] << ' ';
		}*/

		Mat A(8, 9, CV_32F, Scalar(0));
		Mat tmp_H(3, 3, CV_32F, Scalar(0));

		makeA(index, A, good_pairs, keypoints_1, keypoints_2);

		Mat eigenvalues, eigenvectors;
		eigen(A.t()*A, eigenvalues, eigenvectors);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				tmp_H.at<float>(i, j) = eigenvectors.row(eigenvectors.rows - 1).at<float>(0, i * 3 + j);
			}
		}
		int s = calInliners(tmp_H, knn_mat, good_pairs, keypoints_1, keypoints_2, isTarget);
		if (num_inliners < s) {
			num_inliners = s;
			tmp_H.copyTo(H);
			for (int i = 0; i < 4; i++) {
				indexRecorder[i] = index[i];
			}
			A.copyTo(debugA);
		}
		//std::cout << std::endl;
	} while (std::prev_permutation(bitmask.begin(), bitmask.end()));

	for (int i = 0; i < 4; i++) {
		cout << indexRecorder[i] << ' ';
	}
	cout << endl;
	for (int i = 0; i < 4; i++) {
		origin[i] = keypoints_1.at(good_pairs.at<int>(indexRecorder[i], 0)).pt;
		target[i] = keypoints_2.at(good_pairs.at<int>(indexRecorder[i], 1)).pt;

		//Mat position = (Mat_<float>(3, 1) << origin[i].x, origin[i].y, 1);
		Mat position = (Mat_<float>(3, 1) << target[i].x, target[i].y, 1);
		//Mat hx = H * position;
		Mat hx = H.inv() * position;

		cout << origin[i] << ' ' << '[' << hx.at<float>(0, 0) / hx.at<float>(2, 0) << ',' << hx.at<float>(1, 0) / hx.at<float>(2, 0) << ']' << endl;
	}
	cout << "A: " << endl;
	cout << debugA << endl;

	cout << endl;
	cout << "knn rows: " << knn_mat.rows << endl;
	cout << "num of inliners: " << num_inliners << endl;
	H.copyTo(homography_matrix);
}

void makeA(int RandIndex[], Mat& A, Mat good_pairs, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2)
{

	/*for (int i = 0; i < 4; i++) {
		cout << RandIndex[i] << ' ';
	}
	cout << endl;*/

	int X1 = keypoints_1.at(good_pairs.at<int>(RandIndex[0], 0)).pt.x;
	int Y1 = keypoints_1.at(good_pairs.at<int>(RandIndex[0], 0)).pt.y;
	int x1 = keypoints_2.at(good_pairs.at<int>(RandIndex[0], 1)).pt.x;
	int y1 = keypoints_2.at(good_pairs.at<int>(RandIndex[0], 1)).pt.y;

	int X2 = keypoints_1.at(good_pairs.at<int>(RandIndex[1], 0)).pt.x;
	int Y2 = keypoints_1.at(good_pairs.at<int>(RandIndex[1], 0)).pt.y;
	int x2 = keypoints_2.at(good_pairs.at<int>(RandIndex[1], 1)).pt.x;
	int y2 = keypoints_2.at(good_pairs.at<int>(RandIndex[1], 1)).pt.y;

	int X3 = keypoints_1.at(good_pairs.at<int>(RandIndex[2], 0)).pt.x;
	int Y3 = keypoints_1.at(good_pairs.at<int>(RandIndex[2], 0)).pt.y;
	int x3 = keypoints_2.at(good_pairs.at<int>(RandIndex[2], 1)).pt.x;
	int y3 = keypoints_2.at(good_pairs.at<int>(RandIndex[2], 1)).pt.y;

	int X4 = keypoints_1.at(good_pairs.at<int>(RandIndex[3], 0)).pt.x;
	int Y4 = keypoints_1.at(good_pairs.at<int>(RandIndex[3], 0)).pt.y;
	int x4 = keypoints_2.at(good_pairs.at<int>(RandIndex[3], 1)).pt.x;
	int y4 = keypoints_2.at(good_pairs.at<int>(RandIndex[3], 1)).pt.y;

	//cout << X1 << ' ' << x1 << ' ' << -X1*x1 << endl;
	
	A.at<float>(0, 0) = X1;
	A.at<float>(0, 1) = Y1;
	A.at<float>(0, 2) = 1;
	A.at<float>(0, 6) = -x1*X1;
	A.at<float>(0, 7) = -x1*Y1;
	A.at<float>(0, 8) = -x1;

	A.at<float>(1, 3) = X1;
	A.at<float>(1, 4) = Y1;
	A.at<float>(1, 5) = 1;
	A.at<float>(1, 6) = -y1*X1;
	A.at<float>(1, 7) = -y1*Y1;
	A.at<float>(1, 8) = -y1;

	A.at<float>(2, 0) = X2;
	A.at<float>(2, 1) = Y2;
	A.at<float>(2, 2) = 1;
	A.at<float>(2, 6) = -x2*X2;
	A.at<float>(2, 7) = -x2*Y2;
	A.at<float>(2, 8) = -x2;

	A.at<float>(3, 3) = X2;
	A.at<float>(3, 4) = Y2;
	A.at<float>(3, 5) = 1;
	A.at<float>(3, 6) = -y2*X2;
	A.at<float>(3, 7) = -y2*Y2;
	A.at<float>(3, 8) = -y2;

	A.at<float>(4, 0) = X3;
	A.at<float>(4, 1) = Y3;
	A.at<float>(4, 2) = 1;
	A.at<float>(4, 6) = -x3*X3;
	A.at<float>(4, 7) = -x3*Y3;
	A.at<float>(4, 8) = -x3;

	A.at<float>(5, 3) = X3;
	A.at<float>(5, 4) = Y3;
	A.at<float>(5, 5) = 1;
	A.at<float>(5, 6) = -y3*X3;
	A.at<float>(5, 7) = -y3*Y3;
	A.at<float>(5, 8) = -y3;

	A.at<float>(6, 0) = X4;
	A.at<float>(6, 1) = Y4;
	A.at<float>(6, 2) = 1;
	A.at<float>(6, 6) = -x4*X4;
	A.at<float>(6, 7) = -x4*Y4;
	A.at<float>(6, 8) = -x4;

	A.at<float>(7, 3) = X4;
	A.at<float>(7, 4) = Y4;
	A.at<float>(7, 5) = 1;
	A.at<float>(7, 6) = -y4*X4;
	A.at<float>(7, 7) = -y4*Y4;
	A.at<float>(7, 8) = -y4;

	//cout << A;
}

int calInliners(Mat H, Mat knn_mat, Mat good_pairs, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, bool isTarget)
{
	int num_inliners = 0;

	if (isTarget) {
		for (int i = 0; i < good_pairs.rows; i++) {
			int X = keypoints_1.at(good_pairs.at<int>(i, 0)).pt.x;
			int Y = keypoints_1.at(good_pairs.at<int>(i, 0)).pt.y;
			int x = keypoints_2.at(good_pairs.at<int>(i, 1)).pt.x;
			int y = keypoints_2.at(good_pairs.at<int>(i, 1)).pt.y;

			Mat this_xy = (Mat_<float>(3, 1) << X, Y, 1);
			//cout << "this:\n" << this_xy << endl;
			Mat that_xy = H * this_xy;
			//cout << "that:\n" << that_xy << endl;

			float x_ = that_xy.at<float>(0, 0) / that_xy.at<float>(2, 0);
			float y_ = that_xy.at<float>(1, 0) / that_xy.at<float>(2, 0);

			float distance = (x - x_)*(x - x_) + (y - y_)*(y - y_);
			if (distance < 0.5) {
				num_inliners++;
			}
		}
	}
	else {
		for (int i = 0; i < knn_mat.rows; i++) {
			int X = keypoints_1.at(i).pt.x;
			int Y = keypoints_1.at(i).pt.y;
			int x1 = keypoints_2.at(knn_mat.at<int>(i, 0)).pt.x;
			int y1 = keypoints_2.at(knn_mat.at<int>(i, 0)).pt.y;
			int x2 = keypoints_2.at(knn_mat.at<int>(i, 1)).pt.x;
			int y2 = keypoints_2.at(knn_mat.at<int>(i, 1)).pt.y;
			
			Mat this_xy = (Mat_<float>(3, 1) << X, Y, 1);
			//cout << "this:\n" << this_xy << endl;
			Mat that_xy = H * this_xy;

			float x_ = that_xy.at<float>(0, 0) / that_xy.at<float>(2, 0);
			float y_ = that_xy.at<float>(1, 0) / that_xy.at<float>(2, 0);

			float distance1 = (x1 - x_)*(x1 - x_) + (y1 - y_)*(y1 - y_);
			float distance2 = (x2 - x_)*(x2 - x_) + (y2 - y_)*(y2 - y_);

			if (distance1 < 1) {
				num_inliners++;
			}
		}
	}
	

	return num_inliners;
}

void doRANSAC(Mat knn_mat, Mat good_pairs, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, Mat& homography_matrix, bool isTarget)
{
	doSAC(knn_mat, good_pairs, keypoints_1, keypoints_2, homography_matrix, isTarget);

	return;
	
	int num_inliners = 0;
	Mat H(3, 3, CV_32F, Scalar(0));
	int indexRecorder[4] = { 0 };

	int random_times = 2000;
	while (1) {
		Mat A(8, 9, CV_32F, Scalar(0));
		Mat tmp_H(3, 3, CV_32F, Scalar(0));

		//cout << "number of good pairs: " << good_pairs.rows << endl;
		int RandIndex[4] = { 0 };
		for (int i = 0; i < 4; i++) {
			RandIndex[i] = rand() % good_pairs.rows;
			for (int j = 0; j < 4; j++) {
				if ((i != j) && RandIndex[i] == RandIndex[j]) {
					RandIndex[i] = rand() % good_pairs.rows;
					j = 0;
				}
			}
		}

		makeA(RandIndex, A, good_pairs, keypoints_1, keypoints_2);
		/*cout << "A.t():\n" << A << endl;
		cout << "A:\n" << A.t() << endl;
		cout << "AtA\n" << A.t() * A << endl;*/
		Mat eigenvalues, eigenvectors;
		eigen(A.t()*A, eigenvalues, eigenvectors);
		//cout << eigenvalues << endl;
		//cout << eigenvectors << endl;
		
		//cout << eigenvectors.row(eigenvectors.rows-1) << endl;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				tmp_H.at<float>(i, j) = eigenvectors.row(eigenvectors.rows - 1).at<float>(0, i * 3 + j);
			}
		}
		int s = calInliners(tmp_H, knn_mat, good_pairs, keypoints_1, keypoints_2, isTarget);
		if (num_inliners < s) {
			num_inliners = s;
			tmp_H.copyTo(H);
			for (int i = 0; i < 4; i++) {
				indexRecorder[i] = RandIndex[i];
			}
			for (int i = 0; i < 4; i++) {
		origin[i] = keypoints_1.at(good_pairs.at<int>(RandIndex[i], 0)).pt;
		target[i] = keypoints_2.at(good_pairs.at<int>(RandIndex[i], 1)).pt;
	}
		}
		//cout << num_inliners << ' ' << knn_mat.rows * 0.5 << endl;

		random_times--;
		if (isTarget && random_times == 0) {
			tmp_H.copyTo(H);
			break;
		}
		if (!isTarget && (num_inliners > knn_mat.rows * 0.5 || random_times == -3000)) {
			break;
		}
	}

	for (int i = 0; i < 4; i++) {
		cout << indexRecorder[i] << ' ';
	}
	cout << endl;
	cout << "knn rows: " << knn_mat.rows << endl;
	cout << "num of inliners: " << num_inliners << endl;

	H.copyTo(homography_matrix);
}

void drawKeypointsPairs(Mat img_1, Mat img_2, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, Mat descriptors_1, Mat descriptors_2, Mat good_pairs)
{
	/*BFMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
	imshow("match", img_matches);*/

	cout << "num of good pairs: " << good_pairs.rows << endl;

	vector<KeyPoint> pair1, pair2;
	//Mat pair_desc1, pair_desc2;
	for (int i = 0; i < good_pairs.rows; i++) {
		pair1.push_back(keypoints_1.at(good_pairs.at<int>(i, 0)));
		pair2.push_back(keypoints_2.at(good_pairs.at<int>(i, 1)));
	}

	

	Mat pair_feature1, pair_feature2;
	Mat feature1, feature2;


	drawKeypoints(img_1, pair1, pair_feature1, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//imshow("result1", pair_feature1);


	drawKeypoints(img_1, keypoints_1, feature1, Scalar(200, 200, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//imshow("keypoints1", feature1);

	drawKeypoints(img_2, pair2, pair_feature2, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//imshow("result2", pair_feature2);
	drawKeypoints(img_2, keypoints_2, feature2, Scalar(0, 200, 200), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//imshow("keypoints2", feature2);

	/*BFMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	Mat img_matches;
	drawMatches(img_1, pair_feature1, img_2, pair_feature2, matches, img_matches);
	imshow("match", img_matches);*/
}

void readImage()
{
	int data;

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
	cout << "=========================================================" << endl << endl;

	img_puzzles = new Mat[num_puzzles];
	for (int i = 0; i < num_puzzles; i++) {
		img_puzzles[i] = imread(path_name + "puzzle" + to_string(i + 1) + ".bmp");
		//imshow("puzzle" + to_string(i + 1), img_puzzles[i]);
	}
	//imshow("puzzle0", img_puzzles[0]);
}