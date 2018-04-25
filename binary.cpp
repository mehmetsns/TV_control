
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/tracking.hpp>
#include <cstring>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include <iostream>
using namespace std;
using namespace cv;



void main() {
	VideoCapture kamera("C:\\Users\\memoc\\Desktop\\deneme3.mp4");  //("C:\\Users\\memoc\\Desktop\\deneme1.mp4");
	Mat output1, morph_operation_original_values, silik2;
	Mat frame;
	Mat	frame_blur;
	Mat frame_gray;
	Mat hand_roi;
	Mat hsv_image;
	Rect temp_roi;
	Mat gray, edge, draw;
	std::vector<Rect>  hands;
    CascadeClassifier hand_cascade;
	hand_cascade.load("palmdetector17st05err.xml");

	int low_cb = 100;
	int high_cb = 127;
	int low_cr = 135;
	int high_cr = 175;
    int count = 1;

	int select = 1; // if 1 work on Ycbcr, if 2 work on Gray_image

	while (true) {
		kamera.read(frame);

		
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		hand_cascade.detectMultiScale(frame_gray, hands, 1.1, 20, 0, Size(30, 30), Size(350, 450));
		
		if (hands.size() != 0) {
			rectangle(frame, Point(hands[0].x, hands[0].y), Point(hands[0].x + hands[0].width, hands[0].y + hands[0].height), Scalar(255, 255, 255), 2, 1);
		    temp_roi=hands[0];
		}

		if (temp_roi.height>0) {
			hand_roi = frame(temp_roi);

			Mat kernel = Mat(5, 5, CV_8UC1, 255);

			GaussianBlur(hand_roi, frame_blur, Size(5, 5), 0);

			//bilateralFilter(frame, frame_blur, 15,80,80);
			//medianBlur(frame, frame_blur, 5);
			Mat thresold_original_values(frame_blur.rows, frame_blur.cols, CV_8UC1, Scalar(0));
			Mat frame_threshold_hsv(frame_blur.rows, frame_blur.cols, CV_8UC1, Scalar(0));
			Mat threshold_avarage_values(frame_blur.rows, frame_blur.cols, CV_8UC1, Scalar(0));

			if (select == 1) {

				cvtColor(frame_blur, output1, CV_BGR2YCrCb);
				cvtColor(frame_blur, hsv_image, CV_BGR2HSV);
				if (count == 1) {
					Rect2d r = selectROI("ROI", output1, true, false);
					Mat dst = output1(r);

					imshow("renk_bul", dst);

					double tot_y = 0, tot_cr = 0, tot_cb = 0;
					for (int i = 0; i < dst.rows; i++)
						for (int j = 0; j < dst.cols; j++) {

							tot_cr += (int)dst.at<Vec3b>(i, j)[1];
							tot_cb += (int)dst.at<Vec3b>(i, j)[2];

							cout << dst.at<Vec3b>(i, j) << endl;
							//waitKey();
						}
					tot_cr = tot_cr / (dst.rows*dst.cols);
					tot_cb = tot_cb / (dst.rows*dst.cols);

					int avr_cr = tot_cr;
					int avr_cb = tot_cb;




					low_cr = avr_cr - 7;
					high_cr = avr_cr + 7;
					low_cb = avr_cb - 7;
					high_cb = avr_cb + 7;
					count++;

				}

				inRange(output1, Scalar(0, 135, 100), Scalar(255, 175, 127), thresold_original_values);
				inRange(output1, Scalar(0, low_cr, low_cb), Scalar(255, high_cr, high_cb), threshold_avarage_values);
				
				inRange(hsv_image, Scalar(0, 10, 60), Scalar(20, 150, 255), frame_threshold_hsv);

				//morphologyEx(frame_threshold, morph_operation, MORPH_OPEN, kernel, Point(-1, -1), 2);

				erode(thresold_original_values, morph_operation_original_values, Mat(3, 3, CV_8UC1, 255), Point(-1, -1), 3, 0);
				dilate(morph_operation_original_values, morph_operation_original_values, Mat(3, 3, CV_8UC1, 255), Point(-1, -1), 1, 0);

				int max = -1;
				int min = -1;
				//find height
				for (int i = 0; i < morph_operation_original_values.rows && min == -1; i++)
					for (int j = 0; j < morph_operation_original_values.cols; j++)
						if (morph_operation_original_values.at<uchar>(i, j) == 255)
							min = i;



				for (int i = morph_operation_original_values.rows - 1; max == -1 && i >= 0; i--)
					for (int j = 0; j < morph_operation_original_values.cols; j++)
						if (morph_operation_original_values.at<uchar>(i, j) == 255)
							max = i;


				int height = max - min;


				// find weight
				int width = 0;
				for (int j = 0; j < morph_operation_original_values.cols; j++)
					if (morph_operation_original_values.at<uchar>(morph_operation_original_values.rows / 2, j) == 255)
						width++;

				cout << height << "   " << width << "   " << double(height) / width << endl;

				line(morph_operation_original_values, Point(0, morph_operation_original_values.rows / 2), Point(morph_operation_original_values.cols - 1, morph_operation_original_values.rows / 2), Scalar(255));

				imshow("canli", frame);
			
				
				namedWindow("threshold", WINDOW_NORMAL);
				imshow("threshold_original", thresold_original_values);
				
				namedWindow("morph_operation", WINDOW_NORMAL);
				imshow("morph_operation", morph_operation_original_values);
				
				namedWindow("hsv_threshold", WINDOW_NORMAL);
				imshow("hsv_threshold", frame_threshold_hsv);

				namedWindow("threshold_avarage_values",WINDOW_NORMAL);
				imshow("threshold_avarage_values", threshold_avarage_values);

				
				
				//edge detection denemesi

				/*cvtColor(frame_blur, gray, CV_BGR2GRAY);

				Canny(gray, edge, 20, 60, 3);

				edge.convertTo(draw, CV_8U);
				namedWindow("image", CV_WINDOW_AUTOSIZE);
				imshow("image", draw);*/



			}

		}



		waitKey(33);


	}
}



