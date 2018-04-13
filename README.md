# TV_control

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


using namespace cv;
using namespace std;



void main()
{
	/*VideoCapture kamera("C:\\Users\\memoc\\Desktop\\2.mp4");*/
	VideoCapture kamera(0);
	
	Mat frame;
	Mat frame_gray;
	Mat right_frame_gray;
	Mat right_frame;
	Mat fist_frame, fist_frame_gray;
	Mat dst;
	Mat close;
	Mat	open;
	
	std::vector<Rect> faces,fists,hands,fist_in_roi;
	Rect roi;
	
	/*CascadeClassifier */
	CascadeClassifier face_cascade,fist_cascade,hand_cascade;
	face_cascade.load("haarcascade_frontalface_default.xml");
	hand_cascade.load("HandDetectorHaar.xml");
	fist_cascade.load("yumruk_1615(0.5_20).xml");

	int frame_delay = 1;

	// create a tracker object
	Ptr<TrackerKCF> tracker = TrackerKCF::createTracker();
	//close = imread("C:\\Users\\memoc\\Desktop\\close.jpg");
	//imshow("TV", close);
	/*kamera.read(dst);
	Rect2d r = selectROI("ROI", dst, true, false);*/
	
	namedWindow("yumruk", WINDOW_NORMAL); 
	namedWindow("face", WINDOW_NORMAL);

	while (true) {
		kamera.read(frame);
		
		

		//frame_delay++;
		/*string x = to_string(i);
		frame = imread("C:\\Users\\memoc\\Desktop\\yumruk_deneme\\y (" + x + ")" + ".jpg");*/
		
		
		/*Mat frame = dst(r);*/
		resize(frame, frame, Size(320,240));
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		// equalizeHist(frame_gray, frame_gray);
		
		 
		// detect face if there is no fist detection
		if( fists.size()==0 ){
			face_cascade.detectMultiScale(frame_gray, faces, 1.1, 10, 0, Size(20, 30), Size(350, 450));
			//face_cascade.detectMultiScale(frame_gray, faces, 1.1, 10, 0, Size(50, 100), Size(350, 450));
			while (faces.size() == 0) {
				kamera >> frame;
				resize(frame, frame, Size(320, 240));
				cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
				face_cascade.detectMultiScale(frame_gray, faces, 1.1, 10, 0, Size(30, 30), Size(350, 450));
				imshow("face", frame);
				waitKey(33);
				
			}
		}
			kamera >> frame;
			// detect first face that appear, ignore others
			if (faces.size() > 0) {
				rectangle(frame, Point(faces[0].x*2, faces[0].y*2), Point((faces[0].x + faces[0].width)*2, (faces[0].y + faces[0].height)*2), (0, 0, 255), 2);
				//determine ROI respect to face region , ROI is right of the face region
				roi = Rect(Point(0, 0), Point((faces[0].x + faces[0].width / 2)*2, frame.rows - 2));
			}
		
		
		//draw the ROI
		if(faces.size() != 0)
			rectangle(frame, Point((faces[0].x+faces[0].width/2)*2, 0), Point(0 ,frame.cols-4 ), Scalar(0, 0, 255), 4);

		
		right_frame= frame(roi);
		
		cvtColor(right_frame, right_frame_gray, COLOR_BGR2GRAY);
		equalizeHist(right_frame_gray, right_frame_gray);

		//hand_cascade.detectMultiScale(frame_gray2, hands, 1.1, 20, 0, Size(30, 30));

		/*
		for (size_t i = 0; i < hands.size(); i++)
		{

			Point center(hands[i].x + hands[i].width / 2, hands[i].y + hands[i].height / 2);
			ellipse(right_frame, center, Size(hands[i].width / 2, hands[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		} */

		fist_cascade.detectMultiScale(right_frame_gray, fists, 1.1, 5, 0, Size(30,30),Size(250,250));

		if (fists.size() != 0)
			rectangle(frame, Point(fists[0].x, fists[0].y), Point(fists[0].x + fists[0].width, fists[0].y + fists[0].height), (0, 0, 255), 4,1);

		if (fists.size() != 0) {
			Rect2d fist_roi = Rect2d(fists[0]);
			tracker->init(frame, fist_roi);
			int fist_delay = 0;
			while (true) {
				kamera >> frame;
				bool check = tracker->update(frame, fist_roi);
				
				if (!check)
					break;
				
				rectangle(frame, fist_roi, Scalar(255, 255, 255), 2, 1); 
				imshow("face", frame);
				waitKey(33);
				fist_delay++;
				
				if (fist_delay > 10 ) {
					bool is_inside = (Rect(fist_roi) & cv::Rect(0, 0, frame.cols, frame.rows)) == Rect(fist_roi);
					if (!is_inside)
						break;

					fist_frame = frame(fist_roi);
					cvtColor(fist_frame,fist_frame_gray , COLOR_BGR2GRAY);
					equalizeHist(fist_frame_gray, fist_frame_gray);
					
					fist_cascade.detectMultiScale(fist_frame_gray, fist_in_roi, 1.1, 5, 0, Size(30, 30));
					fist_delay = 0;
					if (fist_in_roi.size() == 0) {
						tracker.release();
						tracker = cv::TrackerKCF::createTracker();
						break;
					}
				}
				
			}
		}
		/*if (hands.size() == 1) {

			int j = 0;
				while (j < 3) {
					close = imread("C:\\Users\\memoc\\Desktop\\close.jpg");
					
					putText(close, "TV is Opening", Point(300, 400), 1, 2, Scalar(0, 255, 0), 1);
					putText(close, "               .", Point(300, 400), 1, 2, Scalar(0, 255, 0), 3);
					imshow("TV", close);
					waitKey(1000);

					putText(close, "               ..", Point(300, 400), 1, 2, Scalar(0, 255, 0), 3);
					imshow("TV", close);
					waitKey(1000);

					putText(close, "               ...", Point(300, 400), 1, 2, Scalar(0, 255, 0), 3);
					imshow("TV",close);
					waitKey(1000);
					j++;
				}
				close = imread("C:\\Users\\memoc\\Desktop\\open.jpg");
				imshow("TV", close);
		}
*/

		imshow("face", frame);
		imshow("yumruk", right_frame);
	
		if (waitKey(25) == int('k'))
			return;
		
	}
}





