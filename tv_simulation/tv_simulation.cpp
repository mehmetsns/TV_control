
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

void overlayImage(const cv::Mat &background, const cv::Mat &foreground,
	cv::Mat &output, cv::Point2i location)
{
	background.copyTo(output);


	// start at the row indicated by location, or at row 0 if location.y is negative.
	for (int y = std::max(location.y, 0); y < background.rows; ++y)
	{
		int fY = y - location.y; // because of the translation

								 // we are done of we have processed all rows of the foreground image.
		if (fY >= foreground.rows)
			break;

		// start at the column indicated by location, 

		// or at column 0 if location.x is negative.
		for (int x = std::max(location.x, 0); x < background.cols; ++x)
		{
			int fX = x - location.x; // because of the translation.

									 // we are done with this row if the column is outside of the foreground image.
			if (fX >= foreground.cols)
				break;

			// determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
			double opacity =
				((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3])

				/ 255.;


			// and now combine the background and foreground pixel, using the opacity, 

			// but only if opacity > 0.
			for (int c = 0; opacity > 0 && c < output.channels(); ++c)
			{
				unsigned char foregroundPx =
					foreground.data[fY * foreground.step + fX * foreground.channels() + c];
				unsigned char backgroundPx =
					background.data[y * background.step + x * background.channels() + c];
				output.data[y*output.step + output.channels()*x + c] =
					backgroundPx * (1. - opacity) + foregroundPx * opacity;
			}
		}
	}
}

#define INIT_PALM_X 600
#define INIT_PALM_Y 560
#define CHANNEL_CAPACITY 5
#define VOLUME_CAPACITY 100


void main()
{
	Mat camFrame, frame_gray;
	Mat videoFrame;
	
	Mat menu = imread("control.png", -1);
	Mat menu_ch_up = imread("channel_up.png", -1);
	resize(menu_ch_up, menu_ch_up, Size(0, 0), 0.3, 0.3, INTER_LANCZOS4);
	Mat menu_ch_down = imread("channel_down.png", -1);
	resize(menu_ch_down, menu_ch_down, Size(0, 0), 0.3, 0.3, INTER_LANCZOS4);
	Mat menu_vol_up = imread("volume_up.png", -1);
	resize(menu_vol_up, menu_vol_up, Size(0, 0), 0.3, 0.3, INTER_LANCZOS4);
	Mat menu_vol_down = imread("volume_down.png", -1);
	resize(menu_vol_down, menu_vol_down, Size(0, 0), 0.3, 0.3, INTER_LANCZOS4);

	Mat palm=imread("palm.png",-1);
	Mat fist = imread("fist.png", -1);
	
	resize(palm, palm, Size(0, 0), 0.5, 0.5, INTER_LANCZOS4);
	resize(fist, fist, Size(0, 0), 0.5, 0.5, INTER_LANCZOS4);
	resize(menu, menu, Size(0, 0), 0.3, 0.3, INTER_LANCZOS4);
	
	VideoCapture camera(0);

	int operation;

	CascadeClassifier palmCas, fistCas;

	palmCas.load("palmdetector17st05err.xml");
	fistCas.load("yumruk_1615(0.5_20).xml");
	std::vector<Rect> palms, fists;

	Rect volume_up(584, 488, 88, 78);
	Rect volume_down(576, 633, 87, 79);
	Rect channel_up(662, 555, 76, 89);
	Rect channel_down(513, 544, 73, 93);


	int PalmLocation_x = INIT_PALM_X;
	int PalmLocation_y = INIT_PALM_Y;

	int init_x;
	int init_y;

	bool menuFound = false;
	bool fistFound = false;
	int frameCount = 0;

	int volume = 10;
	int channel_num = 1;

	VideoCapture video("channel" + to_string(channel_num) + ".mp4");

	Mat roi;

	while (true) {
		camera.read(camFrame);
		video.read(videoFrame);
		resize(camFrame, camFrame, Size(0, 0), 0.8, 0.8, INTER_LANCZOS4);
		
		//overlayImage(frame, fist, frame, Point2i(100, 100));
		
		cvtColor(camFrame, frame_gray, COLOR_BGR2GRAY);
		//equalizeHist(frame_gray, frame_gray);

		palmCas.detectMultiScale(frame_gray, palms, 1.1, 15, 0, Size(10, 10), Size(150, 150));

		if (palms.size() != 0) {
			rectangle(camFrame, palms[0], Scalar(255, 255, 255), 2, 1);
			roi = frame_gray(palms[0]);
			if (!menuFound) {
				init_x = palms[0].x + palms[0].width / 2;
				init_y = palms[0].y + palms[0].height / 2;
			}
			else {
				int offset_x = (-(palms[0].x + palms[0].width / 2) + init_x)*2;
				int offset_y = ((palms[0].y + palms[0].height / 2) - init_y)*2;

				PalmLocation_x = offset_x + INIT_PALM_X;
				PalmLocation_y = offset_y + INIT_PALM_Y;

			}
			
			menuFound = true;
			frameCount = 0;

		}
		
		
		if (menuFound){
			
			fistCas.detectMultiScale(frame_gray, fists, 1.1, 15, 0, Size(10, 10), Size(150, 150));
			

			if (fists.size() != 0) {
				fistFound = true;
				frameCount = 0;
				rectangle(camFrame, fists[0], Scalar(255, 0, 0), 2, 1);

			}
			else
				fistFound = false;
			
		}


		if (menuFound) {
			
			int middle_x = PalmLocation_x + palm.cols;
			int middle_y = PalmLocation_y + palm.rows;

			if( middle_x > 584 && middle_x < (584 + 88) && middle_y > 488 && middle_y < (488+78))
				overlayImage(videoFrame, menu_vol_up, videoFrame, Point2i(500, 475)), operation = 1;
			else if (middle_x > 576 && middle_x < (576 + 87) && middle_y > 633 && middle_y < (633 + 79))
				overlayImage(videoFrame, menu_vol_down, videoFrame, Point2i(500, 475)), operation = 2;
			else if (middle_x > 662 && middle_x < (662 + 76) && middle_y > 555 && middle_y < (555 + 89))
				overlayImage(videoFrame, menu_ch_up, videoFrame, Point2i(500, 475)), operation = 3;
			else if (middle_x > 513 && middle_x < (513 + 73) && middle_y > 544 && middle_y < (544 + 93))
				overlayImage(videoFrame, menu_ch_down, videoFrame, Point2i(500, 475)), operation = 4;
			else
				overlayImage(videoFrame, menu, videoFrame, Point2i(500, 475)), operation = 0;
			
			if(!fistFound)
				overlayImage(videoFrame, palm, videoFrame, Point2i(PalmLocation_x, PalmLocation_y));
			else {
				overlayImage(videoFrame, fist, videoFrame, Point2i(PalmLocation_x, PalmLocation_y));
				switch (operation) {
				case 1:
					if (++volume > VOLUME_CAPACITY)
						volume = VOLUME_CAPACITY;
					putText(videoFrame, "VOL: " + to_string(volume), Point(30, 30), 1, 2, Scalar(255, 255, 0), 3);
					break;
				case 2:
					if (--volume < 0)
					volume = 0;
					putText(videoFrame, "VOL: " + to_string(volume), Point(30, 30), 1, 2, Scalar(255, 255, 0), 3);

					break;
				case 3:
					if (++channel_num > CHANNEL_CAPACITY) 
						channel_num = 1;
					video.open("channel" + to_string(channel_num) + ".mp4");
					imshow("channel", videoFrame);
					waitKey(600);
					break;
					
				case 4:
					if (--channel_num < 1)
						channel_num = CHANNEL_CAPACITY;
					video.open("channel" + to_string(channel_num) + ".mp4");
					imshow("channel", videoFrame);
					fistFound = false;
					waitKey(600);
					break;
					

				}
			}
			if (frameCount++ == 7) {
				frameCount = 0;
				menuFound = false;
				fistFound = false;
				PalmLocation_x = 600;
				PalmLocation_y = 560;
			}
		}
		
		imshow("camera", camFrame);
		imshow("channel", videoFrame);

		waitKey(30);
	}



}