// header

#pragma once

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

//log instellingen
#include <opencv2/core/utils/logger.hpp>

//timer
#include <chrono>


#include <iomanip>   // For setprecision
#include <sstream>   // For ostringstream

#include <map>

using namespace std;
using namespace std::chrono;
using namespace cv;


class detect {
private:
	Mat inImg;
	Mat outImg;
	Mat bwImg;
	//Mat binImg;
	//Mat invBinImg;
	Mat hsvImg;
	Mat edgeImg;
	Mat blurImg;

	//int thresVal;
	//double pixSize;
	double pixLength;
	int car1[4];
	int car2[4];
	VideoCapture cap;
	void perform(int& hueD1, int& satD1, int& valD1, int& stdevHueD1, int& hueD2, int& satD2, int& valD2, int& stdevHueD2);
	void setCapture();
	void cvtImage();
	string viewer;
	void showImage(string name, string text, bool live);
	map<string, Mat*> matList;

public:
	detect(string viewername);
	//pre: 
	//post:


	void setFocus();
	//pre: ensure lighting conditions are all set
	//post: after camera is focussed and aperture is set, press q to continue

	void setHSV();



	bool setAperture();



	bool setLength(string& output, int calibrationLength);
	//pre: ensure only the square calibration block is in the image, output is a pointer to a feedback string
	//	   calibrationLength is the length of the calibration block in mm
	//post: if calibration was successful, it will return true
	
	void showResult(int hueD1, int satD1, int valD1, int stdevHueD1, int hueD2, int satD2, int valD2, int stdevHueD2);
	//pre: 
	//post: a liveview will open with correct pieces marked green, incorrect red, unknown blue
	//	    time taken is displayed top left,

	~detect();
};

detect test();