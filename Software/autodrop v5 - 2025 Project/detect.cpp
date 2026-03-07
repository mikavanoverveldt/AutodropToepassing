//header

#include "detect.h"

detect::detect(string viewername) {
	//thresVal = 190;
	//pixSize = 0.0144;
	pixLength = 1;
	viewer = viewername;
	setCapture();
	cvtImage();
	matList["inImg"] = &inImg;
	matList["outImg"] = &outImg;
	matList["bwImg"] = &bwImg;
	matList["hsvImg"] = &hsvImg;
	matList["edgeImg"] = &edgeImg;
	matList["blurImg"] = &blurImg;
	//matList["invBinImg"] = &invBinImg;
}

void detect::showImage(string name, string text, bool live) {
	do {
		cvtImage();
		putText(*matList[name], text, { 10, 25 }, FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
		imshow(viewer, *matList[name]);
		if (waitKey(30) == 'q') {
			destroyAllWindows();
			cvtImage();
			return;
		}
	} while (live);
	waitKey(0);
	cvtImage();
	destroyAllWindows();
	return;
}

void detect::setCapture() {
	//0 is webcam, 1 is external
	int indexNumberCam = 1;
	while (!cap.isOpened())
	{
		cap.open(indexNumberCam, CAP_DSHOW);
		//set resolution and auto white balance
		cap.set(CAP_PROP_FRAME_WIDTH, 1292);
		cap.set(CAP_PROP_FRAME_HEIGHT, 964);
		cap.set(CAP_PROP_AUTO_WB, 1);

		waitKey(100);
	}
	cap.read(inImg);
}

void detect::setFocus() {
	showImage("inImg", "Set focus, then press <q>", true);
}

bool detect::setAperture() {
	//declare used variables
	string message;
	Scalar meanHSV;

	//loop to show a live image with textual feedback
	while (true) {
		cvtImage();
		//calculate 
		meanHSV = mean(hsvImg);
		cout << meanHSV[2] << endl;
		if (meanHSV[2] > 254.900) message = "Aperture too wide";
		else if (meanHSV[2] < 254.700) message = "Aperture too narrow";
		else message = "Aperture correct, press <q>";
		putText(outImg, message, { 10, 25 }, FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
		imshow(viewer, outImg);
		if (waitKey(60) == 'q') {
			destroyAllWindows();
			if (meanHSV[2] > 254.700 && meanHSV[2] < 254.900) return true;
			else return false;
		}
	}
}

void detect::setHSV() {
	showImage("outImg", "Place light cadillac, then press <q>", true);

	vector<vector<Point>> contours;
	findContours(edgeImg, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	//create mask of a contour
	Mat mask = Mat::zeros(inImg.size(), CV_8UC1);
	drawContours(mask, contours, 0, Scalar(255), FILLED);

	//calculate mean and stdev hsv values by combining the mask of a contour with the hsv image
	Scalar meanHSV, stdevHSV;
	meanStdDev(hsvImg, meanHSV, stdevHSV, mask);
	for (int i = 0;i <= 2;i++) {
		car1[i] = meanHSV[i];
	}
	car1[3] = stdevHSV[2];

	showImage("outImg", "Place dark cadillac, then press <q>", true);

	findContours(edgeImg, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	//create mask of a contour
	mask = Mat::zeros(inImg.size(), CV_8UC1);
	drawContours(mask, contours, 0, Scalar(255), FILLED);

	//calculate mean and stdev hsv values by combining the mask of a contour with the hsv image
	meanStdDev(hsvImg, meanHSV, stdevHSV, mask);
	for (int i = 0;i <= 2;i++) {
		car2[i] = meanHSV[i];
	}
	car2[3] = stdevHSV[2];
}

void detect::cvtImage() {
	cap.read(inImg);
	outImg = inImg.clone();
	cvtColor(inImg, bwImg, COLOR_BGR2GRAY);
	cvtColor(inImg, hsvImg, COLOR_BGR2HSV);

	//convert image to just the blue channel
	Mat	tempImg = Mat::zeros(inImg.size(), CV_8UC1);
	vector<Mat> channels(3);
	split(inImg, channels);
	tempImg = channels[0];
	GaussianBlur(tempImg, blurImg, Size(13, 13), 2);

	Canny(blurImg, edgeImg, 30, 150);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(edgeImg, edgeImg, kernel);


}

bool detect::setLength(string& output, int calibrationLength) {
	showImage("edgeImg", "Place calibration block in frame, then press <q>", true);

	vector<vector<Point>> contours;
	findContours(edgeImg, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	//amount of contours found
	int contQty = contours.size();
	cout << contQty;

	if (contQty != 1) {
		output = "Ensure only the calibration block is in frame and try again.\n";
		return false;
	}
	//find size of smallest surrounding rectangle
	RotatedRect minRect = minAreaRect(contours[0]);
	double width = minRect.size.width;
	double height = minRect.size.height;

	//check if found contour is remotely square
	double ratio = width / height;
	cout << "width: " << width << endl;
	cout << "Height: " << height << endl;
	cout << "ratio: " << ratio << endl;
	if (ratio > 1.1 || ratio < 0.9) {
		output = "Invalid contour detected, place calibration block and try again.\n";
		return false;
	}

	//check whether found contour isn't too big, to ensure the 100x100mm inspection area requirement is met
	if (width > 964 / 3 || height > 964 / 3) {
		output = "Increase camera heigth and try again.\n";
		return false;
	}

	//calculate the length of a pixel in mm
	pixLength = 2 * calibrationLength / (width + height);
	output = "Succesfully calibrated!\n";
	return true;
}

void detect::perform(int& hueD1, int& satD1, int& valD1, int& stdevHueD1, int& hueD2, int& satD2, int& valD2, int& stdevHueD2) {
	//declare variables that are used in this function
	int maxLength;
	int minLength;
	int area;
	int width;
	int height;
	double hue;
	double saturation;
	double value;
	Scalar color;
	Scalar green = Scalar(0, 255, 0);
	Scalar red = Scalar(0, 0, 255);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	cvtImage();

	findContours(edgeImg, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	//loop through all detected contours
	for (int i = 0; i < contours.size(); i++) {
		//filter out small contour, which are caused by noise
		area = contourArea(contours[i]);
		if (area > 200) {
			RotatedRect minRect = minAreaRect(contours[i]);
			width = minRect.size.width;
			height = minRect.size.height;
			maxLength = max(width, height) * pixLength;
			minLength = min(width, height) * pixLength;

			//create mask of a contour
			Mat mask = Mat::zeros(inImg.size(), CV_8UC1);
			drawContours(mask, contours, i, Scalar(255), FILLED);

			//calculate hsv values by combining the mask with the hsv image
			Scalar meanHSV, stdevHSV;
			meanStdDev(hsvImg, meanHSV, stdevHSV, mask);
			hue = meanHSV[0];
			saturation = meanHSV[1];
			value = meanHSV[2];

			// filters for both cadillacs, based on the length, measured values in setHSV and the input values for the +- deltas

			if (minLength > 9 && minLength < 15 && maxLength > 40 && maxLength < 45 && hue > car1[0] - hueD1 && hue < car1[0] + hueD1 && saturation > car1[1] - satD1 && saturation < car1[1] + satD1 && value > car1[2] - valD1 && value < car1[2] + valD1 && stdevHSV[2] < car1[3] + 10) {
				//donkere cadillac
				color = green;
			}
			else if (minLength > 9 && minLength < 15 && maxLength > 40 && maxLength < 45 && hue > car2[0] - hueD2 && hue < car2[0] + hueD2 && saturation > car2[1] - satD2 && saturation < car2[1] + satD2 && value > car2[2] - valD2 && value < car2[2] + valD2 && stdevHSV[2] < car2[3] + 10) {
				//lichte cadillac
				color = green;
			}
			else {
				color = red;
			}

			drawContours(outImg, contours, i, color, 2);

			// uncomment this part to get a piece by piece reading of the measured values
			//cout << length << endl;
			//cout << stdevHSV[2] << endl;
			//cout << "hue " << hue << " sat " << saturation << " val " << value << endl;
			//cout << "Donkere Cadillac HueThreshold: " << car2[0] - hueD2 << "-" << car2[0] + hueD2 << " saturation threshold: " << car2[1] - satD << "-" << car2[1] + satD << " Value threshold: " << car2[2] - valD << "-" << car2[2] + valD << endl;
			//cout << "Lichte Cadillac HueThreshold: " << car1[0] - hueD << "-" << car1[0] + hueD << " saturation threshold: " << car1[1] - satD << "-" << car1[1] + satD << " Value threshold: " << car1[2] - valD << "-" << car1[2] + valD << endl;
			////imshow("image", outImg);
			//waitKey(0);



		}
	}

}

void detect::showResult(int hueD1, int satD1, int valD1, int stdevHueD1, int hueD2, int satD2, int valD2, int stdevHueD2) {
	while (true) {
		//register start time
		high_resolution_clock::time_point start = high_resolution_clock::now();

		perform(hueD1, satD1,  valD1, stdevHueD1, hueD2, satD2, valD2, stdevHueD2);

		//register end time, calculate difference and add to image as a 3 decimal string
		high_resolution_clock::time_point end = high_resolution_clock::now();
		duration<double> duration_sec = end - start;
		ostringstream stream;
		stream << fixed << setprecision(3) << duration_sec.count();
		string timeString = stream.str();
		timeString += " s";
		putText(outImg, timeString, { 10, 25 }, FONT_HERSHEY_PLAIN, 2, Scalar(0, 69, 255), 2);

		// imshow("EdgeDetect", edgeImg);

		imshow(viewer, outImg);


		//q is used to quit showing the output
		if (waitKey(60) == 'q') break;
	}
	destroyAllWindows();
}

detect::~detect() {
	//close images and empty input buffer
	destroyAllWindows();
	cin.ignore(1000, '\n');
}