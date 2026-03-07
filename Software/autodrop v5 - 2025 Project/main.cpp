//header

#include "detect.h"

//door naar volgende segment met leegmaken van inputbuffer
#define nextSegment \
	cout << "\nPress <enter> to continue\n"; \
	cin.ignore(1000, '\n'); \
	cin.get();

int main() {
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

	string output;
	bool status;
	char select;
	bool type;

	detect test("voorbeeld");
	test.setFocus();
	nextSegment;
	do {
		status = test.setAperture();
		nextSegment;
		if (status == false) cout << "Incorrect aperture, please try again";
	} while (status == false);
	do {
		status = test.setLength(output, 30);
		cout << output;
		nextSegment;
	} while (status == false);
	test.setHSV();
	nextSegment;



	test.showResult(10, 20, 30, 10, 70, 20, 10, 10);
}


