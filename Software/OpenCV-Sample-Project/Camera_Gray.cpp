#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    // Open the default webcam (0 is the default camera ID)
    VideoCapture camera(0);
    
    // Check if camera opened successfully
    if (!camera.isOpened()) {
        cerr << "Error: Could not open webcam!" << endl;
        return -1;
    }
    
    // Create a window to display the video
    namedWindow("Greyscale Webcam", WINDOW_AUTOSIZE);
    
    Mat frame;      // Store original frame
    Mat greyFrame;  // Store greyscale frame
    
    cout << "Capturing video from webcam... Press 'q' to quit" << endl;
    
    // Continuous loop
    while (true) {
        // Capture frame from webcam
        camera >> frame;
        
        // Check if frame was captured successfully
        if (frame.empty()) {
            cerr << "Error: Could not capture frame!" << endl;
            break;
        }
        
        // Convert frame to greyscale
        cvtColor(frame, greyFrame, COLOR_BGR2GRAY);
        
        // Display the greyscale frame
        imshow("Greyscale Webcam", greyFrame);
        
        // Check for 'q' key press to exit (wait 30ms between frames, ~33 fps)
        if (waitKey(30) == 'q') {
            break;
        }
    }
    
    // Release resources
    camera.release();
    destroyAllWindows();
    
    cout << "Program ended." << endl;
    
    return 0;
}
