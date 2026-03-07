
#include <iostream>
#include <opencv2/opencv.hpp>
#include "GalaxyIncludes.h"

using namespace std;
using namespace cv;
using namespace GxIAPICPP;

int main() {
    try {
        // Initialize SDK
        IGXFactory::GetInstance().Init();

        // Enumerate devices
        gxdeviceinfo_vector devices;
        IGXFactory::GetInstance().UpdateDeviceList(1000, devices);
        if (devices.empty()) {
            cout << "No Daheng camera found." << endl;
            return -1;
        }

        // Open first device
        CGXDevicePointer device = IGXFactory::GetInstance().OpenDeviceBySN(devices[0].GetSN(), GX_ACCESS_EXCLUSIVE);
        CGXStreamPointer stream = device->OpenStream(0);
        CGXFeatureControlPointer features = device->GetRemoteFeatureControl();

        // Set acquisition mode to continuous
        features->GetEnumFeature("AcquisitionMode")->SetValue("Continuous");

        // Start acquisition
        stream->StartGrab();
        features->GetCommandFeature("AcquisitionStart")->Execute();



        // Live loop
        while (true) {
            CImageDataPointer imgData = stream->GetImage(1000); // timeout in ms

            if (imgData->GetStatus() == GX_FRAME_STATUS_SUCCESS) {
                // Convert to RGB24
                void* pRGB = imgData->ConvertToRGB24(GX_BIT_0_7, GX_RAW2RGB_NEIGHBOUR, true);

                // Create OpenCV Mat from RGB data
                cv::Mat frame(imgData->GetHeight(), imgData->GetWidth(), CV_8UC3, pRGB);

                // Show image
                cv::imshow("Daheng Camera", frame);

                // Exit on ESC key
                if (cv::waitKey(1) == 27) {
                    break;
                }
            }
            else {
                std::cout << "Image capture failed or status not successful." << std::endl;
            }
        }


        // Stop acquisition
            features->GetCommandFeature("AcquisitionStop")->Execute();
        stream->StopGrab();




/*
        // Grab one image

        CImageDataPointer imgData = stream->GetImage(1000); // timeout in ms


        if (imgData->GetStatus() == GX_FRAME_STATUS_SUCCESS) {
            // Convert to RGB24
            void* pRGB = imgData->ConvertToRGB24(GX_BIT_0_7, GX_RAW2RGB_NEIGHBOUR, true);

            // Create OpenCV Mat from RGB data
            cv::Mat frame(imgData->GetHeight(), imgData->GetWidth(), CV_8UC3, pRGB);

            // Show image
            cv::imshow("Daheng Camera", frame);
            cv::waitKey(0);  // Wait for key press
            cv::destroyAllWindows();
        }
        else {
            std::cout << "Image status not successful." << std::endl;
        }

*/



/*
//        if (imgData) {
            if (imgData->GetStatus() == GX_FRAME_STATUS_SUCCESS) {
                // process image
            }
            else {
                std::cout << "Image status not successful." << std::endl;
            }
  //      }
    //    else {
      //      std::cout << "Image pointer is null." << std::endl;
       // }
*/

        // Stop acquisition
        features->GetCommandFeature("AcquisitionStop")->Execute();
        stream->StopGrab();

        // Cleanup
        stream->Close();
        device->Close();
        IGXFactory::GetInstance().Uninit();
    }
    catch (CGalaxyException& e) {
        cerr << "Galaxy Exception: " << e.what() << endl;
    }
    catch (exception& e) {
        cerr << "Standard Exception: " << e.what() << endl;
    }

    return 0;
}