#include <opencv2/opencv.hpp>
#include <iostream>
#include "GxIAPI.h"
#include "DxImageProc.h"

using namespace std;
using namespace cv;

#define SDK_INC "/home/mikavanoverveldt/Avans/Vision/Galaxy_Linux-x86_Gige-U3_32bits-64bits_2.4.2507.9231/Galaxy_camera/inc"

// Print a human-readable description of a GX_STATUS error code
static void PrintErrorInfo(GX_STATUS emStatus)
{
    char* pszText = NULL;
    size_t nSize = 0;

    GXGetLastError(&emStatus, NULL, &nSize);
    pszText = new char[nSize];
    GXGetLastError(&emStatus, pszText, &nSize);
    cerr << "Galaxy SDK Error: " << pszText << endl;
    delete[] pszText;
}

int main()
{
    GX_STATUS emStatus = GX_STATUS_SUCCESS;
    GX_DEV_HANDLE hDevice = NULL;
    GX_DS_HANDLE  hStream = NULL;

    // ── 1. Init SDK ──────────────────────────────────────────────────────────
    emStatus = GXInitLib();
    if (emStatus != GX_STATUS_SUCCESS)
    {
        PrintErrorInfo(emStatus);
        return -1;
    }
    cout << "Galaxy SDK version: " << GXGetLibVersion() << endl;

    // ── 2. Enumerate devices ─────────────────────────────────────────────────
    uint32_t nDeviceNum = 0;
    emStatus = GXUpdateAllDeviceList(&nDeviceNum, 1000);
    if (emStatus != GX_STATUS_SUCCESS)
    {
        PrintErrorInfo(emStatus);
        GXCloseLib();
        return -1;
    }
    if (nDeviceNum == 0)
    {
        cerr << "No Daheng camera found. Check cable / power." << endl;
        GXCloseLib();
        return -1;
    }
    cout << "Found " << nDeviceNum << " camera(s). Opening first one..." << endl;

    // ── 3. Open device ───────────────────────────────────────────────────────
    emStatus = GXOpenDeviceByIndex(1, &hDevice);
    if (emStatus != GX_STATUS_SUCCESS)
    {
        PrintErrorInfo(emStatus);
        GXCloseLib();
        return -1;
    }

    // Print basic device info
    GX_STRING_VALUE stModel, stSerial;
    GXGetStringValue(hDevice, "DeviceModelName",   &stModel);
    GXGetStringValue(hDevice, "DeviceSerialNumber", &stSerial);
    cout << "Model : " << stModel.strCurValue  << endl;
    cout << "Serial: " << stSerial.strCurValue << endl;

    // ── 4. Determine camera type (mono vs color) ──────────────────────────────
    GX_NODE_ACCESS_MODE emAccess;
    bool bIsColor = false;
    int64_t i64ColorFilter = GX_COLOR_FILTER_NONE;

    emStatus = GXGetNodeAccessMode(hDevice, "PixelColorFilter", &emAccess);
    if (emStatus == GX_STATUS_SUCCESS)
    {
        bIsColor = (emAccess == GX_NODE_ACCESS_MODE_RO ||
                    emAccess == GX_NODE_ACCESS_MODE_WO ||
                    emAccess == GX_NODE_ACCESS_MODE_RW);
    }
    if (bIsColor)
    {
        GX_ENUM_VALUE stEnum;
        GXGetEnumValue(hDevice, "PixelColorFilter", &stEnum);
        i64ColorFilter = stEnum.stCurValue.nCurValue;
        cout << "Camera type: Color" << endl;
    }
    else
    {
        cout << "Camera type: Mono" << endl;
    }

    // ── 5. Get data stream handle and payload size ────────────────────────────
    uint32_t nStreamNum = 0;
    emStatus = GXGetDataStreamNumFromDev(hDevice, &nStreamNum);
    if (emStatus != GX_STATUS_SUCCESS || nStreamNum < 1)
    {
        cerr << "Failed to get data stream count." << endl;
        GXCloseDevice(hDevice);
        GXCloseLib();
        return -1;
    }

    emStatus = GXGetDataStreamHandleFromDev(hDevice, 1, &hStream);
    if (emStatus != GX_STATUS_SUCCESS)
    {
        PrintErrorInfo(emStatus);
        GXCloseDevice(hDevice);
        GXCloseLib();
        return -1;
    }

    uint32_t nPayloadSize = 0;
    emStatus = GXGetPayLoadSize(hStream, &nPayloadSize);
    if (emStatus != GX_STATUS_SUCCESS)
    {
        PrintErrorInfo(emStatus);
        GXCloseDevice(hDevice);
        GXCloseLib();
        return -1;
    }

    // ── 6. Configure acquisition ──────────────────────────────────────────────
    GXSetEnumValueByString(hDevice, "AcquisitionMode", "Continuous");
    GXSetEnumValueByString(hDevice, "TriggerMode",     "Off");

    uint64_t nBufNum = 5;
    GXSetAcqusitionBufferNumber(hDevice, nBufNum);

    // ── 7. Allocate image conversion buffers ──────────────────────────────────
    unsigned char* pRaw8Buf  = new unsigned char[nPayloadSize];          // for 10/12-bit → 8-bit
    unsigned char* pRGBBuf   = new unsigned char[nPayloadSize * 3];      // for Bayer → RGB24

    // ── 8. Start streaming ────────────────────────────────────────────────────
    emStatus = GXStreamOn(hDevice);
    if (emStatus != GX_STATUS_SUCCESS)
    {
        PrintErrorInfo(emStatus);
        delete[] pRaw8Buf;
        delete[] pRGBBuf;
        GXCloseDevice(hDevice);
        GXCloseLib();
        return -1;
    }

    const int WIN_W = 960;
    const int WIN_H = 540;

    namedWindow("1 - Color",     WINDOW_NORMAL);
    namedWindow("2 - Grayscale", WINDOW_NORMAL);
    namedWindow("3 - Binary",    WINDOW_NORMAL);
    resizeWindow("1 - Color",     WIN_W, WIN_H);
    resizeWindow("2 - Grayscale", WIN_W, WIN_H);
    resizeWindow("3 - Binary",    WIN_W, WIN_H);
    cout << "Streaming... Press ESC or 'q' to quit." << endl;

    // Threshold value for binary conversion (0-255)
    int threshValue = 127;
    createTrackbar("Threshold", "3 - Binary", &threshValue, 255);

    // ── 9. Acquisition loop ───────────────────────────────────────────────────
    while (true)
    {
        PGX_FRAME_BUFFER pFrameBuf = NULL;
        emStatus = GXDQBuf(hDevice, &pFrameBuf, 1000);
        if (emStatus != GX_STATUS_SUCCESS)
        {
            if (emStatus == GX_STATUS_TIMEOUT)
                continue;
            PrintErrorInfo(emStatus);
            break;
        }

        if (pFrameBuf->nStatus == GX_FRAME_STATUS_SUCCESS)
        {
            int w = pFrameBuf->nWidth;
            int h = pFrameBuf->nHeight;
            Mat colorFrame;

            // ── Step 1: get a BGR color frame ─────────────────────────────
            if (bIsColor)
            {
                DxRaw8toRGB24(
                    (unsigned char*)pFrameBuf->pImgBuf,
                    pRGBBuf,
                    w, h,
                    RAW2RGB_NEIGHBOUR,
                    DX_PIXEL_COLOR_FILTER(i64ColorFilter),
                    false);

                Mat rgbMat(h, w, CV_8UC3, pRGBBuf);
                cvtColor(rgbMat, colorFrame, COLOR_RGB2BGR);
            }
            else
            {
                // Mono camera: promote to 3-channel so all windows are consistent
                Mat monoMat(h, w, CV_8UC1, pFrameBuf->pImgBuf);
                cvtColor(monoMat, colorFrame, COLOR_GRAY2BGR);
            }

            // ── Step 2: grayscale ─────────────────────────────────────────
            Mat grayFrame;
            cvtColor(colorFrame, grayFrame, COLOR_BGR2GRAY);

            // ── Step 3: binary threshold ──────────────────────────────────
            Mat binaryFrame;
            threshold(grayFrame, binaryFrame, threshValue, 255, THRESH_BINARY);

            // ── Display all three streams ─────────────────────────────────
            imshow("1 - Color",     colorFrame);
            imshow("2 - Grayscale", grayFrame);
            imshow("3 - Binary",    binaryFrame);
        }
        else
        {
            cerr << "Abnormal frame, status: " << pFrameBuf->nStatus << endl;
        }

        // Return buffer to queue
        GXQBuf(hDevice, pFrameBuf);

        int key = waitKey(1);
        if (key == 27 || key == 'q')   // ESC or q
            break;
    }

    // ── 10. Cleanup ───────────────────────────────────────────────────────────
    destroyAllWindows();

    GXStreamOff(hDevice);

    delete[] pRaw8Buf;
    delete[] pRGBBuf;

    GXCloseDevice(hDevice);
    GXCloseLib();

    cout << "Program ended." << endl;
    return 0;
}
