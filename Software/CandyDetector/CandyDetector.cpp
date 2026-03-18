//////////////////////////////////////////////
///     Work in progress                   /// 
///
///
///


// CandyDetector.cpp
// Candy quality inspection using a Daheng camera and OpenCV.
//
// Pipeline:
//   0. Contrast & Saturation boost → enhance colour separation
//   1. Gaussian Blur              → reduce noise
//   2. BGR → HSV                  → colour-invariant space
//   3. Colour masks               → black, dark-yellow, red/pink
//   4. Combined mask + morphology → isolate candy blobs
//   5. findContours               → individual candy instances
//   6. Per-contour classification → colour + size validity
//   7. Draw bounding boxes        → green (valid) / red (invalid)
//
// Every intermediate result is shown in its own named window.
// All tuneable parameters are exposed as trackbars.


// TODO: 
// Morphologie eerst per kleur, niet op de gecombineerde mask (DONE)
// Robustere kleur detectie

// Finetunen met echte beelden van de camera in standaard met backlight.

// Commentaar per regel toevoegen


// Kalibratie functie met 'known-good'  kleuren


// TRIAL AND ERROR SETTINGS:
// Contrast: 75/300
// Saturation:175/300
// Blur 5/15



#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cctype>
#include "GxIAPI.h"
#include "DxImageProc.h"

using namespace std;
using namespace cv;

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------
static void PrintErrorInfo(GX_STATUS emStatus)
{
    char*  pszText = NULL;
    size_t nSize   = 0;
    GXGetLastError(&emStatus, NULL, &nSize);
    pszText = new char[nSize];
    GXGetLastError(&emStatus, pszText, &nSize);
    cerr << "Galaxy SDK Error: " << pszText << endl;
    delete[] pszText;
}

// ---------------------------------------------------------------------------
// Trackbar state (all in one struct for clarity)
// ---------------------------------------------------------------------------
struct Params
{
    // Stage 0 – Contrast & Saturation
    int contrastAlpha    = 100; // real multiplier = value / 100.0  (100 → ×1.0)
    int saturationScale  = 100; // real multiplier = value / 100.0  (100 → ×1.0)

    // Stage 1 – Gaussian Blur
    int blurKsize = 5;          // must be odd; we force it below

    // Stage 3a – Black candy mask (sugar-coated black)
    int blackHMin = 0;   int blackHMax = 180;
    int blackSMin = 0;   int blackSMax = 100;
    int blackVMin = 0;   int blackVMax = 60;

    // Stage 3b – Dark-yellow candy mask
    int dyHMin = 15;  int dyHMax = 38;
    int dySMin = 40;  int dySMax = 255;
    int dyVMin = 30;  int dyVMax = 180;

    // Stage 3c – Red/pink mask (two hue bands)
    int r1HMin =   0; int r1HMax =  15;
    int r1SMin =  60;
    int r2HMin = 155; int r2HMax = 180;
    int r2SMin =  60;

    // Stage 4 – Morphology
    int morphCloseK = 7;
    int morphOpenK  = 3;

    // Stage 5 – Contour size gate (area in pixels)
    int minArea     = 800;
    int minValidArea = 1500;    // below this a candy is "broken"
};

static Params P;

// ---------------------------------------------------------------------------
// createWindows() – named windows + trackbars
// ---------------------------------------------------------------------------
static void createWindows()
{
    const int W = 800, H = 500;

    // Pipeline step windows
    const char* wins[] = {
        "0 - Enhanced",
        "1 - Original",
        "2 - Blurred",
        "3 - HSV",
        "4a - Mask: Black",
        "4b - Mask: DarkYellow",
        "4c - Mask: Red/Pink",
        "5 - Morphology: Black",
        "5b - Morphology: DarkYellow",
        "5c - Morphology: Red/Pink",
        "6 - Combined Mask",
        "7 - Contours",
        "8 - Result"
    };
    for (auto& w : wins)
    {
        namedWindow(w, WINDOW_NORMAL);
        resizeWindow(w, W, H);
    }

    // ── Contrast & Saturation ────────────────────────────────────────────────
    createTrackbar("Contrast (x/100)",   "0 - Enhanced", &P.contrastAlpha,   300);
    createTrackbar("Saturation (x/100)", "0 - Enhanced", &P.saturationScale, 300);

    // ── Blur ────────────────────────────────────────────────────────────────
    createTrackbar("Blur ksize (x2+1)", "2 - Blurred",
                   &P.blurKsize, 15); // stored as k, actual = 2k+1

    // ── Black mask ──────────────────────────────────────────────────────────
    createTrackbar("H min", "4a - Mask: Black", &P.blackHMin, 180);
    createTrackbar("H max", "4a - Mask: Black", &P.blackHMax, 180);
    createTrackbar("S min", "4a - Mask: Black", &P.blackSMin, 255);
    createTrackbar("S max", "4a - Mask: Black", &P.blackSMax, 255);
    createTrackbar("V min", "4a - Mask: Black", &P.blackVMin, 255);
    createTrackbar("V max", "4a - Mask: Black", &P.blackVMax, 255);

    // ── Dark-yellow mask ────────────────────────────────────────────────────
    createTrackbar("H min", "4b - Mask: DarkYellow", &P.dyHMin, 180);
    createTrackbar("H max", "4b - Mask: DarkYellow", &P.dyHMax, 180);
    createTrackbar("S min", "4b - Mask: DarkYellow", &P.dySMin, 255);
    createTrackbar("S max", "4b - Mask: DarkYellow", &P.dySMax, 255);
    createTrackbar("V min", "4b - Mask: DarkYellow", &P.dyVMin, 255);
    createTrackbar("V max", "4b - Mask: DarkYellow", &P.dyVMax, 255);

    // ── Red/Pink mask ────────────────────────────────────────────────────────
    createTrackbar("Band1 H min", "4c - Mask: Red/Pink", &P.r1HMin, 180);
    createTrackbar("Band1 H max", "4c - Mask: Red/Pink", &P.r1HMax, 180);
    createTrackbar("Band1 S min", "4c - Mask: Red/Pink", &P.r1SMin, 255);
    createTrackbar("Band2 H min", "4c - Mask: Red/Pink", &P.r2HMin, 180);
    createTrackbar("Band2 H max", "4c - Mask: Red/Pink", &P.r2HMax, 180);
    createTrackbar("Band2 S min", "4c - Mask: Red/Pink", &P.r2SMin, 255);

    // ── Morphology ──────────────────────────────────────────────────────────
    createTrackbar("Close K", "6 - Combined Mask", &P.morphCloseK, 30);
    createTrackbar("Open  K", "6 - Combined Mask",  &P.morphOpenK,  30);

    // ── Size thresholds ─────────────────────────────────────────────────────
    createTrackbar("Min detect area", "7 - Contours",  &P.minArea,      20000);
    createTrackbar("Min valid area",  "7 - Contours",  &P.minValidArea, 20000);
}

// ---------------------------------------------------------------------------
// processFrame() – run the full pipeline on one BGR frame
// ---------------------------------------------------------------------------
static void processFrame(const Mat& colorFrame)
{
    // ── Stage 0: Contrast & Saturation enhancement ─────────────────────────────
    Mat enhanced;
    colorFrame.convertTo(enhanced, -1, max(0, P.contrastAlpha) / 100.0, 0);
    {
        Mat hsv0;
        cvtColor(enhanced, hsv0, COLOR_BGR2HSV);
        vector<Mat> ch;
        split(hsv0, ch);
        ch[1].convertTo(ch[1], -1, max(0, P.saturationScale) / 100.0, 0);
        merge(ch, hsv0);
        cvtColor(hsv0, enhanced, COLOR_HSV2BGR);
    }
    imshow("0 - Enhanced", enhanced);

    // ── Stage 1: Show original ───────────────────────────────────────────────
    imshow("1 - Original", colorFrame);

    // ── Stage 2: Gaussian Blur ───────────────────────────────────────────────
    int k = max(1, P.blurKsize) * 2 + 1;   // ensure odd, ≥ 3
    Mat blurred;
    GaussianBlur(enhanced, blurred, Size(k, k), 0);
    imshow("2 - Blurred", blurred);

    // ── Stage 3: BGR → HSV ──────────────────────────────────────────────────
    Mat hsv;
    cvtColor(blurred, hsv, COLOR_BGR2HSV);
    imshow("3 - HSV", hsv);

    // ── Stage 4a: Black mask (sugar-coated, full HSV band) ──────────────────
    Mat maskBlack;
    inRange(hsv,
            Scalar(P.blackHMin, P.blackSMin, P.blackVMin),
            Scalar(P.blackHMax, P.blackSMax, P.blackVMax),
            maskBlack);
    imshow("4a - Mask: Black", maskBlack);

    // ── Stage 4b: Dark-yellow mask ───────────────────────────────────────────
    Mat maskDY;
    inRange(hsv,
            Scalar(P.dyHMin, P.dySMin, P.dyVMin),
            Scalar(P.dyHMax, P.dySMax, P.dyVMax),
            maskDY);
    imshow("4b - Mask: DarkYellow", maskDY);

    // ── Stage 4c: Red/Pink mask (two hue bands) ──────────────────────────────
    Mat maskR1, maskR2, maskRed;
    inRange(hsv,
            Scalar(P.r1HMin, P.r1SMin, 40),
            Scalar(P.r1HMax, 255,       255),
            maskR1);
    inRange(hsv,
            Scalar(P.r2HMin, P.r2SMin, 40),
            Scalar(P.r2HMax, 255,       255),
            maskR2);
    bitwise_or(maskR1, maskR2, maskRed);
    imshow("4c - Mask: Red/Pink", maskRed);

    // ── Stage 5: Morphology on individual masks (before combining) ───────────
    int ck = max(1, P.morphCloseK);
    int ok = max(1, P.morphOpenK);
    Mat elemClose = getStructuringElement(MORPH_ELLIPSE, Size(ck * 2 + 1, ck * 2 + 1));
    Mat elemOpen  = getStructuringElement(MORPH_ELLIPSE, Size(ok * 2 + 1, ok * 2 + 1));

    Mat maskBlackMorphed, maskDYMorphed, maskRedMorphed;
    morphologyEx(maskBlack, maskBlackMorphed, MORPH_CLOSE, elemClose);
    morphologyEx(maskBlackMorphed, maskBlackMorphed, MORPH_OPEN, elemOpen);

    morphologyEx(maskDY, maskDYMorphed, MORPH_CLOSE, elemClose);
    morphologyEx(maskDYMorphed, maskDYMorphed, MORPH_OPEN, elemOpen);

    morphologyEx(maskRed, maskRedMorphed, MORPH_CLOSE, elemClose);
    morphologyEx(maskRedMorphed, maskRedMorphed, MORPH_OPEN, elemOpen);
    imshow("5 - Morphology: Black", maskBlackMorphed);
    imshow("5b - Morphology: DarkYellow", maskDYMorphed);
    imshow("5c - Morphology: Red/Pink", maskRedMorphed);

    // ── Stage 6: Combined mask (all candy colours after morphology) ──────────
    Mat maskAll;
    bitwise_or(maskBlackMorphed, maskDYMorphed, maskAll);
    bitwise_or(maskAll,          maskRedMorphed, maskAll);
    imshow("6 - Combined Mask", maskAll);

    // ── Stage 7: findContours ────────────────────────────────────────────────
    vector<vector<Point>> contours;
    findContours(maskAll, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Draw all accepted contours for the "7 - Contours" window
    Mat contourVis = enhanced.clone();
    int minA = max(1, P.minArea);

    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area < minA)
            continue;
        drawContours(contourVis, contours, (int)i, Scalar(255, 255, 0), 2);
    }
    imshow("7 - Contours", contourVis);

    // ── Stage 8: Per-candy classification + bounding box drawing ─────────────
    Mat result = enhanced.clone();
    int minValid = max(1, P.minValidArea);

    for (const auto& cnt : contours)
    {
        double area = contourArea(cnt);
        if (area < minA)
            continue;   // too small to be a candy at all (noise)

        Rect bbox = boundingRect(cnt);

        // ── Colour vote inside this contour ──────────────────────────────────
        // Build a single-contour mask
        Mat cntMask = Mat::zeros(enhanced.size(), CV_8UC1);
        vector<vector<Point>> tmp = {cnt};
        drawContours(cntMask, tmp, 0, Scalar(255), FILLED);

        // Count red/pink pixels vs total candy pixels
        Mat redInside, allInside;
        bitwise_and(maskRedMorphed, cntMask, redInside);
        bitwise_and(maskAll,        cntMask, allInside);

        int totalPx = countNonZero(allInside);
        int redPx   = countNonZero(redInside);

        double redRatio = (totalPx > 0) ? (double)redPx / totalPx : 0.0;

        bool colorInvalid = (redRatio > 0.20); // > 20 % red pixels → invalid
        bool sizeInvalid  = (area < minValid);

        bool invalid = colorInvalid || sizeInvalid;

        Scalar colour = invalid ? Scalar(0, 0, 255)   // red
                                : Scalar(0, 255, 0);  // green
        rectangle(result, bbox, colour, 3);

        // Label
        string label;
        if (!invalid)
        {
            label = "valid";
        }
        else
        {
            if (colorInvalid && sizeInvalid) label = "inv: col+size";
            else if (colorInvalid)           label = "inv: col";
            else                             label = "inv: size";
        }
        // Position text at bottom-right inside the box
        int baseline = 0;
        Size textSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.55, 2, &baseline);
        int textX = bbox.x + bbox.width - textSize.width - 5;
        int textY = bbox.y + bbox.height - baseline - 5;
        putText(result, label, Point(textX, textY),
                FONT_HERSHEY_SIMPLEX, 0.55, colour, 2);
    }

    imshow("8 - Result", result);
}

// ---------------------------------------------------------------------------
// runDebugMode() – cycle through sample images in ./images/
// ---------------------------------------------------------------------------
static void runDebugMode()
{
    vector<String> imagePaths;
    // Use relative ./images folder (not absolute /images) to match project layout
    try {
        glob("./images/*", imagePaths, false);
    }
    catch (const cv::Exception& e)
    {
        cerr << "Debug mode: could not open ./images/ folder: " << e.what() << endl;
        return;
    }

    // Filter to common image extensions
    vector<String> filtered;
    for (const auto& p : imagePaths)
    {
        String lower = p;
        transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        size_t pos = lower.find_last_of('.');
        if (pos != String::npos && pos + 1 < lower.size())
        {
            String ext = lower.substr(pos); // includes the dot
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
                ext == ".bmp" || ext == ".tif" || ext == ".tiff")
                filtered.push_back(p);
        }
    }

    if (filtered.empty())
    {
        cerr << "Debug mode: no images found in ./images/ folder." << endl;
        return;
    }

    cout << "Debug mode: found " << filtered.size() << " image(s) in ./images/" << endl;
    cout << "  Space / Right arrow : next image" << endl;
    cout << "  Left arrow          : previous image" << endl;
    cout << "  ESC / q             : quit" << endl;

    createWindows();

    int idx = 0;
    Mat current;

    auto loadAndProcess = [&]()
    {
        current = imread(filtered[idx]);
        if (current.empty())
        {
            cerr << "Failed to load: " << filtered[idx] << endl;
            return;
        }
        cout << "Image [" << (idx + 1) << "/" << filtered.size() << "]: "
             << filtered[idx] << endl;
        processFrame(current);
    };

    loadAndProcess();

    while (true)
    {
        int key = waitKey(50);

        if (key == 27 || key == 'q')
            break;

        // Space, right arrow, or 'd' → next
        if (key == ' ' || key == 83 /* right */ || key == 'd')
        {
            idx = (idx + 1) % (int)filtered.size();
            loadAndProcess();
        }
        // Left arrow or 'a' → previous
        else if (key == 81 /* left */ || key == 'a')
        {
            idx = (idx - 1 + (int)filtered.size()) % (int)filtered.size();
            loadAndProcess();
        }
        // Any other key while an image is loaded → reprocess with current params
        else if (key >= 0 && !current.empty())
        {
            processFrame(current);
        }
        // Reprocess on every tick so trackbar changes take effect without a keypress
        else if (!current.empty())
        {
            processFrame(current);
        }
    }

    destroyAllWindows();
    cout << "Debug mode ended." << endl;
}

// ---------------------------------------------------------------------------
// main()
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // ── Debug mode: use sample images instead of camera ──────────────────────
    for (int i = 1; i < argc; i++)
    {
        if (string(argv[i]) == "--debug" || string(argv[i]) == "-d")
        {
            runDebugMode();
            return 0;
        }
    }

    GX_STATUS    emStatus = GX_STATUS_SUCCESS;
    GX_DEV_HANDLE hDevice = NULL;
    GX_DS_HANDLE  hStream = NULL;

    // ── 1. Init SDK ──────────────────────────────────────────────────────────
    emStatus = GXInitLib();
    if (emStatus != GX_STATUS_SUCCESS) { PrintErrorInfo(emStatus); return -1; }
    cout << "Galaxy SDK version: " << GXGetLibVersion() << endl;

    // ── 2. Enumerate devices ─────────────────────────────────────────────────
    uint32_t nDeviceNum = 0;
    emStatus = GXUpdateAllDeviceList(&nDeviceNum, 1000);
    if (emStatus != GX_STATUS_SUCCESS || nDeviceNum == 0)
    {
        if (emStatus != GX_STATUS_SUCCESS) PrintErrorInfo(emStatus);
        else cerr << "No Daheng camera found. Check cable / power." << endl;
        GXCloseLib();
        return -1;
    }
    cout << "Found " << nDeviceNum << " camera(s). Opening first one..." << endl;

    // ── 3. Open device ───────────────────────────────────────────────────────
    emStatus = GXOpenDeviceByIndex(1, &hDevice);
    if (emStatus != GX_STATUS_SUCCESS)
    {
        PrintErrorInfo(emStatus); GXCloseLib(); return -1;
    }

    GX_STRING_VALUE stModel, stSerial;
    GXGetStringValue(hDevice, "DeviceModelName",    &stModel);
    GXGetStringValue(hDevice, "DeviceSerialNumber", &stSerial);
    cout << "Model : " << stModel.strCurValue  << endl;
    cout << "Serial: " << stSerial.strCurValue << endl;

    // ── 4. Colour vs mono ────────────────────────────────────────────────────
    GX_NODE_ACCESS_MODE emAccess;
    bool    bIsColor       = false;
    int64_t i64ColorFilter = GX_COLOR_FILTER_NONE;

    emStatus = GXGetNodeAccessMode(hDevice, "PixelColorFilter", &emAccess);
    if (emStatus == GX_STATUS_SUCCESS)
        bIsColor = (emAccess == GX_NODE_ACCESS_MODE_RO ||
                    emAccess == GX_NODE_ACCESS_MODE_WO ||
                    emAccess == GX_NODE_ACCESS_MODE_RW);
    if (bIsColor)
    {
        GX_ENUM_VALUE stEnum;
        GXGetEnumValue(hDevice, "PixelColorFilter", &stEnum);
        i64ColorFilter = stEnum.stCurValue.nCurValue;
        cout << "Camera type: Color" << endl;
    }
    else cout << "Camera type: Mono" << endl;

    // ── 5. Data stream ───────────────────────────────────────────────────────
    uint32_t nStreamNum = 0;
    emStatus = GXGetDataStreamNumFromDev(hDevice, &nStreamNum);
    if (emStatus != GX_STATUS_SUCCESS || nStreamNum < 1)
    {
        cerr << "Failed to get data stream count." << endl;
        GXCloseDevice(hDevice); GXCloseLib(); return -1;
    }
    emStatus = GXGetDataStreamHandleFromDev(hDevice, 1, &hStream);
    if (emStatus != GX_STATUS_SUCCESS)
    {
        PrintErrorInfo(emStatus); GXCloseDevice(hDevice); GXCloseLib(); return -1;
    }

    uint32_t nPayloadSize = 0;
    emStatus = GXGetPayLoadSize(hStream, &nPayloadSize);
    if (emStatus != GX_STATUS_SUCCESS)
    {
        PrintErrorInfo(emStatus); GXCloseDevice(hDevice); GXCloseLib(); return -1;
    }

    // ── 6. Configure acquisition ──────────────────────────────────────────────
    GXSetEnumValueByString(hDevice, "AcquisitionMode", "Continuous");
    GXSetEnumValueByString(hDevice, "TriggerMode",     "Off");
    GXSetAcqusitionBufferNumber(hDevice, 5);

    // ── 7. Allocate buffers ───────────────────────────────────────────────────
    unsigned char* pRaw8Buf = new unsigned char[nPayloadSize];
    unsigned char* pRGBBuf  = new unsigned char[nPayloadSize * 3];

    // ── 8. Create windows & trackbars ────────────────────────────────────────
    createWindows();

    // ── 9. Start streaming ────────────────────────────────────────────────────
    emStatus = GXStreamOn(hDevice);
    if (emStatus != GX_STATUS_SUCCESS)
    {
        PrintErrorInfo(emStatus);
        delete[] pRaw8Buf; delete[] pRGBBuf;
        GXCloseDevice(hDevice); GXCloseLib(); return -1;
    }
    cout << "Streaming... Press ESC or 'q' to quit." << endl;

    // ── 10. Main loop ─────────────────────────────────────────────────────────
    while (true)
    {
        PGX_FRAME_BUFFER pFrameBuf = NULL;
        emStatus = GXDQBuf(hDevice, &pFrameBuf, 1000);
        if (emStatus != GX_STATUS_SUCCESS)
        {
            if (emStatus == GX_STATUS_TIMEOUT) continue;
            PrintErrorInfo(emStatus); break;
        }

        if (pFrameBuf->nStatus == GX_FRAME_STATUS_SUCCESS)
        {
            int w = pFrameBuf->nWidth;
            int h = pFrameBuf->nHeight;
            Mat colorFrame;

            if (bIsColor)
            {
                DxRaw8toRGB24(
                    (unsigned char*)pFrameBuf->pImgBuf,
                    pRGBBuf, w, h,
                    RAW2RGB_NEIGHBOUR,
                    DX_PIXEL_COLOR_FILTER(i64ColorFilter),
                    false);
                Mat rgbMat(h, w, CV_8UC3, pRGBBuf);
                cvtColor(rgbMat, colorFrame, COLOR_RGB2BGR);
            }
            else
            {
                Mat monoMat(h, w, CV_8UC1, pFrameBuf->pImgBuf);
                cvtColor(monoMat, colorFrame, COLOR_GRAY2BGR);
            }

            processFrame(colorFrame);
        }
        else
        {
            cerr << "Abnormal frame, status: " << pFrameBuf->nStatus << endl;
        }

        GXQBuf(hDevice, pFrameBuf);

        int key = waitKey(1);
        if (key == 27 || key == 'q') break;
    }

    // ── 11. Cleanup ───────────────────────────────────────────────────────────
    destroyAllWindows();
    GXStreamOff(hDevice);
    delete[] pRaw8Buf;
    delete[] pRGBBuf;
    GXCloseDevice(hDevice);
    GXCloseLib();
    cout << "Program ended." << endl;
    return 0;
}
