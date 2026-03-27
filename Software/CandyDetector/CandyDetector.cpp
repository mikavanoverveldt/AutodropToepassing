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
//   4. Morphology per mask       → clean up each colour mask
//   5. Combined mask              → merge all candy colours
//   6. findContours               → individual candy instances
//   7. Per-contour classification → colour + size validity
//   8. Draw bounding boxes        → green (valid) / red (invalid)
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

        // Black mask:
        // 018 180 084 228 0 141

        // Dark yellow mask:
        // 026 038 069 255 000 255

        // Red Mask:
        // X 20 53 >1 X X


        // Close K: 15
        // Open K: 12





#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cctype>
#include "GxIAPI.h"
#include "DxImageProc.h"

using namespace std;
using namespace cv;

// ---------------------------------------------------------------------------
// Error helper (overbodig voor nu)
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
    int contrastAlpha    = 84;  // real multiplier = value / 100.0  (75 → ×0.75)
    int saturationScale = 142; // real multiplier = value / 100.0  (175 → ×1.75)

    // Stage 1 – Gaussian Blur
    int blurKsize = 3;          // must be odd; we force it below

    // Stage 3a – Black candy mask (sugar-coated black)
    int blackHMin = 29;   int blackHMax = 61;
    int blackSMin = 84; int blackSMax = 228;
    int blackVMin = 0;   int blackVMax = 159;

    // Stage 3b – Dark-yellow candy mask
    int dyHMin = 27;  int dyHMax = 64;
    int dySMin = 230;  int dySMax = 255;
    int dyVMin = 119;   int dyVMax = 255;

    // Stage 3c – Red/pink mask (two hue bands)
    int r1HMin = 10;  int r1HMax = 23;
    int r1SMin = 148;
    int r2HMin = 96;  int r2HMax = 180;
    int r2SMin = 0;

    // Stage 4 – Morphology
    int morphCloseK = 11;
    int morphOpenK  = 17;
    int distThreshPct = 56;
    int sureBgDilateK = 76;

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
        "1 - Blurred",
        "2 - HSV",
        "3a - Mask: Black",
        "3b - Mask: DarkYellow",
        "3c - Mask: Red/Pink",
        "4a - Morphology: Black",
        "4b - Morphology: DarkYellow",
        "4c - Morphology: Red/Pink",
        "4d - Watershed DT (Red)",
        "4e - Separated: Black",
        "4f - Separated: DarkYellow",
        "4g - Separated: Red/Pink",
        "5 - Combined Mask",
        "6 - Contours",
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
    createTrackbar("Blur ksize (x2+1)", "1 - Blurred",
                   &P.blurKsize, 15); // stored as k, actual = 2k+1

    // ── Black mask ──────────────────────────────────────────────────────────
    createTrackbar("H min", "3a - Mask: Black", &P.blackHMin, 180);
    createTrackbar("H max", "3a - Mask: Black", &P.blackHMax, 180);
    createTrackbar("S min", "3a - Mask: Black", &P.blackSMin, 255);
    createTrackbar("S max", "3a - Mask: Black", &P.blackSMax, 255);
    createTrackbar("V min", "3a - Mask: Black", &P.blackVMin, 255);
    createTrackbar("V max", "3a - Mask: Black", &P.blackVMax, 255);

    // ── Dark-yellow mask ────────────────────────────────────────────────────
    createTrackbar("H min", "3b - Mask: DarkYellow", &P.dyHMin, 180);
    createTrackbar("H max", "3b - Mask: DarkYellow", &P.dyHMax, 180);
    createTrackbar("S min", "3b - Mask: DarkYellow", &P.dySMin, 255);
    createTrackbar("S max", "3b - Mask: DarkYellow", &P.dySMax, 255);
    createTrackbar("V min", "3b - Mask: DarkYellow", &P.dyVMin, 255);
    createTrackbar("V max", "3b - Mask: DarkYellow", &P.dyVMax, 255);

    // ── Red/Pink mask ────────────────────────────────────────────────────────
    createTrackbar("Band1 H min", "3c - Mask: Red/Pink", &P.r1HMin, 180);
    createTrackbar("Band1 H max", "3c - Mask: Red/Pink", &P.r1HMax, 180);
    createTrackbar("Band1 S min", "3c - Mask: Red/Pink", &P.r1SMin, 255);
    createTrackbar("Band2 H min", "3c - Mask: Red/Pink", &P.r2HMin, 180);
    createTrackbar("Band2 H max", "3c - Mask: Red/Pink", &P.r2HMax, 180);
    createTrackbar("Band2 S min", "3c - Mask: Red/Pink", &P.r2SMin, 255);

    // ── Morphology ──────────────────────────────────────────────────────────
    createTrackbar("Close K", "4a - Morphology: Black", &P.morphCloseK, 150);
    createTrackbar("Open  K", "4a - Morphology: Black",  &P.morphOpenK,  150);
    createTrackbar("DT Threshold %", "4d - Watershed DT (Red)", &P.distThreshPct, 100);
    createTrackbar("Sure BG Dilate K", "4d - Watershed DT (Red)", &P.sureBgDilateK, 150);

    // ── Size thresholds ─────────────────────────────────────────────────────
    createTrackbar("Min detect area", "6 - Contours",  &P.minArea,      2000);
    createTrackbar("Min valid area",  "6 - Contours",  &P.minValidArea, 50000);
}


// ---------------------------------------------------------------------------
// applyWatershed() - separates touching objects
// ---------------------------------------------------------------------------
static Mat applyWatershed(const Mat& morphMask, const Mat& originalImage, 
                          int distThreshPct, int dilateK, 
                          Mat* outDistMap = nullptr)
{
    if (countNonZero(morphMask) == 0) return morphMask.clone();

    // 1. Sure background
    Mat sureBg;
    int dk = max(1, dilateK);
    Mat elemDilate = getStructuringElement(MORPH_RECT, Size(dk * 2 + 1, dk * 2 + 1));
    dilate(morphMask, sureBg, elemDilate);

    // 2. Distance transform
    Mat distTransform;
    distanceTransform(morphMask, distTransform, DIST_L2, 5);

    if (outDistMap)
    {
        normalize(distTransform, *outDistMap, 0, 1.0, NORM_MINMAX);
    }

    // 3. Sure foreground
    double maxVal;
    minMaxLoc(distTransform, NULL, &maxVal);
    Mat sureFg;
    threshold(distTransform, sureFg, (max(1, distThreshPct) / 100.0) * maxVal, 255, THRESH_BINARY);
    sureFg.convertTo(sureFg, CV_8U);

    // 4. Unknown region
    Mat unknown;
    subtract(sureBg, sureFg, unknown);

    // 5. Marker labelling
    Mat markers;
    connectedComponents(sureFg, markers);

    // Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1;

    // Now, mark the region of unknown with zero
    for (int r = 0; r < markers.rows; r++) {
        for (int c = 0; c < markers.cols; c++) {
            if (unknown.at<uchar>(r, c) == 255) {
                markers.at<int>(r, c) = 0;
            }
        }
    }

    // 6. Apply watershed
    // We use a flat image instead of originalImage to avoid boundaries fluctuating due to texture/lighting
    Mat flatImg;
    cvtColor(morphMask, flatImg, COLOR_GRAY2BGR);
    watershed(flatImg, markers);

    // 7. Generate separated mask (draw boundaries as black on original mask)
    Mat boundaries = Mat::zeros(markers.size(), CV_8U);
    for (int r = 0; r < markers.rows; r++) {
        for (int c = 0; c < markers.cols; c++) {
            if (markers.at<int>(r, c) == -1) {
                boundaries.at<uchar>(r, c) = 255;
            }
        }
    }

    // Dilate the 1-pixel boundary so findContours doesn't cross diagonally
    Mat elemBoundary = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(boundaries, boundaries, elemBoundary);

    Mat separatedMask = morphMask.clone();
    separatedMask.setTo(0, boundaries);

    return separatedMask;
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

    // ── Stage 1: Gaussian Blur ───────────────────────────────────────────────
    int k = max(1, P.blurKsize) * 2 + 1;   // ensure odd, ≥ 3
    Mat blurred;
    GaussianBlur(enhanced, blurred, Size(k, k), 0);
    imshow("1 - Blurred", blurred);

    // ── Stage 2: BGR → HSV ──────────────────────────────────────────────────
    Mat hsv;
    cvtColor(blurred, hsv, COLOR_BGR2HSV);
    imshow("2 - HSV", hsv);

    // ── Stage 3a: Black mask (full HSV band) ──────────────────
    Mat maskBlack;
    inRange(hsv,
            Scalar(P.blackHMin, P.blackSMin, P.blackVMin),
            Scalar(P.blackHMax, P.blackSMax, P.blackVMax),
            maskBlack);
    imshow("3a - Mask: Black", maskBlack);

    // ── Stage 3b: Dark-yellow mask ───────────────────────────────────────────
    Mat maskDY;
    inRange(hsv,
            Scalar(P.dyHMin, P.dySMin, P.dyVMin),
            Scalar(P.dyHMax, P.dySMax, P.dyVMax),
            maskDY);
    imshow("3b - Mask: DarkYellow", maskDY);

    // ── Stage 3c: Red/Pink mask (two hue bands) ──────────────────────────────
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
    imshow("3c - Mask: Red/Pink", maskRed);

    // ── Stage 4: Morphology on individual masks ──────────────────────────────
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
    imshow("4a - Morphology: Black", maskBlackMorphed);
    imshow("4b - Morphology: DarkYellow", maskDYMorphed);
    imshow("4c - Morphology: Red/Pink", maskRedMorphed);

    // ── Stage 5: Distance Transform + Watershed separation ───────────────────
    Mat distRedMap;
    Mat maskRedSeparated   = applyWatershed(maskRedMorphed, enhanced, P.distThreshPct, P.sureBgDilateK, &distRedMap);
    Mat maskBlackSeparated = applyWatershed(maskBlackMorphed, enhanced, P.distThreshPct, P.sureBgDilateK);
    Mat maskDYSeparated    = applyWatershed(maskDYMorphed, enhanced, P.distThreshPct, P.sureBgDilateK);

    if (!distRedMap.empty()) imshow("4d - Watershed DT (Red)", distRedMap);
    imshow("4e - Separated: Black", maskBlackSeparated);
    imshow("4f - Separated: DarkYellow", maskDYSeparated);
    imshow("4g - Separated: Red/Pink", maskRedSeparated);


    // ── Stage 6: findContours per colour mask ────────────────────────────────
    vector<vector<Point>> contoursRed, contoursDY, contoursBlack;
    findContours(maskRedSeparated,   contoursRed,   RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    findContours(maskDYSeparated,    contoursDY,    RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    findContours(maskBlackSeparated, contoursBlack, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Draw all accepted contours for the "6 - Contours" window
    Mat contourVis = enhanced.clone();
    int minA = max(1, P.minArea);

    for (size_t i = 0; i < contoursRed.size(); i++)
        if (contourArea(contoursRed[i]) >= minA)
            drawContours(contourVis, contoursRed, (int)i, Scalar(0, 0, 255), 2);
    for (size_t i = 0; i < contoursDY.size(); i++)
        if (contourArea(contoursDY[i]) >= minA)
            drawContours(contourVis, contoursDY, (int)i, Scalar(0, 255, 255), 2);
    for (size_t i = 0; i < contoursBlack.size(); i++)
        if (contourArea(contoursBlack[i]) >= minA)
            drawContours(contourVis, contoursBlack, (int)i, Scalar(255, 255, 255), 2);
    imshow("6 - Contours", contourVis);

    // ── Stage 7: Size validation + bounding boxes (Red, Yellow, Black) ───────
    Mat result = enhanced.clone();
    int minValid = max(1, P.minValidArea);

    auto drawDetections = [&](const vector<vector<Point>>& contours, const string& colorTag)
    {
        for (const auto& cnt : contours)
        {
            double area = contourArea(cnt);
            if (area < minA)
                continue;

            bool invalid = (area < minValid);

            Rect bbox = boundingRect(cnt);

            Scalar boxColor = invalid ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
            rectangle(result, bbox, boxColor, 3);

            string label = colorTag + (invalid ? " inv:size" : " valid");
            int baseline = 0;
            Size textSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.55, 2, &baseline);
            int textX = bbox.x + bbox.width - textSize.width - 5;
            int textY = bbox.y + bbox.height - baseline - 5;
            textX = max(0, textX);
            textY = max(textSize.height, textY);
            putText(result, label, Point(textX, textY),
                    FONT_HERSHEY_SIMPLEX, 0.55, boxColor, 2);
        }
    };

    drawDetections(contoursRed,   "red");
    drawDetections(contoursDY,    "yellow");
    drawDetections(contoursBlack, "black");

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
