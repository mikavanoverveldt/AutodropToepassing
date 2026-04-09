//CandyDetector.cpp

///////////////////////////
// Mika van Overveldt
// Jurre Fikkers
// 3 april 2026

// Vak: Machine vision in de machinebouw
// Opdracht: Ontwikkel een vision pipeline in C++ met OpenCV en Daheng/Galaxy SDK om snoepjes te detecteren op basis van kleur, vorm en grootte. Implementeer interactieve trackbars voor parameterafstemming en toon tussenresultaten in aparte vensters. Zorg voor robuuste foutafhandeling en documenteer de code duidelijk.



/*
Dit programma neemt frames van een Daheng/Galaxy-camera op (of laadt voorbeeldafbeeldingen in debug-modus),
 verwerkt elk frame om snoepjes te detecteren in drie kleurcategorieën (zwarte, suikergecoate; donkergele; 
 en rood/roze), en toont tussenresultaten van de verwerkingspipeline. De pipeline bevat contrast- en
  verzadigingsverbetering, Gaussian blur, HSV-thresholding, morfologische bewerkingen, 
  distance-transform + watershed-separatie, contourdetectie en groottevalidatie. Gedetecteerde snoepjes 
  worden weergegeven met bounding boxes en labels; trackbars (schuifregelaars) maken interactieve 
  parameterafstemming mogelijk.

*/

// TODO:
// CODE CLEANUP
// MORE COMMENTS
// 


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
    int contrastAlpha    = 137;  // real multiplier = value / 100.0  (75 → ×0.75)
    int saturationScale = 141; // real multiplier = value / 100.0  (175 → ×1.75)

    // Stage 1 – Gaussian Blur
    int blurKsize = 3;          // must be odd; we force it below

    // Stage 3a – Black candy mask
    int blackHMin = 0;   int blackHMax = 180;
    int blackSMin = 0;  int blackSMax = 255;
    int blackVMin = 0;   int blackVMax = 40;

    // Stage 3b – Dark-yellow candy mask
    int dyHMin = 21;   int dyHMax = 52;
    int dySMin = 30;   int dySMax = 255;
    int dyVMin = 118;   int dyVMax = 255;

    // Stage 3d – Brown candy mask
    int brHMin = 12;   int brHMax = 23;
    int brSMin = 145;   int brSMax = 255;
    int brVMin = 0;   int brVMax = 255;

    // Stage 3c – Red/pink mask
    int redHMin1 = 0;   int redHMax1 = 5;
    int redHMin2 = 172; int redHMax2 = 180; // Second range for hue wraparound
    int redSMin = 128;  int redSMax = 255;
    int redVMin = 40;  int redVMax = 255;

    // Stage 4 – Morphology
    int morphCloseK = 7;
    int morphOpenK  = 5;

    int redMorphCloseK = 9;
    int redMorphOpenK  = 7;

    // Stage 5 – Watershed (Black + Yellow)
    int distThreshPct = 59; // % of max distance transform value to use as foreground threshold
    int sureBgDilateK = 3; // Dilation kernel size for sure background in watershed 

    // Stage 6 – Contour size gate (area in pixels)
    int minArea     = 5000;     // below this a contour is ignored
    int minValidArea = 32500;   // below this a candy is "broken"
};

static Params P;

// ---------------------------------------------------------------------------
// createWindows() – named windows + trackbars
// ---------------------------------------------------------------------------
static void createWindows(bool showMode, bool adjustMode)
{
    const int W = 800, H = 500;

    // Default mode: only show the final result window.
    if (!showMode)
    {
        namedWindow("8 - Result", WINDOW_NORMAL);
        resizeWindow("8 - Result", W, H);
        return;
    }

     // Pipeline step windows
     const char* wins[] = {
         "0 - Enhanced",
         "1 - Blurred",
         "3a - Mask: Black",
         "3b - Mask: DarkYellow",
         "3c - Mask: Red/Pink",
         "3d - Mask: Brown",
         "4a - Morphology: Black",
         "4b - Morphology: Red",
         "4d - Watershed DT (Yellow)",
         "4f - Separated: Black",
         "4g - Separated: DarkYellow",
         "4h - Separated: Red/Pink",
         "4i - Separated: Brown",
         "6 - Contours",
         "8 - Result"
     };
    for (auto& w : wins) // create and resize all windows in one loop
    {
        namedWindow(w, WINDOW_NORMAL);
        resizeWindow(w, W, H);
    }
    if (adjustMode){
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
     createTrackbar("H min1", "3c - Mask: Red/Pink", &P.redHMin1, 180);
     createTrackbar("H max1", "3c - Mask: Red/Pink", &P.redHMax1, 180);
     createTrackbar("H min2", "3c - Mask: Red/Pink", &P.redHMin2, 180);
     createTrackbar("H max2", "3c - Mask: Red/Pink", &P.redHMax2, 180);
     createTrackbar("S min", "3c - Mask: Red/Pink", &P.redSMin, 255);
     createTrackbar("S max", "3c - Mask: Red/Pink", &P.redSMax, 255);
     createTrackbar("V min", "3c - Mask: Red/Pink", &P.redVMin, 255);
     createTrackbar("V max", "3c - Mask: Red/Pink", &P.redVMax, 255);

    // ── Brown mask ───────────────────────────────────────────────────────────
    createTrackbar("H min", "3d - Mask: Brown", &P.brHMin, 180);
    createTrackbar("H max", "3d - Mask: Brown", &P.brHMax, 180);
    createTrackbar("S min", "3d - Mask: Brown", &P.brSMin, 255);
    createTrackbar("S max", "3d - Mask: Brown", &P.brSMax, 255);
    createTrackbar("V min", "3d - Mask: Brown", &P.brVMin, 255);
    createTrackbar("V max", "3d - Mask: Brown", &P.brVMax, 255);

    // ── Morphology ──────────────────────────────────────────────────────────
    createTrackbar("Close K", "4a - Morphology: Black", &P.morphCloseK, 150);
    createTrackbar("Open  K", "4a - Morphology: Black",  &P.morphOpenK,  150);
    createTrackbar("Red Close K", "4b - Morphology: Red", &P.redMorphCloseK, 150);
    createTrackbar("Red Open  K", "4b - Morphology: Red",  &P.redMorphOpenK,  150);
    
     // Black/Yellow watershed params
     createTrackbar("DT Threshold %", "4d - Watershed DT (Yellow)", &P.distThreshPct, 100);
     createTrackbar("Sure BG Dilate K", "4d - Watershed DT (Yellow)", &P.sureBgDilateK, 150);

    // ── Size thresholds ─────────────────────────────────────────────────────
    createTrackbar("Min detect area", "6 - Contours",  &P.minArea,      10000);
    createTrackbar("Min valid area",  "6 - Contours",  &P.minValidArea, 500000);
    }
}


// ---------------------------------------------------------------------------
// applyWatershed() - separates touching objects
// ---------------------------------------------------------------------------
static Mat applyWatershed(const Mat& morphMask,
                          int distThreshPct, int dilateK, int markerDilateK = 0,
                          Mat* outDistMap = nullptr)
{
    if (countNonZero(morphMask) == 0) return morphMask.clone();

    // 1. Sure background
    // Dilate the binary mask so that border pixels become part of the background
    // (used later to identify 'unknown' regions for watershed markers).
    Mat sureBg;
    int dk = max(1, dilateK);
    Mat elemDilate = getStructuringElement(MORPH_RECT, Size(dk * 2 + 1, dk * 2 + 1));
    dilate(morphMask, sureBg, elemDilate);

    // 2. Distance transform
    // Compute distance-to-nearest-zero for each foreground pixel; peaks indicate object centers.
    Mat distTransform;
    distanceTransform(morphMask, distTransform, DIST_L2, 5);

    if (outDistMap)
    {
        normalize(distTransform, *outDistMap, 0, 1.0, NORM_MINMAX);
    }

    // 3. Sure foreground
    // Threshold the distance map to get reliable object centers (foreground markers).
    double maxVal;
    minMaxLoc(distTransform, NULL, &maxVal);
    Mat sureFg;
    threshold(distTransform, sureFg, (max(1, distThreshPct) / 100.0) * maxVal, 255, THRESH_BINARY);
    sureFg.convertTo(sureFg, CV_8U);
    
    // Optional: Dilate the sure foreground to merge peaks of elongated objects (like ovals)
    if (markerDilateK > 0)
    {
        Mat elemMarkerDilate = getStructuringElement(MORPH_ELLIPSE, Size(markerDilateK * 2 + 1, markerDilateK * 2 + 1));
        dilate(sureFg, sureFg, elemMarkerDilate);
    }

    // 4. Unknown region
    // Subtract sure foreground from sure background to get ambiguous border regions.
    Mat unknown;
    subtract(sureBg, sureFg, unknown);

    // 5. Marker labelling
    // Label connected foreground components to create integer markers for watershed.
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
    // Run watershed on a flat image using markers; watershed will label boundaries with -1.
    // A flat input prevents texture from biasing the boundary placement.
    Mat flatImg;
    cvtColor(morphMask, flatImg, COLOR_GRAY2BGR);
    watershed(flatImg, markers);

    // 7. Generate separated mask (draw boundaries as black on original mask)
    // Extract watershed boundaries and remove them from the original mask to split touching objects.
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
static void processFrame(const Mat& colorFrame, bool showMode)
{
    // ── Stage 0: Contrast & Saturation enhancement ─────────────────────────────
    // Improve global contrast and boost saturation so color thresholds separate objects better.
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
   if(showMode) imshow("0 - Enhanced", enhanced);



    // ── Stage 1: Gaussian Blur ───────────────────────────────────────────────
    // Reduce high-frequency noise to avoid small spurious mask regions after thresholding.
    int k = max(1, P.blurKsize) * 2 + 1;   // ensure odd, ≥ 3
    Mat blurred;
    GaussianBlur(enhanced, blurred, Size(k, k), 0);
    if(showMode) imshow("1 - Blurred", blurred);


    // ── Stage 2: BGR → HSV ──────────────────────────────────────────────────
    // Convert to HSV: hue/saturation separate chromatic info from intensity for robust masking.
    Mat hsv;
    cvtColor(blurred, hsv, COLOR_BGR2HSV);
    if(showMode) imshow("2 - HSV", hsv);
    
    // ── Stage 3a: Black mask ───────────────────
    // Threshold HSV ranges to produce binary masks for each colour class.
    Mat maskBlack;
    inRange(hsv,
            Scalar(P.blackHMin, P.blackSMin, P.blackVMin),
            Scalar(P.blackHMax, P.blackSMax, P.blackVMax),
            maskBlack);
        if (showMode) imshow("3a - Mask: Black", maskBlack);

    // ── Stage 3b: Dark-yellow mask ───────────────────────────────────────────
    Mat maskDY;
    inRange(hsv,
            Scalar(P.dyHMin, P.dySMin, P.dyVMin),
            Scalar(P.dyHMax, P.dySMax, P.dyVMax),
            maskDY);
        if (showMode) imshow("3b - Mask: DarkYellow", maskDY);

     // ── Stage 3c: Red/Pink mask ──────────────────────────────────────────────
     Mat maskRed1, maskRed2;
     inRange(hsv,
         Scalar(P.redHMin1, P.redSMin, P.redVMin),
         Scalar(P.redHMax1, P.redSMax, P.redVMax),
         maskRed1);
     inRange(hsv,
         Scalar(P.redHMin2, P.redSMin, P.redVMin),
         Scalar(P.redHMax2, P.redSMax, P.redVMax),
         maskRed2);
     Mat maskRed = maskRed1 | maskRed2; // Combine both hue ranges
     if (showMode) imshow("3c - Mask: Red/Pink", maskRed);

    // ── Stage 3d: Brown mask ────────────────────────────────────────────────
    Mat maskBrown;
    inRange(hsv,
            Scalar(P.brHMin, P.brSMin, P.brVMin),
            Scalar(P.brHMax, P.brSMax, P.brVMax),
            maskBrown);
        if (showMode) imshow("3d - Mask: Brown", maskBrown);

    // ── Stage 4: Morphology on individual masks ──────────────────────────────
    // Clean and regularize binary masks: closing fills holes, opening removes small noise.

    // Initialize structuring elements based on trackbar values, making sure they are valid (≥1)
    int ck = max(1, P.morphCloseK);
    int ok = max(1, P.morphOpenK);
    int redCk = max(1, P.redMorphCloseK);
    int redOk = max(1, P.redMorphOpenK);
    Mat elemClose = getStructuringElement(MORPH_ELLIPSE, Size(ck * 2 + 1, ck * 2 + 1));
    Mat elemOpen  = getStructuringElement(MORPH_ELLIPSE, Size(ok * 2 + 1, ok * 2 + 1));
    Mat elemRedClose = getStructuringElement(MORPH_ELLIPSE, Size(redCk * 2 + 1, redCk * 2 + 1));
    Mat elemRedOpen  = getStructuringElement(MORPH_ELLIPSE, Size(redOk * 2 + 1, redOk * 2 + 1));

    // Apply morphology to each mask separately, using the same structuring elements. 
    Mat maskBlackMorphed, maskDYMorphed, maskRedMorphed, maskBrownMorphed;

    morphologyEx(maskBlack, maskBlackMorphed, MORPH_CLOSE, elemClose);
    morphologyEx(maskBlackMorphed, maskBlackMorphed, MORPH_OPEN, elemOpen);

    morphologyEx(maskDY, maskDYMorphed, MORPH_CLOSE, elemClose);
    morphologyEx(maskDYMorphed, maskDYMorphed, MORPH_OPEN, elemOpen);

    morphologyEx(maskRed, maskRedMorphed, MORPH_CLOSE, elemRedClose);
    morphologyEx(maskRedMorphed, maskRedMorphed, MORPH_OPEN, elemRedOpen);

    morphologyEx(maskBrown, maskBrownMorphed, MORPH_CLOSE, elemClose);
    morphologyEx(maskBrownMorphed, maskBrownMorphed, MORPH_OPEN, elemOpen);

    if (showMode)
    {
        imshow("4a - Morphology: Black", maskBlackMorphed);
        imshow("4b - Morphology: Red", maskRedMorphed);
    }

    // ── Stage 5: Distance Transform + Watershed separation ───────────────────
    // Split touching objects using distance peaks and watershed; skip for red masks.
    // Skip watershed for red candy as decided - use morphed mask directly
     Mat maskRedSeparated   = maskRedMorphed.clone();
     Mat maskBlackSeparated = applyWatershed(maskBlackMorphed, P.distThreshPct, P.sureBgDilateK, 0);
     Mat distDYMap;
     Mat maskDYSeparated    = applyWatershed(maskDYMorphed, P.distThreshPct, P.sureBgDilateK, 0, &distDYMap);
     Mat maskBrownSeparated = applyWatershed(maskBrownMorphed, P.distThreshPct, P.sureBgDilateK, 0);

     if (showMode)
     {
         if (!distDYMap.empty())  imshow("4d - Watershed DT (Yellow)", distDYMap);
         imshow("4f - Separated: Black", maskBlackSeparated);
         imshow("4g - Separated: DarkYellow", maskDYSeparated);
         imshow("4h - Separated: Red/Pink", maskRedSeparated);
         imshow("4i - Separated: Brown", maskBrownSeparated);
     }


    // ── Stage 6: findContours per colour mask ────────────────────────────────
    // Extract contours from cleaned/separated masks for later area-based filtering.
    vector<vector<Point>> contoursRed, contoursDY, contoursBlack, contoursBrown;
    findContours(maskRedSeparated,   contoursRed,   RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    findContours(maskDYSeparated,    contoursDY,    RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    findContours(maskBlackSeparated, contoursBlack, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    findContours(maskBrownSeparated, contoursBrown, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

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
    for (size_t i = 0; i < contoursBrown.size(); i++)
        if (contourArea(contoursBrown[i]) >= minA)
            drawContours(contourVis, contoursBrown, (int)i, Scalar(42, 42, 165), 2);
    if (showMode) imshow("6 - Contours", contourVis);

    // ── Stage 7: Size validation + bounding boxes (Red, Yellow, Black, Brown) ─
    // Validate contour area, mark invalid detections, and draw labeled bounding boxes.
    Mat result = enhanced.clone();
    int minValid = max(1, P.minValidArea);

    auto drawDetections = [&](const vector<vector<Point>>& contours, const string& colorTag, bool forceInvalid)
    {
        for (const auto& cnt : contours)
        {
            double area = contourArea(cnt);
            if (area < minA)
                continue;

            bool invalid = forceInvalid || (area < minValid);

            Rect bbox = boundingRect(cnt);

            Scalar boxColor = invalid ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
            rectangle(result, bbox, boxColor, 3);

            string label = colorTag;
            if (forceInvalid) label += " inv:col";
            else if (area < minValid) label += " inv:size";
            else label += " valid";

            if (showMode)
            {
                label += " A=" + to_string(static_cast<int>(area));
            }

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

    drawDetections(contoursRed,   "red",    true);
    drawDetections(contoursDY,    "yellow", false);
    drawDetections(contoursBlack, "black",  false);
    drawDetections(contoursBrown, "brown",  false);

    imshow("8 - Result", result);
}

// ---------------------------------------------------------------------------
// runDebugMode() – cycle through sample images in ./images/
// ---------------------------------------------------------------------------
static void runDebugMode(bool showMode, bool adjustMode)
{
    // Load images from ./images and allow stepping through them; useful for UI tuning.
    vector<String> imagePaths;
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
    if (showMode)
        cout << "  UI mode             : ADJUST (all windows + trackbars)" << endl;
    else
        cout << "  UI mode             : RESULT ONLY (use -a/--adjust for full tuning UI)" << endl;

    createWindows(showMode, adjustMode);

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
        processFrame(current, showMode);
    };

    loadAndProcess();

    while (true)
    {
        int key = waitKey(50);

        if (key == 27 || key == 'q')
            break;

        bool imageChanged = false;

        // Space, right arrow, or 'd' → next
        if (key == ' ' || key == 83 /* right */ || key == 'd')
        {
            idx = (idx + 1) % (int)filtered.size();
            imageChanged = true;
        }
        // Left arrow or 'a' → previous
        else if (key == 81 /* left */ || key == 'a')
        {
            idx = (idx - 1 + (int)filtered.size()) % (int)filtered.size();
            imageChanged = true;
        }

        if (imageChanged)
        {
            loadAndProcess();
        }
        else if (!current.empty())
        {
            // Reprocess on every tick so trackbar changes take effect without a keypress.
            processFrame(current, showMode);
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
    // Entry point: parse flags, initialize SDK/camera or debug loop, then process frames.
    bool debugMode = false;
    bool showMode = false;
    bool adjustMode = false;

    // ── Parse runtime flags ──────────────────────────────────────────────────
    for (int i = 1; i < argc; i++)
    {
        string arg = argv[i];
        //if (arg == "--debug" || arg == "-d") debugMode = true;
        if (arg == "--show" || arg == "-s") showMode = true;
        if (arg == "--adjust" || arg == "-a") adjustMode = true;
    }

    // ── Debug mode: use sample images instead of camera ──────────────────────
    if (debugMode)
    {
        runDebugMode(showMode, adjustMode);
        return 0;
    }

    GX_STATUS    emStatus = GX_STATUS_SUCCESS;
    GX_DEV_HANDLE hDevice = NULL;
    GX_DS_HANDLE  hStream = NULL;

    // ── 1. Init SDK ──────────────────────────────────────────────────────────
    // Initialize the Galaxy SDK library; required before any device/API calls.
    emStatus = GXInitLib();
    if (emStatus != GX_STATUS_SUCCESS) { PrintErrorInfo(emStatus); return -1; }
    cout << "Galaxy SDK version: " << GXGetLibVersion() << endl;

    // ── 2. Enumerate devices ─────────────────────────────────────────────────
    // Discover connected cameras and update the device list for selection.
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
    // Open the first available camera (index 1) for streaming.
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
    // Query whether the camera supplies color (Bayer) or mono frames.
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
    // Obtain the device's data stream handle and payload size for buffer allocation.
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
    // Configure acquisition mode, trigger, buffer count and auto controls.
    GXSetEnumValueByString(hDevice, "AcquisitionMode", "Continuous");
    GXSetEnumValueByString(hDevice, "TriggerMode",     "Off");
    GXSetAcqusitionBufferNumber(hDevice, 5);

    // Enable auto exposure and auto white balance
    GXSetEnumValueByString(hDevice, "ExposureAuto", "Continuous");
    GXSetEnumValueByString(hDevice, "BalanceWhiteAuto", "Continuous");

    // ── 7. Allocate buffers ───────────────────────────────────────────────────
    // Allocate host-side buffers for raw image data and RGB-converted pixels.
    unsigned char* pRaw8Buf = new unsigned char[nPayloadSize];
    unsigned char* pRGBBuf  = new unsigned char[nPayloadSize * 3];

    // ── 8. Create windows & trackbars ────────────────────────────────────────
    // Create visualization windows and trackbars (if running in adjust mode).
    createWindows(showMode, adjustMode);

    // ── 9. Start streaming ────────────────────────────────────────────────────
    // Start the camera stream; after this frames can be dequeued for processing.
    emStatus = GXStreamOn(hDevice);
    if (emStatus != GX_STATUS_SUCCESS)
    {
        PrintErrorInfo(emStatus);
        delete[] pRaw8Buf; delete[] pRGBBuf;
        GXCloseDevice(hDevice); GXCloseLib(); return -1;
    }
    if (showMode)
        cout << "UI mode: ADJUST (all windows + trackbars)." << endl;
    else
        cout << "UI mode: RESULT ONLY (use -a/--adjust for full tuning UI)." << endl;
    cout << "Streaming... Press ESC or 'q' to quit." << endl;

    // ── 10. Main loop ─────────────────────────────────────────────────────────
    // Acquire frames, convert to BGR, call `processFrame()` and handle user input.
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

            processFrame(colorFrame, showMode);
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
    // Stop streaming, free buffers and close device/library handles cleanly.
    destroyAllWindows();
    GXStreamOff(hDevice);
    delete[] pRaw8Buf;
    delete[] pRGBBuf;
    GXCloseDevice(hDevice);
    GXCloseLib();
    cout << "Program ended." << endl;
    return 0;
}
