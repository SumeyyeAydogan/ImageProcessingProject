#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

// 1. En yakın komşu ile yeniden boyutlandırma
Mat resizeImage(const Mat& src, int newWidth, int newHeight) {
    Mat dst(newHeight, newWidth, src.type());
    double scaleX = double(src.cols) / newWidth;
    double scaleY = double(src.rows) / newHeight;
    for (int y = 0; y < newHeight; ++y) {
        int srcY = min(int(y * scaleY), src.rows - 1);
        for (int x = 0; x < newWidth; ++x) {
            int srcX = min(int(x * scaleX), src.cols - 1);
            if (src.channels() == 3)
                dst.at<Vec3b>(y, x) = src.at<Vec3b>(srcY, srcX);
            else
                dst.at<uchar>(y, x) = src.at<uchar>(srcY, srcX);
        }
    }
    return dst;
}

// 2. BGR → Gri tonlama
Mat convertToGrayscale(const Mat& src) {
    Mat gray(src.size(), CV_8UC1);
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            Vec3b c = src.at<Vec3b>(y, x);
            gray.at<uchar>(y, x) = uchar(0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2]);
        }
    }
    return gray;
}

// 3. Gauss çekirdeği oluşturma
vector<vector<double>> createGaussianKernel(int ksize, double sigma) {
    int half = ksize / 2;
    vector<vector<double>> kernel(ksize, vector<double>(ksize));
    double sum = 0;
    for (int i = -half; i <= half; ++i) for (int j = -half; j <= half; ++j) {
        double v = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
        kernel[i + half][j + half] = v;
        sum += v;
    }
    for (int i = 0; i < ksize; ++i) for (int j = 0; j < ksize; ++j) kernel[i][j] /= sum;
    return kernel;
}

// 4. Gaussian bulanıklık
// Gaussian Blur uygulamasý (gri tonlamalý görüntü üzerinde)
Mat applyGaussianBlur(const Mat& src, int ksize, double sigma) {
    Mat dst = src.clone();
    vector<vector<double>> kernel = createGaussianKernel(ksize, sigma);
    int halfSize = ksize / 2;
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            double sum = 0.0;
            for (int i = -halfSize; i <= halfSize; i++) {
                for (int j = -halfSize; j <= halfSize; j++) {
                    int yy = min(max(y + i, 0), src.rows - 1);
                    int xx = min(max(x + j, 0), src.cols - 1);
                    sum += kernel[i + halfSize][j + halfSize] * src.at<uchar>(yy, xx);
                }
            }
            dst.at<uchar>(y, x) = static_cast<uchar>(sum);
        }
    }
    return dst;
}
// 5.1. Sobel ile gradyan & yön hesaplama
void computeGradient(const Mat& src, Mat& mag, Mat& dir) {
    mag = Mat::zeros(src.size(), CV_64F);
    dir = Mat::zeros(src.size(), CV_64F);
    int Gx[3][3] = { {1,0,-1},{2,0,-2},{1,0,-1} };
    int Gy[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };
    for (int y = 1; y < src.rows - 1; ++y) for (int x = 1; x < src.cols - 1; ++x) {
        double sx = 0, sy = 0;
        for (int i = -1; i <= 1; ++i) for (int j = -1; j <= 1; ++j) {
            double p = src.at<uchar>(y + i, x + j);
            sx += Gx[i + 1][j + 1] * p;
            sy += Gy[i + 1][j + 1] * p;
        }
        mag.at<double>(y, x) = hypot(sx, sy);
        double a = atan2(sy, sx) * 180.0 / M_PI;
        if (a < 0) a += 180;
        dir.at<double>(y, x) = a;
    }
}

// 5.2. Non-maximum suppression
void nonMaxSuppression(const Mat& mag, const Mat& dir, Mat& out) {
    out = Mat::zeros(mag.size(), CV_64F);
    for (int y = 1; y < mag.rows - 1; ++y) for (int x = 1; x < mag.cols - 1; ++x) {
        double angle = dir.at<double>(y, x), m = mag.at<double>(y, x), m1, m2;
        if (angle < 22.5 || angle >= 157.5) { m1 = mag.at<double>(y, x + 1); m2 = mag.at<double>(y, x - 1); }
        else if (angle < 67.5) { m1 = mag.at<double>(y - 1, x + 1); m2 = mag.at<double>(y + 1, x - 1); }
        else if (angle < 112.5) { m1 = mag.at<double>(y - 1, x); m2 = mag.at<double>(y + 1, x); }
        else { m1 = mag.at<double>(y - 1, x - 1); m2 = mag.at<double>(y + 1, x + 1); }
        if (m >= m1 && m >= m2) out.at<double>(y, x) = m;
    }
}

// 5.3. Çift eşik & Hysteresis
void applyHysteresis(const Mat& nonMax, Mat& edges, Mat& weak, double lowT, double highT) {
    edges = Mat::zeros(nonMax.size(), CV_8U);
    weak = Mat::zeros(nonMax.size(), CV_8U);
    for (int y = 0; y < nonMax.rows; ++y) for (int x = 0; x < nonMax.cols; ++x) {
        double v = nonMax.at<double>(y, x);
        if (v >= highT) edges.at<uchar>(y, x) = 255;
        else if (v >= lowT) weak.at<uchar>(y, x) = 128;
    }
    bool changed;
    do {
        changed = false;
        for (int y = 1; y < nonMax.rows - 1; ++y) for (int x = 1; x < nonMax.cols - 1; ++x) {
            if (weak.at<uchar>(y, x) == 128) {
                bool conn = false;
                for (int i = -1; i <= 1 && !conn; ++i) for (int j = -1; j <= 1; ++j)
                    if (edges.at<uchar>(y + i, x + j) == 255) conn = true;
                if (conn) { edges.at<uchar>(y, x) = 255; weak.at<uchar>(y, x) = 0; changed = true; }
            }
        }
    } while (changed);
}

// Manuel Hough Line
vector<pair<double, double>> detectLines(const Mat& edges, int thresh) {
    int R = edges.rows, C = edges.cols;
    int D = int(hypot(R, C));
    int nrho = 2 * D, ntheta = 180;
    vector<vector<int>> acc(nrho, vector<int>(ntheta));
    for (int y = 0; y < R; ++y) for (int x = 0; x < C; ++x) if (edges.at<uchar>(y, x) == 255)
        for (int t = 0; t < ntheta; ++t) { double th = t * M_PI / 180.0; int r = cvRound(x * cos(th) + y * sin(th)) + D; acc[r][t]++; }
    vector<pair<double, double>> lines;
    for (int r = 0; r < nrho; ++r) for (int t = 0; t < ntheta; ++t)
        if (acc[r][t] > thresh) lines.emplace_back(r - D, t * M_PI / 180.0);
    return lines;
}

// Manuel Hough Circle
vector<Vec3f> detectCircles(const Mat& edges, int minR, int maxR, int thresh) {
    int R = edges.rows, C = edges.cols, nr = maxR - minR + 1;
    vector<vector<vector<int>>> acc(R, vector<vector<int>>(C, vector<int>(nr)));
    for (int y = 0; y < R; ++y) for (int x = 0; x < C; ++x) if (edges.at<uchar>(y, x) == 255)
        for (int r = minR; r <= maxR; ++r) {
            int ri = r - minR;
            for (int t = 0; t < 360; ++t) {
                double th = t * M_PI / 180.0;
                int a = cvRound(x - r * cos(th)), b = cvRound(y - r * sin(th));
                if (a >= 0 && a < C && b >= 0 && b < R) acc[b][a][ri]++;
            }
        }
    vector<Vec3f> circles;
    for (int y = 0; y < R; ++y) for (int x = 0; x < C; ++x) for (int ri = 0; ri < nr; ++ri)
        if (acc[y][x][ri] > thresh) circles.emplace_back(float(x), float(y), float(minR + ri));
    return circles;
}

void drawLines(Mat& img, const vector<pair<double, double>>& lines) {
    for (auto& l : lines) {
        double rho = l.first, th = l.second;
        double a = cos(th), b = sin(th), x0 = a * rho, y0 = b * rho;
        Point p1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
        Point p2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
        line(img, p1, p2, Scalar(0, 0, 255), 2);
    }
}

void drawCircles(Mat& img, const vector<Vec3f>& circles) {
    for (auto& c : circles) {
        Point center(cvRound(c[0]), cvRound(c[1])); int r = cvRound(c[2]);
        circle(img, center, r, Scalar(0, 255, 0), 2);
    }
}

// --- Line Kalibrasyon Sınıfı w/ Trackbar & Measure Mode ---
struct LineKalibrasyonGUI {
    Mat origImg, edges;
    int threshold = 100, minLen = 50, angleFilt = -1, angleRange = 10, rhoMerge = 10, thetaMerge = 10;
    bool measureMode = false, firstPt = true;
    Point pt1{ -1,-1 }, pt2{ -1,-1 };
    double realPerPixel = 1.0; string unit = "unit";
    double pixelDist(const Point& a, const Point& b) { return hypot(double(b.x - a.x), double(b.y - a.y)); }
    void reset() { firstPt = true; pt1 = pt2 = Point(-1, -1); }
    void update() {
        Mat disp = origImg.clone();
        if (measureMode) {
            if (pt1.x >= 0) circle(disp, pt1, 5, Scalar(0, 255, 0), -1);
            if (pt2.x >= 0) circle(disp, pt2, 5, Scalar(0, 255, 0), -1);
            if (pt1.x >= 0 && pt2.x >= 0) {
                line(disp, pt1, pt2, Scalar(0, 255, 0), 2);
                double dp = pixelDist(pt1, pt2), dr = dp * realPerPixel;
                char buf[64]; sprintf_s(buf, "%.1f px = %.2f %s", dp, dr, unit.c_str());
                Point m((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2 - 10);
                putText(disp, buf, m, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
            }
        }
        else {
            auto lines = detectLines(edges, threshold);
            drawLines(disp, lines);
        }
        string mode = measureMode ? "MODE: MEASURE" : "MODE: LINE";
        putText(disp, mode, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        imshow("Line Kalibrasyon", disp);
    }
};

LineKalibrasyonGUI kalGUI;

void onTrk(int v, void*) { kalGUI.threshold = v; kalGUI.update(); }
void onMinL(int v, void*) { kalGUI.minLen = v; kalGUI.update(); }
void onAngF(int v, void*) { kalGUI.angleFilt = v; kalGUI.update(); }
void onAngR(int v, void*) { kalGUI.angleRange = v; kalGUI.update(); }
void onRhoM(int v, void*) { kalGUI.rhoMerge = v; kalGUI.update(); }
void onTheM(int v, void*) { kalGUI.thetaMerge = v; kalGUI.update(); }

void onMouseCal(int ev, int x, int y, int, void*) {
    if (!kalGUI.measureMode || ev != EVENT_LBUTTONDOWN) return;
    if (kalGUI.firstPt) kalGUI.pt1 = Point(x, y), kalGUI.firstPt = false;
    else kalGUI.pt2 = Point(x, y), kalGUI.firstPt = true;
    kalGUI.update();
}

int main() {
    cout << "1: Line Detection\n2: Circle Detection\n3: Line Kalibrasyon\nSelect: ";
    int ch; cin >> ch;
    string fname = (ch == 2) ? "D:\\Dersler\\projects\\ImageProcessingProject\\para.jpg" : "D:\\Dersler\\projects\\ImageProcessingProject\\tahta.jpeg";
    Mat img = imread(fname);
    if (img.empty()) { cerr << "Can't load" << fname; return -1; }
    Mat resized = resizeImage(img, 800, 600);
    Mat gray = convertToGrayscale(resized);
    Mat blurred = applyGaussianBlur(gray, 5, 1.5);
    Mat mag, dir, nonMax, edges, weak;
    computeGradient(blurred, mag, dir);
    nonMaxSuppression(mag, dir, nonMax);
    applyHysteresis(nonMax, edges, weak, 50, 100);

    if (ch == 3) {
        kalGUI.origImg = resized; kalGUI.edges = edges;
        namedWindow("Line Kalibrasyon", WINDOW_NORMAL);
        createTrackbar("Threshold", "Line Kalibrasyon", &kalGUI.threshold, 300, onTrk);
        createTrackbar("Min Length", "Line Kalibrasyon", &kalGUI.minLen, 300, onMinL);
        createTrackbar("Angle Filter", "Line Kalibrasyon", &kalGUI.angleFilt, 180, onAngF);
        createTrackbar("Angle Range", "Line Kalibrasyon", &kalGUI.angleRange, 90, onAngR);
        createTrackbar("Rho Merge", "Line Kalibrasyon", &kalGUI.rhoMerge, 100, onRhoM);
        createTrackbar("Theta Merge", "Line Kalibrasyon", &kalGUI.thetaMerge, 100, onTheM);
        setMouseCallback("Line Kalibrasyon", onMouseCal);
        kalGUI.update();
        while (true) {
            int key = waitKey(0);
            if (key == 27) break;
            if (key == 'm' || key == 'M') { kalGUI.measureMode = !kalGUI.measureMode; kalGUI.reset(); kalGUI.update(); }
            if (key == 'c' || key == 'C') { kalGUI.reset(); kalGUI.update(); }
            if (key == 'k' || key == 'K') {
                cout << "Enter real units per pixel: "; double u; cin >> u;
                cout << "Enter unit name: "; string s; cin >> s;
                kalGUI.realPerPixel = u; kalGUI.unit = s; kalGUI.update();
            }
        }
        destroyAllWindows();
        return 0;
    }
    // Line/Circle Detection
    namedWindow("Edges", WINDOW_AUTOSIZE); imshow("Edges", edges);
    if (ch == 1) { auto lines = detectLines(edges, 180); drawLines(resized, lines); imshow("Detected Lines", resized); }
    if (ch == 2) { auto circles = detectCircles(edges, 20, 100, 120); drawCircles(resized, circles); imshow("Detected Circles", resized); }
    waitKey(0);
    destroyAllWindows();
    return 0;
}
