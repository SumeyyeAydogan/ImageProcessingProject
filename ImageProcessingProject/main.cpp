#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>  // Kalibrasyon için gerekli
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

// 1. En yakın komşu ile yeniden boyutlandırma

Mat resizeImage(const Mat& src, int newWidth, int newHeight) {
    Mat dst(newHeight, newWidth, src.type());
    double scaleX = static_cast<double>(src.cols) / newWidth;
    double scaleY = static_cast<double>(src.rows) / newHeight;
    for (int y = 0; y < newHeight; y++) {
        int srcY = min(static_cast<int>(y * scaleY), src.rows - 1);
        for (int x = 0; x < newWidth; x++) {
            int srcX = min(static_cast<int>(x * scaleX), src.cols - 1);
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
    Mat gray(src.rows, src.cols, CV_8UC1);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            Vec3b c = src.at<Vec3b>(y, x);
            gray.at<uchar>(y, x) =
                static_cast<uchar>(0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2]);
        }
    }
    return gray;
}

// 3. Gauss çekirdeği oluşturma
vector<vector<double>> createGaussianKernel(int ksize, double sigma) {
    int half = ksize / 2;
    vector<vector<double>> kernel(ksize, vector<double>(ksize));
    double sum = 0.0;
    for (int i = -half; i <= half; i++) {
        for (int j = -half; j <= half; j++) {
            double e = exp(-(i * i + j * j) / (2 * sigma * sigma))
                / (2 * M_PI * sigma * sigma);
            kernel[i + half][j + half] = e;
            sum += e;
        }
    }
    for (int i = 0; i < ksize; i++)
        for (int j = 0; j < ksize; j++)
            kernel[i][j] /= sum;
    return kernel;
}

// 4. Gaussian bulanıklık
Mat applyGaussianBlur(const Mat& src, int ksize, double sigma) {
    Mat dst = src.clone();
    auto kernel = createGaussianKernel(ksize, sigma);
    int half = ksize / 2;
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            double acc = 0.0;
            for (int i = -half; i <= half; i++) {
                for (int j = -half; j <= half; j++) {
                    int yy = min(max(y + i, 0), src.rows - 1);
                    int xx = min(max(x + j, 0), src.cols - 1);
                    acc += kernel[i + half][j + half] * src.at<uchar>(yy, xx);
                }
            }
            dst.at<uchar>(y, x) = static_cast<uchar>(acc);
        }
    }
    return dst;
}

// 5.1. Sobel ile gradyan & yön hesaplama
void computeGradient(const Mat& src, Mat& mag, Mat& dir) {
    int rows = src.rows, cols = src.cols;
    mag = Mat::zeros(rows, cols, CV_64F);
    dir = Mat::zeros(rows, cols, CV_64F);
    int Gx[3][3] = { {1,0,-1},{2,0,-2},{1,0,-1} };
    int Gy[3][3] = { {1,2,1},{ 0, 0, 0},{-1,-2,-1} };
    for (int y = 1; y < rows - 1; y++) {
        for (int x = 1; x < cols - 1; x++) {
            double sx = 0, sy = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    double p = src.at<uchar>(y + i, x + j);
                    sx += Gx[i + 1][j + 1] * p;
                    sy += Gy[i + 1][j + 1] * p;
                }
            }
            mag.at<double>(y, x) = hypot(sx, sy);
            double a = atan2(sy, sx) * 180.0 / M_PI;
            if (a < 0) a += 180;
            dir.at<double>(y, x) = a;
        }
    }
}

// 5.2. Non-maximum suppression
void nonMaxSuppression(const Mat& mag, const Mat& dir, Mat& out) {
    int rows = mag.rows, cols = mag.cols;
    out = Mat::zeros(rows, cols, CV_64F);
    for (int y = 1; y < rows - 1; y++) {
        for (int x = 1; x < cols - 1; x++) {
            double angle = dir.at<double>(y, x);
            double m = mag.at<double>(y, x), m1, m2;
            if ((angle < 22.5) || (angle >= 157.5)) {
                m1 = mag.at<double>(y, x + 1);
                m2 = mag.at<double>(y, x - 1);
            }
            else if (angle < 67.5) {
                m1 = mag.at<double>(y - 1, x + 1);
                m2 = mag.at<double>(y + 1, x - 1);
            }
            else if (angle < 112.5) {
                m1 = mag.at<double>(y - 1, x);
                m2 = mag.at<double>(y + 1, x);
            }
            else {
                m1 = mag.at<double>(y - 1, x - 1);
                m2 = mag.at<double>(y + 1, x + 1);
            }
            if (m >= m1 && m >= m2)
                out.at<double>(y, x) = m;
        }
    }
}

// 5.3. Çift eşik & Hysteresis
void applyHysteresis(const Mat& nonMax, Mat& edges, Mat& weak, double lowT, double highT) {
    int rows = nonMax.rows, cols = nonMax.cols;
    edges = Mat::zeros(rows, cols, CV_8U);
    weak = Mat::zeros(rows, cols, CV_8U);
    // Eşikleme
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            double v = nonMax.at<double>(y, x);
            if (v >= highT)       edges.at<uchar>(y, x) = 255;
            else if (v >= lowT)   weak.at<uchar>(y, x) = 128;
        }
    }
    // Hysteresis
    bool changed;
    do {
        changed = false;
        for (int y = 1; y < rows - 1; y++) {
            for (int x = 1; x < cols - 1; x++) {
                if (weak.at<uchar>(y, x) == 128) {
                    // 8-komşulukta güçlü kenar ara
                    bool connect = false;
                    for (int i = -1; i <= 1 && !connect; i++)
                        for (int j = -1; j <= 1; j++)
                            if (edges.at<uchar>(y + i, x + j) == 255)
                                connect = true;
                    if (connect) {
                        edges.at<uchar>(y, x) = 255;
                        weak.at<uchar>(y, x) = 0;
                        changed = true;
                    }
                }
            }
        }
    } while (changed);
}

// ---- Yeni Line Detection Kalibrasyon Kısmı ----

// Çizgi yapısını temsil eden veri yapısı
struct Line {
    double rho;         // rho değeri
    double theta;       // theta değeri (radyan)
    Point p1, p2;      // doğrunun uç noktaları
    double length;      // doğrunun uzunluğu
    double angle;       // açı değeri (derece)

    Line(double r, double t) : rho(r), theta(t) {
        // p1 ve p2'yi hesapla
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;

        p1.x = cvRound(x0 + 1000 * (-b));
        p1.y = cvRound(y0 + 1000 * (a));
        p2.x = cvRound(x0 - 1000 * (-b));
        p2.y = cvRound(y0 - 1000 * (a));

        // Uzunluk ve açıyı hesapla
        length = hypot(p2.x - p1.x, p2.y - p1.y);
        angle = theta * 180.0 / M_PI;
        if (angle > 90) angle = 180 - angle;  // 0-90 aralığında normalize et
    }

    // Doğruyu görüntü sınırlarına kırp
    void clipToImageBounds(int width, int height) {
        vector<Point> intersections;

        // y = 0 (üst kenar)
        if (p1.y != p2.y) {  // yatay değilse
            double x = p1.x + (0 - p1.y) * (p2.x - p1.x) / (p2.y - p1.y);
            if (x >= 0 && x < width)
                intersections.push_back(Point(cvRound(x), 0));
        }

        // y = height-1 (alt kenar)
        if (p1.y != p2.y) {  // yatay değilse
            double x = p1.x + ((height - 1) - p1.y) * (p2.x - p1.x) / (p2.y - p1.y);
            if (x >= 0 && x < width)
                intersections.push_back(Point(cvRound(x), height - 1));
        }

        // x = 0 (sol kenar)
        if (p1.x != p2.x) {  // dikey değilse
            double y = p1.y + (0 - p1.x) * (p2.y - p1.y) / (p2.x - p1.x);
            if (y >= 0 && y < height)
                intersections.push_back(Point(0, cvRound(y)));
        }

        // x = width-1 (sağ kenar)
        if (p1.x != p2.x) {  // dikey değilse
            double y = p1.y + ((width - 1) - p1.x) * (p2.y - p1.y) / (p2.x - p1.x);
            if (y >= 0 && y < height)
                intersections.push_back(Point(width - 1, cvRound(y)));
        }

        // Kesişim noktalarını kontrol et
        if (intersections.size() >= 2) {
            p1 = intersections[0];
            p2 = intersections[1];
            length = hypot(p2.x - p1.x, p2.y - p1.y);
        }
    }
};

// İki doğrunun benzerliğini kontrol et
bool areSimilarLines(const Line& l1, const Line& l2, double rhoThresh, double thetaThresh) {
    double rhoDiff = abs(l1.rho - l2.rho);
    double thetaDiff = abs(l1.theta - l2.theta);

    // Theta'nın döngüsel olduğunu hesaba kat
    if (thetaDiff > M_PI / 2)
        thetaDiff = M_PI - thetaDiff;

    return (rhoDiff < rhoThresh && thetaDiff < thetaThresh * M_PI / 180.0);
}

// Kalibrasyon edilebilir Hough Line Detection
vector<Line> calibratedHoughLines(const Mat& edges,
    int threshold,
    int minLineLength,
    int angleFilter = -1,
    int angleRange = 10,
    int rhoMergeThresh = 10,
    int thetaMergeThresh = 10) {
    int rows = edges.rows;
    int cols = edges.cols;
    int diagLength = cvRound(sqrt(rows * rows + cols * cols));

    // Akümülatör matrisi (rho, theta)
    int numRho = diagLength * 2;
    int numTheta = 180;  // 1 derece çözünürlük
    vector<vector<int>> accumulator(numRho, vector<int>(numTheta, 0));

    // Akümülatörü kenar noktaları ile doldur
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (edges.at<uchar>(y, x) == 255) {  // kenar noktası ise
                for (int t = 0; t < numTheta; t++) {
                    double theta = t * M_PI / 180.0;  // radyan cinsinden
                    int rho = cvRound(x * cos(theta) + y * sin(theta)) + diagLength;
                    accumulator[rho][t]++;
                }
            }
        }
    }

    // Yerel maksimumları bul
    vector<pair<int, int>> peaks;  // (rho, theta) indeks çiftleri
    for (int r = 1; r < numRho - 1; r++) {
        for (int t = 1; t < numTheta - 1; t++) {
            int votes = accumulator[r][t];

            if (votes > threshold) {
                // 3x3 çevresinde yerel maksimum mu kontrol et
                bool isLocalMax = true;
                for (int dr = -1; dr <= 1 && isLocalMax; dr++) {
                    for (int dt = -1; dt <= 1 && isLocalMax; dt++) {
                        if (dr == 0 && dt == 0) continue;

                        if (accumulator[r + dr][t + dt] > votes) {
                            isLocalMax = false;
                        }
                    }
                }

                // Açı filtrelemesi yap
                if (isLocalMax) {
                    int angle = t;
                    if (angle > 90) angle = 180 - angle;

                    if (angleFilter == -1 ||
                        abs(angle - angleFilter) <= angleRange ||
                        abs(angle - (180 - angleFilter)) <= angleRange) {
                        peaks.push_back(make_pair(r, t));
                    }
                }
            }
        }
    }

    // Line nesnelerine dönüştür
    vector<Line> lines;
    for (const auto& peak : peaks) {
        int r = peak.first;
        int t = peak.second;

        double rho = r - diagLength;
        double theta = t * M_PI / 180.0;

        Line line(rho, theta);
        line.clipToImageBounds(cols, rows);

        // Minimum uzunluk filtresi
        if (line.length >= minLineLength) {
            lines.push_back(line);
        }
    }

    // Benzer çizgileri birleştir
    if (!lines.empty()) {
        vector<Line> filteredLines;
        vector<bool> merged(lines.size(), false);

        for (size_t i = 0; i < lines.size(); i++) {
            if (merged[i]) continue;

            vector<int> group;
            group.push_back(i);

            for (size_t j = i + 1; j < lines.size(); j++) {
                if (!merged[j] && areSimilarLines(lines[i], lines[j], rhoMergeThresh, thetaMergeThresh)) {
                    group.push_back(j);
                    merged[j] = true;
                }
            }

            if (group.size() == 1) {
                filteredLines.push_back(lines[i]);
            }
            else {
                // Ortalama rho ve theta hesapla
                double avgRho = 0, avgTheta = 0;
                for (int idx : group) {
                    avgRho += lines[idx].rho;
                    avgTheta += lines[idx].theta;
                }
                avgRho /= group.size();
                avgTheta /= group.size();

                Line mergedLine(avgRho, avgTheta);
                mergedLine.clipToImageBounds(cols, rows);
                filteredLines.push_back(mergedLine);
            }
        }

        lines = filteredLines;
    }

    return lines;
}

// Çizgileri çiz
void drawCustomLines(Mat& img, const vector<Line>& lines, const Scalar& color = Scalar(0, 0, 255), int thickness = 2) {
    for (const auto& line : lines) {
        cv::line(img, line.p1, line.p2, color, thickness, LINE_AA);

        // İsteğe bağlı: Açı ve uzunluk bilgisini göster
        Point mid((line.p1.x + line.p2.x) / 2, (line.p1.y + line.p2.y) / 2);
        string info = "L: " + to_string(int(line.length)) + "px, " + to_string(int(line.angle)) + "°";
        putText(img, info, Point(mid.x - 40, mid.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
    }
}



// Kalibrasyon için gerekli değişkenler ve callback fonksiyonları
struct LineKalibrasyon {
    Mat origImg;
    Mat edges;
    Mat measureImg;  // Ölçüm görüntüsü
    Mat measureEdges;  // Ölçüm görüntüsünün kenarları

    // Kalibrasyon değerleri
    double pixelToUnit = 0.01; // Varsayılan değer
    string unitName = "cm";    // Ölçü birimi cm

    // Kalibrasyon tahtası için değişkenler
    vector<Point2f> boardCorners;  // Kalibrasyon tahtasının köşe noktaları
    bool isCalibrating = true;     // Kalibrasyon modunda mı?
    int cornerCount = 0;           // Seçilen köşe sayısı

    // Mesafe ölçümü için değişkenler
    bool isFirstPoint = true;
    Point pt1, pt2;

    // Kalibrasyon tahtası boyutları
    const double boardSize = 40.0;  // cm cinsinden (40x40 cm)

    // Kalibrasyon değerlerini hesapla
    void calculateCalibration() {
        if (boardCorners.size() == 4) {  // 4 köşe seçildiyse
            // Kalibrasyon tahtasının piksel boyutlarını hesapla
            double pixelWidth = norm(boardCorners[1] - boardCorners[0]);  // Üst kenar
            double pixelHeight = norm(boardCorners[3] - boardCorners[0]); // Sol kenar

            // Piksel/cm oranını hesapla
            double pixelsPerCmWidth = pixelWidth / boardSize;
            double pixelsPerCmHeight = pixelHeight / boardSize;

            // Ortalama değeri al
            double pixelsPerCm = (pixelsPerCmWidth + pixelsPerCmHeight) / 2.0;

            // Kalibrasyon değerini ayarla
            pixelToUnit = 1.0 / pixelsPerCm;

            cout << "Kalibrasyon: 1 cm = " << pixelsPerCm << " piksel" << endl;
            cout << "Yatay: 1 cm = " << pixelsPerCmWidth << " piksel" << endl;
            cout << "Dikey: 1 cm = " << pixelsPerCmHeight << " piksel" << endl;

            // Kalibrasyon modunu kapat ve ölçüm görüntüsüne geç
            isCalibrating = false;
            origImg = measureImg.clone();
            edges = measureEdges;

            cout << "\nKalibrasyon tamamlandı! Şimdi kalemin uç noktalarını seçin." << endl;
        }
    }

    void update() {
        Mat result = origImg.clone();

        if (isCalibrating) {
            // Kalibrasyon tahtası köşelerini çiz
            for (size_t i = 0; i < boardCorners.size(); i++) {
                circle(result, boardCorners[i], 5, Scalar(0, 0, 255), -1);
                putText(result, to_string(i + 1), boardCorners[i], FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            }

            // Köşeler arası çizgileri çiz
            if (boardCorners.size() >= 2) {
                for (size_t i = 0; i < boardCorners.size() - 1; i++) {
                    line(result, boardCorners[i], boardCorners[i + 1], Scalar(0, 0, 255), 2);
                }
            }

            string helpStr = "Kalibrasyon tahtasının köşelerini seçin (sol tık)";
            putText(result, helpStr, Point(20, result.rows - 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 1);
        }
        else {
            // Mesafe ölçümü
            if (!isFirstPoint) {
                circle(result, pt1, 5, Scalar(0, 255, 0), -1);

                if (pt2.x > 0 && pt2.y > 0) {
                    circle(result, pt2, 5, Scalar(0, 255, 0), -1);
                    line(result, pt1, pt2, Scalar(0, 255, 0), 2);

                    double pixelDistance = hypot(pt2.x - pt1.x, pt2.y - pt1.y);
                    double realDistance = pixelDistance * pixelToUnit;

                    string distStr = "Mesafe: " + to_string(realDistance).substr(0, to_string(realDistance).find('.') + 3) + " " + unitName;
                    Point textPos((pt1.x + pt2.x) / 2 - 100, (pt1.y + pt2.y) / 2 - 20);
                    putText(result, distStr, textPos, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
                }
            }

            string helpStr = "Kalemin uç noktalarını seçin (sol tık), Temizle: 'c'";
            putText(result, helpStr, Point(20, result.rows - 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 1);
        }

        imshow("Mesafe Ölçüm", result);
    }

    void resetMeasurement() {
        isFirstPoint = true;
        pt1 = Point(0, 0);
        pt2 = Point(0, 0);
        update();
    }

    void resetCalibration() {
        boardCorners.clear();
        isCalibrating = true;
        origImg = measureImg.clone();  // Kalibrasyon görüntüsüne geri dön
        edges = measureEdges;
        update();
    }
};

// Global kalibrasyon nesnesi
LineKalibrasyon lineKalibrator;

// Fare tıklama callback fonksiyonu
void onMouseCallback(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        if (lineKalibrator.isCalibrating) {
            // Kalibrasyon modunda köşe seçimi
            if (lineKalibrator.boardCorners.size() < 4) {
                lineKalibrator.boardCorners.push_back(Point2f(x, y));
                if (lineKalibrator.boardCorners.size() == 4) {
                    lineKalibrator.calculateCalibration();
                }
            }
        }
        else {
            // Ölçüm modunda nokta seçimi
            if (lineKalibrator.isFirstPoint) {
                lineKalibrator.pt1 = Point(x, y);
                lineKalibrator.isFirstPoint = false;
            }
            else {
                lineKalibrator.pt2 = Point(x, y);
                lineKalibrator.isFirstPoint = true;
            }
        }
        lineKalibrator.update();
    }
}

// Klavye callback fonksiyonu
void onKeyboardInput(int key) {
    switch (key) {
    case 'c': // 'c' tuşuna basılınca mesafe ölçümünü temizle
    case 'C':
        lineKalibrator.resetMeasurement();
        break;
    }
    lineKalibrator.update();
}

int main() {
    // Kalibrasyon görüntüsünü yükle
    string calibFname = "D:\\Dersler\\projects\\ImageProcessingProject\\checkerboard.jpeg";
    Mat calibImg = imread(calibFname);
    if (calibImg.empty()) {
        cerr << "Kalibrasyon resmi yüklenemedi: " << calibFname << endl;
        return -1;
    }

    // Ölçüm görüntüsünü yükle
    string measureFname = "D:\\Dersler\\projects\\ImageProcessingProject\\pencil.jpeg";
    Mat measureImg = imread(measureFname);
    if (measureImg.empty()) {
        cerr << "Ölçüm resmi yüklenemedi: " << measureFname << endl;
        return -1;
    }

    // Ön işlemler
    Mat resizedCalib = resizeImage(calibImg, 800, 600);
    Mat resizedMeasure = resizeImage(measureImg, 800, 600);

    Mat grayCalib = convertToGrayscale(resizedCalib);
    Mat grayMeasure = convertToGrayscale(resizedMeasure);

    Mat blurredCalib = applyGaussianBlur(grayCalib, 5, 1.5);
    Mat blurredMeasure = applyGaussianBlur(grayMeasure, 5, 1.5);

    // Kenar tespiti
    Mat edgesCalib, weakCalib;
    Mat edgesMeasure, weakMeasure;

    Mat gradMagCalib, gradDirCalib;
    Mat gradMagMeasure, gradDirMeasure;

    computeGradient(blurredCalib, gradMagCalib, gradDirCalib);
    computeGradient(blurredMeasure, gradMagMeasure, gradDirMeasure);

    Mat nonMaxCalib, nonMaxMeasure;
    nonMaxSuppression(gradMagCalib, gradDirCalib, nonMaxCalib);
    nonMaxSuppression(gradMagMeasure, gradDirMeasure, nonMaxMeasure);

    applyHysteresis(nonMaxCalib, edgesCalib, weakCalib, 50, 100);
    applyHysteresis(nonMaxMeasure, edgesMeasure, weakMeasure, 50, 100);

    // Mesafe ölçüm penceresi
    namedWindow("Mesafe Ölçüm", WINDOW_NORMAL);
    resizeWindow("Mesafe Ölçüm", 1000, 800);

    // Kalibrasyon değişkenlerini ayarla
    lineKalibrator.origImg = resizedCalib.clone();
    lineKalibrator.edges = edgesCalib;
    lineKalibrator.measureImg = resizedMeasure.clone();
    lineKalibrator.measureEdges = edgesMeasure;

    // Fare olayları için callback
    setMouseCallback("Mesafe Ölçüm", onMouseCallback);

    // İlk güncellemeyi yap
    lineKalibrator.update();

    cout << "\nKullanım:\n"
        << "1. Kalibrasyon tahtasının 4 köşesini seçin (sol tık)\n"
        << "2. Kalibrasyon tamamlandığında otomatik olarak kalem fotoğrafına geçilecek\n"
        << "3. Kalemin uç noktalarını seçin\n"
        << "c - Ölçümü temizle\n"
        << "r - Kalibrasyonu sıfırla\n"
        << "ESC - Çıkış\n";

    int key;
    while ((key = waitKey(0)) != 27) {
        if (key == 'c' || key == 'C') {
            lineKalibrator.resetMeasurement();
        }
        else if (key == 'r' || key == 'R') {
            lineKalibrator.resetCalibration();
        }
    }

    destroyAllWindows();
    return 0;
}