#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


using namespace cv;
using namespace std;

// ---- Önceki fonksiyonlar (resize, gri, blur, canny aşamaları) ----

// 1. Nearest‐neighbor resize
Mat resizeImage(const Mat& src, int newW, int newH) {
    Mat dst(newH, newW, src.type());
    double sx = double(src.cols) / newW,
        sy = double(src.rows) / newH;
    for (int y = 0; y < newH; y++) {
        int syi = min(int(y * sy), src.rows - 1);
        for (int x = 0; x < newW; x++) {
            int sxi = min(int(x * sx), src.cols - 1);
            if (src.channels() == 3)
                dst.at<Vec3b>(y, x) = src.at<Vec3b>(syi, sxi);
            else
                dst.at<uchar>(y, x) = src.at<uchar>(syi, sxi);
        }
    }
    return dst;
}

// 2. BGR → Grayscale
Mat convertToGrayscale(const Mat& src) {
    Mat gray(src.rows, src.cols, CV_8UC1);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            Vec3b c = src.at<Vec3b>(y, x);
            gray.at<uchar>(y, x) =
                uchar(0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2]);
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

// 5.1 Sobel → gradient magnitude & direction
void computeGradient(const Mat& src, Mat& mag, Mat& dir) {
    int R = src.rows, C = src.cols;
    mag = Mat::zeros(R, C, CV_64F);
    dir = Mat::zeros(R, C, CV_64F);
    int Gx[3][3] = { {-1,0,1},{-2,0,2},{-1,0,1} };
    int Gy[3][3] = { {-1,-2,-1},{0,0,0},{1,2,1} };
    for (int y = 1; y < R - 1; y++) {
        for (int x = 1; x < C - 1; x++) {
            double sx = 0, sy = 0;
            for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
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
}

// 5.2 Non-maximum suppression
void nonMaxSuppression(const Mat& mag, const Mat& dir, Mat& out) {
    int R = mag.rows, C = mag.cols;
    out = Mat::zeros(R, C, CV_64F);
    for (int y = 1; y < R - 1; y++) {
        for (int x = 1; x < C - 1; x++) {
            double angle = dir.at<double>(y, x);
            double m = mag.at<double>(y, x),
                m1, m2;
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

// 5.3 Double threshold + Hysteresis
void applyHysteresis(const Mat& nms, Mat& edges, Mat& weak,
    double lowT, double highT)
{
    int R = nms.rows, C = nms.cols;
    edges = Mat::zeros(R, C, CV_8U);
    weak = Mat::zeros(R, C, CV_8U);
    // threshold
    for (int y = 0; y < R; y++) {
        for (int x = 0; x < C; x++) {
            double v = nms.at<double>(y, x);
            if (v >= highT)      edges.at<uchar>(y, x) = 255;
            else if (v >= lowT)  weak.at<uchar>(y, x) = 128;
        }
    }
    // hysteresis
    bool changed;
    do {
        changed = false;
        for (int y = 1; y < R - 1; y++) {
            for (int x = 1; x < C - 1; x++) {
                if (weak.at<uchar>(y, x) == 128) {
                    bool conn = false;
                    for (int i = -1; i <= 1 && !conn; i++)
                        for (int j = -1; j <= 1; j++)
                            if (edges.at<uchar>(y + i, x + j) == 255) {
                                conn = true; break;
                            }
                    if (conn) {
                        edges.at<uchar>(y, x) = 255;
                        weak.at<uchar>(y, x) = 0;
                        changed = true;
                    }
                }
            }
        }
    } while (changed);
}

// ---- Yeni Hough fonksiyonları ----

// A) Doğru tespiti: manuel Hough Line
vector<pair<double, double>> detectLines(const Mat& edges, int thresh) {
    int R = edges.rows, C = edges.cols;
    int D = int(hypot(R, C));
    int nrho = 2 * D, ntheta = 180;
    vector<vector<int>> acc(nrho, vector<int>(ntheta, 0));
    for (int y = 0; y < R; y++) {
        for (int x = 0; x < C; x++) {
            if (edges.at<uchar>(y, x) == 255) {
                for (int t = 0; t < ntheta; t++) {
                    double theta = t * M_PI / 180.0;
                    int rho = cvRound(x * cos(theta) + y * sin(theta)) + D;
                    acc[rho][t]++;
                }
            }
        }
    }
    vector<pair<double, double>> lines;
    for (int r = 0; r < nrho; r++) {
        for (int t = 0; t < ntheta; t++) {
            if (acc[r][t] > thresh) {
                double rho = r - D;
                double theta = t * M_PI / 180.0;
                lines.emplace_back(rho, theta);
            }
        }
    }
    return lines;
}

void drawLines(Mat& img, const vector<pair<double, double>>& lines) {
    for (auto& l : lines) {
        double rho = l.first, theta = l.second;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        Point p1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
        Point p2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
        line(img, p1, p2, Scalar(0, 0, 255), 2);
    }
}

// B) Çember tespiti: manuel Hough Circle
vector<Vec3f> detectCircles(const Mat& edges,
    int minR, int maxR, int thresh)
{
    int R = edges.rows, C = edges.cols;
    int nr = maxR - minR + 1;
    // 3-boyutlu accumulator: [y][x][r-minR]
    vector<vector<vector<int>>> acc(R,
        vector<vector<int>>(C, vector<int>(nr, 0)));
    for (int y = 0; y < R; y++) {
        for (int x = 0; x < C; x++) {
            if (edges.at<uchar>(y, x) == 255) {
                for (int r = minR; r <= maxR; r++) {
                    int ri = r - minR;
                    for (int t = 0; t < 360; t++) {
                        double theta = t * M_PI / 180.0;
                        int a = cvRound(x - r * cos(theta));
                        int b = cvRound(y - r * sin(theta));
                        if (a >= 0 && a < C && b >= 0 && b < R)
                            acc[b][a][ri]++;
                    }
                }
            }
        }
    }
    vector<Vec3f> circles;
    for (int y = 0; y < R; y++) {
        for (int x = 0; x < C; x++) {
            for (int ri = 0; ri < nr; ri++) {
                if (acc[y][x][ri] > thresh) {
                    circles.emplace_back(
                        float(x),
                        float(y),
                        float(minR + ri)
                    );
                }
            }
        }
    }
    return circles;
}

void drawCircles(Mat& img, const vector<Vec3f>& circles) {
    for (auto& c : circles) {
        Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);
        circle(img, center, radius, Scalar(0, 255, 0), 2);
    }
}

// ---- main() ----
int main() {
    cout << "1: Line Detection\n2: Circle Detection\nSelect: ";
    int choice; cin >> choice;

    // Dosya isimlerini isterseniz buradan düzenleyin
    string fname = (choice == 1)
        ? "D:\\Dersler\\projects\\ImageProcessingProject\\satranc.jpg"
        : "D:\\Dersler\\projects\\ImageProcessingProject\\para.jpg";

    Mat img = imread(fname);
    if (img.empty()) {
        cerr << "Resim yüklenemedi: " << fname << endl;
        return -1;
    }

    // Ön işlem hattı
    Mat resized = resizeImage(img, 800, 600);
    Mat gray = convertToGrayscale(resized);
    Mat blurred = applyGaussianBlur(gray, 5, 1.5);

    // Canny aşamaları
    Mat gradMag, gradDir;
    computeGradient(blurred, gradMag, gradDir);

    Mat nonMax;
    nonMaxSuppression(gradMag, gradDir, nonMax);

    Mat edges, weak;
    applyHysteresis(nonMax, edges, weak, 50, 100);

    // Sonuçları çiz ve göster
    if (choice == 1) {
        auto lines = detectLines(edges, /*eşik*/ 150);
        drawLines(resized, lines);
        namedWindow("Detected Lines", WINDOW_AUTOSIZE);
        imshow("Detected Lines", resized);
    }
    else {
        auto circles = detectCircles(edges,
            /*minR*/20,
            /*maxR*/100,
            /*eşik*/120);
        drawCircles(resized, circles);
        namedWindow("Detected Circles", WINDOW_AUTOSIZE);
        imshow("Detected Circles", resized);
    }

    waitKey(0);
    return 0;
}
