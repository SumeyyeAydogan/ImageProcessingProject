#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>


using namespace cv;
using namespace std;

// Yeniden boyutlandýrma: En basit haliyle en yakýn komþu yöntemi kullanýlýr.
Mat resizeImage(const Mat& src, int newWidth, int newHeight) {
    Mat dst(newHeight, newWidth, src.type());
    double scaleX = static_cast<double>(src.cols) / newWidth;
    double scaleY = static_cast<double>(src.rows) / newHeight;
    for (int y = 0; y < newHeight; y++) {
        int srcY = min(static_cast<int>(y * scaleY), src.rows - 1);
        for (int x = 0; x < newWidth; x++) {
            int srcX = min(static_cast<int>(x * scaleX), src.cols - 1);
            if (src.channels() == 3) {
                dst.at<Vec3b>(y, x) = src.at<Vec3b>(srcY, srcX);
            }
            else {
                dst.at<uchar>(y, x) = src.at<uchar>(srcY, srcX);
            }
        }
    }
    return dst;
}

// BGR görüntüyü gri tonlamaya çevirir.
Mat convertToGrayscale(const Mat& src) {
    Mat gray(src.rows, src.cols, CV_8UC1);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            Vec3b color = src.at<Vec3b>(y, x);
            // OpenCV’de renk sýrasý BGR olduðundan:
            uchar grayVal = static_cast<uchar>(0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]);
            gray.at<uchar>(y, x) = grayVal;
        }
    }
    return gray;
}

// Gaussian çekirdek oluþturma fonksiyonu (ksize x ksize, sigma)
vector<vector<double>> createGaussianKernel(int ksize, double sigma) {
    vector<vector<double>> kernel(ksize, vector<double>(ksize));
    int halfSize = ksize / 2;
    double sum = 0.0;
    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            double exponent = -(i * i + j * j) / (2 * sigma * sigma);
            kernel[i + halfSize][j + halfSize] = exp(exponent) / (2 * M_PI * sigma * sigma);
            sum += kernel[i + halfSize][j + halfSize];
        }
    }
    // Çekirdeði normalize et
    for (int i = 0; i < ksize; i++) {
        for (int j = 0; j < ksize; j++) {
            kernel[i][j] /= sum;
        }
    }
    return kernel;
}

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

// Basitleþtirilmiþ Canny kenar algýlama
Mat applyCannyEdgeDetection(const Mat& src, double lowThreshold, double highThreshold) {
    int rows = src.rows;
    int cols = src.cols;
    // Aþama 1: Sobel operatörleri ile gradyan hesaplama
    Mat gradient = Mat::zeros(rows, cols, CV_64F);
    Mat direction = Mat::zeros(rows, cols, CV_64F);

    // Sobel çekirdekleri (3x3)
    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int y = 1; y < rows - 1; y++) {
        for (int x = 1; x < cols - 1; x++) {
            double sumX = 0.0, sumY = 0.0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int pixel = src.at<uchar>(y + i, x + j);
                    sumX += Gx[i + 1][j + 1] * pixel;
                    sumY += Gy[i + 1][j + 1] * pixel;
                }
            }
            double mag = sqrt(sumX * sumX + sumY * sumY);
            gradient.at<double>(y, x) = mag;
            double angle = atan2(sumY, sumX) * 180.0 / M_PI;
            if (angle < 0) angle += 180;
            direction.at<double>(y, x) = angle;
        }
    }

    // Aþama 2: Non-maximum suppression (gradyan doðrultusuna göre komþu piksellerle karþýlaþtýrma)
    Mat nonMaxSupp = Mat::zeros(rows, cols, CV_64F);
    for (int y = 1; y < rows - 1; y++) {
        for (int x = 1; x < cols - 1; x++) {
            double angle = direction.at<double>(y, x);
            double mag = gradient.at<double>(y, x);
            double q = 0.0, r = 0.0;
            // Gradyan yönüne göre komþularý belirle
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                q = gradient.at<double>(y, x + 1);
                r = gradient.at<double>(y, x - 1);
            }
            else if (angle >= 22.5 && angle < 67.5) {
                q = gradient.at<double>(y - 1, x + 1);
                r = gradient.at<double>(y + 1, x - 1);
            }
            else if (angle >= 67.5 && angle < 112.5) {
                q = gradient.at<double>(y - 1, x);
                r = gradient.at<double>(y + 1, x);
            }
            else if (angle >= 112.5 && angle < 157.5) {
                q = gradient.at<double>(y - 1, x - 1);
                r = gradient.at<double>(y + 1, x + 1);
            }

            if (mag >= q && mag >= r)
                nonMaxSupp.at<double>(y, x) = mag;
            else
                nonMaxSupp.at<double>(y, x) = 0;
        }
    }

    // Aþama 3: Çift eþikleme ve hysteresis ile zayýf kenarlarýn belirlenmesi
    Mat edges = Mat::zeros(rows, cols, CV_8U);
    for (int y = 1; y < rows - 1; y++) {
        for (int x = 1; x < cols - 1; x++) {
            double value = nonMaxSupp.at<double>(y, x);
            if (value >= highThreshold) {
                edges.at<uchar>(y, x) = 255;  // güçlü kenar
            }
            else if (value >= lowThreshold) {
                edges.at<uchar>(y, x) = 128;  // zayýf kenar (ilk iþaretleme)
            }
        }
    }
    // Hysteresis: Zayýf kenarlardan, güçlü kenar ile baðlantýlý olanlarý koru.
    bool changed;
    do {
        changed = false;
        for (int y = 1; y < rows - 1; y++) {
            for (int x = 1; x < cols - 1; x++) {
                if (edges.at<uchar>(y, x) == 128) {
                    bool connected = false;
                    for (int i = -1; i <= 1 && !connected; i++) {
                        for (int j = -1; j <= 1; j++) {
                            if (edges.at<uchar>(y + i, x + j) == 255) {
                                connected = true;
                                break;
                            }
                        }
                    }
                    if (connected) {
                        edges.at<uchar>(y, x) = 255;
                        changed = true;
                    }
                    else {
                        edges.at<uchar>(y, x) = 0;
                    }
                }
            }
        }
    } while (changed);

    return edges;
}

int main() {
    // Sadece OpenCV kullanarak görüntüyü dosyadan oku
    Mat img = imread("D:\\Dersler\\projects\\ImageProcessingProject\\satranc.jpg");
    if (img.empty()) {
        cout << "Görüntü yüklenemedi!" << endl;
        return -1;
    }

    // 800x600 boyutlarýna yeniden boyutlandýrma (OpenCV fonksiyonu kullanýlmadan)
    Mat resized = resizeImage(img, 800, 600);

    // Gri tonlamaya çevirme (OpenCV dýþý)
    Mat gray = convertToGrayscale(resized);

    // Gaussian Blur uygulama (kernel: 5x5, sigma: 1.5)
    Mat blurred = applyGaussianBlur(gray, 3, 1.0);

    // Canny kenar algýlama (eþik deðerleri: 100 ve 200)
    Mat edges = applyCannyEdgeDetection(blurred, 50, 100);

    // Sadece OpenCV kullanarak görüntüyü göster
    namedWindow("Edge Detection", WINDOW_AUTOSIZE);
    imshow("Edge Detection", edges);

    waitKey(0);
    destroyAllWindows();

    return 0;
}