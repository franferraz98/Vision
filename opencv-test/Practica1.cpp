#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;

void colorReduce(cv::Mat& img, int div = 128) {
    for (int i = 0; i < img.rows; i++) {
        uchar* data = img.ptr<uchar>(i); // puntero a la fila i
        for (int j = 0; j < img.cols * img.channels(); j++) {
            data[j] = data[j] / div * div + div / 2;
            // o si te gusta mas, puedes hacerlo:
            // *data++= *data/div*div + div/2;
        }
    }
}


void alien(const Mat& in, Mat& out) {
    Mat hsv, bgra, hsvo, bgrao, mask;
    cv::cvtColor(in, hsv, COLOR_BGR2HSV);
    cv::cvtColor(in, bgra, COLOR_BGR2BGRA);
    inRange(hsv, Scalar(0, 0.23 * 255, 0), Scalar(50, 0.68 * 255, 255), hsvo);
    inRange(bgra, Scalar(20, 40, 95, 15), Scalar(255, 255, 255, 255), bgrao);
    bitwise_and(hsvo, bgrao, mask);
    in.copyTo(out);
    add(out, Scalar(0,70,0), out, mask);
}

Mat metodoDistorsion(Mat srcFrame, int k1) {

    Mat map_x, map_y, output;
    double Cy = (double)srcFrame.cols / 2;
    double Cx = (double)srcFrame.rows / 2;
    map_x.create(srcFrame.size(), CV_32FC1);
    map_y.create(srcFrame.size(), CV_32FC1);

    for (int x = 0; x < map_x.rows; x++) {
        for (int y = 0; y < map_y.cols; y++) {
            double r2 = (x - Cx) * (x - Cx) + (y - Cy) * (y - Cy);
            map_x.at<float>(x, y) = (double)((y - Cy) / (1 + double(k1 / 1000000.0) * r2) + Cy); // se suma para obtener la posicion absoluta
            map_y.at<float>(x, y) = (double)((x - Cx) / (1 + double(k1 / 1000000.0) * r2) + Cx); // la posicion relativa del punto al centro
        }
    }
    remap(srcFrame, output, map_x, map_y, INTER_LINEAR);
    return output;
}

void barrelDistortion(const Mat& frame, Mat& out, double k1, double k2) {
    double centrox = frame.rows / 2.0;
    double centroy = frame.cols / 2.0;
    
    Mat m1, m2;

    m1.create(frame.size(), CV_32FC1);
    m2.create(frame.size(), CV_32FC1);

    for (int i = 0; i < frame.rows; i++) {
        for (int j = 0; j < frame.cols; j++) {
            double inorm = (2.0*i - frame.rows) / frame.rows;
            double jnorm = (2.0*j - frame.cols) / frame.cols;
            double ru = sqrt(pow(inorm, 2.0) + pow(jnorm, 2.0));
            float xd = inorm * (1.0 + k1 * ru * ru - k2 * pow(ru, 4.0));
            float yd = jnorm * (1.0 + k1 * ru * ru - k2 * pow(ru, 4.0));
            m1.at<float>(i, j) = (xd + 1.0) * frame.rows / 2.0; 
            m2.at<float>(i, j) = (yd + 1.0) * frame.cols / 2.0;
        }
    }
    
    remap(frame, out, m2, m1, INTER_LINEAR);
}

float k1_ = 0.0;

static void on_trackbar(int slider, void*)
{
    k1_ = (slider - 50.0) / 100.0;
}

int main(int, char**)
{
    int effect = 6;
    Mat frame, frameMod;
    vector<Mat> channels(3);
    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;
    // open the default camera using default API
    // cap.open(0);
    // OR advance usage: select any API backend
    int deviceID = 0;             // 0 = open default camera
    int apiID = cv::CAP_ANY;      // 0 = autodetect default API
    // open selected camera using selected API
    cap.open(deviceID + apiID);
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    //--- GRAB AND WRITE LOOP
    cout << "Start grabbing" << endl
        << "Press any key to terminate" << endl;
    namedWindow("Test Distorsion", WINDOW_AUTOSIZE);
    int k1_slider = 1;

    createTrackbar("Distortion", "Test Distorsion", &k1_slider, 100.0, on_trackbar);
    on_trackbar(k1_slider, 0);
    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        // check if we succeeded
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        // show live and wait for a key with timeout long enough to show images
        switch (effect) {
            case 1: //Blanco y negro
                cv::cvtColor(frame, frameMod, COLOR_BGR2GRAY);
                break;
            case 2: //Contraste
                cv::cvtColor(frame, frameMod, COLOR_BGR2Lab);
                split(frameMod, channels);
                channels[0].convertTo(channels[0], -1, 1.5, 30);
                merge(channels, frameMod);
                cv::cvtColor(frameMod, frameMod, COLOR_Lab2BGR);
                break;
            case 3: //Ecualizar histograma
                cv::cvtColor(frame, frameMod, COLOR_BGR2HSV);
                split(frameMod, channels);
                equalizeHist(channels[1], channels[1]);
                merge(channels, frameMod);
                cv::cvtColor(frameMod, frameMod, COLOR_HSV2BGR);
                break;
            case 4: //Alien
                alien(frame, frameMod);
                break;
            case 5: //Poster
                frame.copyTo(frameMod);
                colorReduce(frameMod);
                break;
            case 6: //Barril
                
                barrelDistortion(frame, frameMod, k1_, 0.2);

                break;
            case 7: //Cojín
                break;
            default:
                frameMod = frame;
        }
        imshow("Mod", frameMod);
        imshow("Original", frame);
        if (waitKey(5) >= 0)
            break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}