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

void menu() {
    cout << "|--------------------------------------------------------|" << endl;
    cout << "|                        EFECTOS                         |" << endl;
    cout << "|--------------------------------------------------------|" << endl;
    cout << "| 1.- Blanco y negro                                     |" << endl;
    cout << "| 2.- Mejorar el contraste.                              |" << endl;
    cout << "| 3.- Ecualizar histograma.                              |" << endl;
    cout << "| 4.- Alien.                                             |" << endl;
    cout << "| 5.- Poster.                                            |" << endl;
    cout << "| 6.- Barril.                                            |" << endl;
    cout << "| 7.- Cojin                                              |" << endl;
    cout << "|--------------------------------------------------------|" << endl << endl;
}

void colorReduce(cv::Mat& frame, int divisor = 128) {
    for (int i = 0; i < frame.rows; i++) {
        uchar* data = frame.ptr<uchar>(i);
        for (int j = 0; j < frame.cols * frame.channels(); j++) {
            data[j] = data[j] / divisor * divisor + divisor / 2;
        }
    }
}


void alien(const Mat& in, Mat& out) {
    Mat hsv, bgra, hsvo, bgrao, mask1, mask2, colorFilter;
    colorFilter.create(in.size(), CV_32FC1);
    cv::cvtColor(in, hsv, COLOR_BGR2HSV);
    cv::cvtColor(in, bgra, COLOR_BGR2BGRA);
    inRange(hsv, Scalar(0, 0.23 * 255, 0), Scalar(50, 0.68 * 255, 255), hsvo);
    inRange(bgra, Scalar(20, 40, 95, 15), Scalar(255, 255, 255, 255), bgrao);
    
    /*for (int i = 0; i < bgra.rows; i++) {
        for (int j = 0; j < bgra.cols; j++) {
            uchar* data = bgra.ptr<uchar>(i,j);
            if (data[2] > data[1] && data[2] > data[0]  && fabs(data[2] - data[1]) < 15.0) {
                colorFilter.at<float>(i, j) = 255.0;
            }
            else {
                colorFilter.at<float>(i, j) = 0.0;
            }
        }
    }
    */
    // imshow("ColorFilter", colorFilter);
    bitwise_and(hsvo, bgrao, mask1);
    in.copyTo(out);
    add(out, Scalar(0,70,0), out, mask1);
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
            float xd = inorm * (1.0 - k1 * ru * ru + k2 * pow(ru, 4.0));
            float yd = jnorm * (1.0 - k1 * ru * ru + k2 * pow(ru, 4.0));
            m1.at<float>(i, j) = (xd + 1.0) * frame.rows / 2.0; 
            m2.at<float>(i, j) = (yd + 1.0) * frame.cols / 2.0;
        }
    }
    
    remap(frame, out, m2, m1, INTER_LINEAR);
}

void pincushionDistortion(const Mat& frame, Mat& out, double k1, double k2) {
    double centrox = frame.rows / 2.0;
    double centroy = frame.cols / 2.0;

    Mat m1, m2;

    m1.create(frame.size(), CV_32FC1);
    m2.create(frame.size(), CV_32FC1);

    for (int i = 0; i < frame.rows; i++) {
        for (int j = 0; j < frame.cols; j++) {
            double inorm = (2.0 * i - frame.rows) / frame.rows;
            double jnorm = (2.0 * j - frame.cols) / frame.cols;
            double ru = sqrt(pow(inorm, 2.0) + pow(jnorm, 2.0));
            float xd = inorm * (1.0 + k1 * ru * ru - k2 * pow(ru, 4.0));
            float yd = jnorm * (1.0 + k1 * ru * ru - k2 * pow(ru, 4.0));
            m1.at<float>(i, j) = (xd + 1.0) * frame.rows / 2.0;
            m2.at<float>(i, j) = (yd + 1.0) * frame.cols / 2.0;
        }
    }

    remap(frame, out, m2, m1, INTER_LINEAR);
}


static float k1_ = 0.0;
static float k2_ = 0.0;

static void on_trackbar1(int slider, void*)
{
    k1_ = (slider - 50.0) / 100.0;
}

static void on_trackbar2(int slider, void*)
{
    k2_ = (slider - 50.0) / 100.0;
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
    menu();
    char ef;
    cin >> ef;
    cout << "Seleccionado: " << ef << endl;
    effect = ef - '0';
    cout << "Start grabbing" << endl
        << "Press any key to terminate" << endl;
    int k1_slider = 1;
    int k2_slider = 1;
    if (effect == 6 || effect == 7) {
        namedWindow("Test Distorsion", WINDOW_AUTOSIZE);
        createTrackbar("K1", "Test Distorsion", &k1_slider, 100.0, on_trackbar1);
        createTrackbar("K2", "Test Distorsion", &k2_slider, 100.0, on_trackbar2);
        on_trackbar1(k1_slider, 0);
        on_trackbar2(k2_slider, 0);
    }
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
                channels[0].convertTo(channels[0], -1, 1.5, 10);
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
                barrelDistortion(frame, frameMod, k1_, k2_);
                break;
            case 7: //Cojín
                pincushionDistortion(frame, frameMod, k1_, k2_);
                break;
            default:
                frameMod = frame;
        }
        imshow("Mod", frameMod);
        imshow("Original", frame);
        if (waitKey(5) >= 0)
            break;
    }
    cout << frame.size() << endl;
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}