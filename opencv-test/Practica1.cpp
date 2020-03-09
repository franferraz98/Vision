#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

float k1_ = 0.0, k2_ = 0.0;
int colors = 128;
int size_ = 1;
Scalar colorAlien(70,0,0);

Mat plotHistograms(const Mat& img, String window_name, bool accumulate){
	vector<Mat> planes;
	split( img, planes );
	int histSize = 256;
	float range[] = { 0, 256 }; //the upper boundary is exclusive
	const float* histRange = { range };
	bool uniform = true;
	Mat hist_2;
	calcHist( &planes[2], 1, 0, Mat(), hist_2, 1, &histSize, &histRange, uniform, false );

	normalize(hist_2, hist_2, 0, img.rows, NORM_MINMAX, -1, Mat() );
 	int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0,0,0));
    float sum = 0;
	for( int i = 1; i < histSize; i++ )
	  {

		if(accumulate){
			float norm = (hist_2.at<float>(i) / (img.rows * img.cols)) * 5000;
	      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(sum) ) ,
	                       Point( bin_w*(i), hist_h - cvRound(sum + norm) ),
	                       Scalar( 255, 0, 0), 2, 8, 0  );
	      sum += norm;
		} else {
			line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist_2.at<float>(i-1)) ) ,
				                       Point( bin_w*(i), hist_h - cvRound(hist_2.at<float>(i)) ),
				                       Scalar( 255, 0, 0), 2, 8, 0  );
		}

	  }
	namedWindow(window_name, CV_WINDOW_AUTOSIZE );
	imshow(window_name, histImage );
	return histImage;
}

void equalizeHistogram(const Mat& frame, Mat& frameMod){
	vector<Mat> channels(3);
	cv::cvtColor(frame, frameMod, COLOR_BGR2HSV);
	Mat histEqAcumOrig = plotHistograms(frameMod, "Histograma acumulado original" , true);
	split(frameMod, channels);
	equalizeHist(channels[2], channels[2]);
	merge(channels, frameMod);
	Mat histEq = plotHistograms(frameMod, "Histograma", false);

	Mat histEqAcum = plotHistograms(frameMod, "Histograma acumulado" , true);

	cv::cvtColor(frameMod, frameMod, COLOR_HSV2BGR);
}



void poster(Mat& img, int div) {
    for (int i = 0; i < img.rows; i++) {
        uchar* data = img.ptr<uchar>(i); // puntero a la fila i
        for (int j = 0; j < img.cols * img.channels(); j++) {
            data[j] = data[j] / div * div + div / 2;
        }
    }
}

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

void alien(const Mat& in, Mat& out, Scalar alien) {
    Mat hsv, bgra, hsvo, bgrao, mask;
    cv::cvtColor(in, hsv, COLOR_BGR2HSV);
    cv::cvtColor(in, bgra, COLOR_BGR2BGRA);
    inRange(hsv, Scalar(0, 0.23 * 255, 0), Scalar(50, 0.68 * 255, 255), hsvo);
    inRange(bgra, Scalar(20, 40, 95, 15), Scalar(255, 255, 255, 255), bgrao);

    // imshow("ColorFilter", colorFilter);
    bitwise_and(hsvo, bgrao, mask1);
    in.copyTo(out);
    add(out, alien, out, mask);
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

void pixelar(const Mat& in, Mat& out, int size){
	for (int r = 0; r < in.rows; r += size)
	{
	    for (int c = 0; c < in.cols; c += size)
	    {
	    	Rect rect;
	        rect.x = c;
	        rect.y = r;
	        rect.width = size;
	        if(c + size > in.cols){
	        	rect.width = in.cols - c;
	        }

	        rect.height = size;
	        if(r + size > in.rows){
				rect.height = in.rows - r;
			}

	        Scalar color = mean(Mat(in, rect));
	        rectangle(out, rect, color, CV_FILLED);
	    }
	}
}

void fusion(const Mat& in, const Mat& in2, Mat& out){
	float centroX = in.rows / 2;
	float centroY = in.cols / 2;
	float distMax = sqrt((pow(1.0, 2.0) + pow(1.0, 2.0)));
	for(int i = 0; i < in.rows; i++){
		for(int j = 0; j < in.cols; j++){
			float disX = abs(i - centroX) / centroX;
			float disY = abs(j - centroY) / centroY;
			float alpha = sqrt((pow(disX, 2.0) + pow(disY, 2.0)));
			alpha = alpha / distMax;
			out.at<Vec3b>(i,j) = (1.0 - alpha) * in.at<Vec3b>(i,j) + alpha * in2.at<Vec3b>(i,j);
		}
	}
}

static void valorK1(int slider, void*)
{
    k1_ = (slider - 50.0) / 100.0;
    cout << "Valor k1: " << k1_ << endl;
}

static void valorK2(int slider, void*)
{
    k2_ = (slider - 50.0) / 100.0;
    cout << "Valor k2: " << k2_ << endl;
}

static void numColores(int slider, void*){
	colors = slider;
}

static void sizePixelado(int slider, void*){
	size_ = slider + 1;
}

static void coloresAlien(int slider, void*){
	switch(slider){
	case 0:
		colorAlien = Scalar(70,0,0);
		break;
	case 1:
		colorAlien = Scalar(0,70,0);
		break;
	case 2:
		colorAlien = Scalar(0,0,70);
		break;
	}
}

static void on_trackbar2(int slider, void*)
{
    k2_ = (slider - 50.0) / 100.0;
}

int main(int, char**)
{
    int effect = 6;
    Mat frame, frameMod, img, img2;
    vector<Mat> channels(3);
    String last_slider = "";
    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;
    // open the default camera using default API

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
    cout << "Start grabbing" << endl
        << "Press any key to terminate" << endl;
    
    while(true)
    {
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        // check if we succeeded
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        switch (effect) {
            case 1: //Blanco y negro
                cvtColor(frame, frameMod, COLOR_BGR2GRAY);
                break;
            case 2: //Contraste
                cvtColor(frame, frameMod, COLOR_BGR2HSV);
                split(frameMod, channels);
                //alpha = 1.6 beta = 30
                channels[2].convertTo(channels[2], -1, 1.6, 30);
                merge(channels, frameMod);
                cvtColor(frameMod, frameMod, COLOR_HSV2BGR);
                break;
            case 3: //Ecualizar histograma
                equalizeHistogram(frame,frameMod);
                break;
            case 4: //Alien
                alien(frame, frameMod);
                break;
            case 5: //Poster
                frame.copyTo(frameMod);
                poster(frameMod, colors);
                break;
            case 6: //Distorsion
                //distortion(frame, frameMod, k1_, k2_);
                //barrelDistortion(frame, frameMod, k1_, k2_);
                pincushionDistortion(frame, frameMod, k1_, k2_);
                break;
            case 7: // pixelar
            	pixelar(frame, frameMod, size_);
            	break;
            case 8:
            	fusion(frame, img2, frameMod);
            	break;
            default:
                frameMod = frame;
        }

        imshow("Mod", frameMod);
        imshow("Original", frame);

        int key = waitKey(5);
        if ((key - 48) > 0 && (key - 48) < 9) {
        	int k1_slider = 1, k2_slider = 1, colores = 1, size = 1;
        	effect = (key - 48);
        	if(last_slider != ""){
        		destroyWindow(last_slider);
        	}
        	switch(effect){
        		case 1:
        			last_slider = "";
        			break;
        		case 2: //Contraste
        			last_slider = "";
        		    break;
        		case 3: //Ecualizar histograma
        			last_slider = "";
        			break;
        		case 4: //Alien
        			last_slider = "Aliens";
        			namedWindow(last_slider, WINDOW_AUTOSIZE);
					createTrackbar("Color del alien", last_slider, &colorAlien, 3, coloresAlien);
					coloresAlien(colorAlien, 0);
        			break;
        		case 5: //Poster
        			last_slider = "Colores";
					namedWindow(last_slider, WINDOW_AUTOSIZE);
					createTrackbar("Numero de colores", last_slider, &colores, 128, numColores);
					numColores(colores, 0);
        			break;
        		case 6: //Distorsion
        			last_slider = "Test Distorsion";
        			namedWindow(last_slider, WINDOW_AUTOSIZE);
					createTrackbar("Distortion", last_slider, &k1_slider, 100.0, valorK1);
					valorK1(k1_slider, 0);
					createTrackbar("Distortion 2", last_slider, &k2_slider, 100.0, valorK2);
					valorK2(k2_slider, 0);
        			break;
        		case 7:
        			last_slider = "Pixelar";
					namedWindow(last_slider, WINDOW_AUTOSIZE);
					createTrackbar("Pixelado", last_slider, &size, 15, sizePixelado);
					sizePixelado(colores, 0);
					break;
        			break;
        		case 8:
        			img = imread("sky.jpg", IMREAD_COLOR);
        			resize(img, img2, Size(frame.cols, frame.rows));
        			//img2 = Mat(frame.rows, frame.cols, frame.type(), Scalar(0,0,0));
        			break;
        	}
        }
        else if (key >= 0) {
            break;
        }
    }
    cout << frame.size() << endl;
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
