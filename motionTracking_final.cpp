//motionTracking.cpp

//Written by  Kyle Hounslow, December 2013

//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software")
//, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
//and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//IN THE SOFTWARE.

#include <queue>
#include <cstdio>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;
using namespace cv;

//our sensitivity value to be used in the absdiff() function
const static int SENSITIVITY_VALUE = 20;
//size of blur used to smooth the intensity image output from absdiff() function
const static int BLUR_SIZE = 10;
//we'll have just one object to search for
//and keep track of its position.
int theObject[2] = {0,0};
//bounding rectangle of the object, we will use the center of this as its position.
// Rect objectBoundingRectangle = Rect(0,0,0,0);


//int to string helper function
string intToString(int number){

	//this function has a number input and string output
	std::stringstream ss;
	ss << number;
	return ss.str();
}

void searchForMovement(Mat& thresholdImage, Mat &cameraFeed){
	//notice how we use the '&' operator for objectDetected and cameraFeed. This is because we wish
	//to take the values passed into the function and manipulate them, rather than just working with a copy.
	//eg. we draw to the cameraFeed to be displayed in the main() function.
	bool objectDetected = false;
	Mat temp;
	thresholdImage.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );// retrieves all contours
	//findContours(temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );// retrieves external contours

	//if contours vector is not empty, we have found some objects
	if(contours.size()>0) objectDetected=true;
	else objectDetected = false;

	if(objectDetected){
		//the largest contour is found at the end of the contours vector
		//we will simply assume that the biggest contour is the object we are looking for.
		// cout << "object detect : " << contours.size() << endl;

		//make a bounding rectangle around the largest contour then find its centroid
		//this will be the object's final estimated position.

		//중복된 object를 제거한다.
		vector<int> parents(contours.size(), -1);

		for(int i = 0; i < contours.size(); i++) {
			if(parents[i] != -1)	//이미 속한 그룹이 있는 경우 더이상 연산을 할 필요가 없다.
				continue;
			Rect objectA = boundingRect(contours[i]);

			for(int j = i + 1; j < contours.size(); j++) {
				if(parents[j] != -1)
					continue;
				Rect objectB = boundingRect(contours[j]);
				Rect objectInter = objectA | objectB; //겹치는 부분의 큰 영역으로 만든다

				if(objectA == objectInter)
					parents[j] = i;
				else if(objectB == objectInter) {
					parents[i] = j;
					break;
				}
			}
		}

		vector<Rect> objects;
		for(int i = 0; i < contours.size(); i++) {
			if(parents[i] == -1)
				objects.push_back(boundingRect(contours[i]));
		}

		//이미지를 하나로 합치기 위한 작업, max_height 계산

		int max_height = 0;
		int max_width = 0;
		for(auto iter = objects.begin(); iter != objects.end(); iter++) {
			max_height = iter->height > max_height ? iter->height : max_height;
			max_width += iter->width;
		}
		//size, depth(pixel range), channel
		//cv::IplImage *DispImage = cv::cvCreateImage( cv::cvSize(max_width, max_height), 8, 3);
		
		Mat croppedImage = Mat::zeros(max_height, max_width, CV_8UC3);//(max_height, max_width, CV_8UC3, 0);
		// croppedImage.setTo(Scalar::all(0));
		static int cnt = 0;
		int x = 0;
		for(int i = 0; i < objects.size(); i++) {
			Rect object = objects[i];

			Mat img = cameraFeed(object);
			img.copyTo(croppedImage(Rect(x, 0, object.width, object.height)));
			x += object.width;
		}
		cv::imshow("cropped", croppedImage);
		// char str[100];
		// sprintf(str, "./img/img%05d.jpg", cnt++);
		// cv::imwrite(str, croppedImage);

		for(int i = 0; i < objects.size(); i++) {
			Rect object = objects[i];
			rectangle(cameraFeed, object.tl(), object.br(), Scalar(255, 0, 0), 2);
			putText(cameraFeed, "Object", object.tl(), 1, 1, Scalar(255,0,0),2);	
		}
	}
}
int main(int argc, char** argv){

	//some boolean variables for added functionality
	bool objectDetected = false;
	//these two can be toggled by pressing 'd' or 't'
	bool debugMode = false;
	bool bigCamSize = true;
	//pause and resume code
	bool pause = false;
	//set up the matrices that we will need
	//the two frames we will be comparing
	Mat capturedFrame;
	//their grayscale images (needed for absdiff() function)
	Mat grayImage1,grayImage2;
	//resulting difference image
	Mat differenceImage;
	//thresholded difference image (for use in findContours() function)
	Mat thresholdImage;
	//video capture object.
	
	VideoCapture capture;
	if(argc != 2)
		capture = VideoCapture(-1);
	else
		capture = VideoCapture(argv[1]);

	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);


	if(!capture.isOpened()){
		cout<<"ERROR ACQUIRING VIDEO FEED\n";
		getchar();
		return -1;
	}


	queue<Mat> imageQ;

	//we can loop the video by re-opening the capture every time the video reaches its last frame

	//check if the video has reach its last frame.
	//we add '-1' because we are reading two frames from the video at a time.
	//if this is not included, we get a memory error!
	int cnt = 0;
	const int frame_keeping_cnt = 3;
	int test = 0;
	while(1){
		test++;
		// while (test > 10) {
		// 	//stay in this loop until 
		// 	switch (waitKey()){
		// 		//a switch statement inside a switch statement? Mind blown.
		// 	case 112: 
		// 		//change pause back to false
		// 		pause = false;
		// 		cout<<"Code Resumed"<<endl;
		// 		break;
		// 	}
		// }
		capture.read(capturedFrame);

		cv::cvtColor(capturedFrame,grayImage1,COLOR_BGR2GRAY);
		cv::GaussianBlur(grayImage1, grayImage1, cv::Size(21, 21), 0);
		
		imageQ.push(grayImage1.clone());
		//read first frame
		if(imageQ.size() < frame_keeping_cnt) {
			continue;
		}

		Mat grayImage2 = imageQ.front();
		imageQ.pop();

		cv::absdiff(grayImage1,grayImage2,differenceImage);
		//threshold intensity image at a given sensitivity value
		cv::threshold(differenceImage,thresholdImage,SENSITIVITY_VALUE,255,THRESH_BINARY);
		if(debugMode==true){
			//show the difference image and threshold image
			cv::imshow("Difference Image",differenceImage);
			cv::imshow("Threshold Image", thresholdImage);
		}else{
			//if not in debug mode, destroy the windows so we don't see them anymore
			cv::destroyWindow("Difference Image");
			cv::destroyWindow("Threshold Image");
		}
		//blur the image to get rid of the noise. This will output an intensity image
		cv::blur(thresholdImage,thresholdImage,cv::Size(BLUR_SIZE,BLUR_SIZE));
		//threshold again to obtain binary image from blur output
		cv::threshold(thresholdImage,thresholdImage,SENSITIVITY_VALUE,255,THRESH_BINARY);
		if(debugMode==true){
			//show the threshold image after it's been "blurred"

			imshow("Final Threshold Image",thresholdImage);

		}
		else {
			//if not in debug mode, destroy the windows so we don't see them anymore
			cv::destroyWindow("Final Threshold Image");
		}

		searchForMovement(thresholdImage,capturedFrame);

		//show our captured frame
		char strFrame[100];
		sprintf(strFrame, "frame : %d", cnt++);
		putText(capturedFrame, strFrame, Point(0,30), 1, 1, Scalar(255,0,0),2);
		imshow("capturedFrame",capturedFrame);
		//imshow("prevFrame", prevFrame);
		//check to see if a button has been pressed.
		//this 10ms delay is necessary for proper operation of this program
		//if removed, frames will not have enough time to referesh and a blank 
		//image will appear.
		switch(waitKey(10)){

		case 27: //'esc' key has been pressed, exit program.
			return 0;
		case 116: //'t' has been pressed. this will toggle tracking

			queue<Mat>().swap(imageQ);
			// capture.release();
			// capture = VideoCapture(0);
			if(!capture.isOpened()){
				cout<<"ERROR ACQUIRING VIDEO FEED\n";
				return -1;
			}

			bigCamSize = !bigCamSize;
			if(bigCamSize) {
				capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
				capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
			} else {
				capture.set(CV_CAP_PROP_FRAME_WIDTH, 480);
				capture.set(CV_CAP_PROP_FRAME_HEIGHT, 360);
			}
			break;
		case 100: //'d' has been pressed. this will debug mode
			debugMode = !debugMode;
			if(debugMode == false) cout<<"Debug mode disabled."<<endl;
			else cout<<"Debug mode enabled."<<endl;
			break;
		case 112: //'p' has been pressed. this will pause/resume the code.
			pause = !pause;
			if(pause == true){ 
				cout<<"Code paused, press 'p' again to resume"<<endl;
				while (pause == true){
					//stay in this loop until 
					switch (waitKey()){
						//a switch statement inside a switch statement? Mind blown.
					case 112: 
						//change pause back to false
						pause = false;
						cout<<"Code Resumed"<<endl;
						break;
					}
				}
			}
		}
	}
	//release the capture before re-opening and looping again.
	capture.release();

	return 0;

}
