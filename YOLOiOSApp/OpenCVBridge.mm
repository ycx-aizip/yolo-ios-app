#import "./OpenCVBridge.h"

// Real device - use OpenCV
#import <opencv2/core/core.hpp>
#import <opencv2/imgproc/imgproc.hpp>
#import <opencv2/opencv.hpp>

@implementation OpenCVBridge

+ (BOOL)isOpenCVWorking {
    // Real device implementation - use OpenCV
    // Create a simple OpenCV matrix
    cv::Mat testMat = cv::Mat::eye(3, 3, CV_8UC1);
    
    // Check if the matrix is valid
    BOOL isValid = !testMat.empty() && testMat.size().width == 3 && testMat.size().height == 3;
    
    // Print OpenCV version for verification
    NSLog(@"OpenCV Version: %s", CV_VERSION);
    
    return isValid;
}

@end 