#import "OpenCVBridge.h"
#import <opencv2/opencv.hpp>
#import <opencv2/imgproc/imgproc.hpp>
#import <opencv2/core/core.hpp>

@implementation OpenCVBridge

#pragma mark - OpenCV Utility Methods

/// Check if OpenCV is properly working
+ (BOOL)isOpenCVWorking {
    NSLog(@"Checking OpenCV functionality");
    
    // Simple test: try to create a matrix and check if successful
    cv::Mat testMat(5, 5, CV_8UC1);
    NSLog(@"Created a matrix. Is empty: %d", testMat.empty());
    
    return !testMat.empty();
}

/// Get the OpenCV version string
+ (NSString *)getOpenCVVersion {
    return [NSString stringWithUTF8String:CV_VERSION];
}

/// Convert CVPixelBuffer to cv::Mat
- (cv::Mat)cvMatFromPixelBuffer:(CVPixelBufferRef)pixelBuffer {
    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    
    void *baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer);
    size_t width = CVPixelBufferGetWidth(pixelBuffer);
    size_t height = CVPixelBufferGetHeight(pixelBuffer);
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
    OSType pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
    
    cv::Mat mat;
    
    if (pixelFormat == kCVPixelFormatType_32BGRA) {
        mat = cv::Mat((int)height, (int)width, CV_8UC4, baseAddress, bytesPerRow);
    } else if (pixelFormat == kCVPixelFormatType_32RGBA) {
        cv::Mat rgba((int)height, (int)width, CV_8UC4, baseAddress, bytesPerRow);
        cv::cvtColor(rgba, mat, cv::COLOR_RGBA2BGRA);
    } else if (pixelFormat == kCVPixelFormatType_24RGB) {
        cv::Mat rgb((int)height, (int)width, CV_8UC3, baseAddress, bytesPerRow);
        cv::cvtColor(rgb, mat, cv::COLOR_RGB2BGR);
    } else {
        NSLog(@"Unsupported CVPixelBuffer format: %d", pixelFormat);
        mat = cv::Mat();
    }
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    return mat;
}

/// Find peaks in a signal (similar to scipy.signal.find_peaks)
/// @param signal The signal to find peaks in
/// @param minDist Minimum distance between peaks
/// @returns NSArray of peak indices
- (NSArray *)findPeaksInSignal:(const std::vector<float> &)signal withMinDistance:(int)minDist {
    NSMutableArray *peaks = [NSMutableArray array];
    
    if (signal.size() < 3) {
        return peaks;
    }
    
    for (int i = 1; i < signal.size() - 1; i++) {
        // Check if this point is higher than both its neighbors
        if (signal[i] > signal[i-1] && signal[i] > signal[i+1]) {
            // Check if we already have peaks and if this one is too close to any existing peak
            BOOL tooClose = NO;
            for (NSNumber *existingPeak in peaks) {
                if (abs(i - [existingPeak intValue]) < minDist) {
                    tooClose = YES;
                    
                    // If this peak is higher than the existing one, replace it
                    if (signal[i] > signal[[existingPeak intValue]]) {
                        NSInteger index = [peaks indexOfObject:existingPeak];
                        peaks[index] = @(i);
                    }
                    break;
                }
            }
            
            if (!tooClose) {
                [peaks addObject:@(i)];
            }
        }
    }
    
    // Sort peaks by height (amplitude)
    [peaks sortUsingComparator:^NSComparisonResult(NSNumber *peak1, NSNumber *peak2) {
        float height1 = signal[[peak1 intValue]];
        float height2 = signal[[peak2 intValue]];
        if (height1 < height2) {
            return NSOrderedDescending;
        } else if (height1 > height2) {
            return NSOrderedAscending;
        }
        return NSOrderedSame;
    }];
    
    return peaks;
}

/// Smooth a 1D signal with a moving average filter
/// @param signal The signal to smooth
/// @param kernelSize The size of the smoothing kernel
- (std::vector<float>)smoothSignal:(const std::vector<float> &)signal withKernelSize:(int)kernelSize {
    if (kernelSize < 2 || signal.size() == 0) {
        return signal;
    }
    
    // Ensure kernel size is odd
    if (kernelSize % 2 == 0) {
        kernelSize += 1;
    }
    
    std::vector<float> smoothed(signal.size(), 0.0);
    int halfKernel = kernelSize / 2;
    
    // For each point in the signal
    for (int i = 0; i < signal.size(); i++) {
        float sum = 0.0;
        int count = 0;
        
        // Sum the values in the kernel window
        for (int j = -halfKernel; j <= halfKernel; j++) {
            int idx = i + j;
            if (idx >= 0 && idx < signal.size()) {
                sum += signal[idx];
                count++;
            }
        }
        
        // Compute the average
        smoothed[i] = (count > 0) ? sum / count : signal[i];
    }
    
    return smoothed;
}

/// Calculate projection (sum along axis) of a grayscale image
/// @param image The input grayscale image
/// @param isHorizontalProjection Whether to calculate horizontal projection (sum along rows) or vertical (sum along columns)
- (std::vector<float>)calculateProjection:(const cv::Mat &)image isHorizontalProjection:(BOOL)isHorizontalProjection {
    std::vector<float> projection;
    
    if (image.empty()) {
        return projection;
    }
    
    if (isHorizontalProjection) {
        // Horizontal projection: sum along rows (for each y-coordinate)
        projection.resize(image.rows);
        for (int y = 0; y < image.rows; y++) {
            float sum = 0;
            for (int x = 0; x < image.cols; x++) {
                sum += image.at<uchar>(y, x);
            }
            projection[y] = sum;
        }
    } else {
        // Vertical projection: sum along columns (for each x-coordinate)
        projection.resize(image.cols);
        for (int x = 0; x < image.cols; x++) {
            float sum = 0;
            for (int y = 0; y < image.rows; y++) {
                sum += image.at<uchar>(y, x);
            }
            projection[x] = sum;
        }
    }
    
    return projection;
}

#pragma mark - Core Image Processing Methods for Calibration

/// Test processing a frame
- (BOOL)processTestFrame:(CVPixelBufferRef)pixelBuffer {
    if (pixelBuffer == nil) {
        NSLog(@"processTestFrame: Nil pixel buffer");
        return NO;
    }
    
    // Try to convert to cv::Mat
    cv::Mat inputMat = [self cvMatFromPixelBuffer:pixelBuffer];
    if (inputMat.empty()) {
        NSLog(@"processTestFrame: Failed to convert pixel buffer to cv::Mat");
        return NO;
    }
    
    // Try basic image operations
    cv::Mat grayMat;
    cv::cvtColor(inputMat, grayMat, cv::COLOR_BGRA2GRAY);
    
    if (grayMat.empty()) {
        NSLog(@"processTestFrame: Failed to convert to grayscale");
        return NO;
    }
    
    NSLog(@"processTestFrame: Successfully processed frame (%dx%d)", inputMat.cols, inputMat.rows);
    return YES;
}

/// Process a single frame for auto-calibration
/// Implements the Python algorithm for detecting optimal threshold positioning
/// @param pixelBuffer The input video frame
/// @param isVerticalDirection Whether counting direction is vertical (true) or horizontal (false)
- (NSArray *)processCalibrationFrame:(CVPixelBufferRef)pixelBuffer isVerticalDirection:(BOOL)isVerticalDirection {
    NSLog(@"Processing calibration frame");
    
    // Convert pixel buffer to cv::Mat
    cv::Mat inputFrame = [self cvMatFromPixelBuffer:pixelBuffer];
    if (inputFrame.empty()) {
        NSLog(@"Failed to convert pixel buffer to cv::Mat");
        return @[@0.3f, @0.7f]; // Default fallback values
    }
    
    // 1. Convert to grayscale - matches Python cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv::Mat grayFrame;
    cv::cvtColor(inputFrame, grayFrame, cv::COLOR_BGRA2GRAY);
    
    // 2. Apply Gaussian blur with 5x5 kernel - matches Python cv2.GaussianBlur(gray, (5, 5), 0)
    cv::Mat blurredFrame;
    cv::GaussianBlur(grayFrame, blurredFrame, cv::Size(5, 5), 0);
    
    // 3. Apply Canny edge detection with thresholds 50, 150 - matches Python cv2.Canny(blurred, 50, 150)
    cv::Mat edges;
    cv::Canny(blurredFrame, edges, 50, 150);
    
    // 4. Calculate projection based on direction
    std::vector<float> projection = [self calculateProjection:edges isHorizontalProjection:isVerticalDirection];
    
    // 5. Calculate kernel size based on dimension
    // max(5, height//20) for vertical or max(5, width//20) for horizontal - matches Python
    int dimension = isVerticalDirection ? inputFrame.rows : inputFrame.cols;
    int kernelSize = std::max(5, dimension / 20);
    
    // 6. Smooth the projection
    std::vector<float> smoothedProjection = [self smoothSignal:projection withKernelSize:kernelSize];
    
    // 7. Find peaks in the smoothed projection
    // Calculate minimum distance between peaks - matches Python distance parameter
    // Quarter of the dimension (height/width)
    int minPeakDistance = dimension / 4;
    
    NSArray *peaks = [self findPeaksInSignal:smoothedProjection withMinDistance:minPeakDistance];
    
    // 8. Convert peak positions to normalized thresholds
    NSMutableArray *thresholds = [NSMutableArray array];
    float denominator = (float)dimension;
    
    // If we found at least two peaks, use them
    if ([peaks count] >= 2) {
        // Use the first two peaks (which are sorted by amplitude)
        int peak1 = [[peaks objectAtIndex:0] intValue];
        int peak2 = [[peaks objectAtIndex:1] intValue];
        
        float threshold1 = peak1 / denominator;
        float threshold2 = peak2 / denominator;
        
        // Sort thresholds by position (not amplitude)
        if (threshold1 > threshold2) {
            [thresholds addObject:@(threshold2)];
            [thresholds addObject:@(threshold1)];
        } else {
            [thresholds addObject:@(threshold1)];
            [thresholds addObject:@(threshold2)];
        }
    }
    // If we found only one peak, use it with a second one positioned at 30% or 70%
    else if ([peaks count] == 1) {
        int peak = [[peaks objectAtIndex:0] intValue];
        float threshold = peak / denominator;
        
        // Add a second threshold at 30% or 70% depending on first peak - matches Python fallback
        float secondThreshold = (threshold < 0.5f) ? 0.7f : 0.3f;
        
        [thresholds addObject:@(std::min(threshold, secondThreshold))];
        [thresholds addObject:@(std::max(threshold, secondThreshold))];
    }
    // Fallback: use default thresholds at 30% and 70% - matches Python np.linspace(0.3 * dim, 0.7 * dim, n_lines)
    else {
        [thresholds addObject:@(0.3f)];
        [thresholds addObject:@(0.7f)];
    }
    
    return thresholds;
}

@end 