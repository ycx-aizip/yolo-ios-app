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

+ (NSString *)getOpenCVVersion {
    // Return the OpenCV version as a string
    return [NSString stringWithFormat:@"%s", CV_VERSION];
}

+ (UIImage *)UIImageFromCVPixelBuffer:(CVPixelBufferRef)pixelBuffer {
    // Lock the pixel buffer base address
    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    
    // Get buffer dimensions
    size_t width = CVPixelBufferGetWidth(pixelBuffer);
    size_t height = CVPixelBufferGetHeight(pixelBuffer);
    
    // Get base address and bytes per row
    void *baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer);
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
    
    // Create a CGContextRef from the pixel buffer data
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(baseAddress, width, height, 8,
                                                bytesPerRow, colorSpace,
                                                kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
    
    // Create a CGImage from the context
    CGImageRef quartzImage = CGBitmapContextCreateImage(context);
    
    // Create a UIImage from the CGImage using the correct scale
    UIImage *image = [UIImage imageWithCGImage:quartzImage scale:1.0 orientation:UIImageOrientationUp];
    
    // Clean up resources
    CGImageRelease(quartzImage);
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    
    // Log success for debugging
    if (image) {
        NSLog(@"Successfully converted CVPixelBuffer to UIImage: %f x %f", image.size.width, image.size.height);
    } else {
        NSLog(@"Failed to convert CVPixelBuffer to UIImage");
    }
    
    return image;
}

+ (CVPixelBufferRef)CVPixelBufferFromUIImage:(UIImage *)image {
    // Get image dimensions
    CGSize size = image.size;
    
    // Create pixel buffer
    NSDictionary *options = @{
        (NSString*)kCVPixelBufferCGImageCompatibilityKey: @YES,
        (NSString*)kCVPixelBufferCGBitmapContextCompatibilityKey: @YES,
    };
    
    CVPixelBufferRef pixelBuffer = NULL;
    CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         size.width,
                                         size.height,
                                         kCVPixelFormatType_32ARGB,
                                         (__bridge CFDictionaryRef)options,
                                         &pixelBuffer);
    
    if (status != kCVReturnSuccess) {
        NSLog(@"Failed to create CVPixelBuffer");
        return NULL;
    }
    
    // Lock pixel buffer
    CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    
    // Get pixel buffer data pointer
    void *pixelData = CVPixelBufferGetBaseAddress(pixelBuffer);
    
    // Create bitmap context
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(pixelData,
                                               size.width,
                                               size.height,
                                               8,
                                               CVPixelBufferGetBytesPerRow(pixelBuffer),
                                               colorSpace,
                                               kCGImageAlphaNoneSkipFirst);
    
    // Draw image into context
    CGContextDrawImage(context, CGRectMake(0, 0, size.width, size.height), image.CGImage);
    
    // Release resources
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    
    // Log success for debugging
    NSLog(@"Successfully converted UIImage to CVPixelBuffer: %f x %f", size.width, size.height);
    
    return pixelBuffer;
}

#pragma mark - Helper Methods for OpenCV conversion

// Convert UIImage to cv::Mat for processing
+ (cv::Mat)cvMatFromUIImage:(UIImage *)image {
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (RGBA)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,
                                                   cols,
                                                   rows,
                                                   8,
                                                   cvMat.step[0],
                                                   colorSpace,
                                                   kCGImageAlphaNoneSkipLast |
                                                   kCGBitmapByteOrderDefault);
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    // Convert to BGR (OpenCV standard format)
    cv::Mat bgr;
    cv::cvtColor(cvMat, bgr, cv::COLOR_RGBA2BGR);
    
    return bgr;
}

// Convert cv::Mat to UIImage for returning to Swift
+ (UIImage *)UIImageFromCVMat:(cv::Mat)cvMat {
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize() * cvMat.total()];
    
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        // Grayscale
        colorSpace = CGColorSpaceCreateDeviceGray();
        
        // Create image with proper alpha channel handling
        CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
        
        CGImageRef imageRef = CGImageCreate(cvMat.cols,
                                          cvMat.rows,
                                          8,
                                          8 * cvMat.elemSize(),
                                          cvMat.step[0],
                                          colorSpace,
                                          kCGImageAlphaNone | kCGBitmapByteOrderDefault,
                                          provider,
                                          NULL,
                                          false,
                                          kCGRenderingIntentDefault);
        
        UIImage *image = [UIImage imageWithCGImage:imageRef];
        
        CGImageRelease(imageRef);
        CGDataProviderRelease(provider);
        CGColorSpaceRelease(colorSpace);
        
        return image;
    } else {
        // RGB
        cv::Mat rgbaMat;
        colorSpace = CGColorSpaceCreateDeviceRGB();
        
        // Ensure we have 4 channels (RGBA)
        if (cvMat.channels() == 3) {
            cv::cvtColor(cvMat, rgbaMat, cv::COLOR_BGR2RGBA);
        } else if (cvMat.channels() == 4) {
            rgbaMat = cvMat;
        } else {
            return nil; // Unsupported format
        }
        
        // Recreate data with the modified mat
        NSData *dataRGBA = [NSData dataWithBytes:rgbaMat.data length:rgbaMat.elemSize() * rgbaMat.total()];
        
        CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)dataRGBA);
        
        CGImageRef imageRef = CGImageCreate(rgbaMat.cols,
                                          rgbaMat.rows,
                                          8,
                                          8 * rgbaMat.elemSize() / rgbaMat.channels(),
                                          rgbaMat.step[0],
                                          colorSpace,
                                          kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault,
                                          provider,
                                          NULL,
                                          false,
                                          kCGRenderingIntentDefault);
        
        UIImage *image = [UIImage imageWithCGImage:imageRef];
        
        CGImageRelease(imageRef);
        CGDataProviderRelease(provider);
        CGColorSpaceRelease(colorSpace);
        
        return image;
    }
}

#pragma mark - Image Processing Methods

+ (UIImage *)convertToGrayscale:(UIImage *)image {
    // Convert UIImage to cv::Mat
    cv::Mat inputMat = [self cvMatFromUIImage:image];
    
    // Convert to grayscale
    cv::Mat grayMat;
    cv::cvtColor(inputMat, grayMat, cv::COLOR_BGR2GRAY);
    
    // Create 8-bits-per-component grayscale image
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
    NSData *data = [NSData dataWithBytes:grayMat.data length:grayMat.elemSize() * grayMat.total()];
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    CGImageRef imageRef = CGImageCreate(grayMat.cols,
                                      grayMat.rows,
                                      8,      // 8 bits per component
                                      8,      // 8 bits per pixel
                                      grayMat.step[0],  // bytes per row
                                      colorSpace,
                                      kCGImageAlphaNone | kCGBitmapByteOrderDefault,
                                      provider,
                                      NULL,
                                      false,
                                      kCGRenderingIntentDefault);
    
    UIImage *newImage = [UIImage imageWithCGImage:imageRef scale:1.0 orientation:UIImageOrientationUp];
    
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return newImage;
}

+ (UIImage *)applyGaussianBlur:(UIImage *)image kernelSize:(int)kernelSize {
    // Ensure kernel size is odd
    if (kernelSize % 2 == 0) {
        kernelSize += 1;
    }
    
    // Convert UIImage to cv::Mat
    cv::Mat inputMat = [self cvMatFromUIImage:image];
    
    // Apply Gaussian blur with sigma=0 to match Python
    cv::Mat blurredMat;
    cv::GaussianBlur(inputMat, blurredMat, cv::Size(kernelSize, kernelSize), 0);
    
    // Create RGB image with proper format
    CGColorSpaceRef colorSpace;
    CGBitmapInfo bitmapInfo;
    
    if (blurredMat.channels() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
        bitmapInfo = kCGImageAlphaNone | kCGBitmapByteOrderDefault;
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
        bitmapInfo = kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault;
        // Convert to RGBA if it's BGR
        if (blurredMat.channels() == 3) {
            cv::cvtColor(blurredMat, blurredMat, cv::COLOR_BGR2RGBA);
        }
    }
    
    // Create CGImage from mat
    NSData *data = [NSData dataWithBytes:blurredMat.data 
                                  length:blurredMat.elemSize() * blurredMat.total()];
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    CGImageRef imageRef = CGImageCreate(blurredMat.cols,
                                      blurredMat.rows,
                                      8,
                                      8 * blurredMat.channels(),
                                      blurredMat.step[0],
                                      colorSpace,
                                      bitmapInfo,
                                      provider,
                                      NULL,
                                      false,
                                      kCGRenderingIntentDefault);
    
    UIImage *resultImage = [UIImage imageWithCGImage:imageRef 
                                               scale:1.0 
                                         orientation:UIImageOrientationUp];
    
    // Clean up
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return resultImage;
}

+ (UIImage *)applyCannyEdgeDetection:(UIImage *)image threshold1:(double)threshold1 threshold2:(double)threshold2 {
    // Convert UIImage to cv::Mat
    cv::Mat inputMat = [self cvMatFromUIImage:image];
    
    // Convert to grayscale if needed
    cv::Mat grayMat;
    if (inputMat.channels() > 1) {
        cv::cvtColor(inputMat, grayMat, cv::COLOR_BGR2GRAY);
    } else {
        grayMat = inputMat;
    }
    
    // Apply Canny edge detection with default aperture size=3 to match Python
    cv::Mat edgesMat;
    cv::Canny(grayMat, edgesMat, threshold1, threshold2, 3);
    
    // Create grayscale image with proper format
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
    
    // Create CGImage from mat
    NSData *data = [NSData dataWithBytes:edgesMat.data 
                                length:edgesMat.elemSize() * edgesMat.total()];
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    CGImageRef imageRef = CGImageCreate(edgesMat.cols,
                                      edgesMat.rows,
                                      8,
                                      8,  // 8 bits per pixel for grayscale
                                      edgesMat.step[0],
                                      colorSpace,
                                      kCGImageAlphaNone | kCGBitmapByteOrderDefault,
                                      provider,
                                      NULL,
                                      false,
                                      kCGRenderingIntentDefault);
    
    UIImage *resultImage = [UIImage imageWithCGImage:imageRef 
                                             scale:1.0 
                                       orientation:UIImageOrientationUp];
    
    // Clean up
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return resultImage;
}

#pragma mark - Projection Calculation

+ (NSArray<NSNumber *> *)calculateHorizontalProjection:(UIImage *)image {
    // Convert UIImage to cv::Mat
    cv::Mat inputMat = [self cvMatFromUIImage:image];
    
    // Convert to grayscale if needed
    cv::Mat grayMat;
    if (inputMat.channels() > 1) {
        cv::cvtColor(inputMat, grayMat, cv::COLOR_BGR2GRAY);
    } else {
        grayMat = inputMat;
    }
    
    // Sum along width (axis=1 in Python, columns in OpenCV)
    cv::Mat projection;
    cv::reduce(grayMat, projection, 1, cv::REDUCE_SUM, CV_32F);
    
    // Convert to NSArray
    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:projection.rows];
    for (int i = 0; i < projection.rows; i++) {
        float value = projection.at<float>(i, 0);
        [result addObject:@(value)];
    }
    
    return result;
}

+ (NSArray<NSNumber *> *)calculateVerticalProjection:(UIImage *)image {
    // Convert UIImage to cv::Mat
    cv::Mat inputMat = [self cvMatFromUIImage:image];
    
    // Convert to grayscale if needed
    cv::Mat grayMat;
    if (inputMat.channels() > 1) {
        cv::cvtColor(inputMat, grayMat, cv::COLOR_BGR2GRAY);
    } else {
        grayMat = inputMat;
    }
    
    // Sum along height (axis=0 in Python, rows in OpenCV)
    cv::Mat projection;
    cv::reduce(grayMat, projection, 0, cv::REDUCE_SUM, CV_32F);
    
    // Convert to NSArray
    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:projection.cols];
    for (int i = 0; i < projection.cols; i++) {
        float value = projection.at<float>(0, i);
        [result addObject:@(value)];
    }
    
    return result;
}

#pragma mark - Array Processing

+ (NSArray<NSNumber *> *)smoothArray:(NSArray<NSNumber *> *)array kernelSize:(int)kernelSize {
    // Ensure kernel size is odd
    if (kernelSize % 2 == 0) {
        kernelSize += 1;
    }
    
    // Convert NSArray to cv::Mat
    cv::Mat inputMat(1, (int)array.count, CV_32F);
    for (int i = 0; i < array.count; i++) {
        inputMat.at<float>(0, i) = [array[i] floatValue];
    }
    
    // Create kernel for smoothing (similar to np.ones(kernelSize) / kernelSize in Python)
    cv::Mat kernel = cv::Mat::ones(1, kernelSize, CV_32F) / (float)kernelSize;
    
    // Apply convolution (similar to np.convolve in Python)
    cv::Mat smoothedMat;
    cv::filter2D(inputMat, smoothedMat, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    
    // Convert back to NSArray
    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:smoothedMat.cols];
    for (int i = 0; i < smoothedMat.cols; i++) {
        float value = smoothedMat.at<float>(0, i);
        [result addObject:@(value)];
    }
    
    return result;
}

+ (NSArray<NSNumber *> *)findPeaksInArray:(NSArray<NSNumber *> *)array minDistance:(int)minDistance prominence:(double)prominence {
    // This is a simplified peak finding algorithm similar to scipy.signal.find_peaks
    // For a complete implementation, we would need to implement more complex logic
    
    NSMutableArray<NSNumber *> *peaks = [NSMutableArray array];
    
    // Minimum array size needed
    if (array.count < 3) {
        return peaks;
    }
    
    // Find local maxima
    for (int i = 1; i < array.count - 1; i++) {
        float prev = [array[i-1] floatValue];
        float current = [array[i] floatValue];
        float next = [array[i+1] floatValue];
        
        // Check if this point is a local maximum
        if (current > prev && current > next) {
            // Check if it's far enough from existing peaks
            BOOL isFarEnough = YES;
            for (NSNumber *existingPeak in peaks) {
                int existingIndex = [existingPeak intValue];
                if (abs(i - existingIndex) < minDistance) {
                    // Compare heights and keep only the highest
                    if (current > [array[existingIndex] floatValue]) {
                        [peaks removeObject:existingPeak];
                    } else {
                        isFarEnough = NO;
                    }
                    break;
                }
            }
            
            if (isFarEnough) {
                [peaks addObject:@(i)];
            }
        }
    }
    
    // Sort peaks by position
    [peaks sortUsingComparator:^NSComparisonResult(NSNumber *obj1, NSNumber *obj2) {
        return [obj1 compare:obj2];
    }];
    
    return peaks;
}

@end 