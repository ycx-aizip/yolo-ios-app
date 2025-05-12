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
    
    // Create a UIImage from the CGImage
    UIImage *image = [UIImage imageWithCGImage:quartzImage];
    
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

@end 