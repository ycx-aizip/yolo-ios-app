#ifndef OpenCVBridge_h
#define OpenCVBridge_h

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <CoreVideo/CoreVideo.h>

@interface OpenCVBridge : NSObject

// Test methods to verify OpenCV integration
+ (BOOL)isOpenCVWorking;
+ (NSString *)getOpenCVVersion;

// Convert between CVPixelBuffer and UIImage
+ (UIImage *)UIImageFromCVPixelBuffer:(CVPixelBufferRef)pixelBuffer;
+ (CVPixelBufferRef)CVPixelBufferFromUIImage:(UIImage *)image;

// Image processing for calibration
+ (UIImage *)applyCannyEdgeDetection:(UIImage *)image threshold1:(double)threshold1 threshold2:(double)threshold2;
+ (UIImage *)applyGaussianBlur:(UIImage *)image kernelSize:(int)kernelSize;

// Projection calculations for calibration
+ (NSArray<NSNumber *> *)calculateHorizontalProjection:(UIImage *)image;
+ (NSArray<NSNumber *> *)calculateVerticalProjection:(UIImage *)image;

// Peak finding for calibration
+ (NSArray<NSNumber *> *)findPeaksInArray:(NSArray<NSNumber *> *)array minDistance:(int)minDistance prominence:(double)prominence;

// Utility functions
+ (UIImage *)convertToGrayscale:(UIImage *)image;
+ (NSArray<NSNumber *> *)smoothArray:(NSArray<NSNumber *> *)array kernelSize:(int)kernelSize;

// Process test frame for OpenCV verification
- (BOOL)processTestFrame:(CVPixelBufferRef)pixelBuffer;

// Process a single frame for calibration (streaming method)
- (NSArray *)processCalibrationFrame:(CVPixelBufferRef)pixelBuffer isVerticalDirection:(BOOL)isVerticalDirection;

@end

#endif /* OpenCVBridge_h */ 