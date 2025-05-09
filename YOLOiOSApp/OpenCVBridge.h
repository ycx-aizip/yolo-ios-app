#ifndef OpenCVBridge_h
#define OpenCVBridge_h

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <CoreVideo/CoreVideo.h>

// No OpenCV imports in header - we'll handle them in the implementation

@interface OpenCVBridge : NSObject

// Test method to verify OpenCV integration
+ (BOOL)isOpenCVWorking;

@end

#endif /* OpenCVBridge_h */ 