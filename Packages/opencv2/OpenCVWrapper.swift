// OpenCVWrapper.swift
// Wrapper to expose OpenCVBridge functionality to the YOLO package

import Foundation
import CoreVideo
import UIKit
import ObjectiveC  // For Objective-C runtime functions
import AVFoundation

/// Interface to the Objective-C OpenCVBridge when used in an iOS app
/// This wrapper enables proper access from Swift package code
@objc public class OpenCVWrapper: NSObject {
    
    /// Check if OpenCV integration is working
    /// - Returns: true if OpenCV is properly integrated
    @objc public static func isOpenCVWorking() -> Bool {
        // Use the Objective-C runtime to dynamically find the OpenCVBridge class
        guard let openCVBridgeClass = NSClassFromString("OpenCVBridge") else {
            print("❌ OpenCVBridge class not found!")
            return false
        }
        
        // Check if the isOpenCVWorking method exists
        let selector = NSSelectorFromString("isOpenCVWorking")
        guard openCVBridgeClass.responds(to: selector) else {
            print("❌ isOpenCVWorking method not found in OpenCVBridge!")
            return false
        }
        
        // Create an instance of OpenCVBridge
        guard let bridge = openCVBridgeClass.alloc() as? NSObject else {
            print("❌ Could not create OpenCVBridge instance!")
            return false
        }
        
        // Call the isOpenCVWorking method
        let result = bridge.perform(selector)
        guard let resultBool = result?.takeUnretainedValue() as? Bool else {
            return false
        }
        
        return resultBool
    }
    
    /// Process a frame with OpenCV to test integration
    /// - Parameter pixelBuffer: The frame to process
    /// - Returns: true if the frame was successfully processed
    @objc public static func processTestFrame(_ pixelBuffer: CVPixelBuffer) -> Bool {
        guard let openCVBridgeClass = NSClassFromString("OpenCVBridge") else {
            print("❌ OpenCVBridge class not found!")
            return false
        }
        
        guard let bridge = openCVBridgeClass.alloc() as? NSObject else {
            print("❌ Could not create OpenCVBridge instance!")
            return false
        }
        
        let selector = NSSelectorFromString("processTestFrame:")
        if bridge.responds(to: selector) {
            let success = bridge.perform(selector, with: pixelBuffer)?.takeUnretainedValue() as? Bool
            return success ?? false
        }
        
        return false
    }
    
    /// Helper function to create a UIImage from a CVPixelBuffer
    private static func createUIImageFromPixelBuffer(_ pixelBuffer: CVPixelBuffer) -> UIImage? {
        if let openCVBridgeClass = NSClassFromString("OpenCVBridge") as? NSObject.Type {
            let selector = NSSelectorFromString("UIImageFromCVPixelBuffer:")
            if openCVBridgeClass.responds(to: selector) {
                // Create an NSInvocation equivalent in Swift
                let methodImp = openCVBridgeClass.method(for: selector)
                if methodImp != nil {
                    // This is a complex type conversion that uses Objective-C runtime
                    typealias FunctionType = @convention(c) (AnyObject, Selector, CVPixelBuffer) -> UIImage?
                    let function = unsafeBitCast(methodImp, to: FunctionType.self)
                    return function(openCVBridgeClass, selector, pixelBuffer)
                }
            }
        }
        
        return nil
    }
    
    /// Helper function to convert a UIImage to CVPixelBuffer
    private static func convertUIImageToPixelBuffer(_ image: UIImage) -> CVPixelBuffer? {
        if let openCVBridgeClass = NSClassFromString("OpenCVBridge") as? NSObject.Type {
            let selector = NSSelectorFromString("CVPixelBufferFromUIImage:")
            if openCVBridgeClass.responds(to: selector) {
                // Create an NSInvocation equivalent in Swift
                let methodImp = openCVBridgeClass.method(for: selector)
                if methodImp != nil {
                    // This is a complex type conversion that uses Objective-C runtime
                    typealias FunctionType = @convention(c) (AnyObject, Selector, UIImage) -> CVPixelBuffer?
                    let function = unsafeBitCast(methodImp, to: FunctionType.self)
                    return function(openCVBridgeClass, selector, image)
                }
            }
        }
        
        return nil
    }
    
    /// Get the OpenCV version string
    /// - Returns: OpenCV version or "Unknown" if not accessible
    @objc public static func getOpenCVVersion() -> String {
        guard let openCVBridgeClass = NSClassFromString("OpenCVBridge") else {
            return "OpenCV Not Accessible"
        }
        
        guard let bridge = openCVBridgeClass.alloc() as? NSObject else {
            return "Could not create OpenCVBridge instance"
        }
        
        let selector = NSSelectorFromString("getOpenCVVersion")
        if bridge.responds(to: selector) {
            let result = bridge.perform(selector)
            if let versionString = result?.takeUnretainedValue() as? String {
                return versionString
            }
        }
        
        return "Unknown"
    }
    
    // MARK: - Image Processing Methods for Calibration
    
    /// Convert an image to grayscale using OpenCV
    /// - Parameter image: The input image
    /// - Returns: A grayscale version of the image, or nil if conversion failed
    @objc public static func convertToGrayscale(_ image: UIImage) -> UIImage? {
        guard let openCVBridgeClass = NSClassFromString("OpenCVBridge") as? NSObject.Type else {
            print("OpenCVBridge class not found")
            return nil
        }
        
        let selector = NSSelectorFromString("convertToGrayscale:")
        
        guard openCVBridgeClass.responds(to: selector) else {
            print("OpenCVBridge does not respond to convertToGrayscale:")
            return nil
        }
        
        let methodImp = class_getMethodImplementation(object_getClass(openCVBridgeClass), selector)
        
        guard methodImp != nil else {
            print("Could not get method implementation for convertToGrayscale:")
            return nil
        }
        
        typealias FunctionType = @convention(c) (AnyObject, Selector, UIImage) -> UIImage?
        let function = unsafeBitCast(methodImp, to: FunctionType.self)
        
        return function(openCVBridgeClass, selector, image)
    }
    
    /// Apply Gaussian blur to an image using OpenCV
    /// - Parameters:
    ///   - image: The input image
    ///   - kernelSize: The size of the blur kernel (must be odd)
    /// - Returns: The blurred image, or nil if processing failed
    @objc public static func applyGaussianBlur(_ image: UIImage, kernelSize: Int) -> UIImage? {
        guard let openCVBridgeClass = NSClassFromString("OpenCVBridge") as? NSObject.Type else {
            print("OpenCVBridge class not found")
            return nil
        }
        
        let selector = NSSelectorFromString("applyGaussianBlur:kernelSize:")
        
        guard openCVBridgeClass.responds(to: selector) else {
            print("OpenCVBridge does not respond to applyGaussianBlur:kernelSize:")
            return nil
        }
        
        let methodImp = class_getMethodImplementation(object_getClass(openCVBridgeClass), selector)
        
        guard methodImp != nil else {
            print("Could not get method implementation for applyGaussianBlur:kernelSize:")
            return nil
        }
        
        typealias FunctionType = @convention(c) (AnyObject, Selector, UIImage, Int) -> UIImage?
        let function = unsafeBitCast(methodImp, to: FunctionType.self)
        
        return function(openCVBridgeClass, selector, image, kernelSize)
    }
    
    /// Apply Canny edge detection to an image using OpenCV
    /// - Parameters:
    ///   - image: The input image
    ///   - threshold1: First threshold for the hysteresis procedure
    ///   - threshold2: Second threshold for the hysteresis procedure
    /// - Returns: An image with detected edges, or nil if processing failed
    @objc public static func applyCannyEdgeDetection(_ image: UIImage, threshold1: Double, threshold2: Double) -> UIImage? {
        guard let openCVBridgeClass = NSClassFromString("OpenCVBridge") as? NSObject.Type else {
            print("OpenCVBridge class not found")
            return nil
        }
        
        let selector = NSSelectorFromString("applyCannyEdgeDetection:threshold1:threshold2:")
        
        guard openCVBridgeClass.responds(to: selector) else {
            print("OpenCVBridge does not respond to applyCannyEdgeDetection:threshold1:threshold2:")
            return nil
        }
        
        let methodImp = class_getMethodImplementation(object_getClass(openCVBridgeClass), selector)
        
        guard methodImp != nil else {
            print("Could not get method implementation for applyCannyEdgeDetection:threshold1:threshold2:")
            return nil
        }
        
        typealias FunctionType = @convention(c) (AnyObject, Selector, UIImage, Double, Double) -> UIImage?
        let function = unsafeBitCast(methodImp, to: FunctionType.self)
        
        return function(openCVBridgeClass, selector, image, threshold1, threshold2)
    }
    
    /// Calculate horizontal projection of an image (sum along width) using OpenCV
    /// - Parameter image: The input image
    /// - Returns: An array of values representing the horizontal projection, or nil if processing failed
    @objc public static func calculateHorizontalProjection(_ image: UIImage) -> [NSNumber]? {
        guard let openCVBridgeClass = NSClassFromString("OpenCVBridge") as? NSObject.Type else {
            print("OpenCVBridge class not found")
            return nil
        }
        
        let selector = NSSelectorFromString("calculateHorizontalProjection:")
        
        guard openCVBridgeClass.responds(to: selector) else {
            print("OpenCVBridge does not respond to calculateHorizontalProjection:")
            return nil
        }
        
        let methodImp = class_getMethodImplementation(object_getClass(openCVBridgeClass), selector)
        
        guard methodImp != nil else {
            print("Could not get method implementation for calculateHorizontalProjection:")
            return nil
        }
        
        typealias FunctionType = @convention(c) (AnyObject, Selector, UIImage) -> [NSNumber]?
        let function = unsafeBitCast(methodImp, to: FunctionType.self)
        
        return function(openCVBridgeClass, selector, image)
    }
    
    /// Calculate vertical projection of an image (sum along height) using OpenCV
    /// - Parameter image: The input image
    /// - Returns: An array of values representing the vertical projection, or nil if processing failed
    @objc public static func calculateVerticalProjection(_ image: UIImage) -> [NSNumber]? {
        guard let openCVBridgeClass = NSClassFromString("OpenCVBridge") as? NSObject.Type else {
            print("OpenCVBridge class not found")
            return nil
        }
        
        let selector = NSSelectorFromString("calculateVerticalProjection:")
        
        guard openCVBridgeClass.responds(to: selector) else {
            print("OpenCVBridge does not respond to calculateVerticalProjection:")
            return nil
        }
        
        let methodImp = class_getMethodImplementation(object_getClass(openCVBridgeClass), selector)
        
        guard methodImp != nil else {
            print("Could not get method implementation for calculateVerticalProjection:")
            return nil
        }
        
        typealias FunctionType = @convention(c) (AnyObject, Selector, UIImage) -> [NSNumber]?
        let function = unsafeBitCast(methodImp, to: FunctionType.self)
        
        return function(openCVBridgeClass, selector, image)
    }
    
    /// Smooth an array of values using a convolution kernel (similar to np.convolve)
    /// - Parameters:
    ///   - array: The input array of values
    ///   - kernelSize: The size of the smoothing kernel
    /// - Returns: A smoothed array of values, or nil if processing failed
    @objc public static func smoothArray(_ array: [NSNumber], kernelSize: Int) -> [NSNumber]? {
        guard let openCVBridgeClass = NSClassFromString("OpenCVBridge") as? NSObject.Type else {
            print("OpenCVBridge class not found")
            return nil
        }
        
        let selector = NSSelectorFromString("smoothArray:kernelSize:")
        
        guard openCVBridgeClass.responds(to: selector) else {
            print("OpenCVBridge does not respond to smoothArray:kernelSize:")
            return nil
        }
        
        let methodImp = class_getMethodImplementation(object_getClass(openCVBridgeClass), selector)
        
        guard methodImp != nil else {
            print("Could not get method implementation for smoothArray:kernelSize:")
            return nil
        }
        
        typealias FunctionType = @convention(c) (AnyObject, Selector, [NSNumber], Int) -> [NSNumber]?
        let function = unsafeBitCast(methodImp, to: FunctionType.self)
        
        return function(openCVBridgeClass, selector, array, kernelSize)
    }
    
    /// Find peaks in an array of values (similar to scipy.signal.find_peaks)
    /// - Parameters:
    ///   - array: The input array of values
    ///   - minDistance: The minimum distance between peaks
    ///   - prominence: The minimum height difference required for a peak
    /// - Returns: An array of indices representing the peak positions, or nil if processing failed
    @objc public static func findPeaksInArray(_ array: [NSNumber], minDistance: Int, prominence: Double) -> [NSNumber]? {
        guard let openCVBridgeClass = NSClassFromString("OpenCVBridge") as? NSObject.Type else {
            print("OpenCVBridge class not found")
            return nil
        }
        
        let selector = NSSelectorFromString("findPeaksInArray:minDistance:prominence:")
        
        guard openCVBridgeClass.responds(to: selector) else {
            print("OpenCVBridge does not respond to findPeaksInArray:minDistance:prominence:")
            return nil
        }
        
        let methodImp = class_getMethodImplementation(object_getClass(openCVBridgeClass), selector)
        
        guard methodImp != nil else {
            print("Could not get method implementation for findPeaksInArray:minDistance:prominence:")
            return nil
        }
        
        typealias FunctionType = @convention(c) (AnyObject, Selector, [NSNumber], Int, Double) -> [NSNumber]?
        let function = unsafeBitCast(methodImp, to: FunctionType.self)
        
        return function(openCVBridgeClass, selector, array, minDistance, prominence)
    }
    
    /// Process a CVPixelBuffer through the calibration pipeline
    /// - Parameters:
    ///   - pixelBuffer: The input pixel buffer
    ///   - direction: The counting direction (vertical or horizontal)
    /// - Returns: NSArray containing threshold values, or nil if processing failed. 
    ///           If successful, the array will contain exactly two CGFloat values.
    @objc public static func processCalibrationFrame(_ pixelBuffer: CVPixelBuffer, isVerticalDirection: Bool) -> NSArray? {
        guard let openCVBridgeClass = NSClassFromString("OpenCVBridge") else {
            print("❌ OpenCVBridge class not found!")
            return nil
        }
        
        guard let bridge = openCVBridgeClass.alloc() as? NSObject else {
            print("❌ Could not create OpenCVBridge instance!")
            return nil
        }
        
        let selector = NSSelectorFromString("processCalibrationFrame:isVerticalDirection:")
        if bridge.responds(to: selector) {
            let result = bridge.perform(selector, with: pixelBuffer, with: isVerticalDirection)
            if let thresholdArray = result?.takeUnretainedValue() as? [Any] {
                if thresholdArray.count == 2 {
                    let threshold1 = thresholdArray[0] as! CGFloat
                    let threshold2 = thresholdArray[1] as! CGFloat
                    return [NSNumber(value: Float(threshold1)), NSNumber(value: Float(threshold2))]
                } else if thresholdArray.count == 1 {
                    let threshold = thresholdArray[0] as! CGFloat
                    let threshold1 = threshold
                    let threshold2 = threshold < 0.5 ? CGFloat(0.7) : CGFloat(0.3)
                    return [NSNumber(value: Float(min(threshold1, threshold2))), NSNumber(value: Float(max(threshold1, threshold2)))]
                } else {
                    print("Unexpected threshold array format")
                    return nil
                }
            }
        }
        
        return nil
    }
}
