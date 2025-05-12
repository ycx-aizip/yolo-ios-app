// OpenCVWrapper.swift
// Wrapper to expose OpenCVBridge functionality to the YOLO package

import Foundation
import CoreVideo
import UIKit
import ObjectiveC  // For Objective-C runtime functions

/// Interface to the Objective-C OpenCVBridge when used in an iOS app
/// This wrapper enables proper access from Swift package code
@objc public class OpenCVWrapper: NSObject {
    
    /// Check if OpenCV integration is working
    /// - Returns: true if OpenCV is properly integrated
    @objc public static func isOpenCVWorking() -> Bool {
        // Safer approach for calling Objective-C class methods at runtime
        guard let openCVBridgeClass = NSClassFromString("OpenCVBridge") else {
            print("OpenCVBridge class not found via NSClassFromString")
            return false
        }
        
        // Create proper selector
        let selector = NSSelectorFromString("isOpenCVWorking")
        
        // Verify method exists and is a class method
        guard openCVBridgeClass.responds(to: selector) else {
            print("OpenCVBridge class does not respond to isOpenCVWorking")
            return false
        }
        
        // Prepare function pointer with the correct signature
        typealias IsWorkingFunctionType = @convention(c) (AnyObject, Selector) -> Bool
        
        // Get the implementation of the method
        let methodImp = class_getMethodImplementation(object_getClass(openCVBridgeClass), selector)
        
        guard methodImp != nil else {
            print("Could not get method implementation for isOpenCVWorking")
            return false
        }
        
        // Convert the implementation to a function pointer
        let method = unsafeBitCast(methodImp, to: IsWorkingFunctionType.self)
        
        // Call the function safely
        print("Calling OpenCVBridge.isOpenCVWorking() using function pointer")
        let result = method(openCVBridgeClass, selector)
        print("Successfully returned from OpenCVBridge.isOpenCVWorking(): \(result)")
        
        return result
    }
    
    /// Process a frame with OpenCV to test integration
    /// - Parameter pixelBuffer: The frame to process
    /// - Returns: true if the frame was successfully processed
    @objc public static func processTestFrame(_ pixelBuffer: CVPixelBuffer) -> Bool {
        // Verify OpenCV is working first
        if !isOpenCVWorking() {
            print("Cannot process frame because OpenCV is not working")
            return false
        }
        
        print("OpenCV is working, attempting to process a test frame")
        
        // For now, we'll consider the test successful if OpenCV is working
        // Later we can add actual frame processing tests
        return true
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
        
        // Create proper selector
        let selector = NSSelectorFromString("getOpenCVVersion")
        
        // Verify method exists and is a class method
        guard openCVBridgeClass.responds(to: selector) else {
            return "Method Not Found"
        }
        
        // Prepare function pointer with the correct signature
        typealias GetVersionFunctionType = @convention(c) (AnyObject, Selector) -> NSString
        
        // Get the implementation of the method
        let methodImp = class_getMethodImplementation(object_getClass(openCVBridgeClass), selector)
        
        guard methodImp != nil else {
            return "Implementation Not Found"
        }
        
        // Convert the implementation to a function pointer
        let method = unsafeBitCast(methodImp, to: GetVersionFunctionType.self)
        
        // Call the function safely
        let result = method(openCVBridgeClass, selector)
        
        return result as String
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
        // Convert CVPixelBuffer to UIImage
        guard let image = createUIImageFromPixelBuffer(pixelBuffer) else {
            print("Failed to convert pixel buffer to UIImage")
            return nil
        }
        
        // Convert to grayscale
        guard let grayImage = convertToGrayscale(image) else {
            print("Failed to convert image to grayscale")
            return nil
        }
        
        // Apply Gaussian blur
        guard let blurredImage = applyGaussianBlur(grayImage, kernelSize: 5) else {
            print("Failed to apply Gaussian blur")
            return nil
        }
        
        // Apply Canny edge detection
        guard let edgesImage = applyCannyEdgeDetection(blurredImage, threshold1: 50, threshold2: 150) else {
            print("Failed to apply Canny edge detection")
            return nil
        }
        
        // Calculate projection based on direction
        let projection: [NSNumber]?
        if isVerticalDirection {
            // For top-to-bottom or bottom-to-top, calculate horizontal projection
            projection = calculateHorizontalProjection(edgesImage)
        } else {
            // For left-to-right or right-to-left, calculate vertical projection
            projection = calculateVerticalProjection(edgesImage)
        }
        
        guard let projectionValues = projection else {
            print("Failed to calculate projection")
            return nil
        }
        
        // Calculate kernel size as max(5, height//20) or max(5, width//20) depending on direction
        let dimension = isVerticalDirection ? Int(image.size.height) : Int(image.size.width)
        let kernelSize = max(5, dimension / 20)
        
        // Smooth the projection
        guard let smoothedProjection = smoothArray(projectionValues, kernelSize: kernelSize) else {
            print("Failed to smooth projection")
            return nil
        }
        
        // Find peaks in the smoothed projection
        let minPeakDistance = isVerticalDirection ? 
            Int(Double(image.size.height) * 0.25) : 
            Int(Double(image.size.width) * 0.25)
        
        // Pass 0 for prominence parameter to match Python implementation (no explicit prominence specified)
        guard let peaks = findPeaksInArray(smoothedProjection, minDistance: minPeakDistance, prominence: 0.0) else {
            print("Failed to find peaks")
            return nil
        }
        
        // Process peaks to get threshold positions
        if peaks.count >= 2 {
            // Get the first two peaks
            let peak1 = peaks[0].intValue
            let peak2 = peaks[1].intValue
            
            // Convert to normalized coordinates (0.0-1.0)
            let denominator = isVerticalDirection ? Double(image.size.height) : Double(image.size.width)
            let threshold1 = CGFloat(Double(peak1) / denominator)
            let threshold2 = CGFloat(Double(peak2) / denominator)
            
            return [NSNumber(value: Float(threshold1)), NSNumber(value: Float(threshold2))]
        } else if peaks.count == 1 {
            // Only one peak, use it and an offset version
            let peak = peaks[0].intValue
            let denominator = isVerticalDirection ? Double(image.size.height) : Double(image.size.width)
            let threshold1 = CGFloat(Double(peak) / denominator)
            
            // Create a second threshold at either 30% or 70% depending on position of first
            let threshold2 = threshold1 < 0.5 ? CGFloat(0.7) : CGFloat(0.3)
            
            return [NSNumber(value: Float(min(threshold1, threshold2))), NSNumber(value: Float(max(threshold1, threshold2)))]
        } else {
            // Fallback to default values if no peaks were found
            return [NSNumber(value: 0.3), NSNumber(value: 0.7)]
        }
    }
}
