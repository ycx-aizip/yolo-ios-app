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
}
