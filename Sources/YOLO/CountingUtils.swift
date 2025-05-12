// Swift file for CountingDirection enum and calibration utilities
// This file is shared between tracking and counting functionality

import Foundation
import AVFoundation
import CoreVideo
import UIKit
import CoreGraphics
// Import OpenCVWrapper directly since we're in the same module
// No need for @_exported import since we're in the same module

/// Direction for counting objects (fish)
public enum CountingDirection {
    case topToBottom
    case bottomToTop
    case leftToRight
    case rightToLeft
}

/// Calibration utilities for auto threshold detection
public class CalibrationUtils {
    /// Default frame count for calibration (10 seconds at 30fps)
    public static let defaultCalibrationFrameCount = 300  // 10 seconds at 30fps to match Python version
    
    /// Test function to verify OpenCV access from the YOLO package
    /// This comprehensive test checks:
    /// 1. If OpenCV is properly integrated and working
    /// 2. If the version can be retrieved
    /// 3. If frame conversion works correctly
    /// 
    /// - Parameter frame: A sample frame to test with
    /// - Returns: true if all OpenCV tests passed
    public static func testOpenCVAccess(frame: CVPixelBuffer?) -> Bool {
        guard let frame = frame else {
            print("❌ No frame provided for OpenCV test")
            return false
        }
        
        print("Starting OpenCV integration tests...")
        print("Frame dimensions: \(CVPixelBufferGetWidth(frame)) x \(CVPixelBufferGetHeight(frame))")
        
        // Test 1: Basic OpenCV functionality
        let isWorking = OpenCVWrapper.isOpenCVWorking()
        print("Test 1 - Basic OpenCV functionality: \(isWorking ? "✅ PASSED" : "❌ FAILED")")
        
        // If basic test fails, don't continue
        if !isWorking {
            print("❌ Basic OpenCV test failed, aborting further tests")
            return false
        }
        
        // Test 2: Get OpenCV version
        let version = OpenCVWrapper.getOpenCVVersion()
        let versionTest = version != "OpenCV Not Accessible" && version != "Unknown"
        print("Test 2 - OpenCV version: \(versionTest ? "✅ PASSED" : "❌ FAILED") - Version: \(version)")
        
        // Test 3: Process a frame
        let frameProcessed = OpenCVWrapper.processTestFrame(frame)
        print("Test 3 - Frame processing: \(frameProcessed ? "✅ PASSED" : "❌ FAILED")")
        
        let allTestsPassed = isWorking && versionTest && frameProcessed
        print("OpenCV integration test summary: \(allTestsPassed ? "✅ ALL TESTS PASSED" : "❌ SOME TESTS FAILED")")
        
        return allTestsPassed
    }
    
    /// Calculate auto-calibration thresholds based on the provided frames
    /// This method is kept for backward compatibility but is no longer the primary calibration method.
    /// The main calibration now happens in the streaming mode in TrackingDetector.
    ///
    /// - Parameters:
    ///   - frames: Collection of frames (pixel buffers) for processing
    ///   - direction: Current counting direction
    /// - Returns: Tuple with two threshold values (0.0-1.0)
    public static func calculateThresholds(
        from frames: [CVPixelBuffer],
        direction: CountingDirection
    ) -> (threshold1: CGFloat, threshold2: CGFloat) {
        print("Warning: Using legacy buffered calibration with \(frames.count) frames. This method is now deprecated in favor of streaming calibration.")
        
        // Return default values if no frames are provided
        if frames.isEmpty {
            print("⚠️ No frames provided for calibration, using default values")
            return (0.3, 0.7)
        }
        
        // Determine if we're using vertical or horizontal direction
        let isVerticalDirection = direction == .topToBottom || direction == .bottomToTop
        print("Calibration direction: \(isVerticalDirection ? "vertical" : "horizontal")")
        
        // Accumulate results from each frame
        var threshold1Values: [CGFloat] = []
        var threshold2Values: [CGFloat] = []
        
        // Process all frames without skipping to match Python implementation
        print("Processing all \(frames.count) frames for calibration without skipping")
        
        // Process each frame individually
        for (index, frame) in frames.enumerated() {
            print("Processing calibration frame \(index+1)/\(frames.count)")
            
            // Process the frame through OpenCV
            if let thresholdArray = OpenCVWrapper.processCalibrationFrame(frame, isVerticalDirection: isVerticalDirection),
               thresholdArray.count >= 2,
               let value1 = thresholdArray[0] as? NSNumber,
               let value2 = thresholdArray[1] as? NSNumber {
                
                threshold1Values.append(CGFloat(value1.floatValue))
                threshold2Values.append(CGFloat(value2.floatValue))
            }
            
            // Memory management: Force a cleanup after each frame processing
            autoreleasepool {
                // This ensures temporary objects are released
            }
        }
        
        // Return default values if we couldn't get any valid results
        if threshold1Values.isEmpty || threshold2Values.isEmpty {
            print("⚠️ No valid threshold results from calibration, using default values")
            return (0.3, 0.7)
        }
        
        // Calculate the average thresholds from all processed frames
        let avgThreshold1 = threshold1Values.reduce(0, +) / CGFloat(threshold1Values.count)
        let avgThreshold2 = threshold2Values.reduce(0, +) / CGFloat(threshold2Values.count)
        
        print("Calibration complete - Thresholds: (\(avgThreshold1), \(avgThreshold2))")
        
        // Ensure thresholds are properly ordered (smaller one first)
        return (min(avgThreshold1, avgThreshold2), max(avgThreshold1, avgThreshold2))
    }
    
    /// Process a single frame for streaming calibration
    /// This method implements the Python equivalent of processing a frame in the calibration loop
    ///
    /// - Parameters:
    ///   - frame: The current frame to process
    ///   - direction: The counting direction
    /// - Returns: Tuple with two threshold values (0.0-1.0) from this frame, or nil if processing failed
    public static func processCalibrationFrame(
        _ frame: CVPixelBuffer,
        direction: CountingDirection
    ) -> (threshold1: CGFloat, threshold2: CGFloat)? {
        // Determine if we're using vertical or horizontal direction
        let isVerticalDirection = direction == .topToBottom || direction == .bottomToTop
        
        // Process the frame through OpenCV
        if let thresholdArray = OpenCVWrapper.processCalibrationFrame(frame, isVerticalDirection: isVerticalDirection),
           thresholdArray.count >= 2,
           let value1 = thresholdArray[0] as? NSNumber,
           let value2 = thresholdArray[1] as? NSNumber {
            
            let threshold1 = CGFloat(value1.floatValue)
            let threshold2 = CGFloat(value2.floatValue)
            
            // Ensure thresholds are properly ordered (smaller one first)
            return (min(threshold1, threshold2), max(threshold1, threshold2))
        }
        
        return nil
    }
    
    /// Calculate the progress percentage based on collected frames
    ///
    /// - Parameters:
    ///   - currentFrameCount: Number of frames collected so far
    ///   - totalFrameCount: Total number of frames needed
    /// - Returns: Progress percentage (0-100)
    public static func calculateProgress(
        currentFrameCount: Int,
        totalFrameCount: Int
    ) -> Int {
        return Int(min(100, (Double(currentFrameCount) / Double(totalFrameCount)) * 100))
    }
} 