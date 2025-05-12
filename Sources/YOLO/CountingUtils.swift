// Swift file for CountingDirection enum and calibration utilities
// This file is shared between tracking and counting functionality

import Foundation
import AVFoundation
import CoreVideo
import UIKit

/// Direction for counting objects (fish)
public enum CountingDirection {
    case topToBottom
    case bottomToTop
    case leftToRight
    case rightToLeft
}

/// Calibration utilities for auto threshold detection
public class CalibrationUtils {
    /// Default frame count for calibration (approximately 5 seconds at 30fps)
    public static let defaultCalibrationFrameCount = 150
    
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
    /// This is a dummy implementation that simply returns fixed values
    /// after processing a specified number of frames
    ///
    /// - Parameters:
    ///   - frames: Collection of frames (pixel buffers) for processing
    ///   - direction: Current counting direction
    /// - Returns: Tuple with two threshold values (0.0-1.0)
    public static func calculateThresholds(
        from frames: [CVPixelBuffer],
        direction: CountingDirection
    ) -> (threshold1: CGFloat, threshold2: CGFloat) {
        // Dummy implementation - In the future, this will analyze frames
        // to find optimal threshold positions based on visual features
        
        // For now, just return fixed values regardless of input
        return (0.1, 0.9)
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