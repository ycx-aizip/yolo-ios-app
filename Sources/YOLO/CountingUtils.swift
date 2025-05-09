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