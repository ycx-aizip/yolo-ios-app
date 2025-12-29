// Unified Coordinate Transformation System for YOLO iOS App
// This system establishes a single intermediate coordinate space that all sources convert to,
// then provides unified transformations to UI, counting, and detection display.

import Foundation
import CoreGraphics
import Vision
import AVFoundation

// Note: CountingDirection and FrameSource are defined in the same YOLO module
// so they should be accessible without additional imports

// MARK: - Unified Coordinate System Definition

/// The unified intermediate coordinate system used throughout the app
/// 
/// **Standard Definition:**
/// - Origin: Top-left corner (0,0)
/// - X-axis: Left to right (0 to 1)
/// - Y-axis: Top to bottom (0 to 1)
/// - Orientation: Landscape Right (USB cable on left side)
/// - Aspect Ratio: Maintained from source content
/// 
/// **Key Benefits:**
/// - Single coordinate space eliminates transformation complexity
/// - Fixed orientation removes orientation-dependent logic
/// - Consistent across all frame sources and UI elements
public struct UnifiedCoordinateSystem {
    
    // MARK: - Core Coordinate Definitions
    
    /// Normalized rectangle in unified coordinate space
    public struct UnifiedRect {
        public let x: CGFloat      // 0.0 = left edge, 1.0 = right edge
        public let y: CGFloat      // 0.0 = top edge, 1.0 = bottom edge  
        public let width: CGFloat  // Normalized width
        public let height: CGFloat // Normalized height
        
        public init(x: CGFloat, y: CGFloat, width: CGFloat, height: CGFloat) {
            self.x = x
            self.y = y
            self.width = width
            self.height = height
        }
        
        /// Convert to CGRect for compatibility
        public var cgRect: CGRect {
            return CGRect(x: x, y: y, width: width, height: height)
        }
        
        /// Center point of the rectangle
        public var center: (x: CGFloat, y: CGFloat) {
            return (x: x + width/2, y: y + height/2)
        }
    }
    
    /// Point in unified coordinate space
    public struct UnifiedPoint {
        public let x: CGFloat  // 0.0 = left, 1.0 = right
        public let y: CGFloat  // 0.0 = top, 1.0 = bottom
        
        public init(x: CGFloat, y: CGFloat) {
            self.x = x
            self.y = y
        }
        
        /// Convert to CGPoint for compatibility
        public var cgPoint: CGPoint {
            return CGPoint(x: x, y: y)
        }
    }
    
    // MARK: - Source-to-Unified Transformations
    
    /// Transforms coordinates from Vision framework to unified coordinate system
    /// 
    /// Vision uses bottom-left origin with Y inverted, we need top-left origin
    /// - Parameter visionRect: Rectangle from Vision framework (bottom-left origin)
    /// - Returns: Rectangle in unified coordinate system (top-left origin)
    public static func fromVision(_ visionRect: CGRect) -> UnifiedRect {
        // Vision: (0,0) = bottom-left, Y increases upward
        // Unified: (0,0) = top-left, Y increases downward
        // For landscape right orientation, Vision coordinates need proper transformation
        let unifiedY = 1.0 - visionRect.origin.y - visionRect.height
        
        // For landscape right orientation (USB on left), Vision X coordinates 
        // map directly to unified X coordinates without flipping
        return UnifiedRect(
            x: visionRect.origin.x,
            y: unifiedY,
            width: visionRect.width,
            height: visionRect.height
        )
    }
    
    /// Transforms coordinates from Camera source to unified coordinate system
    /// 
    /// Camera coordinates depend on session preset and need aspect ratio correction
    /// - Parameters:
    ///   - cameraRect: Rectangle from camera detection
    ///   - sessionPreset: Camera session preset for aspect ratio calculation
    /// - Returns: Rectangle in unified coordinate system
    public static func fromCamera(_ cameraRect: CGRect, sessionPreset: AVCaptureSession.Preset) -> UnifiedRect {
        // Camera coordinates are already in the correct orientation for landscape right
        // For landscape right orientation, we need to ensure proper coordinate mapping
        
        var adjustedRect = cameraRect
        
        // Apply aspect ratio correction based on session preset
        // For landscape right, the camera frame might need different handling
        switch sessionPreset {
        case .hd1280x720:
            // 16:9 aspect ratio - this is our standard, no adjustment needed
            break
        case .photo:
            // 4:3 aspect ratio - needs adjustment for landscape right display
            // In landscape right, width/height relationship changes
            let ratio = (16.0/9.0) / (4.0/3.0) // ~1.33
            adjustedRect.origin.y = (adjustedRect.origin.y - 0.5) * ratio + 0.5
            adjustedRect.size.height *= ratio
            break
        default:
            // Default to 16:9 handling
            break
        }
        
        return UnifiedRect(
            x: adjustedRect.origin.x,
            y: adjustedRect.origin.y,
            width: adjustedRect.width,
            height: adjustedRect.height
        )
    }
    
    /// Transforms coordinates from Camera with resizeAspectFill to unified coordinate system
    /// 
    /// This method handles the coordinate transformation for cameras using resizeAspectFill
    /// video gravity, which crops the camera feed to fill the entire display area.
    /// - Parameters:
    ///   - cameraRect: Rectangle from camera detection (already converted from Vision)
    ///   - cameraSize: Actual camera frame size (width: longSide, height: shortSide)
    ///   - displayBounds: Display area bounds where the camera feed is shown
    /// - Returns: Rectangle in unified coordinate system
    public static func fromCameraWithAspectFill(_ cameraRect: CGRect, cameraSize: CGSize, displayBounds: CGRect) -> UnifiedRect {
        // Camera uses resizeAspectFill, so we need to account for cropping
        let cameraAspect = cameraSize.width / cameraSize.height
        let displayAspect = displayBounds.width / displayBounds.height
        
        var adjustedRect = cameraRect
        
        if cameraAspect > displayAspect {
            // Camera is wider than display - cropping left/right edges
            // The visible area is the center portion of the camera frame
            let visibleWidth = cameraSize.height * displayAspect
            let cropOffset = (cameraSize.width - visibleWidth) / (2 * cameraSize.width)
            
            // Adjust X coordinates to account for cropping
            adjustedRect.origin.x = (cameraRect.origin.x - cropOffset) / (1.0 - 2 * cropOffset)
            adjustedRect.size.width = cameraRect.width / (1.0 - 2 * cropOffset)
            
            // Clamp to valid range
            adjustedRect.origin.x = max(0, min(1.0 - adjustedRect.size.width, adjustedRect.origin.x))
            adjustedRect.size.width = min(1.0, adjustedRect.size.width)
            
        } else if cameraAspect < displayAspect {
            // Camera is taller than display - cropping top/bottom edges
            // The visible area is the center portion of the camera frame
            let visibleHeight = cameraSize.width / displayAspect
            let cropOffset = (cameraSize.height - visibleHeight) / (2 * cameraSize.height)
            
            // Adjust Y coordinates to account for cropping
            adjustedRect.origin.y = (cameraRect.origin.y - cropOffset) / (1.0 - 2 * cropOffset)
            adjustedRect.size.height = cameraRect.height / (1.0 - 2 * cropOffset)
            
            // Clamp to valid range
            adjustedRect.origin.y = max(0, min(1.0 - adjustedRect.size.height, adjustedRect.origin.y))
            adjustedRect.size.height = min(1.0, adjustedRect.size.height)
        }
        // If aspects are equal, no cropping occurs
        
        return UnifiedRect(
            x: adjustedRect.origin.x,
            y: adjustedRect.origin.y,
            width: adjustedRect.width,
            height: adjustedRect.height
        )
    }
    
    /// Transforms coordinates from Album video source to unified coordinate system
    /// 
    /// Album videos may have different aspect ratios and orientations
    /// - Parameters:
    ///   - albumRect: Rectangle from album video detection
    ///   - videoSize: Original video dimensions
    ///   - displayBounds: Display area bounds
    /// - Returns: Rectangle in unified coordinate system
    public static func fromAlbum(_ albumRect: CGRect, videoSize: CGSize, displayBounds: CGRect) -> UnifiedRect {
        // Album videos are displayed with aspect fit, so we need to account for letterboxing/pillarboxing
        let videoAspect = videoSize.width / videoSize.height
        let displayAspect = displayBounds.width / displayBounds.height
        
        var adjustedRect = albumRect
        
        if videoAspect > displayAspect {
            // Video is wider - letterboxing (black bars top/bottom)
            let scaledHeight = displayBounds.width / videoAspect
            let verticalOffset = (displayBounds.height - scaledHeight) / (2 * displayBounds.height)
            let yScale = scaledHeight / displayBounds.height
            
            adjustedRect.origin.y = (albumRect.origin.y * yScale) + verticalOffset
            adjustedRect.size.height *= yScale
        } else if videoAspect < displayAspect {
            // Video is taller - pillarboxing (black bars left/right)
            let scaledWidth = displayBounds.height * videoAspect
            let horizontalOffset = (displayBounds.width - scaledWidth) / (2 * displayBounds.width)
            let xScale = scaledWidth / displayBounds.width
            
            adjustedRect.origin.x = (albumRect.origin.x * xScale) + horizontalOffset
            adjustedRect.size.width *= xScale
        }
        
        return UnifiedRect(
            x: adjustedRect.origin.x,
            y: adjustedRect.origin.y,
            width: adjustedRect.width,
            height: adjustedRect.height
        )
    }
    
    /// Transforms coordinates from GoPro source to unified coordinate system
    /// 
    /// GoPro streams are typically 16:9 and need aspect ratio handling
    /// - Parameters:
    ///   - goProRect: Rectangle from GoPro detection
    ///   - streamSize: GoPro stream dimensions
    ///   - displayBounds: Display area bounds
    /// - Returns: Rectangle in unified coordinate system
    public static func fromGoPro(_ goProRect: CGRect, streamSize: CGSize, displayBounds: CGRect) -> UnifiedRect {
        // GoPro streams are typically 16:9, similar to album handling
        return fromAlbum(goProRect, videoSize: streamSize, displayBounds: displayBounds)
    }
    
    /// Transforms coordinates from UVC source to unified coordinate system
    /// 
    /// UVC cameras may have various aspect ratios and orientations
    /// - Parameters:
    ///   - uvcRect: Rectangle from UVC detection
    ///   - sourceSize: UVC source dimensions
    ///   - displayBounds: Display area bounds
    /// - Returns: Rectangle in unified coordinate system
    public static func fromUVC(_ uvcRect: CGRect, sourceSize: CGSize, displayBounds: CGRect) -> UnifiedRect {
        // UVC sources can vary widely, use generic video handling
        return fromAlbum(uvcRect, videoSize: sourceSize, displayBounds: displayBounds)
    }
    
    // MARK: - Unified-to-Target Transformations
    
    /// Transforms unified coordinates to screen coordinates for UI display
    /// 
    /// Converts from normalized unified coordinates to pixel coordinates for drawing
    /// - Parameters:
    ///   - unifiedRect: Rectangle in unified coordinate system
    ///   - screenBounds: Screen bounds for pixel conversion
    /// - Returns: Rectangle in screen pixel coordinates
    public static func toScreen(_ unifiedRect: UnifiedRect, screenBounds: CGRect) -> CGRect {
        return CGRect(
            x: unifiedRect.x * screenBounds.width,
            y: unifiedRect.y * screenBounds.height,
            width: unifiedRect.width * screenBounds.width,
            height: unifiedRect.height * screenBounds.height
        )
    }
    
    /// Transforms unified coordinates to counting coordinates for threshold detection
    /// 
    /// Provides coordinates for fish counting logic with direction-specific adjustments
    /// - Parameters:
    ///   - unifiedRect: Rectangle in unified coordinate system
    ///   - countingDirection: Direction of fish movement for counting
    /// - Returns: Rectangle adjusted for counting logic
    public static func toCounting(_ unifiedRect: UnifiedRect, countingDirection: CountingDirection) -> UnifiedRect {
        var countingRect = unifiedRect
        
        // Apply direction-specific coordinate adjustments for counting logic
        switch countingDirection {
        case .topToBottom, .bottomToTop:
            // Vertical movement - use Y coordinates directly
            // No transformation needed as unified system already uses top-to-bottom Y
            break
            
        case .leftToRight, .rightToLeft:
            // Horizontal movement - use X coordinates directly
            // No transformation needed as unified system already uses left-to-right X
            break
        }
        
        return countingRect
    }
    
    /// Transforms unified threshold values to display threshold positions
    /// 
    /// Converts threshold values to screen positions for line drawing
    /// - Parameters:
    ///   - thresholds: Array of threshold values (0.0-1.0)
    ///   - countingDirection: Direction of counting for proper line orientation
    ///   - screenBounds: Screen bounds for positioning
    /// - Returns: Array of line positions for drawing
    public static func thresholdsToScreen(_ thresholds: [CGFloat], countingDirection: CountingDirection, screenBounds: CGRect) -> [CGRect] {
        return thresholds.map { threshold in
            switch countingDirection {
            case .topToBottom:
                // Horizontal lines for vertical movement - threshold as Y position
                return CGRect(
                    x: 0,
                    y: threshold * screenBounds.height,
                    width: screenBounds.width,
                    height: 2 // Line thickness
                )
                
            case .bottomToTop:
                // Horizontal lines for vertical movement - flipped Y for user intuition
                // User expects threshold 0.3 to appear 30% from bottom
                return CGRect(
                    x: 0,
                    y: (1.0 - threshold) * screenBounds.height,
                    width: screenBounds.width,
                    height: 2 // Line thickness
                )
                
            case .leftToRight:
                // Vertical lines for horizontal movement - threshold as X position
                return CGRect(
                    x: threshold * screenBounds.width,
                    y: 0,
                    width: 2, // Line thickness
                    height: screenBounds.height
                )
                
            case .rightToLeft:
                // Vertical lines for horizontal movement - flipped X for user intuition
                // User expects threshold 0.3 to appear 30% from right
                return CGRect(
                    x: (1.0 - threshold) * screenBounds.width,
                    y: 0,
                    width: 2, // Line thickness
                    height: screenBounds.height
                )
            }
        }
    }
    
    /// Transforms display threshold values to counting threshold values
    /// 
    /// Converts user-friendly threshold display values to internal counting values
    /// - Parameters:
    ///   - displayThresholds: Threshold values as shown to user (0.0-1.0)
    ///   - countingDirection: Direction of counting
    /// - Returns: Threshold values for internal counting logic
    public static func displayToCounting(_ displayThresholds: [CGFloat], countingDirection: CountingDirection) -> [CGFloat] {
        switch countingDirection {
        case .bottomToTop:
            // For bottomToTop, display is flipped but counting uses original coordinates
            return displayThresholds.map { 1.0 - $0 }
            
        case .rightToLeft:
            // For rightToLeft, display is flipped but counting uses original coordinates
            return displayThresholds.map { 1.0 - $0 }
            
        case .topToBottom, .leftToRight:
            // No transformation needed - display matches counting
            return displayThresholds
        }
    }
    
    /// Transforms counting threshold values to display threshold values
    /// 
    /// Converts internal counting values to user-friendly display values
    /// - Parameters:
    ///   - countingThresholds: Internal threshold values (0.0-1.0)
    ///   - countingDirection: Direction of counting
    /// - Returns: Threshold values for display to user
    public static func countingToDisplay(_ countingThresholds: [CGFloat], countingDirection: CountingDirection) -> [CGFloat] {
        switch countingDirection {
        case .bottomToTop:
            // For bottomToTop, counting uses original coordinates but display is flipped
            return countingThresholds.map { 1.0 - $0 }
            
        case .rightToLeft:
            // For rightToLeft, counting uses original coordinates but display is flipped
            return countingThresholds.map { 1.0 - $0 }
            
        case .topToBottom, .leftToRight:
            // No transformation needed - counting matches display
            return countingThresholds
        }
    }
    
    // MARK: - Utility Methods
    
    /// Validates that a rectangle is within the unified coordinate bounds
    /// - Parameter rect: Rectangle to validate
    /// - Returns: True if rectangle is valid (within 0.0-1.0 bounds)
    public static func isValid(_ rect: UnifiedRect) -> Bool {
        return rect.x >= 0.0 && rect.y >= 0.0 && 
               rect.x + rect.width <= 1.0 && rect.y + rect.height <= 1.0 &&
               rect.width > 0.0 && rect.height > 0.0
    }
    
    /// Clamps a rectangle to valid unified coordinate bounds
    /// - Parameter rect: Rectangle to clamp
    /// - Returns: Rectangle clamped to 0.0-1.0 bounds
    public static func clamp(_ rect: UnifiedRect) -> UnifiedRect {
        let clampedX = max(0.0, min(1.0, rect.x))
        let clampedY = max(0.0, min(1.0, rect.y))
        let maxWidth = 1.0 - clampedX
        let maxHeight = 1.0 - clampedY
        let clampedWidth = max(0.0, min(maxWidth, rect.width))
        let clampedHeight = max(0.0, min(maxHeight, rect.height))
        
        return UnifiedRect(x: clampedX, y: clampedY, width: clampedWidth, height: clampedHeight)
    }
    
    /// Converts a center point and size to a unified rectangle
    /// - Parameters:
    ///   - center: Center point in unified coordinates
    ///   - size: Size in unified coordinates
    /// - Returns: Rectangle centered at the given point
    public static func fromCenter(_ center: UnifiedPoint, size: (width: CGFloat, height: CGFloat)) -> UnifiedRect {
        return UnifiedRect(
            x: center.x - size.width/2,
            y: center.y - size.height/2,
            width: size.width,
            height: size.height
        )
    }
}

// MARK: - Integration Extensions
// Extensions will be added after FrameSource protocol is available

// MARK: - Migration Helper

/// Helper class to migrate existing coordinate transformation code to unified system
public class CoordinateSystemMigration {
    
    /// Identifies coordinate transformation calls that need to be migrated
    /// - Parameter sourceFile: Source file path to analyze
    /// - Returns: List of transformation calls that need updating
    public static func identifyTransformationCalls(in sourceFile: String) -> [String] {
        // This would be implemented to help identify existing transformation code
        // For now, return common patterns to look for
        return [
            "transformDetectionToScreenCoordinates",
            "VNImageRectForNormalizedRect", 
            "convertNormalizedRectToScreenRect",
            "1.0 - rect.origin.y",  // Y-coordinate flipping
            "rect.applying(transform)", // CGAffineTransform usage
            "videoToScreenScale", // Manual scaling
            "videoToScreenOffset" // Manual offset
        ]
    }
    
    /// Provides migration guidance for common transformation patterns
    /// - Parameter pattern: The transformation pattern found
    /// - Returns: Suggested replacement using unified coordinate system
    public static func migrationGuidance(for pattern: String) -> String {
        switch pattern {
        case "transformDetectionToScreenCoordinates":
            return """
            Replace with:
            let unified = frameSource.toUnifiedCoordinates(rect, additionalInfo: info)
            let screen = UnifiedCoordinateSystem.toScreen(unified, screenBounds: bounds)
            """
            
        case "VNImageRectForNormalizedRect":
            return """
            Replace with:
            let unified = UnifiedCoordinateSystem.fromVision(visionRect)
            let screen = UnifiedCoordinateSystem.toScreen(unified, screenBounds: bounds)
            """
            
        case "1.0 - rect.origin.y":
            return """
            Replace with:
            let unified = UnifiedCoordinateSystem.fromVision(rect) // Handles Y-flip automatically
            """
            
        default:
            return "Consider using UnifiedCoordinateSystem for this transformation"
        }
    }
} 