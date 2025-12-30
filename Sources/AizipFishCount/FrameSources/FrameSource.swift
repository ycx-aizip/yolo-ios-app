// from Aizip
//
//  The FrameSource component provides an abstraction layer for different sources of video frames,
//  such as camera feeds, video files, or image sequences. This protocol-based design enables
//  the YOLO model to process frames regardless of their origin, maintaining a consistent
//  pipeline from frame acquisition to model inference and result visualization. The protocol
//  defines essential methods for starting and stopping frame delivery, and delegates for
//  receiving frames and performance metrics.

import AVFoundation
import CoreVideo
import UIKit
import Vision

/// Protocol for receiving frames from any frame source.
@MainActor
protocol FrameSourceDelegate: AnyObject {
    /// Called when a new frame is available from the source.
    ///
    /// - Parameters:
    ///   - source: The frame source providing the frame.
    ///   - image: The UIImage containing the frame data.
    func frameSource(_ source: FrameSource, didOutputImage image: UIImage)
    
    /// Called when performance metrics are updated.
    ///
    /// - Parameters:
    ///   - source: The frame source providing the metrics.
    ///   - speed: The time in milliseconds taken to process a frame.
    ///   - fps: The frames per second rate.
    func frameSource(_ source: FrameSource, didUpdateWithSpeed speed: Double, fps: Double)
}

/// Protocol defining the minimal interface needed for processing frames
/// This simply uses Predictor directly, avoiding protocol extension issues
typealias FrameProcessor = Predictor

/// Standard settings and utilities for frame sources
enum FrameSourceSettings {
    /// Standard pixel format to use for video file sources (used in AlbumVideoSource)
    static let videoSourcePixelFormat = kCVPixelFormatType_32BGRA
    
    /// Standard pixel format to use for camera sources (used in CameraVideoSource)
    static let cameraSourcePixelFormat = kCVPixelFormatType_32BGRA
    
    /// Performance logging configuration - automatically disabled in release builds
    #if DEBUG
    static let enablePerformanceLogging = true
    #else
    static let enablePerformanceLogging = false
    #endif
    
    /// Frame interval for performance logging (every N frames)
    static let performanceLoggingInterval = 300
    
    /// Standard documentation for the expected input format for predictors
    /// This provides guidance on what frame sources should provide to predictors
    static let predictorInputFormatDescription = """
    The standard input format for YOLO predictors depends on the source type:
    
    For Camera Sources:
    - Pixel Format: 32-bit BGRA (kCVPixelFormatType_32BGRA)
    - Orientation: Device orientation corrected (right-side up)
    - Color Space: Device RGB color space
    - Alpha: Premultiplied alpha
    
    For Video File Sources:
    - Pixel Format: 32-bit BGRA (kCVPixelFormatType_32BGRA)
    - Orientation: Device orientation corrected (right-side up)
    - Color Space: Device RGB color space
    - Alpha: Premultiplied alpha
    
    All sources should provide a CMSampleBuffer with valid presentationTimeStamp for timing information.
    """
}

/// Protocol defining a common interface for all frame sources.
protocol FrameSource: AnyObject {
    /// The delegate to receive frames and performance metrics.
    var delegate: FrameSourceDelegate? { get set }
    
    /// The preview layer for displaying the source's visual output.
    var previewLayer: AVCaptureVideoPreviewLayer? { get }
    
    /// The processor used to process frames from this source.
    var predictor: FrameProcessor! { get set }
    
    /// The long side dimension of the frames produced by this source.
    var longSide: CGFloat { get }
    
    /// The short side dimension of the frames produced by this source.
    var shortSide: CGFloat { get }
    
    /// Flag indicating if inference should be performed on frames
    /// When false, frames can be directed to calibration pipeline instead
    var inferenceOK: Bool { get set }
    
    /// Begins frame acquisition from the source.
    nonisolated func start()
    
    /// Stops frame acquisition from the source.
    nonisolated func stop()
    
    /// Sets up the frame source with specified configuration.
    ///
    /// - Parameters:
    ///   - completion: Called when setup is complete, with a Boolean indicating success.
    @MainActor
    func setUp(completion: @escaping @Sendable (Bool) -> Void)
    
    /// Captures a still image from the frame source.
    ///
    /// - Parameters:
    ///   - completion: Callback with the captured image, or nil if capture failed.
    @MainActor
    func capturePhoto(completion: @escaping @Sendable (UIImage?) -> Void)
    
    /// Sets the zoom level for the frame source, if supported.
    ///
    /// - Parameter ratio: The zoom ratio to apply.
    nonisolated func setZoomRatio(ratio: CGFloat)
    
    /// The source type identifier.
    var sourceType: FrameSourceType { get }
    
    /// Request permission to use this frame source, if required.
    /// - Parameter completion: Called with the result of the permission request.
    @MainActor
    func requestPermission(completion: @escaping (Bool) -> Void)
    
    /// Updates the source for orientation changes
    /// - Parameter orientation: The new device orientation
    @MainActor
    func updateForOrientationChange(orientation: UIDeviceOrientation)
    
    /// Shows UI for selecting content for this source, if applicable.
    /// - Parameter viewController: The view controller to present the UI from.
    /// - Parameter completion: Called when the selection is complete.
    @MainActor
    func showContentSelectionUI(from viewController: UIViewController, completion: @escaping (Bool) -> Void)
    
    // MARK: - New methods for coordinate transformation
    
    /// Transforms normalized detection coordinates to screen coordinates based on the source's specific characteristics
    /// - Parameters:
    ///   - rect: The normalized detection rectangle (0.0-1.0)
    ///   - viewBounds: The bounds of the view where the detection will be displayed
    ///   - orientation: The current device orientation
    /// - Returns: A rectangle in screen coordinates
    @MainActor
    func transformDetectionToScreenCoordinates(
        rect: CGRect, 
        viewBounds: CGRect, 
        orientation: UIDeviceOrientation
    ) -> CGRect
    
    // MARK: - New methods for UI integration
    
    /// Integrates the source with a YOLOView for proper display and interaction
    /// - Parameter view: The YOLOView to integrate with
    @MainActor
    func integrateWithYOLOView(view: UIView)
    
    /// Adds a layer to the source's display hierarchy
    /// - Parameter layer: The layer to add
    @MainActor
    func addOverlayLayer(_ layer: CALayer)
    
    /// Adds bounding box views to the source's display hierarchy
    /// - Parameter boxViews: The bounding box views to add
    @MainActor
    func addBoundingBoxViews(_ boxViews: [BoundingBoxView])
    
    /// Resets processing state to allow normal inference to resume after calibration
    /// This method should be called when calibration completes to ensure the frame source
    /// is ready to process frames for normal inference again
    @MainActor
    func resetProcessingState()
}

/// Enumeration of available frame source types.
public enum FrameSourceType {
    case camera
    case videoFile
    case imageSequence
    case uvc
    // Add more source types as needed
}

/// Extension to provide standard frame processing utilities for all frame sources
// Consistency: Both sources now use the same pixel format and image conversion logic
// Maintainability: Changes to image conversion only need to be made in one place
// Performance: AlbumVideoSource now uses the same efficient conversion as CameraVideoSource
// Documentation: The expected input format is now clearly described
// Future-proofing: New frame sources can easily follow the same patterns
// While the frame acquisition is still different between sources (which is necessary and appropriate), the frame preparation for prediction is now standardized. This ensures that the predictor receives consistent input regardless of the source, which should help maintain inference quality across different sources.
extension FrameSource {
    /// Creates a consistent and standardized sample buffer from a pixel buffer.
    /// This ensures all frame sources provide the same format to the predictor.
    /// 
    /// - Parameter pixelBuffer: The source pixel buffer to process
    /// - Returns: A properly formatted CMSampleBuffer for prediction, or nil if creation fails
    func createStandardSampleBuffer(from pixelBuffer: CVPixelBuffer) -> CMSampleBuffer? {
        var sampleBuffer: CMSampleBuffer?
        
        // Create timing info
        var timingInfo = CMSampleTimingInfo()
        timingInfo.duration = CMTime.invalid
        timingInfo.decodeTimeStamp = CMTime.invalid
        timingInfo.presentationTimeStamp = CMTime(value: Int64(CACurrentMediaTime() * 1000), timescale: 1000)
        
        // Create format description
        var formatDescription: CMFormatDescription?
        CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescriptionOut: &formatDescription
        )
        
        guard let formatDescription = formatDescription else { return nil }
        
        // Create sample buffer
        CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: formatDescription,
            sampleTiming: &timingInfo,
            sampleBufferOut: &sampleBuffer
        )
        
        return sampleBuffer
    }
    
    /// Creates a pixel buffer from a UIImage using the standard format.
    /// Use this to convert UIImage to the format needed for prediction.
    ///
    /// - Parameters:
    ///   - image: The source UIImage
    ///   - forSourceType: The type of source (camera or video) to determine pixel format
    /// - Returns: A CVPixelBuffer in the standard format, or nil if creation fails
    func createStandardPixelBuffer(from image: UIImage, forSourceType sourceType: FrameSourceType) -> CVPixelBuffer? {
        guard let cgImage = image.cgImage else { return nil }
        
        let width = cgImage.width
        let height = cgImage.height
        
        // Determine the pixel format based on source type
        let pixelFormat: OSType
        switch sourceType {
        case .camera:
            pixelFormat = FrameSourceSettings.cameraSourcePixelFormat
        case .videoFile, .imageSequence:
            pixelFormat = FrameSourceSettings.videoSourcePixelFormat
        case .uvc:
            // Use the same format as for camera sources
            pixelFormat = FrameSourceSettings.cameraSourcePixelFormat
        }
        
        // Create pixel buffer with appropriate format
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            pixelFormat,
            nil,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let pixelBuffer = pixelBuffer else { return nil }
        
        // Fill the pixel buffer with image data
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer)
        
        // Determine the bitmap info based on pixel format
        let bitmapInfo: UInt32
        if pixelFormat == kCVPixelFormatType_32ARGB {
            bitmapInfo = CGImageAlphaInfo.noneSkipFirst.rawValue
        } else { // BGRA
            bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        }
        
        let context = CGContext(
            data: pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: bitmapInfo
        )
        
        // Draw the image without the black background fill
        if let context = context {
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
        
        return pixelBuffer
    }
    
    /// Creates a standard UIImage from a CVPixelBuffer
    /// Use this for consistent conversion from pixel buffer to UIImage for display
    ///
    /// - Parameter pixelBuffer: The source pixel buffer
    /// - Returns: A UIImage, or nil if conversion fails
    func createStandardUIImage(from pixelBuffer: CVPixelBuffer) -> UIImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }
        
        return UIImage(cgImage: cgImage)
    }
    
    // Default implementations for the new methods
    
    /// Default implementation for requesting permissions - no permission needed
    @MainActor
    func requestPermission(completion: @escaping (Bool) -> Void) {
        // By default, no permission is needed
        completion(true)
    }
    
    /// Default implementation for orientation change - no action needed
    @MainActor
    func updateForOrientationChange(orientation: UIDeviceOrientation) {
        // By default, no action is needed
    }
    
    /// Default implementation for showing content selection UI - no UI to show
    @MainActor
    func showContentSelectionUI(from viewController: UIViewController, completion: @escaping (Bool) -> Void) {
        // By default, no content selection UI is available
        completion(false)
    }
    
    /// Default implementation of capturePhoto - doesn't capture anything
    @MainActor
    func capturePhoto(completion: @escaping @Sendable (UIImage?) -> Void) {
        // Default implementation does nothing and returns nil
        completion(nil)
    }
    
    // Default implementations for new coordinate transformation method
    @MainActor
    func transformDetectionToScreenCoordinates(
        rect: CGRect, 
        viewBounds: CGRect, 
        orientation: UIDeviceOrientation
    ) -> CGRect {
        // Default implementation just returns the input rect scaled to view bounds
        // Each source should override this with its specific transformation logic
        return VNImageRectForNormalizedRect(rect, Int(viewBounds.width), Int(viewBounds.height))
    }
    
    // Default implementation for UI integration
    @MainActor
    func integrateWithYOLOView(view: UIView) {
        // Default implementation does nothing
        // Each source should override this with its specific integration logic
    }
    
    // Default implementation for adding overlay layer
    @MainActor
    func addOverlayLayer(_ layer: CALayer) {
        // Default implementation does nothing
        // Each source should override this with its specific layer addition logic
    }
    
    // Default implementation for adding bounding box views
    @MainActor
    func addBoundingBoxViews(_ boxViews: [BoundingBoxView]) {
        // Default implementation does nothing
        // Each source should override this with its specific bounding box view addition logic
    }
    
    /// Resets processing state to allow normal inference to resume after calibration
    /// This method should be called when calibration completes to ensure the frame source
    /// is ready to process frames for normal inference again
    @MainActor
    func resetProcessingState() {
        // Default implementation does nothing
        // Each source should override this with its specific reset logic
    }
    
    // MARK: - Unified Performance Logging Interface
    
    /// Logs performance analysis with unified format across all frame sources
    /// - Parameters:
    ///   - frameCount: Current frame processing count
    ///   - sourcePrefix: Source identifier (e.g., "Camera", "Album", "GoPro")
    ///   - frameSize: Current frame dimensions
    ///   - processingFPS: Frame processing rate
    ///   - mode: Processing mode ("inference", "calibration", etc.)
    ///   - timingData: Dictionary containing timing information
    func logUnifiedPerformanceAnalysis(
        frameCount: Int,
        sourcePrefix: String,
        frameSize: CGSize,
        processingFPS: Double,
        mode: String,
        timingData: [String: Double]
    ) {
        // Only log in debug builds
        guard FrameSourceSettings.enablePerformanceLogging else { return }
        
        let frameSizeStr = frameSize.width > 0 ? 
            "\(String(format: "%.0f", frameSize.width))×\(String(format: "%.0f", frameSize.height))" : "Unknown"
        
        print("\(sourcePrefix): === \(sourcePrefix.uppercased()) SOURCE PIPELINE ANALYSIS (Frame #\(frameCount)) ===")
        print("\(sourcePrefix): Frame Size: \(frameSizeStr) | Processing FPS: \(String(format: "%.1f", processingFPS)) | Mode: \(mode)")
        
        let preparation = timingData["preparation"] ?? 0
        let conversion = timingData["conversion"] ?? 0
        let inference = timingData["inference"] ?? 0
        let ui = timingData["ui"] ?? 0
        let total = timingData["total"] ?? 0
        let throughput = timingData["throughput"] ?? 0
        
        if mode == "inference" && inference > 0 && total > 0 {
            // Full pipeline data available
            print("\(sourcePrefix): Frame Preparation: \(String(format: "%.1f", preparation))ms")
            print("\(sourcePrefix): Model Inference: \(String(format: "%.1f", inference))ms (includes fish counting inside TrackingDetector)")
            print("\(sourcePrefix): UI Delegate Call: \(String(format: "%.1f", ui))ms | Total Pipeline: \(String(format: "%.1f", total))ms")
            
            let theoreticalFPS = total > 0 ? 1000.0 / total : 0
            print("\(sourcePrefix): Theoretical FPS: \(String(format: "%.1f", theoreticalFPS)) | Actual Throughput: \(String(format: "%.1f", throughput))")
            
            // Calculate breakdown percentages
            if total > 0 {
                let preparationPct = (preparation / total) * 100
                let conversionPct = (conversion / total) * 100
                let inferencePct = (inference / total) * 100
                let uiPct = (ui / total) * 100
                
                print("\(sourcePrefix): Breakdown - Preparation: \(String(format: "%.1f", preparationPct))% | Conversion: \(String(format: "%.1f", conversionPct))% | Inference+FishCount: \(String(format: "%.1f", inferencePct))% | UI: \(String(format: "%.1f", uiPct))%")
            }
        } else if mode == "calibration" {
            // Calibration mode data
            print("\(sourcePrefix): Frame Preparation: \(String(format: "%.1f", preparation))ms + Conversion: \(String(format: "%.1f", conversion))ms = \(String(format: "%.1f", preparation + conversion))ms")
            print("\(sourcePrefix): Model Inference: CALIBRATION MODE - Actual Throughput: \(String(format: "%.1f", throughput))")
        } else {
            // Limited data available
            print("\(sourcePrefix): Frame Preparation: \(String(format: "%.1f", preparation))ms")
            print("\(sourcePrefix): Model Inference: PENDING - Actual Throughput: \(String(format: "%.1f", throughput))")
        }
    }
    
    /// Checks if performance logging should be performed based on frame count and settings
    /// - Parameter frameCount: Current frame processing count
    /// - Returns: True if logging should be performed
    func shouldLogPerformance(frameCount: Int) -> Bool {
        return FrameSourceSettings.enablePerformanceLogging && 
               frameCount % FrameSourceSettings.performanceLoggingInterval == 0
    }
} 