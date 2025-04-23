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
    static let videoSourcePixelFormat = kCVPixelFormatType_32ARGB
    
    /// Standard pixel format to use for camera sources (used in CameraVideoSource)
    static let cameraSourcePixelFormat = kCVPixelFormatType_32BGRA
    
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
    - Pixel Format: 32-bit ARGB (kCVPixelFormatType_32ARGB)
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
}

/// Enumeration of available frame source types.
public enum FrameSourceType {
    case camera
    case videoFile
    case imageSequence
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
} 