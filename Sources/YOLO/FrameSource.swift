// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
//  This file is part of the Ultralytics YOLO Package, defining the abstraction for frame sources.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
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

/// Protocol defining a common interface for all frame sources.
protocol FrameSource: AnyObject {
    /// The delegate to receive frames and performance metrics.
    var delegate: FrameSourceDelegate? { get set }
    
    /// The preview layer for displaying the source's visual output.
    var previewLayer: AVCaptureVideoPreviewLayer? { get }
    
    /// The predictor used to process frames from this source.
    var predictor: Predictor! { get set }
    
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
    func capturePhoto(completion: @escaping @Sendable (UIImage?) -> Void)
    
    /// Sets the zoom level for the frame source, if supported.
    ///
    /// - Parameter ratio: The zoom ratio to apply.
    nonisolated func setZoomRatio(ratio: CGFloat)
    
    /// The source type identifier.
    var sourceType: FrameSourceType { get }
}

/// Enumeration of available frame source types.
enum FrameSourceType {
    case camera
    case videoFile
    case imageSequence
    // Add more source types as needed
} 