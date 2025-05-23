// from softbank_fishcount_iphone14
//
//  GoProSource.swift
//  YOLO
//
//  Implementation of a frame source that connects to GoPro cameras via WiFi
//  using the OpenGoPro HTTP API and RTSP streaming.

import AVFoundation
import Foundation
import MobileVLCKit
import UIKit
import Vision

/// GoPro webcam version response structure - simplified version
struct GoProWebcamVersion: Decodable {
    let version: Int
    
    // Optional fields - won't cause decoding to fail if missing
    let max_lens_support: Bool?
    let usb_3_1_compatible: Bool?
}

/// GoPro streaming status for validation purposes
enum GoProStreamingStatus {
    case connecting
    case playing
    case error(String)
    case stopped
}

/// Delegate protocol for GoProSource streaming status updates
protocol GoProSourceDelegate: AnyObject {
    func goProSource(_ source: GoProSource, didUpdateStatus status: GoProStreamingStatus)
    func goProSource(_ source: GoProSource, didReceiveFirstFrame size: CGSize)
    func goProSource(_ source: GoProSource, didReceiveFrameWithTime time: Int64)
}

/// Class for handling GoPro camera as a frame source
@MainActor
class GoProSource: NSObject, @preconcurrency FrameSource, @preconcurrency VLCMediaPlayerDelegate {
    // MARK: - FrameSource Protocol Properties
    
    /// The delegate to receive frames and performance metrics.
    weak var delegate: FrameSourceDelegate?
    
    /// The processor used to process frames from this source.
    var predictor: FrameProcessor!
    
    /// The preview layer for displaying the source's visual output.
    var previewLayer: AVCaptureVideoPreviewLayer? {
        // GoPro uses VLC player, not AVCaptureSession, so return nil
        return nil
    }
    
    /// The long side dimension of the frames produced by this source.
    var longSide: CGFloat = 1920  // Default HD resolution
    
    /// The short side dimension of the frames produced by this source.
    var shortSide: CGFloat = 1080  // Default HD resolution
    
    /// Flag indicating if inference should be performed on frames
    var inferenceOK: Bool = true
    
    /// The source type identifier.
    var sourceType: FrameSourceType { return .goPro }
    
    /// Additional delegate for video capture (for compatibility with YOLOView)
    weak var videoCaptureDelegate: VideoCaptureDelegate?
    
    // MARK: - GoProSource Properties
    
    // Default GoPro IP address when connected via WiFi
    private let goProIP = "10.5.5.9"
    
    // HTTP endpoints for controlling webcam mode
    private let goProPort = 8080
    private let versionEndpoint = "/gopro/webcam/version"
    private let previewEndpoint = "/gopro/webcam/preview"
    private let startEndpoint = "/gopro/webcam/start"
    private let stopEndpoint = "/gopro/webcam/stop"
    private let exitEndpoint = "/gopro/webcam/exit"
    
    // RTSP configuration - simplified to known working values
    private let rtspPort = 554
    private let rtspPath = "/live"
    
    // VLCKit video player
    private var videoPlayer: VLCMediaPlayer?
    
    // Frame rate limiting properties
    private var lastFrameExtractionTime: CFTimeInterval = 0
    private let frameExtractionInterval: CFTimeInterval = 0.125 // 8 fps (adjust as needed)
    
    // Stream status tracking
    private var streamStartTime: Date?
    private var hasReceivedFirstFrame = false
    private var frameCount = 0
    private var lastFrameTime: Int64 = 0
    
    // Delegate for status updates
    weak var goProDelegate: GoProSourceDelegate?
    
    // Add a property to store the test delegate to prevent it from being deallocated
    private var testDelegate: TestDelegate?
    
    // Add frameTimestamps property to the class
    private var frameTimestamps: [CFTimeInterval] = []
    
    // VLC player view for frame extraction - maintains reference to the drawable
    // Internal access to allow YOLOView to access it
    var playerView: UIView?
    
    // Container view that will hold the player view
    private weak var containerView: UIView?
    
    // Current orientation of the device
    private var currentOrientation: UIDeviceOrientation = .portrait
    
    // Store the last frame size for coordinate transformations
    private var lastFrameSize: CGSize = .zero
    
    // Store the current pixel buffer for reference
    private var currentPixelBuffer: CVPixelBuffer?
    
    // Store last image for frame rate limiting
    private var lastCapturedImage: UIImage?
    
    // MARK: - Initialization
    
    override init() {
        super.init()
        
        // Since UIView creation must happen on the main thread
        DispatchQueue.main.async {
            self.setupVLCPlayer()
        }
        
        // Listen for orientation changes
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(orientationDidChange),
            name: UIDevice.orientationDidChangeNotification,
            object: nil
        )
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
    }
    
    // MARK: - FrameSource Protocol Methods
    
    /// Sets up the frame source with specified configuration.
    func setUp(completion: @escaping @Sendable (Bool) -> Void) {
        // Reset any previous state
        hasReceivedFirstFrame = false
        frameCount = 0
        
        // For GoPro, we just need to initialize the VLC player
        // The connection to the camera happens when startRTSPStream is called
        completion(true)
    }
    
    /// Begins frame acquisition from the source.
    nonisolated func start() {
        Task { @MainActor [weak self] in
            guard let self = self else { return }
            
            // Reset state
            self.hasReceivedFirstFrame = false
            self.frameCount = 0
            self.frameTimestamps.removeAll()
            
            // Make sure player view is visible
            self.playerView?.isHidden = false
            self.playerView?.alpha = 1.0
            
            // Ensure container view knows we need layout
            self.containerView?.setNeedsLayout()
            self.containerView?.layoutIfNeeded()
            
            // Start streaming
            if self.videoPlayer?.state != .playing {
                self.startRTSPStream { _ in }
            }
        }
    }
    
    /// Stops frame acquisition from the source.
    nonisolated func stop() {
        Task { @MainActor [weak self] in
            guard let self = self else { return }
            print("GoPro: Stopping stream and cleaning up resources")
            
            // Stop the VLC player
                    self.videoPlayer?.stop()
                    
            // Clean up player view from the container
            if let playerView = self.playerView {
                // Make sure we're on the main thread
                dispatchPrecondition(condition: .onQueue(.main))
                
                // First make the view invisible to prevent flicker
                playerView.isHidden = true
                
                // CRITICAL FIX: Before removing from hierarchy, store references to all constraints
                let playerViewConstraints = playerView.constraints
                let playerSuperview = playerView.superview
                var superviewConstraints: [NSLayoutConstraint] = []
                
                // Find and collect all constraints in the superview that reference our player view
                if let superview = playerSuperview {
                    superviewConstraints = superview.constraints.filter { constraint in
                        return (constraint.firstItem === playerView || constraint.secondItem === playerView)
                    }
                    
                    // Deactivate superview constraints that reference the player view
                    if !superviewConstraints.isEmpty {
                        print("GoPro: Deactivating \(superviewConstraints.count) superview constraints")
                        NSLayoutConstraint.deactivate(superviewConstraints)
                    }
                }
                
                // Deactivate the player view's own constraints
                if !playerViewConstraints.isEmpty {
                    print("GoPro: Deactivating \(playerViewConstraints.count) player view constraints")
                    NSLayoutConstraint.deactivate(playerViewConstraints)
                }
                
                // Now it's safe to remove from superview
                playerView.removeFromSuperview()
                
                print("GoPro: Player view removed from hierarchy")
            }
            
            // Break circular references 
            self.containerView = nil
            
            // Reset all state variables
            self.hasReceivedFirstFrame = false
            self.frameCount = 0
            self.frameTimestamps.removeAll()
            self.streamStartTime = nil
            self.lastFrameSize = .zero
            self.currentPixelBuffer = nil
            
            // Notify both delegate types
            Task { @MainActor in
                self.goProDelegate?.goProSource(self, didUpdateStatus: .stopped)
            }
        }
    }
    
    /// Sets the zoom level for the frame source.
    /// Not supported for GoPro cameras via RTSP.
    nonisolated func setZoomRatio(ratio: CGFloat) {
        // Zoom not supported via RTSP
    }
    
    /// Captures a still image from the frame source.
    @MainActor
    func capturePhoto(completion: @escaping @Sendable (UIImage?) -> Void) {
        // Capture the current frame from VLC player
        if let frameImage = extractCurrentFrame() {
            completion(frameImage)
        } else {
            completion(nil)
        }
    }
    
    /// Updates the source for orientation changes
    func updateForOrientationChange(orientation: UIDeviceOrientation) {
        self.currentOrientation = orientation
        updatePlayerViewLayout()
    }
    
    /// Request permission to use this frame source - not applicable for GoPro
    func requestPermission(completion: @escaping (Bool) -> Void) {
        // GoPro RTSP doesn't require app permissions, but we do need network access
        // which is already granted via Info.plist
        completion(true)
    }
    
    /// Shows UI for selecting content for this source - not applicable for GoPro
    func showContentSelectionUI(from viewController: UIViewController, completion: @escaping (Bool) -> Void) {
        // GoPro doesn't need content selection as it's a live stream
        completion(false)
    }

    // MARK: - VLC Player Setup
    
    @MainActor
    private func setupVLCPlayer() {
        videoPlayer = VLCMediaPlayer()
        videoPlayer?.delegate = self
        
        // Enable debugging
        videoPlayer?.libraryInstance.debugLogging = true
        videoPlayer?.libraryInstance.debugLoggingLevel = 3
        
        // Use a default sensible size (16:9 aspect ratio for common HD videos)
        let defaultWidth: CGFloat = 1280
        let defaultHeight: CGFloat = 720
        
        // Create a view to use as the drawable for frame extraction
        playerView = createPlayerView()
        playerView?.frame = CGRect(x: 0, y: 0, width: defaultWidth, height: defaultHeight)
        
        // Set critical properties for proper rendering
        playerView?.backgroundColor = .black
        playerView?.layer.masksToBounds = true
        
        // Make sure we're on the main thread for view operations
        dispatchPrecondition(condition: .onQueue(.main))
        
        // This is crucial - use VLCMediaPlayer's drawable property
        videoPlayer?.drawable = playerView
        
        // Force the video player to start with a good frame size
        videoPlayer?.videoAspectRatio = UnsafeMutablePointer<Int8>(mutating: "16:9")
        videoPlayer?.videoCropGeometry = UnsafeMutablePointer<Int8>(mutating: "")
        
        // Set additional properties for better performance
        videoPlayer?.audio?.volume = 0  // Mute audio
        
        print("GoPro: VLC player view initialized with size: \(playerView?.bounds.size ?? .zero)")
    }
    
    // MARK: - Frame Extraction Methods
    
    @MainActor
    private func extractCurrentFrame() -> UIImage? {
        // Frame rate limiting - only extract frames at the specified interval
        let currentTime = CACurrentMediaTime()
        if currentTime - lastFrameExtractionTime < frameExtractionInterval,
           let cachedImage = lastCapturedImage {
            // Return cached frame if interval hasn't elapsed
            return cachedImage
        }
        
        // Update last extraction time
        lastFrameExtractionTime = currentTime
        
        guard let videoPlayer = videoPlayer else { 
            print("GoPro: Cannot extract frame - video player is nil")
            return fallbackSnapshot() 
        }
        
        // Check if VLC player is ready (playing or paused or buffering)
        let playerState = videoPlayer.state
        let validStates: [VLCMediaPlayerState] = [.playing, .paused, .buffering]
        if !validStates.contains(playerState) {
            print("GoPro: Cannot take snapshot - player state: \(playerState.rawValue) not in valid states")
            return fallbackSnapshot()
        }
        
        // Get actual video dimensions
        let videoSize = videoPlayer.videoSize
        if videoSize.width <= 0 || videoSize.height <= 0 {
            print("GoPro: Invalid video dimensions: \(videoSize)")
            return fallbackSnapshot()
        }
        
        // Method 1: Capture directly from player view using Core Graphics
        if let playerView = playerView,
           let snapshot = captureFromView(playerView),
           !isBlackImage(snapshot) {
            // Update last frame size
            self.lastFrameSize = snapshot.size
            
            // Log occasional success for monitoring
            if frameCount % 30 == 0 {
                print("GoPro: Successfully captured frame directly from view - size: \(snapshot.size)")
            }
            
            // Cache the image for frame rate limiting
            self.lastCapturedImage = snapshot
            
            // Create and cache pixel buffer from image
            if let pixelBuffer = createPixelBuffer(from: snapshot) {
                self.currentPixelBuffer = pixelBuffer
            }
            
            return snapshot
        }
        
        // Method 2: If direct capture fails, use a temporary file but optimize the process
        // This approach is much faster than the original implementation as we use 
        // more efficient file operations and avoid unnecessary disk I/O
        return captureUsingOptimizedSnapshot(videoPlayer: videoPlayer, videoSize: videoSize)
    }
    
    // Optimized snapshot method using temporary file with minimal disk I/O
    private func captureUsingOptimizedSnapshot(videoPlayer: VLCMediaPlayer, videoSize: CGSize) -> UIImage? {
        // Create a temporary file path in memory area if possible
        let tempFilename = "gopro_\(Int(Date().timeIntervalSince1970 * 1000)).png"
        let tempDir = FileManager.default.temporaryDirectory.path
        let tempPath = (tempDir as NSString).appendingPathComponent(tempFilename)
        
        // Use VLC's saveVideoSnapshot method with optimized parameters
        videoPlayer.saveVideoSnapshot(at: tempPath, withWidth: Int32(videoSize.width), andHeight: Int32(videoSize.height))
        
        // Check if file was created successfully - use optimized path
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: tempPath) else {
            print("GoPro: VLC snapshot file was not created")
            return fallbackSnapshot()
        }
        
        // Use efficient file loading to minimize I/O impact
        guard let imageData = fileManager.contents(atPath: tempPath),
              let image = UIImage(data: imageData) else {
            print("GoPro: Failed to load snapshot data from temporary file")
            try? fileManager.removeItem(atPath: tempPath)
            return fallbackSnapshot()
        }
        
        // Clean up the temporary file immediately
        try? fileManager.removeItem(atPath: tempPath)
        
        // Update last frame size and cache
        self.lastFrameSize = image.size
        self.lastCapturedImage = image
        
        // Create and cache pixel buffer from image
        if let pixelBuffer = createPixelBuffer(from: image) {
            self.currentPixelBuffer = pixelBuffer
        }
        
        // Log occasional success for monitoring
        if frameCount % 30 == 0 {
            print("GoPro: Successfully captured frame using optimized snapshot - size: \(image.size)")
        }
        
        return image
    }
    
    // Capture from UIView using Core Graphics
    private func captureFromView(_ view: UIView) -> UIImage? {
        if view.bounds.size.width <= 0 || view.bounds.size.height <= 0 {
            return nil
        }
        
        UIGraphicsBeginImageContextWithOptions(view.bounds.size, false, 0)
        defer { UIGraphicsEndImageContext() }
        
        // Render the view hierarchy
        if view.drawHierarchy(in: view.bounds, afterScreenUpdates: true) {
            return UIGraphicsGetImageFromCurrentImageContext()
        }
        
        return nil
    }
    
    // Create a pixel buffer from UIImage for consistent processing
    private func createPixelBuffer(from image: UIImage) -> CVPixelBuffer? {
        let width = Int(image.size.width)
        let height = Int(image.size.height)
        
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32ARGB,
            attributes as CFDictionary,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0)) }
        
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else {
            return nil
        }
        
        // Draw the image into the context
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        UIGraphicsPushContext(context)
        image.draw(in: rect)
        UIGraphicsPopContext()
        
        return buffer
    }
    
    // This is a fallback method that tries several approaches if the main VLC snapshot fails
    @MainActor
    private func fallbackSnapshot() -> UIImage? {
        print("GoPro: Using fallback snapshot methods")
        
        // Approach 1: Try to get direct snapshot from player view
        if let playerView = playerView, 
           playerView.bounds.size.width > 0, 
           playerView.bounds.size.height > 0 {
            
            // Try to get a snapshot of the player view
            if let snapshotImage = playerView.snapshotImage(), !isBlackImage(snapshotImage) {
                print("GoPro: Using player view snapshot - size: \(snapshotImage.size)")
                self.lastFrameSize = snapshotImage.size
                return snapshotImage
            }
        }
        
        // Approach 2: Create test pattern as last resort for debugging
        print("GoPro: All capture methods failed, using test pattern")
        let patternSize = CGSize(width: 1280, height: 720)
        let testImage = createTestPatternImage(size: patternSize)
        self.lastFrameSize = patternSize
        return testImage
    }
    
    // Create a test pattern image for debugging when actual frame capture fails
    private func createTestPatternImage(size: CGSize) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: size)
        let img = renderer.image { ctx in
            // Draw a recognizable pattern
            let context = ctx.cgContext
            
            // Fill with dark gray
            context.setFillColor(UIColor.darkGray.cgColor)
            context.fill(CGRect(origin: .zero, size: size))
            
            // Draw red border
            context.setStrokeColor(UIColor.red.cgColor)
            context.setLineWidth(20)
            context.stroke(CGRect(x: 10, y: 10, width: size.width - 20, height: size.height - 20))
            
            // Draw crosshairs
            context.setStrokeColor(UIColor.yellow.cgColor)
            context.setLineWidth(5)
            context.move(to: CGPoint(x: 0, y: 0))
            context.addLine(to: CGPoint(x: size.width, y: size.height))
            context.move(to: CGPoint(x: size.width, y: 0))
            context.addLine(to: CGPoint(x: 0, y: size.height))
            context.strokePath()
            
            // Draw grid lines
            context.setStrokeColor(UIColor.green.cgColor)
            context.setLineWidth(2)
            let gridSize: CGFloat = 100
            for x in stride(from: gridSize, to: size.width, by: gridSize) {
                context.move(to: CGPoint(x: x, y: 0))
                context.addLine(to: CGPoint(x: x, y: size.height))
            }
            for y in stride(from: gridSize, to: size.height, by: gridSize) {
                context.move(to: CGPoint(x: 0, y: y))
                context.addLine(to: CGPoint(x: size.width, y: y))
            }
            context.strokePath()
            
            // Add timestamp for tracking when this test pattern was generated
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "HH:mm:ss.SSS"
            let timestamp = dateFormatter.string(from: Date())
            
            // Add error text with timestamp
            let text = "FRAME ERROR - \(timestamp)" as NSString
            let attributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: 40),
                .foregroundColor: UIColor.white
            ]
            text.draw(at: CGPoint(x: size.width/2 - 200, y: size.height/2 - 20), withAttributes: attributes)
            
            // Add count of failed attempts
            let countText = "Frame #\(self.frameCount)" as NSString
            countText.draw(at: CGPoint(x: size.width/2 - 50, y: size.height/2 + 40), withAttributes: attributes)
        }
        
        return img
    }
    
    // Helper to check if an image is entirely or almost entirely black
    private func isBlackImage(_ image: UIImage) -> Bool {
        // Create a 1x1 pixel context to sample average color
        let context = CIContext()
        guard let cgImage = image.cgImage,
              let reducedImage = context.createCGImage(
                  CIImage(cgImage: cgImage).clampedToExtent(),
                  from: CGRect(x: 0, y: 0, width: 1, height: 1)) else {
            return true // Assume black if we can't analyze
        }
        
        // Get the pixel data
        guard let data = CFDataGetBytePtr(reducedImage.dataProvider?.data) else {
            return true // Assume black if we can't get data
        }
        
        // Check RGB values (skip alpha)
        let brightness = (Int(data[0]) + Int(data[1]) + Int(data[2])) / 3
        
        // Image is considered black if average brightness is very low
        return brightness < 10
    }
    
    // Process the extracted frame into YOLO-compatible format
    private func processExtractedFrame(_ image: UIImage) -> CMSampleBuffer? {
        // First create a pixel buffer from the image using standard conversion
        guard let pixelBuffer = createStandardPixelBuffer(from: image, forSourceType: sourceType) else {
            print("GoPro: Failed to create pixel buffer from frame")
            return nil
        }
        
        // Then create a sample buffer using standard conversion
        guard let sampleBuffer = createStandardSampleBuffer(from: pixelBuffer) else {
            print("GoPro: Failed to create sample buffer from pixel buffer")
            return nil
        }
        
        // Track this buffer as the current one
        self.currentPixelBuffer = pixelBuffer
        
        return sampleBuffer
    }
    
    // Process the current frame for either inference or calibration
    @MainActor
    private func processCurrentFrame() {
        // Measure frame extraction time
        let extractionStartTime = CACurrentMediaTime()
        
        guard let frameImage = extractCurrentFrame() else {
            print("GoPro: Failed to extract frame for processing")
            return
        }
        
        // Calculate and log extraction time
        let extractionTime = (CACurrentMediaTime() - extractionStartTime) * 1000 // in ms
        
        // Check if the extracted image is our test pattern
        let isTestPattern = (frameImage.size.width == 1280 && frameImage.size.height == 720)
        
        // Always send the frame to the delegate for display, even if it's a test pattern
        delegate?.frameSource(self, didOutputImage: frameImage)
        
        // Log extraction performance occasionally
        if frameCount % 60 == 0 {
            print("GoPro: Frame #\(frameCount) extraction took \(String(format: "%.1f", extractionTime))ms, size: \(frameImage.size), isTestPattern: \(isTestPattern)")
        }
        
        // Skip inference if this is just a test pattern
        if isTestPattern {
            if frameCount % 60 == 0 {  // Log only occasionally to avoid spam
                print("GoPro: Skipping inference on test pattern frame")
            }
            return
        }
        
        // Check if we're in calibration mode
        let shouldProcessForCalibration = !inferenceOK && predictor is TrackingDetector
        
        if shouldProcessForCalibration {
            // Process for auto-calibration
            guard let trackingDetector = predictor as? TrackingDetector else {
                print("GoPro: Cannot process for calibration - predictor is not TrackingDetector")
                return
            }
            
            // Clear boxes when starting calibration (first frame)
            if trackingDetector.getCalibrationFrameCount() == 0 {
                print("GoPro: Starting calibration, clearing boxes")
                self.videoCaptureDelegate?.onClearBoxes()
            }
            
            // Convert UIImage to CVPixelBuffer for calibration
            if let pixelBufferForCalibration = createStandardPixelBuffer(from: frameImage, forSourceType: sourceType) {
                // Process the frame for calibration
                trackingDetector.processFrame(pixelBufferForCalibration)
                
                // Print calibration progress by using frame count
                let frameCount = trackingDetector.getCalibrationFrameCount()
                if frameCount > 0 {
                    // Estimate progress as percentage of expected frames (assume 100 frames needed)
                    let estimatedProgress = min(Double(frameCount) / 100.0, 1.0)
                    print("GoPro: Calibration progress: \(Int(estimatedProgress * 100))% (frame \(frameCount))")
                }
            }
            
            // Skip regular inference during calibration
            return
        }
        
        // Process with predictor if inference is enabled
        if inferenceOK, let predictor = self.predictor {
            // Measure sample buffer creation time
            let bufferStartTime = CACurrentMediaTime()
            
            if let sampleBuffer = processExtractedFrame(frameImage) {
                let bufferTime = (CACurrentMediaTime() - bufferStartTime) * 1000 // in ms
                
                // Log inference attempt details periodically
                if frameCount % 60 == 0 {
                    print("GoPro: Running detection on frame #\(frameCount), size: \(frameImage.size)")
                    print("GoPro: Buffer creation took \(String(format: "%.1f", bufferTime))ms")
                }
                
                // Ensure the delegate is set up correctly
                if self.videoCaptureDelegate == nil {
                    print("GoPro: Warning - videoCaptureDelegate is nil, detection results will be lost")
                }
                
                // Run inference on the sample buffer
                predictor.predict(
                    sampleBuffer: sampleBuffer,
                    onResultsListener: self,
                    onInferenceTime: self
                )
            } else {
                print("GoPro: Failed to create sample buffer from frame image")
            }
        }
    }
    
    // MARK: - Improved Delegate Management
    
    /// Sets both goProDelegate and FrameSource delegate to ensure complete integration
    func setDelegate(_ delegate: GoProSourceDelegate & FrameSourceDelegate) {
        self.goProDelegate = delegate
        self.delegate = delegate
    }
    
    /// Sets all required delegates to properly integrate with YOLOView
    @MainActor
    func integrateWithYOLOView(view: UIView) {
        // We need the view to conform to both VideoCaptureDelegate and FrameSourceDelegate
        guard let captureDelegate = view as? VideoCaptureDelegate,
              let frameDelegate = view as? FrameSourceDelegate else {
            print("GoPro: Error - View does not conform to required delegate protocols")
            return
        }
        
        // Set both delegate types
        self.delegate = frameDelegate
        self.videoCaptureDelegate = captureDelegate
        
        // If the view also conforms to GoProSourceDelegate, set that too
        if let goProDelegate = view as? GoProSourceDelegate {
            self.goProDelegate = goProDelegate
        }
        
        // Ensure video capture delegate is properly typed and connected
        print("GoPro: Setting up integration with YOLOView")
        
        // Get the container view where we'll display the video
        let containerView: UIView
        if let viewForDrawing = captureDelegate.viewForDrawing {
            containerView = viewForDrawing
        } else {
            containerView = view
        }
        
        self.containerView = containerView
        
        // Make sure we have a valid player view
        guard let playerView = playerView else {
            print("GoPro: Error - No player view available for integration")
            return
        }
        
        // Make sure we're on the main thread for UI operations
        dispatchPrecondition(condition: .onQueue(.main))
        
        // Make sure the player view is not in any view hierarchy
        playerView.removeFromSuperview()
        
        // Add player view to the container with a clear visual hierarchy
        playerView.translatesAutoresizingMaskIntoConstraints = false
        
        // IMPORTANT: Add as subview with proper z-index - ensure it's at the bottom
        if containerView.subviews.isEmpty {
            containerView.addSubview(playerView)
        } else {
            containerView.insertSubview(playerView, at: 0)
        }
        
        print("GoPro: Adding player view to container with size: \(containerView.bounds.size)")
        
        // Clear existing constraints
        NSLayoutConstraint.deactivate(playerView.constraints)
        
        // Setup clear edge constraints to fill the container
        let constraints = [
            playerView.leadingAnchor.constraint(equalTo: containerView.leadingAnchor),
            playerView.trailingAnchor.constraint(equalTo: containerView.trailingAnchor),
            playerView.topAnchor.constraint(equalTo: containerView.topAnchor),
            playerView.bottomAnchor.constraint(equalTo: containerView.bottomAnchor)
        ]
        
        // Add identifiers to constraints for debugging
        for (index, constraint) in constraints.enumerated() {
            constraint.identifier = "playerView-container-\(index)"
        }
        
        NSLayoutConstraint.activate(constraints)
        
        // Force layout to apply the new constraints
        containerView.layoutIfNeeded()
        
        // CRITICAL: Ensure the player view is properly connected to the VLC player
        videoPlayer?.drawable = playerView
        
        // NEW: Add the boundingBoxViews to the player view or its parent
        // This is critical for making the bounding boxes visible
        if let yoloView = view as? YOLOView {
            print("GoPro: Setting up bounding box views on player view layer")
            
            // Access the boundingBoxViews from YOLOView and add them to our player view
            for box in yoloView.boundingBoxViews {
                box.addToLayer(playerView.layer)
            }
            
            // Access and setup overlay layer
            if let overlayLayer = yoloView.layer.sublayers?.first(where: { $0.name == "overlayLayer" }) {
                playerView.layer.addSublayer(overlayLayer)
            }
        }
        
        print("GoPro: Successfully integrated player view with container: \(containerView.bounds.size)")
        
        // Fix for detection results handling
        if let predictor = self.predictor {
            print("GoPro: Using predictor of type \(type(of: predictor))")
        } else {
            print("GoPro: Error - No predictor available for integration")
            
            // Try to get predictor from YOLOView if possible
            if let yoloView = view as? YOLOView, 
               let currentPredictor = yoloView.getCurrentPredictor() {
                print("GoPro: Obtained predictor from YOLOView")
                self.predictor = currentPredictor
            }
        }
        
        // Debug delegate chain
        print("GoPro: Delegate chain set up: delegate=\(String(describing: delegate)), videoCaptureDelegate=\(String(describing: videoCaptureDelegate))")
    }
    
    // MARK: - Layout Management
    
    @MainActor
    private func updatePlayerViewLayout() {
        guard let playerView = playerView, let containerView = containerView else {
            print("GoPro: Cannot update layout - missing views")
            return
        }

        // This operation must be performed on the main thread
        dispatchPrecondition(condition: .onQueue(.main))
        
        // Safety check for container view size
        let containerSize = containerView.bounds.size
        if containerSize.width <= 0 || containerSize.height <= 0 {
            print("GoPro: Warning - Container has invalid size: \(containerSize)")
            return
        }
        
        // CRITICAL FIX: Verify the player view is still in the view hierarchy
        guard playerView.superview === containerView else {
            print("GoPro: Warning - Player view is not in the container's hierarchy")
            return
        }
        
        // Remove any aspect constraints we've previously added
        containerView.constraints.forEach { constraint in
            if constraint.identifier?.contains("aspect") == true && 
               (constraint.firstItem === playerView || constraint.secondItem === playerView) {
                containerView.removeConstraint(constraint)
            }
        }
        
        // Clear existing named constraints on the player view itself
        playerView.constraints.forEach { constraint in
            if constraint.identifier?.contains("aspect") == true {
                playerView.removeConstraint(constraint)
            }
        }
        
        // Get video dimensions from the player or use defaults
        var videoWidth: CGFloat = max(1, videoPlayer?.videoSize.width ?? 1920)
        var videoHeight: CGFloat = max(1, videoPlayer?.videoSize.height ?? 1080)
        
        // Fallback to sensible defaults if we have invalid dimensions
        if !videoWidth.isFinite || !videoHeight.isFinite || videoWidth <= 0 || videoHeight <= 0 {
            print("GoPro: Invalid video dimensions (\(videoWidth)x\(videoHeight)), using defaults")
            videoWidth = 1920
            videoHeight = 1080
        }
        
        // Simple, reliable filling of the container - ignore aspect ratio concerns
        let leadingConstraint = playerView.leadingAnchor.constraint(equalTo: containerView.leadingAnchor)
        let trailingConstraint = playerView.trailingAnchor.constraint(equalTo: containerView.trailingAnchor)
        let topConstraint = playerView.topAnchor.constraint(equalTo: containerView.topAnchor)
        let bottomConstraint = playerView.bottomAnchor.constraint(equalTo: containerView.bottomAnchor)
        
        // Add identifiers
        leadingConstraint.identifier = "playerViewLeading"
        trailingConstraint.identifier = "playerViewTrailing"
        topConstraint.identifier = "playerViewTop"
        bottomConstraint.identifier = "playerViewBottom"
        
        // CRITICAL FIX: Store references to the constraints we're activating
        let constraints = [
            leadingConstraint,
            trailingConstraint,
            topConstraint,
            bottomConstraint
        ]
        
        // Activate constraints and store references
        NSLayoutConstraint.activate(constraints)
        
        // Force immediate layout
        containerView.layoutIfNeeded()
        
        print("GoPro: Updated player layout with simple fill constraint for size: \(containerSize)")
    }
    
    // Orientation change notification handler
    @objc private func orientationDidChange() {
        let orientation = UIDevice.current.orientation
        if orientation.isPortrait || orientation.isLandscape {
            updateForOrientationChange(orientation: orientation)
        }
    }
    
    // MARK: - RTSP Configuration
    
    /// Configure media options for optimal RTSP streaming from GoPro
    private func configureRTSPMediaOptions(_ media: VLCMedia) {
        media.addOption(":verbose=0")

        // Critical options for RTSP
        media.addOption(":rtsp-tcp")  // Force TCP for more reliable streaming
        
        // Reduced caching for lower latency
        media.addOption(":network-caching=0")   // Set to 0 for lower latency (was 50)
        media.addOption(":live-caching=0")      // Set to 0 for lower latency (was 50)
        media.addOption(":file-caching=0")      // Set to 0 for all file caching
        
        // Hardware acceleration (when available)
        media.addOption(":avcodec-hw=any")      // Let VLC choose best hardware option
        
        // Keep audio enabled but muted (audio helps sync)
        media.addOption(":audio-track=0")       // Audio enabled but track 0
        
        // Improve H.264 handling 
        media.addOption(":rtsp-frame-buffer-size=3000000")  // Larger buffer for better frame quality (was 1000000)
        
        // SPS/PPS handling (critical for H.264)
        media.addOption(":rtsp-sps-pps=true")   // Force SPS/PPS with each keyframe
        
        // Timeout settings
        media.addOption(":rtp-timeout=2000")    // 2 second timeout (was 5000)
        
        // Optimize for low latency
        media.addOption(":clock-jitter=0")      // Disable clock jitter detection
        media.addOption(":clock-synchro=0")     // Disable clock synchro
        
        // Force decoding threads to 4 for better performance
        media.addOption(":avcodec-threads=4")
        
        // Disable subtitles
        media.addOption(":no-sub-autodetect-file")
        
        // Options for better frame extraction
        media.addOption(":video-filter=scene")
        media.addOption(":scene-format=png")
        media.addOption(":scene-ratio=1")       // Save every frame
        media.addOption(":scene-prefix=gopro_snapshot")
        media.addOption(":scene-path=\(NSTemporaryDirectory())")
        
        // Improved video configuration
        media.addOption(":avcodec-skip-frame=0") // Don't skip frames
        media.addOption(":avcodec-skip-idct=0")  // Don't skip IDCT
        media.addOption(":avcodec-dr=1")         // Enable direct rendering
        media.addOption(":sout-x264-keyint=1")   // Force keyframe every frame
        
        // Clear cookies to ensure fresh connection
        media.clearStoredCookies()
        
        print("GoPro: Media options configured for RTSP streaming")
    }
    
    /// Start RTSP stream from GoPro
    /// - Parameter completion: Callback with result (success/failure)
    @MainActor
    func startRTSPStream(completion: @escaping (Result<Void, Error>) -> Void) {
        // Construct RTSP URL - using the known working configuration
        let rtspURLString = "rtsp://\(goProIP):\(rtspPort)\(rtspPath)"
        print("GoPro: Connecting to RTSP stream at \(rtspURLString)")
        
        guard let url = URL(string: rtspURLString), let videoPlayer = videoPlayer else {
            let error = NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid RTSP URL or player not initialized"])
            completion(.failure(error))
            return
        }
        
        // Ensure we're on the main thread for UI operations
        dispatchPrecondition(condition: .onQueue(.main))
        
        // Set stream status to connecting
        goProDelegate?.goProSource(self, didUpdateStatus: .connecting)
        
        // Reset frame tracking
        hasReceivedFirstFrame = false
        frameCount = 0
        frameTimestamps.removeAll()
        streamStartTime = Date()
        
        // Make sure our player view is properly configured
        if let playerView = playerView {
            // Force layout update
            playerView.layoutIfNeeded()
            
            // Make sure player view is visible
            playerView.isHidden = false
            
            // Connect player to view again to ensure proper linkage
            videoPlayer.drawable = playerView
        }
        
        // Configure player with optimized options
        let media = VLCMedia(url: url)
        configureRTSPMediaOptions(media)
        videoPlayer.media = media
        
        // Start playback
        videoPlayer.play()
        
        print("GoPro: VLC Player - Opening stream")
        
        // Success is reported immediately, actual success is determined by frame reception
        completion(.success(()))
    }
    
    /// Stop RTSP stream
    @MainActor
    func stopRTSPStream() {
        print("GoPro: Stopping RTSP stream")
        videoPlayer?.stop()
        
        // Notify both delegate types
        goProDelegate?.goProSource(self, didUpdateStatus: .stopped)
    }

    // MARK: - VLC Media Player Delegate Methods
    
    // These functions are called by VLC and may not be on the main thread
    nonisolated func mediaPlayerStateChanged(_ aNotification: Notification!) {
        guard let player = aNotification.object as? VLCMediaPlayer else { return }
        
        // Capture state safely
        let playerState = player.state
        let playerVideoSize = player.videoSize
        
        // Create main actor task to process this on main thread
                    Task { @MainActor in
            // Note: Since we're not using [weak self], we don't need the guard let
            
            switch playerState {
                case .opening:
                    print("GoPro: VLC Player - Opening stream")
                
                case .buffering:
                    print("GoPro: VLC Player - Buffering stream")
                    
                case .playing:
                    print("GoPro: VLC Player - Stream is playing")
                    self.goProDelegate?.goProSource(self, didUpdateStatus: .playing)
                    
                    // Get video dimensions and notify about frame size
                    if playerVideoSize.width > 0 && playerVideoSize.height > 0 {
                        self.lastFrameSize = playerVideoSize
                        // Notify observers about frame size change
                        self.broadcastFrameSizeChange(playerVideoSize)
                    }
                    
                case .error:
                    print("GoPro: VLC Player - Error streaming")
                    self.goProDelegate?.goProSource(self, didUpdateStatus: .error("Playback error"))
                    
                case .ended:
                    print("GoPro: VLC Player - Stream ended")
                    self.goProDelegate?.goProSource(self, didUpdateStatus: .stopped)
                    
                case .stopped:
                    print("GoPro: VLC Player - Stream stopped")
                    self.goProDelegate?.goProSource(self, didUpdateStatus: .stopped)
                    
                default:
                    print("GoPro: VLC Player - State: \(playerState.rawValue)")
            }
        }
    }
    
    nonisolated func mediaPlayerTimeChanged(_ aNotification: Notification!) {
        guard let player = aNotification.object as? VLCMediaPlayer else { return }
        
        // Capture values safely - don't use self here
        let timeValue = Int64(player.time.intValue)
        let videoSize = player.videoSize
        
        // Create main actor task to process this on main thread
        Task { @MainActor in
            // Note: Since we're not using [weak self], we don't need the guard let
            
            // Increment frame count
            self.frameCount += 1
            
            // Store the current media time
            self.lastFrameTime = timeValue
            
            // Track frame timestamps for FPS calculation - keep last 30 frames
            self.frameTimestamps.append(CACurrentMediaTime())
            while self.frameTimestamps.count > 30 {
                self.frameTimestamps.removeFirst()
            }
            
            // Calculate frame size if this is the first frame
            if !self.hasReceivedFirstFrame {
                print("GoPro: VLC Player - Received first frame, size: \(videoSize)")
                
                // Only proceed if we have a valid frame size
                if videoSize.width > 0 && videoSize.height > 0 {
                    self.hasReceivedFirstFrame = true
                    
                    // Update frame dimensions for FrameSource protocol
                    self.longSide = max(videoSize.width, videoSize.height)
                    self.shortSide = min(videoSize.width, videoSize.height)
                    
                    // Store the frame size and broadcast it
                    self.lastFrameSize = videoSize
                    self.broadcastFrameSizeChange(videoSize)
                    
                    // Update player view layout now that we have a valid video size
                    self.updatePlayerViewLayout()
                    
                    // Notify first frame received with size
                    print("GoPro: Calling delegate didReceiveFirstFrame with size: \(videoSize)")
                    self.goProDelegate?.goProSource(self, didReceiveFirstFrame: videoSize)
                    print("GoPro: Delegate didReceiveFirstFrame call completed")
                } else {
                    print("GoPro: Received invalid frame size: \(videoSize). Waiting for valid frame...")
                    return // Skip processing this frame
                }
            } else if frameCount % 60 == 0 {
                // Periodically check if the video size has changed and broadcast updates
                if videoSize.width > 0 && videoSize.height > 0 && 
                   (self.lastFrameSize.width != videoSize.width || self.lastFrameSize.height != videoSize.height) {
                    print("GoPro: Frame size changed from \(self.lastFrameSize) to \(videoSize)")
                    
                    // Update our stored dimensions
                    self.longSide = max(videoSize.width, videoSize.height)
                    self.shortSide = min(videoSize.width, videoSize.height)
                    self.lastFrameSize = videoSize
                    
                    // Broadcast the size change
                    self.broadcastFrameSizeChange(videoSize)
                    
                    // Update layout if needed
                    self.updatePlayerViewLayout()
                }
            }
            
            // Use our enhanced frame processing method
            self.processCurrentFrame()
            
            // Calculate and report performance metrics
            let instantFps = self.calculateInstantaneousFps()
            self.delegate?.frameSource(self, didUpdateWithSpeed: 1000.0 / instantFps, fps: instantFps)
            
            // Every 30 frames, report detailed metrics
            if self.frameCount % 30 == 0 {
                // Calculate instantaneous FPS from last several frames if available
                let instantFps = self.calculateInstantaneousFps()
                
                // Calculate overall FPS if stream has been running for at least 1 second
                if let startTime = self.streamStartTime {
                    let elapsed = Date().timeIntervalSince(startTime)
                    let overallFps = Double(self.frameCount) / elapsed
                    print("GoPro: VLC Player - Receiving frames at \(String(format: "%.2f", instantFps)) FPS (avg: \(String(format: "%.2f", overallFps)))")
                }
            }
            
            // Always notify goProDelegate about frame (for backward compatibility)
            self.goProDelegate?.goProSource(self, didReceiveFrameWithTime: self.lastFrameTime)
        }
    }
    
    /// Helper method to broadcast frame size changes to observers
    @MainActor
    private func broadcastFrameSizeChange(_ size: CGSize) {
        // Post a notification with the new frame size
        print("GoPro: Broadcasting frame size change: \(size)")
        
        // Store the frame size locally
        self.lastFrameSize = size
        
        // Update frame dimensions for FrameSource protocol
        self.longSide = max(size.width, size.height)
        self.shortSide = min(size.width, size.height)
        
        // Broadcast to all observers
        NotificationCenter.default.post(
            name: NSNotification.Name("GoProFrameSizeChanged"),
            object: self,
            userInfo: ["frameSize": size]
        )
        
        // Also try to directly update YOLOView if we can find it
        if let yoloView = self.videoCaptureDelegate as? YOLOView {
            print("GoPro: Directly updating YOLOView with frame size: \(size)")
            yoloView.goProLastFrameSize = size
            
            // Force layout update to ensure proper coordinate transformation
            DispatchQueue.main.async {
                yoloView.setNeedsLayout()
                yoloView.layoutIfNeeded()
            }
        }
    }
    
    /// Calculate instantaneous FPS based on recent frame timestamps
    private func calculateInstantaneousFps() -> Double {
        guard frameTimestamps.count >= 2 else { return 0 }
        
        // Calculate time difference between oldest and newest timestamp
        let timeSpan = frameTimestamps.last! - frameTimestamps.first!
        if timeSpan <= 0 { return 0 }
        
        // Calculate frames per second
        return Double(frameTimestamps.count - 1) / timeSpan
    }
    
    // MARK: - GoPro HTTP API Methods
    
    /// Check if connected to a GoPro camera by requesting webcam version
    /// - Parameter completion: Callback with result (success/failure)
    func checkConnection(completion: @escaping (Result<GoProWebcamVersion, Error>) -> Void) {
        print("GoPro: Starting connection check")
        let urlString = "http://\(goProIP):\(goProPort)\(versionEndpoint)"
        
        guard let url = URL(string: urlString) else {
            print("GoPro: Invalid URL format")
            DispatchQueue.main.async {
            completion(.failure(NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
            }
            return
        }
        
        // Create a URL request with a timeout
        var request = URLRequest(url: url)
        request.timeoutInterval = 5.0 // 5 second timeout
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            // Handle network error
            if let error = error {
                print("GoPro: Network error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
                return
            }
            
            // Check HTTP response
            guard let httpResponse = response as? HTTPURLResponse else {
                print("GoPro: Invalid HTTP response")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid HTTP response"])))
                }
                return
            }
            
            print("GoPro: HTTP Status Code: \(httpResponse.statusCode)")
            
            // Check status code
            guard httpResponse.statusCode == 200 else {
                print("GoPro: HTTP error status code: \(httpResponse.statusCode)")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP error: \(httpResponse.statusCode)"])))
                }
                return
            }
            
            // Check data
            guard let data = data, !data.isEmpty else {
                print("GoPro: No data received from server")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: 3, userInfo: [NSLocalizedDescriptionKey: "No data received"])))
                }
                return
            }
            
            // Try to parse JSON - more lenient approach
            do {
                // First try parsing as a dictionary to check for version field
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    if let versionValue = json["version"] as? Int {
                        print("GoPro: Found version: \(versionValue)")
                        // Create a simplified version object with just what we need
                        let version = GoProWebcamVersion(
                            version: versionValue,
                            max_lens_support: json["max_lens_support"] as? Bool,
                            usb_3_1_compatible: json["usb_3_1_compatible"] as? Bool
                        )
                        print("GoPro: Connection successful")
                        DispatchQueue.main.async {
                            completion(.success(version))
                        }
                        return
                    }
                }
                
                // If that didn't work, try standard decoding
                let decoder = JSONDecoder()
                let version = try decoder.decode(GoProWebcamVersion.self, from: data)
                print("GoPro: Connection successful (standard decoding)")
                DispatchQueue.main.async {
                    completion(.success(version))
                }
            } catch {
                // Log detailed error information
                if let dataString = String(data: data, encoding: .utf8) {
                    // If we got ANY response data, consider it a success
                    if !dataString.isEmpty {
                        // Create a minimal version object with default values
                        let defaultVersion = GoProWebcamVersion(
                            version: 1,  // Default version
                            max_lens_support: nil,
                            usb_3_1_compatible: nil
                        )
                        print("GoPro: Connection successful (with default values)")
                        DispatchQueue.main.async {
                            completion(.success(defaultVersion))
                        }
                        return
                    }
                }
                
                print("GoPro: Connection failed - \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
        
        task.resume()
    }
    
    /// Enter webcam preview mode
    /// - Parameter completion: Callback with result (success/failure)
    func enterWebcamPreview(completion: @escaping (Result<Void, Error>) -> Void) {
        print("GoPro: Entering webcam preview mode")
        let urlString = "http://\(goProIP):\(goProPort)\(previewEndpoint)"
        
        guard let url = URL(string: urlString) else {
            print("GoPro: Invalid preview URL format")
            DispatchQueue.main.async {
            completion(.failure(NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid preview URL"])))
            }
            return
        }
        
        // Create a URL request with a timeout
        var request = URLRequest(url: url)
        request.timeoutInterval = 5.0 // 5 second timeout
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            // Handle network error
            if let error = error {
                print("GoPro: Preview network error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
                return
            }
            
            // Check HTTP response
            guard let httpResponse = response as? HTTPURLResponse else {
                print("GoPro: Invalid HTTP response for preview")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid HTTP response"])))
                }
                return
            }
            
            print("GoPro: Preview HTTP Status Code: \(httpResponse.statusCode)")
            
            // Check status code
            guard httpResponse.statusCode == 200 else {
                print("GoPro: Preview HTTP error status code: \(httpResponse.statusCode)")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP error: \(httpResponse.statusCode)"])))
                }
                return
            }
            
            // Success - empty response is expected
            print("GoPro: Preview mode enabled successfully")
            DispatchQueue.main.async {
                completion(.success(()))
            }
        }
        
        task.resume()
    }
    
    /// Start webcam streaming
    /// - Parameter completion: Callback with result (success/failure)
    func startWebcam(completion: @escaping (Result<Void, Error>) -> Void) {
        print("GoPro: Starting webcam stream")
        // Construct URL with query parameters
        let queryParams = "res=12&fov=0&port=8556&protocol=RTSP"
        let urlString = "http://\(goProIP):\(goProPort)\(startEndpoint)?\(queryParams)"
        
        guard let url = URL(string: urlString) else {
            print("GoPro: Invalid start URL format")
            DispatchQueue.main.async {
            completion(.failure(NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid start URL"])))
            }
            return
        }
        
        // Create a URL request with a timeout
        var request = URLRequest(url: url)
        request.timeoutInterval = 5.0 // 5 second timeout
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            // Handle network error
            if let error = error {
                print("GoPro: Start webcam network error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
                return
            }
            
            // Check HTTP response
            guard let httpResponse = response as? HTTPURLResponse else {
                print("GoPro: Invalid HTTP response for start")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid HTTP response"])))
                }
                return
            }
            
            print("GoPro: Start HTTP Status Code: \(httpResponse.statusCode)")
            
            // Check status code
            guard httpResponse.statusCode == 200 else {
                print("GoPro: Start HTTP error status code: \(httpResponse.statusCode)")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP error: \(httpResponse.statusCode)"])))
                }
                return
            }
            
            // Success - empty response is expected
            print("GoPro: Webcam started successfully")
            DispatchQueue.main.async {
                completion(.success(()))
            }
        }
        
        task.resume()
    }
    
    /// Stop webcam streaming
    /// - Parameter completion: Callback with result (success/failure)
    func stopWebcam(completion: @escaping (Result<Void, Error>) -> Void) {
        print("GoPro: Stopping webcam stream")
        let urlString = "http://\(goProIP):\(goProPort)\(stopEndpoint)"
        
        guard let url = URL(string: urlString) else {
            print("GoPro: Invalid stop URL format")
            DispatchQueue.main.async {
            completion(.failure(NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid stop URL"])))
            }
            return
        }
        
        // Create a URL request with a timeout
        var request = URLRequest(url: url)
        request.timeoutInterval = 5.0 // 5 second timeout
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            // Handle network error
            if let error = error {
                print("GoPro: Stop webcam network error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
                return
            }
            
            // Check HTTP response
            guard let httpResponse = response as? HTTPURLResponse else {
                print("GoPro: Invalid HTTP response for stop")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid HTTP response"])))
                }
                return
            }
            
            print("GoPro: Stop HTTP Status Code: \(httpResponse.statusCode)")
            
            // Check status code
            guard httpResponse.statusCode == 200 else {
                print("GoPro: Stop HTTP error status code: \(httpResponse.statusCode)")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP error: \(httpResponse.statusCode)"])))
                }
                return
            }
            
            // Success - empty response is expected
            print("GoPro: Webcam stopped successfully")
            DispatchQueue.main.async {
                completion(.success(()))
            }
        }
        
        task.resume()
    }
    
    /// Exit webcam mode
    /// - Parameter completion: Callback with result (success/failure)
    func exitWebcam(completion: @escaping (Result<Void, Error>) -> Void) {
        print("GoPro: Exiting webcam mode")
        let urlString = "http://\(goProIP):\(goProPort)\(exitEndpoint)"
        
        guard let url = URL(string: urlString) else {
            print("GoPro: Invalid exit URL format")
            DispatchQueue.main.async {
            completion(.failure(NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid exit URL"])))
            }
            return
        }
        
        // Create a URL request with a timeout
        var request = URLRequest(url: url)
        request.timeoutInterval = 5.0 // 5 second timeout
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            // Handle network error
            if let error = error {
                print("GoPro: Exit webcam network error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
                return
            }
            
            // Check HTTP response
            guard let httpResponse = response as? HTTPURLResponse else {
                print("GoPro: Invalid HTTP response for exit")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid HTTP response"])))
                }
                return
            }
            
            print("GoPro: Exit HTTP Status Code: \(httpResponse.statusCode)")
            
            // Check status code
            guard httpResponse.statusCode == 200 else {
                print("GoPro: Exit HTTP error status code: \(httpResponse.statusCode)")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP error: \(httpResponse.statusCode)"])))
                }
                return
            }
            
            // Success - empty response is expected
            print("GoPro: Webcam exited successfully")
            DispatchQueue.main.async {
                completion(.success(()))
            }
        }
        
        task.resume()
    }
    
    /// Gracefully exit webcam mode with both stop and exit commands
    /// - Parameter completion: Callback with result (success/failure)
    func gracefulWebcamExit(completion: @escaping (Result<Void, Error>) -> Void) {
        print("GoPro: Starting graceful webcam exit")
        
        // Stop RTSP stream if running
        stopRTSPStream()
        
        // First stop the webcam
        stopWebcam { [weak self] stopResult in
            guard let self = self else { return }
            
            switch stopResult {
            case .success:
                // Then exit webcam mode
                self.exitWebcam { exitResult in
                    switch exitResult {
                    case .success:
                        print("GoPro: Graceful exit completed successfully")
                        DispatchQueue.main.async {
                        completion(.success(()))
                        }
                    case .failure(let error):
                        print("GoPro: Exit command failed: \(error.localizedDescription)")
                        DispatchQueue.main.async {
                        completion(.failure(error))
                        }
                    }
                }
            case .failure(let error):
                print("GoPro: Stop command failed: \(error.localizedDescription)")
                // Try exit anyway as a best effort
                self.exitWebcam { exitResult in
                    switch exitResult {
                    case .success:
                        // If exit succeeds, still report the stop error
                        DispatchQueue.main.async {
                        completion(.failure(error))
                        }
                    case .failure(let exitError):
                        // Report both errors
                        let combinedError = NSError(
                            domain: "GoProSource",
                            code: 3,
                            userInfo: [
                                NSLocalizedDescriptionKey: "Multiple errors: Stop - \(error.localizedDescription), Exit - \(exitError.localizedDescription)"
                            ]
                        )
                        DispatchQueue.main.async {
                        completion(.failure(combinedError))
                    }
                }
            }
        }
        }
    }

    // MARK: - Simple Test Function for RTSP Stream
    
    /// Enhanced test function to validate RTSP streaming from GoPro
    /// Tests for 30 frames and reports size and FPS
    /// - Parameters:
    ///   - timeout: Timeout in seconds to wait for frames (default 30 seconds)
    ///   - completion: Called with success/failure and detailed message
    func testRTSPStream(timeout: TimeInterval = 30.0, completion: @escaping (Bool, String) -> Void) {
        var testLog = "Starting GoPro RTSP stream test (capturing 30 frames)...\n"
        let testCompleted = Atomic<Bool>(value: false)
        
        // Store original delegate to restore later
        let originalDelegate = self.goProDelegate
        
        // Frame tracking
        let targetFrameCount = 30
        let frameStartTime = Date()
        var framesCaptured = 0
        
        print("GoPro: Test - Setting up test delegate")
        
        // IMPORTANT: Create test delegate as STORED property to prevent deallocation
        // This is critical - a local variable might be released when this function returns
        self.testDelegate = TestDelegate(
            onFirstFrame: { [weak self] size in
                print("GoPro: Test - onFirstFrame callback executing with size: \(size)")
                testLog += "First frame received! Size: \(size)\n"
            },
            onFrame: { [weak self, frameStartTime] time in
                guard let self = self else { return }
                
                // Increment our local frame counter
                framesCaptured += 1
                
                // Get current frame count from the GoProSource
                let currentFrameCount = self.frameCount
                
                // Get video size from the video player
                let videoSize = self.videoPlayer?.videoSize ?? CGSize.zero
                
                // Calculate FPS
                let elapsed = Date().timeIntervalSince(frameStartTime)
                let fps = elapsed > 0 ? Double(framesCaptured) / elapsed : 0
                
                print("GoPro: Test - Frame #\(framesCaptured), Source frame count: \(currentFrameCount), Size: \(videoSize), FPS: \(String(format: "%.2f", fps))")
                testLog += "Frame #\(framesCaptured), Size: \(videoSize), FPS: \(String(format: "%.2f", fps))\n"
                
                // If we've received enough frames, complete the test
                if framesCaptured >= targetFrameCount {
                    // Check if test already completed
                    if testCompleted.exchange(true) {
                        print("GoPro: Test - Frame count reached target but test already completed")
                        return
                    }
                    
                    print("GoPro: Test - Received \(framesCaptured) frames, completing test")
                    
                    // Calculate final stats
                    let totalElapsed = Date().timeIntervalSince(frameStartTime)
                    let avgFPS = Double(framesCaptured) / totalElapsed
                    
                    testLog += "\nTest complete:\n"
                    testLog += "- Total frames captured: \(framesCaptured)\n"
                    testLog += "- Time elapsed: \(String(format: "%.2f", totalElapsed)) seconds\n"
                    testLog += "- Average FPS: \(String(format: "%.2f", avgFPS))\n"
                    
                    // Use a helper method to complete the test safely
                    self.completeTest(
                        originalDelegate: originalDelegate,
                        success: true,
                        message: "Successfully received \(framesCaptured) frames\n\nLog:\n\(testLog)",
                        completion: completion
                    )
                }
            },
            onError: { [weak self] error in
                print("GoPro: Test - onError callback executing: \(error)")
                
                guard let self = self else { 
                    print("GoPro: Test - Self is nil in onError")
                    return 
                }
                
                // Check if test already completed
                if testCompleted.exchange(true) {
                    print("GoPro: Test - onError - test already completed, returning")
                    return
                }
                
                testLog += "Error: \(error)\n"
                print("GoPro: Test - Error received: \(error), completing test")
                
                // Use a helper method to complete the test safely
                self.completeTest(
                    originalDelegate: originalDelegate,
                    success: false,
                    message: "Stream error occurred: \(error)\n\nLog:\n\(testLog)",
                    completion: completion
                )
            }
        )
        
        // Reset tracking variables for accurate testing
        frameCount = 0
        frameTimestamps.removeAll()
        
        // Set test delegate BEFORE starting the stream
        self.goProDelegate = self.testDelegate
        print("GoPro: Test - Delegate set, starting RTSP stream")
        
        // Use the main startRTSPStream function for testing
        startRTSPStream { [weak self] result in
            guard let self = self else { 
                print("GoPro: Test - Self is nil in startRTSPStream completion")
                return 
            }
            
            switch result {
            case .success:
                testLog += "RTSP stream started successfully\n"
                print("GoPro: Test - RTSP stream started, waiting for \(targetFrameCount) frames with \(timeout) sec timeout")
                
                // Set timeout timer with timeout for testing
                Task { [weak self] in
                    // Give time for frames to be received
                    try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                    
                    guard let self = self else { 
                        print("GoPro: Test - Self is nil in timeout handler")
                        return 
                    }
                    
                    // Check if test already completed
                    if testCompleted.exchange(true) {
                        print("GoPro: Test - Timeout handler - test already completed, ignoring timeout")
                        return
                    }
                    
                    // Check how many frames we received
                    let framesReceived = framesCaptured
                    if framesReceived > 0 {
                        print("GoPro: Test - Timeout reached, but received \(framesReceived) frames")
                        testLog += "Timeout: Received \(framesReceived) frames within \(timeout) seconds (target was \(targetFrameCount))\n"
                        
                        // Calculate final stats
                        let totalElapsed = Date().timeIntervalSince(frameStartTime)
                        let avgFPS = Double(framesReceived) / totalElapsed
                        
                        testLog += "\nTest complete (timeout):\n"
                        testLog += "- Total frames: \(framesReceived)\n"
                        testLog += "- Time elapsed: \(String(format: "%.2f", totalElapsed)) seconds\n"
                        testLog += "- Average FPS: \(String(format: "%.2f", avgFPS))\n"
                        
                        // Consider partial success if we got some frames
                        let success = framesReceived >= 5 // At least 5 frames to be considered successful
                        
                        // Use helper method to complete test
                        self.completeTest(
                            originalDelegate: originalDelegate,
                            success: success,
                            message: "Received \(framesReceived)/\(targetFrameCount) frames before timeout\n\nLog:\n\(testLog)",
                            completion: completion
                        )
                    } else {
                        print("GoPro: Test - Timeout reached, no frames received")
                        testLog += "Timeout: No frames received within \(timeout) seconds\n"
                        
                        // Complete with failure
                        self.completeTest(
                            originalDelegate: originalDelegate,
                            success: false,
                            message: "Failed to receive any frames within timeout\n\nLog:\n\(testLog)",
                            completion: completion
                        )
                    }
                }
                
            case .failure(let error):
                // Check if test already completed
                if testCompleted.exchange(true) {
                    print("GoPro: Test - startRTSPStream failure - test already completed, ignoring")
                    return
                }
                
                testLog += "Failed to start RTSP stream: \(error.localizedDescription)\n"
                print("GoPro: Test - Failed to start RTSP stream: \(error.localizedDescription)")
                
                // Complete with failure
                self.completeTest(
                    originalDelegate: originalDelegate,
                    success: false,
                    message: "Failed to start stream\n\nLog:\n\(testLog)",
                    completion: completion
                )
            }
        }
    }
    
    /// Helper method to safely complete the test and clean up resources
    private func completeTest(
        originalDelegate: GoProSourceDelegate?,
        success: Bool,
        message: String,
        completion: @escaping (Bool, String) -> Void
    ) {
        // Stop the stream
        stopRTSPStream()
        
        // Restore original delegate
        self.goProDelegate = originalDelegate
        
        // Clear the test delegate reference
        self.testDelegate = nil
        
        // Report result on main thread
        DispatchQueue.main.async {
            completion(success, message)
        }
    }
    
    // Thread-safe boolean wrapper for atomic operations
    private class Atomic<T>: @unchecked Sendable {
        private let queue = DispatchQueue(label: "com.gopro.atomic")
        private var _value: T
        
        init(value: T) {
            self._value = value
        }
        
        // Atomically exchanges the current value with a new value and returns the old value
        func exchange(_ newValue: T) -> T {
            return queue.sync {
                let oldValue = _value
                _value = newValue
                return oldValue
            }
        }
    }
    
    // Helper delegate for testing
    private class TestDelegate: GoProSourceDelegate {
        let onFirstFrame: (CGSize) -> Void
        let onFrame: (Int64) -> Void
        let onError: (String) -> Void
        
        init(onFirstFrame: @escaping (CGSize) -> Void,
             onFrame: @escaping (Int64) -> Void,
             onError: @escaping (String) -> Void) {
            self.onFirstFrame = onFirstFrame
            self.onFrame = onFrame
            self.onError = onError
        }
        
        func goProSource(_ source: GoProSource, didUpdateStatus status: GoProStreamingStatus) {
            print("GoPro: TestDelegate - Status update: \(status)")
            if case .error(let error) = status {
                print("GoPro: TestDelegate - Calling onError with: \(error)")
                onError(error)
            }
        }
        
        func goProSource(_ source: GoProSource, didReceiveFirstFrame size: CGSize) {
            print("GoPro: TestDelegate - Received first frame with size: \(size)")
            onFirstFrame(size)
        }
        
        func goProSource(_ source: GoProSource, didReceiveFrameWithTime time: Int64) {
            onFrame(time)
        }
    }
    
    // MARK: - FPS Calculation Methods
    
    /// Calculate current FPS based on recent frame timestamps
    @MainActor
    private func calculateCurrentFPS() -> Double {
        guard frameTimestamps.count >= 2 else {
            return 0
        }
        
        // Calculate FPS based on last 10 frames or all available frames
        let framesToConsider = min(10, frameTimestamps.count)
        let recentTimestamps = Array(frameTimestamps.suffix(framesToConsider))
        
        // Calculate time difference between first and last frame
        let timeInterval = recentTimestamps.last! - recentTimestamps.first!
        
        // Calculate frames per second
        return timeInterval > 0 ? Double(framesToConsider - 1) / timeInterval : 0
    }
    
    /// Get average FPS over all recorded frames
    @MainActor
    var averageFPS: Double {
        guard frameTimestamps.count >= 2 else {
            return 0
        }
        
        // Calculate time difference between first and last frame
        let timeInterval = frameTimestamps.last! - frameTimestamps.first!
        
        // Calculate frames per second
        return timeInterval > 0 ? Double(frameTimestamps.count - 1) / timeInterval : 0
    }

    // MARK: - FrameSource Protocol Implementation for UI Integration and Coordinate Transformation

    @MainActor
    func transformDetectionToScreenCoordinates(
        rect: CGRect, 
        viewBounds: CGRect, 
        orientation: UIDeviceOrientation
    ) -> CGRect {
        // Get the container view dimensions
        guard let containerView = containerView, 
              containerView.bounds.width > 0, 
              containerView.bounds.height > 0,
              lastFrameSize.width > 0, 
              lastFrameSize.height > 0 else {
            // Fallback to basic transformation if we don't have valid dimensions
            return VNImageRectForNormalizedRect(rect, Int(viewBounds.width), Int(viewBounds.height))
        }
        
        // Calculate aspect ratios
        let videoAspectRatio = lastFrameSize.width / lastFrameSize.height
        let viewAspectRatio = containerView.bounds.width / containerView.bounds.height
        
        // Get the current orientation
        let isPortrait = orientation.isPortrait || 
                        (!orientation.isLandscape && containerView.bounds.height > containerView.bounds.width)
        
        // Start with the original normalized rect
        var adjustedBox = rect
        
        // Calculate container dimensions
        let containerWidth = containerView.bounds.width
        let containerHeight = containerView.bounds.height
        
        // CRITICAL FIX: Separate handling for portrait and landscape modes
        if isPortrait {
            // PORTRAIT MODE HANDLING
            if videoAspectRatio > viewAspectRatio {
                // Video is wider than container (letterboxing - black bars on top/bottom)
                // Calculate scaled height and vertical offset
                let scaledHeight = containerWidth / videoAspectRatio
                let verticalOffset = (containerHeight - scaledHeight) / 2
                
                // Scale and offset y-coordinates
                let yScale = scaledHeight / containerHeight
                let normalizedOffset = verticalOffset / containerHeight
                
                // Adjust y-coordinates - FIXED for portrait mode
                adjustedBox.origin.y = (rect.origin.y * yScale) + normalizedOffset
                adjustedBox.size.height = rect.size.height * yScale
            } else if videoAspectRatio < viewAspectRatio {
                // Video is taller than container (pillarboxing - black bars on sides)
                // Calculate scaled width and horizontal offset
                let scaledWidth = containerHeight * videoAspectRatio
                let horizontalOffset = (containerWidth - scaledWidth) / 2
                
                // Scale and offset x-coordinates
                let xScale = scaledWidth / containerWidth
                let normalizedOffset = horizontalOffset / containerWidth
                
                // Adjust x-coordinates - FIXED for portrait mode
                adjustedBox.origin.x = (rect.origin.x * xScale) + normalizedOffset
                adjustedBox.size.width = rect.size.width * xScale
            }
        } else {
            // LANDSCAPE MODE HANDLING
            if videoAspectRatio > viewAspectRatio {
                // Video is wider than container (letterboxing - black bars on top/bottom)
                // Calculate scaled height and vertical offset
                let scaledHeight = containerWidth / videoAspectRatio
                let verticalOffset = (containerHeight - scaledHeight) / 2
                
                // Scale and offset y-coordinates
                let yScale = scaledHeight / containerHeight
                let normalizedOffset = verticalOffset / containerHeight
                
                // Adjust y-coordinates for landscape
                adjustedBox.origin.y = (rect.origin.y * yScale) + normalizedOffset
                adjustedBox.size.height = rect.size.height * yScale
            } else if videoAspectRatio < viewAspectRatio {
                // Video is taller than container (pillarboxing - black bars on sides)
                // Calculate scaled width and horizontal offset
                let scaledWidth = containerHeight * videoAspectRatio
                let horizontalOffset = (containerWidth - scaledWidth) / 2
                
                // Scale and offset x-coordinates
                let xScale = scaledWidth / containerWidth
                let normalizedOffset = horizontalOffset / containerWidth
                
                // Adjust x-coordinates for landscape
                adjustedBox.origin.x = (rect.origin.x * xScale) + normalizedOffset
                adjustedBox.size.width = rect.size.width * xScale
            }
        }
        
        // Handle Y-coordinate inversion for both portrait and landscape modes
        // Y=0 in the model output is the top of the image,
        // but Y=0 in the display is the bottom of the image
        adjustedBox.origin.y = 1.0 - adjustedBox.origin.y - adjustedBox.size.height
        
        // Convert normalized coordinates to screen coordinates
        return VNImageRectForNormalizedRect(adjustedBox, Int(containerWidth), Int(containerHeight))
    }

    @MainActor
    func addOverlayLayer(_ layer: CALayer) {
        // Add the overlay layer to the player view layer
        if let playerView = playerView {
            playerView.layer.addSublayer(layer)
        }
    }

    @MainActor
    func addBoundingBoxViews(_ boxViews: [BoundingBoxView]) {
        // Add bounding box views to the player view layer
        if let playerView = playerView {
            for box in boxViews {
                box.addToLayer(playerView.layer)
            }
        }
    }
}

// MARK: - ResultsListener and InferenceTimeListener Implementation
extension GoProSource: @preconcurrency ResultsListener, @preconcurrency InferenceTimeListener {
    nonisolated func on(inferenceTime: Double, fpsRate: Double) {
        // Since this is nonisolated, we need to dispatch to the main actor
        Task { @MainActor [weak self] in
            guard let self = self else { return }
            
            // Forward to both delegate types to ensure complete integration
            self.delegate?.frameSource(self, didUpdateWithSpeed: inferenceTime, fps: fpsRate)
            
            // Also forward to video capture delegate for YOLOView integration
            if let videoCaptureDelegate = self.videoCaptureDelegate {
                videoCaptureDelegate.onInferenceTime(speed: inferenceTime, fps: fpsRate)
            } else {
                print("GoPro: Warning - videoCaptureDelegate is nil for inferenceTime update")
            }
        }
    }
    
    nonisolated func on(result: YOLOResult) {
    // Since this is nonisolated, we need to dispatch to the main actor
    Task { @MainActor [weak self] in
        guard let self = self else { return }
        
        // Ensure we're on the main thread for UI operations
        dispatchPrecondition(condition: .onQueue(.main))
        
        // Increment the frame counter
        self.frameCount += 1
        
        // Record timestamp for performance monitoring
        let timestamp = CACurrentMediaTime()
        self.frameTimestamps.append(timestamp)
        
        // Only keep the last 60 timestamps for rolling performance calculation
        if self.frameTimestamps.count > 60 {
            self.frameTimestamps.removeFirst()
        }
        
        // Calculate FPS
        let currentFPS = self.calculateCurrentFPS()
        
        // Log occasionally
        if self.frameCount % 30 == 0 {
            print("GoPro: VLC Player - Receiving frames at \(String(format: "%.2f", currentFPS)) FPS (avg: \(String(format: "%.2f", self.averageFPS)))")
        }
        
        // Update FPS delegate
        if let delegate = self.delegate {
            delegate.frameSource(self, didUpdateWithSpeed: result.speed, fps: currentFPS)
        }
            
        // IMPORTANT: Always notify the delegate about the result, even if there are no boxes
        // This ensures boxes are cleared when no fish are present
        if let videoCaptureDelegate = self.videoCaptureDelegate {
            // REFACTORED: Pass the original result directly
            // YOLOView will use transformDetectionToScreenCoordinates for each box
            
            // Log detection information (only if there are boxes to avoid spam)
            if result.boxes.count > 0 {
                print("GoPro: Received detection result with \(result.boxes.count) boxes")
            }
            
            // Forward to video capture delegate for YOLOView integration
            // This will trigger box clearing if there are no boxes
            videoCaptureDelegate.onPredict(result: result)
            
            // CRITICAL FIX: Force a redraw of the container view to ensure boxes are visible/hidden
            if let containerView = self.containerView {
                containerView.setNeedsDisplay()
                containerView.layoutIfNeeded()
            }
            
            // If the containerView is a YOLOView, make sure bounding boxes are visible
            if let yoloView = self.containerView as? YOLOView {
                for box in yoloView.boundingBoxViews {
                    if !box.shapeLayer.isHidden {
                        // Ensure the box layer is not hidden and has the right z-position
                        box.shapeLayer.zPosition = 1000
                        box.textLayer.zPosition = 1001
                    }
                }
            }
        } else {
            print("GoPro: WARNING - videoCaptureDelegate is nil for prediction result")
            print("GoPro: Detection results are being LOST!")
            
            // Try to reestablish the delegate connection if possible
            if let view = self.delegate as? VideoCaptureDelegate {
                print("GoPro: Attempting to recover by connecting to delegate")
                self.videoCaptureDelegate = view
                
                // Pass the original result directly
                view.onPredict(result: result)
            }
        }
    }
}
    

}

// MARK: - Backward Compatibility

// Add a class that can adapt GoProSourceDelegate to FrameSourceDelegate
// This allows the new FrameSource implementation to work with older code
class DelegateAdapter: FrameSourceDelegate {
    private weak var goProSource: GoProSource?
    
    init(goProSource: GoProSource) {
        self.goProSource = goProSource
    }
    
    func frameSource(_ source: FrameSource, didOutputImage image: UIImage) {
        // No direct mapping for this in the old protocol
    }
    
    func frameSource(_ source: FrameSource, didUpdateWithSpeed speed: Double, fps: Double) {
        // No direct mapping for this in the old protocol
    }
}

// Extension to add required property to VideoCaptureDelegate
extension VideoCaptureDelegate {
    var viewForDrawing: UIView? {
        // Try to get the view from self if it's a UIView
        return self as? UIView
    }
}

// Extension to YOLOView to access predictor
// Removed since YOLOView now has a built-in getCurrentPredictor method

// Add an extension to UIView for snapshot method
extension UIView {
    func snapshotImage() -> UIImage? {
        // Begin image context
        UIGraphicsBeginImageContextWithOptions(bounds.size, false, 0)
        defer { UIGraphicsEndImageContext() }
        
        // Draw view hierarchy into context
        if let context = UIGraphicsGetCurrentContext() {
            // Save a solid white background to detect if the frame is empty
            context.setFillColor(UIColor.white.cgColor)
            context.fill(bounds)
            
            // Render the view
            drawHierarchy(in: bounds, afterScreenUpdates: true)
        }
        
        // Get the snapshot image
        return UIGraphicsGetImageFromCurrentImageContext()
    }
}

/// Creates a player view for displaying the VLC video output
@MainActor
private func createPlayerView() -> UIView {
    let view = UIView()
    view.backgroundColor = .black
    view.translatesAutoresizingMaskIntoConstraints = false
    
    // Add tag for easier identification in the view hierarchy
    view.tag = 9876
    
    // CRITICAL: Ensure this view correctly handles CALayer drawing for bounding boxes
    view.layer.shouldRasterize = false
    view.layer.drawsAsynchronously = false
    
    // Ensure the view's layer can properly display sublayers
    view.layer.masksToBounds = false
    view.clipsToBounds = false
    
    // Name the layer for easier debugging
    view.layer.name = "goProPlayerLayer"
    
    return view
}
