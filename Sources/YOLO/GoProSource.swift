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

// MARK: - Data Structures

/// Response structure for GoPro webcam version information from HTTP API
/// 
/// This structure represents the JSON response from the `/gopro/webcam/version` endpoint
/// and supports flexible parsing with optional fields to handle different GoPro firmware versions.
struct GoProWebcamVersion: Decodable {
    /// The webcam API version number (required field)
    let version: Int
    /// Whether the GoPro supports maximum lens capabilities (optional)
    let max_lens_support: Bool?
    /// Whether the GoPro is compatible with USB 3.1 (optional)
    let usb_3_1_compatible: Bool?
}

/// Enumeration representing the current status of GoPro RTSP streaming
/// 
/// Used to communicate streaming state changes to delegates for UI updates
/// and error handling throughout the connection lifecycle.
enum GoProStreamingStatus {
    /// Currently attempting to establish connection to GoPro
    case connecting
    /// Successfully streaming video from GoPro
    case playing
    /// Error occurred during streaming with descriptive message
    case error(String)
    /// Streaming has been stopped or disconnected
    case stopped
}

/// Delegate protocol for receiving GoProSource streaming status updates and frame notifications
/// 
/// Implement this protocol to receive callbacks about streaming state changes,
/// first frame detection, and periodic frame timing information for UI updates.
protocol GoProSourceDelegate: AnyObject {
    /// Called when the streaming status changes (connecting, playing, error, stopped)
    /// - Parameters:
    ///   - source: The GoProSource instance reporting the status change
    ///   - status: The new streaming status
    func goProSource(_ source: GoProSource, didUpdateStatus status: GoProStreamingStatus)
    
    /// Called when the first valid frame is received with video dimensions
    /// - Parameters:
    ///   - source: The GoProSource instance that received the frame
    ///   - size: The dimensions of the video frame (width x height)
    func goProSource(_ source: GoProSource, didReceiveFirstFrame size: CGSize)
    
    /// Called periodically with frame timing information for performance monitoring
    /// - Parameters:
    ///   - source: The GoProSource instance providing the timing
    ///   - time: The VLC media time value in milliseconds
    func goProSource(_ source: GoProSource, didReceiveFrameWithTime time: Int64)
}

/// Enumeration of GoPro HTTP API endpoints for webcam functionality
/// 
/// Provides structured access to GoPro's OpenGoPro HTTP API endpoints
/// with automatic URL path and query parameter generation.
private enum GoProEndpoint {
    /// Get webcam version information
    case version
    /// Enter webcam preview mode
    case preview
    /// Start webcam streaming with specified resolution and field of view
    case start(resolution: Int, fov: Int)
    /// Stop webcam streaming
    case stop
    /// Exit webcam mode completely
    case exit
    
    /// The HTTP path component for this endpoint
    var path: String {
        switch self {
        case .version: return "/gopro/webcam/version"
        case .preview: return "/gopro/webcam/preview"
        case .start: return "/gopro/webcam/start"
        case .stop: return "/gopro/webcam/stop"
        case .exit: return "/gopro/webcam/exit"
        }
    }
    
    /// Query parameters required for this endpoint (if any)
    var queryItems: [URLQueryItem]? {
        switch self {
        case .start(let res, let fov):
            return [
                URLQueryItem(name: "res", value: "\(res)"),
                URLQueryItem(name: "fov", value: "\(fov)"),
                URLQueryItem(name: "protocol", value: "RTSP")
            ]
        default: return nil
        }
    }
}

// MARK: - Main Class

/// High-performance GoPro camera frame source implementation using VLC and CADisplayLink
/// 
/// This class provides a complete frame source implementation for GoPro cameras connected via WiFi.
/// It uses a dual-pipeline architecture:
/// - **Rendering Pipeline**: VLC Media Player for smooth RTSP video playback (25-30 FPS)
/// - **Detection Pipeline**: CADisplayLink-driven frame extraction for object detection (16-20 FPS)
/// 
/// Key Features:
/// - Optimized frame extraction with UIGraphicsImageRenderer caching
/// - Smart processing control to prevent CPU waste during inference
/// - Coordinate transformation between video and screen space
/// - Comprehensive performance monitoring and logging
/// - Swift 6 concurrency compliant with MainActor isolation

@MainActor
class GoProSource: NSObject, @preconcurrency FrameSource, @preconcurrency VLCMediaPlayerDelegate {
    
    // MARK: - FrameSource Protocol Properties
    
    /// Delegate for receiving frame output and performance metrics
    weak var delegate: FrameSourceDelegate?
    /// Frame processor for running YOLO inference on extracted frames
    var predictor: FrameProcessor!
    /// Preview layer (always nil for GoPro since we use VLC player view)
    var previewLayer: AVCaptureVideoPreviewLayer? { return nil }
    /// Longer dimension of video frames (updated when first frame received)
    var longSide: CGFloat = 1280
    /// Shorter dimension of video frames (updated when first frame received)
    var shortSide: CGFloat = 720
    /// Whether inference should be performed on frames (false during calibration)
    var inferenceOK: Bool = true
    /// Source type identifier for this frame source
    var sourceType: FrameSourceType { return .goPro }
    
    // MARK: - Properties
    
    /// Delegate for YOLOView integration and detection result handling
    weak var videoCaptureDelegate: VideoCaptureDelegate?
    /// Delegate for GoPro-specific status updates and notifications
    weak var goProDelegate: GoProSourceDelegate?
    
    // Network configuration
    /// GoPro camera IP address when connected to its WiFi network
    private let goProIP = "10.5.5.9"
    /// HTTP API port for GoPro commands
    private let goProPort = 8080
    /// RTSP streaming port for video data
    private let rtspPort = 554
    /// RTSP path for live video stream
    private let rtspPath = "/live"
    
    // VLC components
    /// VLC media player instance for RTSP streaming
    private var videoPlayer: VLCMediaPlayer?
    /// UI view where VLC renders video content
    var playerView: UIView?
    /// Container view that holds the player view (typically YOLOView)
    private weak var containerView: UIView?
    
    // State tracking
    /// Flag indicating if first valid frame has been received from stream
    private var hasReceivedFirstFrame = false
    /// Size of the last received video frame for coordinate transformation
    private var lastFrameSize: CGSize = .zero
    /// Completion callback for stream ready notification
    private var streamReadyCompletion: ((Result<Void, Error>) -> Void)?
    /// Backup timer to ensure stream ready callback is called
    private var streamReadyTimer: Timer?
    
    // Frame processing
    /// CADisplayLink for high-frequency frame extraction (40 Hz)
    private var displayLink: CADisplayLink?
    /// Counter for successful frame extractions for performance monitoring
    private var frameExtractionCount = 0
    /// Timestamps of recent frame extractions for FPS calculation
    private var frameExtractionTimestamps: [CFTimeInterval] = []
    
    // Performance optimizations
    /// Cached UIGraphicsImageRenderer to avoid recreation overhead
    private var cachedRenderer: UIGraphicsImageRenderer?
    /// Bounds of last cached renderer to detect when recreation is needed
    private var lastRendererBounds: CGRect = .zero
    /// Flag to prevent new frame extractions during model processing
    private var isModelProcessing: Bool = false
    
    // Performance metrics
    /// Time spent in last frame extraction operation (milliseconds)
    private var lastExtractionTime: Double = 0
    /// Time spent in last format conversion operation (milliseconds)
    private var lastConversionTime: Double = 0
    /// Time spent in last model inference operation (milliseconds)
    private var lastInferenceTime: Double = 0
    /// Time spent in last UI delegate operation (milliseconds)
    private var lastUITime: Double = 0
    /// Total time for complete pipeline processing (milliseconds)
    private var lastTotalPipelineTime: Double = 0
    /// Timestamps for frame completion events (for FPS calculation)
    private var frameTimestamps: [CFTimeInterval] = []
    /// Timestamps when pipeline processing starts
    private var pipelineStartTimes: [CFTimeInterval] = []
    /// Timestamps when complete pipeline finishes
    private var pipelineCompleteTimes: [CFTimeInterval] = []
    
    // MARK: - Initialization
    
    /// Initializes a new GoProSource instance with VLC player and orientation monitoring
    /// 
    /// Sets up the VLC media player, creates the player view for video rendering,
    /// and registers for device orientation change notifications to handle layout updates.
    override init() {
        super.init()
        setupVLCPlayerAndView()
        NotificationCenter.default.addObserver(
            self, selector: #selector(orientationDidChange),
            name: UIDevice.orientationDidChangeNotification, object: nil
        )
    }
    
    /// Cleanup method that removes notification observers
    deinit {
        NotificationCenter.default.removeObserver(self)
    }
    
    // MARK: - FrameSource Protocol Implementation
    
    /// Sets up the frame source for operation
    /// - Parameter completion: Callback with setup success status
    func setUp(completion: @escaping @Sendable (Bool) -> Void) {
        resetState()
        completion(true)
    }
    
    /// Begins frame acquisition and processing
    /// 
    /// Initializes state, starts RTSP stream if not playing, and begins CADisplayLink
    /// frame extraction at 40Hz for optimal performance balance.
    nonisolated func start() {
        Task { @MainActor in
            hasReceivedFirstFrame = false
            frameExtractionCount = 0
            frameTimestamps.removeAll()
            playerView?.isHidden = false
            containerView?.setNeedsLayout()
            
            if videoPlayer?.state != .playing {
                startRTSPStream { _ in }
            }
            startFrameRateMeasurement()
        }
    }
    
    /// Stops frame acquisition and cleans up resources
    /// 
    /// Stops CADisplayLink, VLC player, cleans up views, and resets all state.
    /// Notifies delegate of stopped status.
    nonisolated func stop() {
        Task { @MainActor in
            stopFrameRateMeasurement()
            videoPlayer?.stop()
            cleanupPlayerView()
            resetState()
            goProDelegate?.goProSource(self, didUpdateStatus: .stopped)
        }
    }
    
    /// Sets zoom ratio (not supported for GoPro RTSP streams)
    /// - Parameter ratio: Desired zoom ratio (ignored)
    nonisolated func setZoomRatio(ratio: CGFloat) {
        // Not supported for GoPro RTSP
    }
    
    /// Captures a still image from the current video frame
    /// - Parameter completion: Callback with captured image or nil if failed
    @MainActor
    func capturePhoto(completion: @escaping @Sendable (UIImage?) -> Void) {
        completion(extractCurrentFrame())
    }
    
    /// Updates layout for device orientation changes
    /// - Parameter orientation: New device orientation
    func updateForOrientationChange(orientation: UIDeviceOrientation) {
        Task { @MainActor in updatePlayerViewLayout() }
    }
    
    /// Requests permission to use this frame source (always granted for network sources)
    /// - Parameter completion: Callback with permission status (always true)
    func requestPermission(completion: @escaping (Bool) -> Void) {
        completion(true) // No permission needed for network streams
    }
    
    /// Shows content selection UI (not applicable for live streams)
    /// - Parameters:
    ///   - viewController: Presenting view controller (unused)
    ///   - completion: Callback with selection status (always false)
    func showContentSelectionUI(from viewController: UIViewController, completion: @escaping (Bool) -> Void) {
        completion(false) // No content selection for live streams
    }
    
    // MARK: - VLC Player Setup and Management
    
    /// Sets up VLC media player and associated UI view for video rendering
    /// 
    /// Creates and configures a VLC media player instance with optimized settings for RTSP streaming.
    /// Also creates the player view with proper constraints and initializes the CADisplayLink
    /// for frame extraction. Disables VLC logging for better performance.
    @MainActor
    private func setupVLCPlayerAndView() {
        videoPlayer = VLCMediaPlayer()
        videoPlayer?.delegate = self
        videoPlayer?.libraryInstance.debugLogging = false
        videoPlayer?.libraryInstance.debugLoggingLevel = 0
        
        playerView = UIView()
        playerView?.backgroundColor = .black
        playerView?.translatesAutoresizingMaskIntoConstraints = false
        playerView?.tag = 9876
        playerView?.frame = CGRect(x: 0, y: 0, width: 1920, height: 1080)
        
        videoPlayer?.drawable = playerView
        videoPlayer?.videoAspectRatio = UnsafeMutablePointer<Int8>(mutating: "16:9")
        videoPlayer?.audio?.volume = 0
        
        setupDisplayLinkForFrameRateMeasurement()
    }
    
    /// Configures VLC media instance with optimized options for GoPro RTSP streaming
    /// 
    /// Applies critical VLC options for low-latency, high-performance RTSP streaming:
    /// - Forces TCP transport for reliability
    /// - Uses minimal but stable caching for network resilience
    /// - Enables hardware acceleration with controlled buffering
    /// - Optimizes frame buffer size for GoPro streams with stability
    /// - Adds clock synchronization for extended runtime stability
    /// - Disables unnecessary features (audio, subtitles)
    /// 
    /// - Parameter media: VLC media instance to configure
    private func configureRTSPStream(_ media: VLCMedia) {
        let options = [
            // Basic configuration
            ":verbose=0", ":rtsp-tcp", ":no-sub-autodetect-file", ":no-audio",
            
            // Conservative caching for network stability (was 0, now minimal buffering)
            ":network-caching=150",        // 150ms network buffer for stability
            ":live-caching=100",           // 100ms live cache to handle processing delays
            ":file-caching=0",             // Keep file caching disabled
            
            // Buffer management for extended runtime stability
            ":rtsp-frame-buffer-size=2000000",  // Reduced from 4MB to 2MB to prevent accumulation
            ":avcodec-hw=any",
            ":avcodec-threads=4",
            
            // Clock synchronization for drift prevention (was disabled, now enabled with tolerance)
            ":clock-jitter=100",           // Allow 100ms jitter tolerance
            ":clock-synchro=1",            // Enable clock synchronization
            
            // Additional stability options for extended runtime
            ":rtsp-caching=100",           // RTSP-specific caching
            ":input-repeat=0",             // Disable input repeat to prevent loops
            ":rtsp-timeout=5",              // 5-second timeout for hung connections

            // No frame skipping
            ":avcodec-skip-frame=0",
            ":avcodec-skip-idct=0",
            ":skip-frames=0",
            ":drop-late-frames=0",
            ":rtsp-mcast-timeout=5000",
            ":network-timeout=5000"
        ]
        options.forEach { media.addOption($0) }
        media.clearStoredCookies()
    }
    
    /// Safely removes player view from container and cleans up constraints
    /// 
    /// Performs proper cleanup of the player view hierarchy to prevent memory leaks
    /// and constraint conflicts when switching frame sources. Deactivates all constraints
    /// referencing the player view before removal.
    @MainActor
    private func cleanupPlayerView() {
        guard let playerView = playerView else { return }
        playerView.isHidden = true
        
        if let superview = playerView.superview {
            NSLayoutConstraint.deactivate(superview.constraints.filter {
                $0.firstItem === playerView || $0.secondItem === playerView
            })
        }
        NSLayoutConstraint.deactivate(playerView.constraints)
        playerView.removeFromSuperview()
        containerView = nil
    }
    
    // MARK: - Frame Processing Pipeline
    
    /// Initializes CADisplayLink for high-frequency frame extraction
    /// 
    /// Creates a CADisplayLink running at 40Hz to provide optimal timing granularity
    /// for frame extraction without overwhelming the UI thread. The display link starts
    /// paused and is activated when streaming begins.
    @MainActor
    private func setupDisplayLinkForFrameRateMeasurement() {
        displayLink = CADisplayLink(target: self, selector: #selector(displayLinkCallback))
        displayLink?.preferredFramesPerSecond = 40
        displayLink?.isPaused = true
        displayLink?.add(to: .main, forMode: .default)
    }
    
    /// CADisplayLink callback for frame extraction timing
    /// 
    /// Lightweight validation method called at 40Hz to trigger frame processing.
    /// Performs quick checks to ensure streaming is active and model is available
    /// before dispatching actual frame processing to avoid blocking the UI thread.
    /// Tracks performance metrics and logs statistics every 300 successful frames.
    @objc private func displayLinkCallback() {
        guard let videoPlayer = videoPlayer,
              [.playing, .buffering].contains(videoPlayer.state),
              let playerView = playerView,
              playerView.bounds.size.width > 0,
              videoCaptureDelegate != nil,
              !isModelProcessing else { return }
        
        Task { @MainActor in
            if processCurrentFrame() {
                frameExtractionCount += 1
                frameExtractionTimestamps.append(CACurrentMediaTime())
                
                // Keep only last 60 timestamps
                if frameExtractionTimestamps.count > 60 {
                    frameExtractionTimestamps.removeFirst()
                }
                
                // Log performance every 300 frames
                if frameExtractionCount % 300 == 0 {
                    logPerformanceMetrics()
                }
            }
        }
    }
    
    /// Extracts current video frame as UIImage using optimized rendering
    /// 
    /// Captures the current frame from the VLC player view using UIGraphicsImageRenderer
    /// with several performance optimizations:
    /// - Caches renderer instances to avoid recreation overhead (~18ms extraction time)
    /// - Temporarily hides UI overlays to prevent them appearing in extracted frames
    /// - Uses `afterScreenUpdates: false` to avoid waiting for screen refresh
    /// - Only recreates renderer when view bounds change
    /// 
    /// - Returns: UIImage of current frame, or nil if extraction fails
    @MainActor
    private func extractCurrentFrame() -> UIImage? {
        guard let playerView = playerView,
              playerView.bounds.size.width > 0,
              let videoPlayer = videoPlayer,
              [.playing, .paused, .buffering].contains(videoPlayer.state) else { return nil }
        
        // Hide UI overlays temporarily
        if let yoloView = containerView as? YOLOView {
            yoloView.hideUIOverlaysForExtraction()
            defer { yoloView.restoreUIOverlaysAfterExtraction() }
        }
        
        // Use cached renderer for performance
        let currentBounds = playerView.bounds
        if cachedRenderer == nil || lastRendererBounds != currentBounds {
            cachedRenderer = UIGraphicsImageRenderer(bounds: currentBounds)
            lastRendererBounds = currentBounds
        }
        
        guard let renderer = cachedRenderer else { return nil }
        
        let snapshot = renderer.image { _ in
            playerView.drawHierarchy(in: currentBounds, afterScreenUpdates: false)
        }
        
        if lastFrameSize != snapshot.size {
            lastFrameSize = snapshot.size
        }
        
        return snapshot
    }
    
    /// Processes current video frame through the complete detection pipeline
    /// 
    /// Main processing method that handles the complete frame processing pipeline:
    /// 1. **Frame Extraction** (~18ms): Captures current video frame as UIImage
    /// 2. **Display Update**: Sends frame to delegate for immediate UI display
    /// 3. **Calibration Check**: Routes to calibration if in auto-calibration mode
    /// 4. **Format Conversion** (~3ms): Converts UIImage to CMSampleBuffer
    /// 5. **Model Inference** (~28ms): Dispatches to background for YOLO processing
    /// 
    /// Performance: Total pipeline ~50ms (20 FPS theoretical, 16-18 FPS actual)
    /// 
    /// - Returns: True if frame was successfully processed, false if skipped
    @MainActor
    private func processCurrentFrame() -> Bool {
        guard let videoPlayer = videoPlayer,
              [.playing, .buffering].contains(videoPlayer.state),
              let playerView = playerView,
              playerView.bounds.size.width > 0 else { return false }
        
        // Measure total pipeline time from start and record for throughput calculation
        let pipelineStartTime = CACurrentMediaTime()
        pipelineStartTimes.append(pipelineStartTime)
        if pipelineStartTimes.count > 30 {
            pipelineStartTimes.removeFirst()
        }
        
        // Step 1: Frame extraction
        let extractionStartTime = CACurrentMediaTime()
        guard let frameImage = extractCurrentFrame() else { return false }
        lastExtractionTime = (CACurrentMediaTime() - extractionStartTime) * 1000
        
        // Always send frame to delegate for display
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.delegate?.frameSource(self, didOutputImage: frameImage)
        }
        
        // Check if we're in calibration mode
        let shouldProcessForCalibration = !inferenceOK && predictor is TrackingDetector
        
        if shouldProcessForCalibration {
            // Process for auto-calibration
            guard let trackingDetector = predictor as? TrackingDetector else {
                print("GoPro: Cannot process for calibration - predictor is not TrackingDetector")
                return false
            }
            
            // Clear boxes when starting calibration (first frame)
            if trackingDetector.getCalibrationFrameCount() == 0 {
                print("GoPro: Starting calibration, clearing boxes")
                DispatchQueue.main.async { [weak self] in
                    self?.videoCaptureDelegate?.onClearBoxes()
                }
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
            return false
        }
        
        // Regular inference
        if inferenceOK, let predictor = self.predictor {
            let conversionStartTime = CACurrentMediaTime()
            if let sampleBuffer = preProcessExtractedFrame(frameImage) {
                lastConversionTime = (CACurrentMediaTime() - conversionStartTime) * 1000
                
                let preInferenceTime = (CACurrentMediaTime() - pipelineStartTime) * 1000
                
                // Log detailed timing breakdown periodically (before inference)
                if frameExtractionCount % 300 == 0 {
                    let frameSize = frameImage.size
                    print("GoPro: Frame #\(frameExtractionCount) - Pre-Inference (Size: \(String(format: "%.0f", frameSize.width))×\(String(format: "%.0f", frameSize.height))): Extraction \(String(format: "%.1f", lastExtractionTime))ms + Conversion \(String(format: "%.1f", lastConversionTime))ms = \(String(format: "%.1f", preInferenceTime))ms total")
                }
                
                // Ensure the delegate is set up correctly
                if self.videoCaptureDelegate == nil {
                    // Only log warning occasionally to avoid spam
                    if frameExtractionCount % 300 == 0 {
                        print("GoPro: Warning - videoCaptureDelegate is nil, detection results will be lost")
                    }
                }
                
                // Run inference - dispatch to background to avoid blocking UI
                DispatchQueue.global(qos: .userInteractive).async { [weak self] in
                    guard let self = self else { return }
                    
                    // Set processing flag to prevent new frame extractions
                    Task { @MainActor in
                        self.isModelProcessing = true
                    }
                    
                    predictor.predict(
                        sampleBuffer: sampleBuffer,
                        onResultsListener: self,
                        onInferenceTime: self
                    )
                }
            }
        }
        
        return true
    }
    
    /// Converts extracted UIImage to CMSampleBuffer for model inference
    /// 
    /// Performs the format conversion from UIImage to the CMSampleBuffer format
    /// required by the Vision framework and YOLO model. Uses standard conversion
    /// functions with proper pixel format handling for the GoPro source type.
    /// 
    /// - Parameter image: UIImage from frame extraction
    /// - Returns: CMSampleBuffer for model processing, or nil if conversion fails
    private func preProcessExtractedFrame(_ image: UIImage) -> CMSampleBuffer? {
        guard let pixelBuffer = createStandardPixelBuffer(from: image, forSourceType: sourceType) else { return nil }
        return createStandardSampleBuffer(from: pixelBuffer)
    }
    
    // MARK: - RTSP Streaming Management
    
    @MainActor
    func startRTSPStream(completion: @escaping (Result<Void, Error>) -> Void) {
        if let videoPlayer = videoPlayer, videoPlayer.state != .stopped {
            videoPlayer.stop()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
                self?.startRTSPStreamInternal(completion: completion)
            }
            return
        }
        startRTSPStreamInternal(completion: completion)
    }
    
    @MainActor
    private func startRTSPStreamInternal(completion: @escaping (Result<Void, Error>) -> Void) {
        guard isProperlyIntegrated() else {
            completion(.failure(NSError(domain: "GoProSource", code: 3, 
                userInfo: [NSLocalizedDescriptionKey: "Not properly integrated with YOLOView"])))
            return
        }
        
        let rtspURLString = "rtsp://\(goProIP):\(rtspPort)\(rtspPath)"
        guard let url = URL(string: rtspURLString), let videoPlayer = videoPlayer else {
            completion(.failure(NSError(domain: "GoProSource", code: 1, 
                userInfo: [NSLocalizedDescriptionKey: "Invalid RTSP URL or player not initialized"])))
            return
        }
        
        streamReadyCompletion = completion
        
        // Setup backup timer
        streamReadyTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: false) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self = self,
                      let videoPlayer = self.videoPlayer,
                      videoPlayer.state == .playing,
                      let completion = self.streamReadyCompletion else { return }
                
                self.streamReadyCompletion = nil
                self.streamReadyTimer = nil
                completion(.success(()))
            }
        }
        
        goProDelegate?.goProSource(self, didUpdateStatus: .connecting)
        
        if let playerView = playerView {
            playerView.layoutIfNeeded()
            playerView.isHidden = false
            videoPlayer.drawable = playerView
        }
        
        let media = VLCMedia(url: url)
        configureRTSPStream(media)
        videoPlayer.media = media
        videoPlayer.play()
    }
    
    @MainActor
    private func isProperlyIntegrated() -> Bool {
        guard let playerView = playerView,
              let containerView = containerView,
              videoCaptureDelegate != nil,
              predictor != nil,
              playerView.superview === containerView,
              containerView.bounds.size.width > 0,
              containerView.bounds.size.height > 0 else { return false }
        return true
    }
    
    @MainActor
    func stopRTSPStream() {
        videoPlayer?.stop()
        streamReadyCompletion = nil
        streamReadyTimer?.invalidate()
        streamReadyTimer = nil
        goProDelegate?.goProSource(self, didUpdateStatus: .stopped)
    }
    
    // MARK: - Helper Methods
    
    /// Resets all internal state variables to initial values
    /// 
    /// Comprehensive state reset method called during setup and stop operations.
    /// Clears all timing arrays, performance metrics, cached objects, and state flags
    /// to ensure clean initialization for new streaming sessions.
    private func resetState() {
        hasReceivedFirstFrame = false
        frameExtractionCount = 0
        isModelProcessing = false
        cachedRenderer = nil
        lastRendererBounds = .zero
        streamReadyCompletion = nil
        streamReadyTimer?.invalidate()
        streamReadyTimer = nil
        frameTimestamps.removeAll()
        pipelineStartTimes.removeAll()
        pipelineCompleteTimes.removeAll()
        lastExtractionTime = 0
        lastConversionTime = 0
        lastInferenceTime = 0
        lastUITime = 0
        lastTotalPipelineTime = 0
    }
    
    /// Starts CADisplayLink-based frame rate measurement and processing
    @MainActor
    private func startFrameRateMeasurement() {
        frameExtractionCount = 0
        frameExtractionTimestamps.removeAll()
        displayLink?.isPaused = false
    }
    
    /// Stops CADisplayLink frame processing to pause frame extraction
    @MainActor
    private func stopFrameRateMeasurement() {
        displayLink?.isPaused = true
    }
    
    @MainActor
    private func updatePlayerViewLayout() {
        guard let playerView = playerView, let containerView = containerView,
              containerView.bounds.size.width > 0, containerView.bounds.size.height > 0,
              playerView.superview === containerView else { return }
        
        DispatchQueue.main.async {
            containerView.setNeedsLayout()
            containerView.layoutIfNeeded()
            playerView.frame = containerView.bounds
        }
    }
    
    @objc private func orientationDidChange() {
        DispatchQueue.main.async { [weak self] in
            let orientation = UIDevice.current.orientation
            if orientation.isPortrait || orientation.isLandscape {
                self?.updateForOrientationChange(orientation: orientation)
            }
        }
    }
    
    /// Calculates current frame completion rate based on recent timestamps
    /// 
    /// - Returns: Current FPS based on last 5 frame completions
    private func calculateCurrentFPS() -> Double {
        guard frameTimestamps.count >= 2 else { return 0 }
        let recentTimestamps = Array(frameTimestamps.suffix(5))
        let timeInterval = recentTimestamps.last! - recentTimestamps.first!
        return timeInterval > 0 ? Double(recentTimestamps.count - 1) / timeInterval : 0
    }
    
    /// Calculates actual pipeline throughput FPS based on complete processing cycles
    /// 
    /// Provides the most accurate FPS measurement by tracking complete pipeline
    /// cycles from frame extraction through model inference completion.
    /// Uses last 10 completions for stable measurement.
    /// 
    /// - Returns: True throughput FPS (typically 16-18 FPS vs 20 FPS theoretical)
    @MainActor
    private func calculateThroughputFPS() -> Double {
        guard pipelineCompleteTimes.count >= 2 else { return 0 }
        
        // Calculate throughput based on last 10 complete pipeline cycles
        let cyclesToConsider = min(10, pipelineCompleteTimes.count)
        let recentCompletions = Array(pipelineCompleteTimes.suffix(cyclesToConsider))
        
        // Calculate time difference between first and last completion
        let timeInterval = recentCompletions.last! - recentCompletions.first!
        
        // Calculate completed pipelines per second (true throughput)
        return timeInterval > 0 ? Double(cyclesToConsider - 1) / timeInterval : 0
    }
    
    private func logPerformanceMetrics() {
        guard frameExtractionTimestamps.count >= 2 else { return }
        let windowSeconds = frameExtractionTimestamps.last! - frameExtractionTimestamps.first!
        let successfulProcessingFPS = Double(frameExtractionTimestamps.count - 1) / windowSeconds
        let frameSizeStr = lastFrameSize.width > 0 ? 
            "\(String(format: "%.0f", lastFrameSize.width))×\(String(format: "%.0f", lastFrameSize.height))" : "Unknown"
        
        print("GoPro: Performance (Frame Size: \(frameSizeStr)) - Processing FPS: \(String(format: "%.1f", successfulProcessingFPS))")
    }
    
    private func broadcastFrameSizeChange(_ size: CGSize) {
        lastFrameSize = size
        longSide = max(size.width, size.height)
        shortSide = min(size.width, size.height)
        
        NotificationCenter.default.post(
            name: NSNotification.Name("GoProFrameSizeChanged"),
            object: self, userInfo: ["frameSize": size]
        )
        
        if let yoloView = videoCaptureDelegate as? YOLOView {
            yoloView.goProLastFrameSize = size
            DispatchQueue.main.async {
                yoloView.setNeedsLayout()
                yoloView.layoutIfNeeded()
            }
        }
    }
}

// MARK: - VLC Media Player Delegate Methods

extension GoProSource {
    nonisolated func mediaPlayerStateChanged(_ aNotification: Notification!) {
        guard let player = aNotification.object as? VLCMediaPlayer else { return }
        let playerState = player.state
        let playerVideoSize = player.videoSize
        
        Task { @MainActor in
            switch playerState {
            case .playing:
                self.goProDelegate?.goProSource(self, didUpdateStatus: .playing)
                self.startFrameRateMeasurement()
                
                if playerVideoSize.width > 0 && playerVideoSize.height > 0 {
                    self.broadcastFrameSizeChange(playerVideoSize)
                }
                
                // Backup completion mechanism
                if let completion = self.streamReadyCompletion {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
                        guard let self = self,
                              let completion = self.streamReadyCompletion else { return }
                        
                        self.streamReadyCompletion = nil
                        self.streamReadyTimer?.invalidate()
                        self.streamReadyTimer = nil
                        completion(.success(()))
                    }
                }
                
            case .error:
                self.stopFrameRateMeasurement()
                self.goProDelegate?.goProSource(self, didUpdateStatus: .error("Playback error"))
                
                if let completion = self.streamReadyCompletion {
                    self.streamReadyCompletion = nil
                    self.streamReadyTimer?.invalidate()
                    self.streamReadyTimer = nil
                    completion(.failure(NSError(domain: "GoProSource", code: 2, 
                        userInfo: [NSLocalizedDescriptionKey: "VLC playback error"])))
                }
                
            case .ended, .stopped:
                self.stopFrameRateMeasurement()
                self.goProDelegate?.goProSource(self, didUpdateStatus: .stopped)
                
            default:
                break
            }
        }
    }
    
    nonisolated func mediaPlayerTimeChanged(_ aNotification: Notification!) {
        guard let player = aNotification.object as? VLCMediaPlayer else { return }
        let playerVideoSize = player.videoSize
        let playerTimeValue = Int64(player.time.intValue)
        
        Task { @MainActor in
            if !self.hasReceivedFirstFrame && playerVideoSize.width > 0 && playerVideoSize.height > 0 {
                self.hasReceivedFirstFrame = true
                self.broadcastFrameSizeChange(playerVideoSize)
                
                DispatchQueue.main.async { [weak self] in
                    self?.updatePlayerViewLayout()
                }
                
                self.goProDelegate?.goProSource(self, didReceiveFirstFrame: playerVideoSize)
                
                if let completion = self.streamReadyCompletion {
                    self.streamReadyCompletion = nil
                    self.streamReadyTimer?.invalidate()
                    self.streamReadyTimer = nil
                    DispatchQueue.main.async { completion(.success(())) }
                }
            }
            
            if self.frameExtractionCount % 30 == 0 {
                self.goProDelegate?.goProSource(self, didReceiveFrameWithTime: playerTimeValue)
            }
        }
    }
}

// MARK: - ResultsListener and InferenceTimeListener Implementation

extension GoProSource: @preconcurrency ResultsListener, @preconcurrency InferenceTimeListener {
    nonisolated func on(inferenceTime: Double, fpsRate: Double) {
        Task { @MainActor in
            self.lastInferenceTime = inferenceTime
            self.delegate?.frameSource(self, didUpdateWithSpeed: inferenceTime, fps: fpsRate)
            self.videoCaptureDelegate?.onInferenceTime(speed: inferenceTime, fps: fpsRate)
        }
    }
    
    nonisolated func on(result: YOLOResult) {
        Task { @MainActor in
            // OPTIMIZATION: Clear processing flag to allow new frame extractions
            self.isModelProcessing = false
            
            // Record the start of post-processing phase
            let postProcessingStartTime = CACurrentMediaTime()
            
            // Record timestamp for performance monitoring
            let timestamp = CACurrentMediaTime()
            self.frameTimestamps.append(timestamp)
            if self.frameTimestamps.count > 30 {
                self.frameTimestamps.removeFirst()
            }

            // UI DELEGATE CALL - Track this time separately (NOT actual fish counting)
            let uiDelegateStartTime = CACurrentMediaTime()
            
            // Forward detection results to video capture delegate
            var delegateCallSuccessful = false
            if let videoCaptureDelegate = self.videoCaptureDelegate {
                videoCaptureDelegate.onPredict(result: result)
                delegateCallSuccessful = true
                
                // Force immediate UI update for responsive box rendering
                if let containerView = self.containerView {
                    DispatchQueue.main.async {
                        containerView.setNeedsDisplay()
                        
                        // If the containerView is a YOLOView, ensure bounding boxes are properly rendered
                        if let yoloView = containerView as? YOLOView {
                            // Force immediate layout for responsive box updates
                            yoloView.setNeedsLayout()
                            yoloView.layoutIfNeeded()
                            
                            // Ensure boxes are visible with proper z-position
                            for box in yoloView.boundingBoxViews {
                                if !box.shapeLayer.isHidden {
                                    box.shapeLayer.zPosition = 1000
                                    box.textLayer.zPosition = 1001
                                }
                            }
                        }
                    }
                }
            } else {
                // Only log warning occasionally to avoid spam
                if self.frameExtractionCount % 300 == 0 {
                    print("GoPro: WARNING - videoCaptureDelegate is nil for prediction result")
                    print("GoPro: Detection results are being LOST!")
                }
            }
            
            // Calculate UI delegate call time (this is NOT fish counting time)
            let uiDelegateTime = (CACurrentMediaTime() - uiDelegateStartTime) * 1000 // ms
            self.lastUITime = uiDelegateTime
            
            // Calculate total UI processing time (including async UI updates)
            let totalUITime = (CACurrentMediaTime() - postProcessingStartTime) * 1000 // ms
            
            // Update total pipeline time to include UI processing
            self.lastTotalPipelineTime = self.lastExtractionTime + self.lastConversionTime + self.lastInferenceTime + totalUITime
            
            // Store pipeline completion time for accurate throughput calculation
            self.pipelineCompleteTimes.append(timestamp)
            if self.pipelineCompleteTimes.count > 30 {
                self.pipelineCompleteTimes.removeFirst()
            }
            
            // Calculate actual throughput FPS based on pipeline completion rate
            let actualThroughputFPS = self.calculateThroughputFPS()
            
            // Update delegate with actual throughput FPS (most accurate)
            self.delegate?.frameSource(self, didUpdateWithSpeed: result.speed, fps: actualThroughputFPS)
            
            // ACCURATE logging for GoProSource pipeline analysis
            if self.frameExtractionCount % 300 == 0 {
                let delegateStatus = delegateCallSuccessful ? "✓" : "✗"
                let theoreticalFPS = self.lastTotalPipelineTime > 0 ? 1000.0 / self.lastTotalPipelineTime : 0
                
                print("GoPro: === GOPRO SOURCE PIPELINE ANALYSIS (Frame #\(self.frameExtractionCount)) ===")
                print("GoPro: Frame Extraction: \(String(format: "%.1f", self.lastExtractionTime))ms | Format Conversion: \(String(format: "%.1f", self.lastConversionTime))ms")
                print("GoPro: Model Inference: \(String(format: "%.1f", self.lastInferenceTime))ms (includes fish counting inside TrackingDetector)")
                print("GoPro: UI Delegate Call: \(String(format: "%.1f", self.lastUITime))ms \(delegateStatus) | Async UI Rendering: \(String(format: "%.1f", totalUITime - self.lastUITime))ms")
                print("GoPro: GoProSource Total: \(String(format: "%.1f", self.lastTotalPipelineTime))ms")
                print("GoPro: Theoretical FPS: \(String(format: "%.1f", theoreticalFPS)) | Actual Throughput: \(String(format: "%.1f", actualThroughputFPS)) | Model Reported FPS: \(String(format: "%.1f", result.fps ?? 0.0))")
                
                // Calculate breakdown percentages
                let extractionPct = (self.lastExtractionTime / self.lastTotalPipelineTime) * 100
                let conversionPct = (self.lastConversionTime / self.lastTotalPipelineTime) * 100
                let inferencePct = (self.lastInferenceTime / self.lastTotalPipelineTime) * 100
                let uiPct = (totalUITime / self.lastTotalPipelineTime) * 100
                
                print("GoPro: Breakdown - Extraction: \(String(format: "%.1f", extractionPct))% | Conversion: \(String(format: "%.1f", conversionPct))% | Inference+FishCount: \(String(format: "%.1f", inferencePct))% | UI: \(String(format: "%.1f", uiPct))%")
            }
        }
    }
}

// MARK: - UI Integration and Coordinate Transformation

extension GoProSource {
    @MainActor
    func transformDetectionToScreenCoordinates(rect: CGRect, viewBounds: CGRect, orientation: UIDeviceOrientation) -> CGRect {
        guard let containerView = containerView,
              containerView.bounds.width > 0, containerView.bounds.height > 0,
              lastFrameSize.width > 0, lastFrameSize.height > 0 else {
            return VNImageRectForNormalizedRect(rect, Int(viewBounds.width), Int(viewBounds.height))
        }
        
        let videoAspectRatio = lastFrameSize.width / lastFrameSize.height
        let viewAspectRatio = containerView.bounds.width / containerView.bounds.height
        let isPortrait = orientation.isPortrait || containerView.bounds.height > containerView.bounds.width
        
        var adjustedBox = rect
        
        if isPortrait {
            if videoAspectRatio > viewAspectRatio {
                let scaledHeight = containerView.bounds.width / videoAspectRatio
                let verticalOffset = (containerView.bounds.height - scaledHeight) / 2
                let yScale = scaledHeight / containerView.bounds.height
                let normalizedOffset = verticalOffset / containerView.bounds.height
                
                adjustedBox.origin.y = (rect.origin.y * yScale) + normalizedOffset
                adjustedBox.size.height = rect.size.height * yScale
            } else if videoAspectRatio < viewAspectRatio {
                let scaledWidth = containerView.bounds.height * videoAspectRatio
                let horizontalOffset = (containerView.bounds.width - scaledWidth) / 2
                let xScale = scaledWidth / containerView.bounds.width
                let normalizedOffset = horizontalOffset / containerView.bounds.width
                
                adjustedBox.origin.x = (rect.origin.x * xScale) + normalizedOffset
                adjustedBox.size.width = rect.size.width * xScale
            }
        }
        
        adjustedBox.origin.y = 1.0 - adjustedBox.origin.y - adjustedBox.size.height
        return VNImageRectForNormalizedRect(adjustedBox, Int(containerView.bounds.width), Int(containerView.bounds.height))
    }
    
    @MainActor
    func integrateWithYOLOView(view: UIView) {
        guard let captureDelegate = view as? VideoCaptureDelegate,
              let frameDelegate = view as? FrameSourceDelegate else { return }
        
        self.delegate = frameDelegate
        self.videoCaptureDelegate = captureDelegate
        
        if let goProDelegate = view as? GoProSourceDelegate {
            self.goProDelegate = goProDelegate
        }
        
        setupPlayerViewInContainer(view)
        
        if predictor == nil, let yoloView = view as? YOLOView {
            predictor = yoloView.getCurrentPredictor()
        }
    }
    
    @MainActor
    private func setupPlayerViewInContainer(_ containerView: UIView) {
        self.containerView = containerView
        guard let playerView = playerView else { return }
        
        playerView.removeFromSuperview()
        
        if containerView.subviews.isEmpty {
            containerView.addSubview(playerView)
        } else {
            containerView.insertSubview(playerView, at: 0)
        }
        
        NSLayoutConstraint.activate([
            playerView.leadingAnchor.constraint(equalTo: containerView.leadingAnchor),
            playerView.trailingAnchor.constraint(equalTo: containerView.trailingAnchor),
            playerView.topAnchor.constraint(equalTo: containerView.topAnchor),
            playerView.bottomAnchor.constraint(equalTo: containerView.bottomAnchor)
        ])
        
        containerView.layoutIfNeeded()
        videoPlayer?.drawable = playerView
        
        if let yoloView = containerView as? YOLOView {
            for box in yoloView.boundingBoxViews {
                box.addToLayer(playerView.layer)
            }
        }
    }
    
    @MainActor
    func addOverlayLayer(_ layer: CALayer) {
        playerView?.layer.addSublayer(layer)
    }
    
    @MainActor
    func addBoundingBoxViews(_ boxViews: [BoundingBoxView]) {
        guard let playerView = playerView else { return }
        for box in boxViews {
            box.addToLayer(playerView.layer)
        }
    }
}

// MARK: - HTTP API Methods

extension GoProSource {
    private func performGoProRequest(endpoint: GoProEndpoint, completion: @escaping (Result<Data?, Error>) -> Void) {
        var urlComponents = URLComponents()
        urlComponents.scheme = "http"
        urlComponents.host = goProIP
        urlComponents.port = goProPort
        urlComponents.path = endpoint.path
        urlComponents.queryItems = endpoint.queryItems
        
        guard let url = urlComponents.url else {
            completion(.failure(NSError(domain: "GoProSource", code: 1, 
                userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
            return
        }
        
        var request = URLRequest(url: url)
        request.timeoutInterval = 5.0
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                DispatchQueue.main.async { completion(.failure(error)) }
                return
            }
            
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
                let error = NSError(domain: "GoProSource", code: statusCode, 
                    userInfo: [NSLocalizedDescriptionKey: "HTTP error: \(statusCode)"])
                DispatchQueue.main.async { completion(.failure(error)) }
                return
            }
            
            DispatchQueue.main.async { completion(.success(data)) }
        }.resume()
    }
    
    func checkConnection(completion: @escaping (Result<GoProWebcamVersion, Error>) -> Void) {
        performGoProRequest(endpoint: .version) { result in
            switch result {
            case .success(let data):
                guard let data = data, !data.isEmpty else {
                    completion(.failure(NSError(domain: "GoProSource", code: 3, 
                        userInfo: [NSLocalizedDescriptionKey: "No data received"])))
                    return
                }
                
                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let versionValue = json["version"] as? Int {
                        let version = GoProWebcamVersion(
                            version: versionValue,
                            max_lens_support: json["max_lens_support"] as? Bool,
                            usb_3_1_compatible: json["usb_3_1_compatible"] as? Bool
                        )
                        completion(.success(version))
                        return
                    }
                    
                    let decoder = JSONDecoder()
                    let version = try decoder.decode(GoProWebcamVersion.self, from: data)
                    completion(.success(version))
                } catch {
                    if let dataString = String(data: data, encoding: .utf8), !dataString.isEmpty {
                        let defaultVersion = GoProWebcamVersion(version: 1, max_lens_support: nil, usb_3_1_compatible: nil)
                        completion(.success(defaultVersion))
                    } else {
                        completion(.failure(error))
                    }
                }
                
            case .failure(let error):
                completion(.failure(error))
            }
        }
    }
    
    func enterWebcamPreview(completion: @escaping (Result<Void, Error>) -> Void) {
        performGoProRequest(endpoint: .preview) { result in
            completion(result.map { _ in () })
        }
    }
    
    func startWebcam(completion: @escaping (Result<Void, Error>) -> Void) {
        performGoProRequest(endpoint: .start(resolution: 7, fov: 4)) { result in
            completion(result.map { _ in () })
        }
    }
    
    func stopWebcam(completion: @escaping (Result<Void, Error>) -> Void) {
        performGoProRequest(endpoint: .stop) { result in
            completion(result.map { _ in () })
        }
    }
    
    func exitWebcam(completion: @escaping (Result<Void, Error>) -> Void) {
        performGoProRequest(endpoint: .exit) { result in
            completion(result.map { _ in () })
        }
    }
    
    func gracefulWebcamExit(completion: @escaping (Result<Void, Error>) -> Void) {
        stopRTSPStream()
        
        stopWebcam { [weak self] stopResult in
            guard let self = self else { return }
            
                         switch stopResult {
             case .success:
                 self.exitWebcam(completion: completion)
             case .failure(let error):
                self.exitWebcam { exitResult in
                    switch exitResult {
                    case .success:
                        completion(.failure(error))
                    case .failure(let exitError):
                        let combinedError = NSError(domain: "GoProSource", code: 3, 
                            userInfo: [NSLocalizedDescriptionKey: "Multiple errors: Stop - \(error.localizedDescription), Exit - \(exitError.localizedDescription)"])
                        completion(.failure(combinedError))
                    }
                }
            }
        }
    }
}

// MARK: - Backward Compatibility

extension VideoCaptureDelegate {
    var viewForDrawing: UIView? {
        return self as? UIView
    }
}

