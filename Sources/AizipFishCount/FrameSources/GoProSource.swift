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

// MARK: - Network Configuration

/// Network configuration constants for GoPro connectivity
private struct GoProNetworkConfig {
    static let goProIP = "10.5.5.9"
    static let httpPort = 8080
    static let rtspPort = 554
    static let rtspPath = "/live"
    
    static var rtspURL: String {
        "rtsp://\(goProIP):\(rtspPort)\(rtspPath)"
    }
}

// MARK: - Data Structures & Protocols

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

/// High-performance GoPro camera frame source implementation using VLC and pure event-driven processing
/// 
/// This class provides a complete frame source implementation for GoPro cameras connected via WiFi.
/// It uses a pure event-driven architecture:
/// - **Rendering Pipeline**: VLC Media Player for smooth RTSP video playback (25-30 FPS)
/// - **Detection Pipeline**: Event-driven frame extraction triggered by inference completion (20+ FPS)
/// 
/// Key Features:
/// - Pure event-driven processing eliminates gaps and timer overhead
/// - Optimized frame extraction with UIGraphicsImageRenderer caching
/// - Smart calibration handling through the same event-driven pipeline
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
    
    // MARK: - Core Components
    
    // VLC Streaming Components
    /// VLC media player instance for RTSP streaming
    private var videoPlayer: VLCMediaPlayer?
    /// UI view where VLC renders video content
    var playerView: UIView?
    /// Container view that holds the player view (typically YOLOView)
    private weak var containerView: UIView?
    
    // MARK: - State Management
    
    /// Resets processing state to allow normal inference to resume after calibration
    @MainActor
    func resetProcessingState() {
        isModelProcessing = false
        print("GoPro: Processing state reset - ready for normal inference")
    }
    
    // Stream State
    /// Flag indicating if first valid frame has been received from stream
    private var hasReceivedFirstFrame = false
    /// Size of the last received video frame for coordinate transformation
    private var lastFrameSize: CGSize = .zero
    /// Completion callback for stream ready notification
    private var streamReadyCompletion: ((Result<Void, Error>) -> Void)?
    /// Backup timer to ensure stream ready callback is called
    private var streamReadyTimer: Timer?
    
    // Frame Processing State
    /// Counter for successful frame extractions for performance monitoring
    private var frameExtractionCount = 0
    /// Timestamps of recent frame extractions for FPS calculation
    private var frameExtractionTimestamps: [CFTimeInterval] = []
    /// Flag to prevent new frame extractions during model processing
    private var isModelProcessing: Bool = false
    /// Last trigger source for logging (completion vs initial)
    private var lastTriggerSource: String = "unknown"
    /// Flag to track if processing pipeline is active
    private var isPipelineActive: Bool = false
    
    // Performance Optimization State
    /// Cached UIGraphicsImageRenderer to avoid recreation overhead
    private var cachedRenderer: UIGraphicsImageRenderer?
    /// Bounds of last cached renderer to detect when recreation is needed
    private var lastRendererBounds: CGRect = .zero
    
    // MARK: - Performance Metrics
    
    // Pipeline Timing Metrics
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
    
    // FPS Calculation Arrays
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
    /// Initializes state and starts pure event-driven frame processing pipeline.
    /// The pipeline operates by extracting frames immediately when inference completes,
    /// creating a tight feedback loop for maximum throughput without timer overhead.
    nonisolated func start() {
        Task { @MainActor in
            hasReceivedFirstFrame = false
            frameExtractionCount = 0
            frameTimestamps.removeAll()
            isPipelineActive = false
            isModelProcessing = false
            playerView?.isHidden = false
            containerView?.setNeedsLayout()
            
            if videoPlayer?.state != .playing {
                startRTSPStream { _ in }
            }
            startEventDrivenProcessing()
        }
    }
    
    /// Stops frame acquisition and cleans up resources
    /// 
    /// Stops the event-driven processing pipeline, VLC player, and resets all state.
    /// Notifies delegate of stopped status.
    nonisolated func stop() {
        Task { @MainActor in
            stopEventDrivenProcessing()
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
    
    // MARK: - VLC Stream Management
    
    /// Sets up VLC media player and associated UI view for video rendering
    /// 
    /// Creates and configures a VLC media player instance with optimized settings for RTSP streaming.
    /// Also creates the player view with proper constraints for event-driven frame extraction.
    /// Disables VLC logging for better performance.
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
        playerView?.frame = CGRect(x: 0, y: 0, width: 1280, height: 720)
        
        videoPlayer?.drawable = playerView
        videoPlayer?.videoAspectRatio = UnsafeMutablePointer<Int8>(mutating: "16:9")
        videoPlayer?.audio?.volume = 0
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
    
    /// Starts the pure event-driven processing pipeline
    /// 
    /// Initiates frame processing by triggering the first extraction.
    /// From that point forward, each completed inference will immediately trigger
    /// the next frame extraction, creating a tight feedback loop.
    @MainActor
    private func startEventDrivenProcessing() {
        guard !isPipelineActive else { return }
        
        frameExtractionCount = 0
        frameExtractionTimestamps.removeAll()
        isPipelineActive = true
        isModelProcessing = false
        
        // Trigger initial frame processing to start the pipeline
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            self?.triggerNextFrameProcessing(source: "initial")
        }
    }
    
    /// Stops the event-driven processing pipeline
    @MainActor
    private func stopEventDrivenProcessing() {
        isPipelineActive = false
        isModelProcessing = false
    }
    
    /// Triggers the next frame processing cycle
    /// 
    /// This is the core of the event-driven pipeline. It extracts a frame,
    /// processes it through the complete pipeline, and sets up the next cycle
    /// to be triggered when inference completes.
    /// 
    /// - Parameter source: Source of the trigger ("initial", "completion")
    @MainActor
    private func triggerNextFrameProcessing(source: String) {
        guard isPipelineActive,
              let videoPlayer = videoPlayer,
              [.playing, .buffering].contains(videoPlayer.state),
              let playerView = playerView,
              playerView.bounds.size.width > 0,
              videoCaptureDelegate != nil,
              !isModelProcessing else { return }
        
        if processCurrentFrame(allowInference: true) {
            lastTriggerSource = source
            frameExtractionCount += 1
            frameExtractionTimestamps.append(CACurrentMediaTime())
            
            if frameExtractionTimestamps.count > 60 {
                frameExtractionTimestamps.removeFirst()
            }
            
            if frameExtractionCount % 300 == 0 {
                logPipelineAnalysisAtExtraction()
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
    /// - Forces 1x scale to avoid meaningless upsampling from device scale factor
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
        
        // Use cached renderer for performance with 1x scale
        let currentBounds = playerView.bounds
        if cachedRenderer == nil || lastRendererBounds != currentBounds {
            let format = UIGraphicsImageRendererFormat()
            format.scale = 1.0  // Force 1x scale to avoid upsampling
            cachedRenderer = UIGraphicsImageRenderer(bounds: currentBounds, format: format)
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
    /// 1. **Frame Extraction** (~14-20ms): Captures current video frame as UIImage
    /// 2. **Display Update**: Sends frame to delegate for immediate UI display
    /// 3. **Calibration Check**: Routes to calibration if in auto-calibration mode
    /// 4. **Format Conversion** (~0.2ms): Converts UIImage to CMSampleBuffer for inference
    /// 5. **Model Inference** (~24ms): Dispatches to background for YOLO processing
    /// 
    /// Performance: Total pipeline ~38-45ms (22+ FPS actual throughput with event-driven approach)
    /// 
    /// - Parameter allowInference: Whether to run inference (false when model is processing)
    /// - Returns: True if frame was successfully processed, false if skipped
    @MainActor
    private func processCurrentFrame(allowInference: Bool = true) -> Bool {
        guard let videoPlayer = videoPlayer,
              [.playing, .buffering].contains(videoPlayer.state),
              let playerView = playerView,
              playerView.bounds.size.width > 0 else { return false }
        
        let pipelineStartTime = CACurrentMediaTime()
        pipelineStartTimes.append(pipelineStartTime)
        if pipelineStartTimes.count > 30 {
            pipelineStartTimes.removeFirst()
        }
        
        // Extract current frame
        let extractionStartTime = CACurrentMediaTime()
        guard let frameImage = extractCurrentFrame() else { return false }
        lastExtractionTime = (CACurrentMediaTime() - extractionStartTime) * 1000
        
        // Send frame to delegate for display
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.delegate?.frameSource(self, didOutputImage: frameImage)
        }
        
        // Handle calibration mode - CRITICAL FIX: process calibration frames through event-driven pipeline
        let shouldProcessForCalibration = !inferenceOK && predictor is TrackingDetector
        
        if shouldProcessForCalibration {
            return processCalibrationFrame(frameImage)
        }
        
        // Regular inference - only if allowed and ready
        if allowInference && inferenceOK && !isModelProcessing, let predictor = self.predictor {
            return processInferenceFrame(frameImage, predictor: predictor)
        }
        
        return true
    }
    
    /// Processes frame for calibration mode
    /// 
    /// Handles frame processing during auto-calibration mode by converting the frame
    /// to CVPixelBuffer format and passing it to the tracking detector for calibration
    /// analysis. Clears detection boxes on the first calibration frame.
    /// 
    /// CRITICAL: This method triggers the next frame processing cycle to ensure
    /// continuous calibration frame processing in the event-driven pipeline.
    /// 
    /// - Parameter frameImage: The extracted frame to process for calibration
    /// - Returns: True to indicate frame was processed (for calibration)
    @MainActor
    private func processCalibrationFrame(_ frameImage: UIImage) -> Bool {
        guard let trackingDetector = predictor as? TrackingDetector else {
            print("GoPro: Cannot process for calibration - predictor is not TrackingDetector")
            return false
        }
        
        if trackingDetector.getCalibrationFrameCount() == 0 {
            print("GoPro: Starting calibration, clearing boxes")
            DispatchQueue.main.async { [weak self] in
                self?.videoCaptureDelegate?.onClearBoxes()
            }
        }
        
        let conversionStartTime = CACurrentMediaTime()
        guard let pixelBuffer = createStandardPixelBuffer(from: frameImage, forSourceType: sourceType) else {
            return false
        }
        lastConversionTime = (CACurrentMediaTime() - conversionStartTime) * 1000
        
        // Process frame for calibration
        trackingDetector.processFrame(pixelBuffer)
        
        let frameCount = trackingDetector.getCalibrationFrameCount()
        if frameCount > 0 {
            let progress = min(Double(frameCount) / 300.0, 1.0)
            print("GoPro: Calibration progress: \(Int(progress * 100))% (frame \(frameCount))")
        }
        
        // CRITICAL FIX: Trigger next frame processing to continue calibration
        // Use a small delay to prevent excessive CPU usage during calibration
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) { [weak self] in
            self?.triggerNextFrameProcessing(source: "calibration")
        }
        
        return true
    }
    
    /// Processes frame for inference
    /// 
    /// Handles regular model inference by converting the frame to CMSampleBuffer format
    /// and dispatching the prediction to a background queue. Sets processing flag to
    /// prevent concurrent inference operations.
    /// 
    /// - Parameters:
    ///   - frameImage: The extracted frame to process for inference
    ///   - predictor: The FrameProcessor (YOLO model) to run inference
    /// - Returns: True indicating inference was dispatched successfully
    @MainActor
    private func processInferenceFrame(_ frameImage: UIImage, predictor: FrameProcessor) -> Bool {
        let conversionStartTime = CACurrentMediaTime()
        guard let sampleBuffer = preProcessExtractedFrame(frameImage) else { return true }
        lastConversionTime = (CACurrentMediaTime() - conversionStartTime) * 1000
        
        if videoCaptureDelegate == nil && frameExtractionCount % 300 == 0 {
            print("GoPro: Warning - videoCaptureDelegate is nil, detection results will be lost")
        }
        
        isModelProcessing = true
        
        DispatchQueue.global(qos: .userInteractive).async { [weak self] in
            guard let self = self else { return }
            predictor.predict(
                sampleBuffer: sampleBuffer,
                onResultsListener: self,
                onInferenceTime: self
            )
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
    
    /// Starts RTSP streaming from GoPro with proper cleanup and retry logic
    /// 
    /// Ensures any existing stream is properly stopped before starting a new one.
    /// Validates integration with container view and sets up backup timer for reliability.
    /// 
    /// - Parameter completion: Callback with stream startup result
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
        
        let rtspURLString = GoProNetworkConfig.rtspURL
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
    
    /// Validates that GoProSource is properly integrated with container view and delegates
    /// 
    /// Checks all required components are set up: player view in container,
    /// delegates configured, predictor available, and valid view bounds.
    /// 
    /// - Returns: True if all integration requirements are met
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
    
    /// Stops RTSP streaming and cleans up all stream-related state
    /// 
    /// Stops the VLC player, cancels completion callbacks, invalidates timers,
    /// and notifies delegate of stopped status.
    @MainActor
    func stopRTSPStream() {
        videoPlayer?.stop()
        streamReadyCompletion = nil
        streamReadyTimer?.invalidate()
        streamReadyTimer = nil
        goProDelegate?.goProSource(self, didUpdateStatus: .stopped)
    }
    
    // MARK: - Utility Methods
    
    /// Resets all internal state variables to initial values
    /// 
    /// Comprehensive state reset method called during setup and stop operations.
    /// Clears all timing arrays, performance metrics, cached objects, and state flags
    /// to ensure clean initialization for new streaming sessions.
    private func resetState() {
        hasReceivedFirstFrame = false
        frameExtractionCount = 0
        isModelProcessing = false
        isPipelineActive = false
        lastTriggerSource = "unknown"
        cachedRenderer = nil
        lastRendererBounds = .zero
        streamReadyCompletion = nil
        streamReadyTimer?.invalidate()
        streamReadyTimer = nil
        frameTimestamps.removeAll()
        frameExtractionTimestamps.removeAll()
        pipelineStartTimes.removeAll()
        pipelineCompleteTimes.removeAll()
        lastExtractionTime = 0
        lastConversionTime = 0
        lastInferenceTime = 0
        lastUITime = 0
        lastTotalPipelineTime = 0
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
    
    // MARK: - Performance Calculations
    
    /// Calculates current frame completion rate based on recent timestamps
    private func calculateCurrentFPS() -> Double {
        guard frameTimestamps.count >= 2 else { return 0 }
        let recentTimestamps = Array(frameTimestamps.suffix(5))
        let timeInterval = recentTimestamps.last! - recentTimestamps.first!
        return timeInterval > 0 ? Double(recentTimestamps.count - 1) / timeInterval : 0
    }
    
    /// Calculates actual pipeline throughput FPS based on complete processing cycles
    @MainActor
    private func calculateThroughputFPS() -> Double {
        guard pipelineCompleteTimes.count >= 2 else { return 0 }
        let cyclesToConsider = min(10, pipelineCompleteTimes.count)
        let recentCompletions = Array(pipelineCompleteTimes.suffix(cyclesToConsider))
        let timeInterval = recentCompletions.last! - recentCompletions.first!
        return timeInterval > 0 ? Double(cyclesToConsider - 1) / timeInterval : 0
    }
    
    // MARK: - Performance Logging
    
    /// Logs comprehensive pipeline analysis at extraction level to ensure every 300th frame is logged
    @MainActor
    private func logPipelineAnalysisAtExtraction() {
        let frameSizeStr = lastFrameSize.width > 0 ? 
            "\(String(format: "%.0f", lastFrameSize.width))×\(String(format: "%.0f", lastFrameSize.height))" : "Unknown"
        
        // Calculate extraction FPS from recent timestamps
        let extractionFPS: Double
        if frameExtractionTimestamps.count >= 2 {
            let windowSeconds = frameExtractionTimestamps.last! - frameExtractionTimestamps.first!
            extractionFPS = Double(frameExtractionTimestamps.count - 1) / windowSeconds
        } else {
            extractionFPS = 0
        }
        
        // Calculate actual throughput FPS
        let actualThroughputFPS = calculateThroughputFPS()
        
        // Show whether this is due to extraction only or if we have inference data
        let hasInferenceData = lastInferenceTime > 0 && lastTotalPipelineTime > 0
        
        print("GoPro: === GOPRO SOURCE PIPELINE ANALYSIS (Frame #\(frameExtractionCount)) ===")
        print("GoPro: Frame Size: \(frameSizeStr) | Extraction FPS: \(String(format: "%.1f", extractionFPS)) | Triggered by: \(lastTriggerSource)")
        
        if hasInferenceData {
            // Full pipeline data available
            print("GoPro: Frame Extraction: \(String(format: "%.1f", lastExtractionTime))ms + Conversion: \(String(format: "%.1f", lastConversionTime))ms = \(String(format: "%.1f", lastExtractionTime + lastConversionTime))ms")
            print("GoPro: Model Inference: \(String(format: "%.1f", lastInferenceTime))ms (includes fish counting inside TrackingDetector)")
            print("GoPro: UI Delegate Call: \(String(format: "%.1f", lastUITime))ms | Total Pipeline: \(String(format: "%.1f", lastTotalPipelineTime))ms")
            
            let theoreticalFPS = lastTotalPipelineTime > 0 ? 1000.0 / lastTotalPipelineTime : 0
            print("GoPro: Theoretical FPS: \(String(format: "%.1f", theoreticalFPS)) | Actual Throughput: \(String(format: "%.1f", actualThroughputFPS))")
            
            // Calculate breakdown percentages
            let extractionPct = (lastExtractionTime / lastTotalPipelineTime) * 100
            let conversionPct = (lastConversionTime / lastTotalPipelineTime) * 100
            let inferencePct = (lastInferenceTime / lastTotalPipelineTime) * 100
            let uiPct = (lastUITime / lastTotalPipelineTime) * 100
            
            print("GoPro: Breakdown - Extraction: \(String(format: "%.1f", extractionPct))% | Conversion: \(String(format: "%.1f", conversionPct))% | Inference+FishCount: \(String(format: "%.1f", inferencePct))% | UI: \(String(format: "%.1f", uiPct))%")
            // print("GoPro: Event-Driven Pipeline - Zero gaps between completion and next extraction")
        } else {
            // Only extraction data available (calibration, skipped inference, etc.)
            print("GoPro: Frame Extraction: \(String(format: "%.1f", lastExtractionTime))ms + Conversion: \(String(format: "%.1f", lastConversionTime))ms = \(String(format: "%.1f", lastExtractionTime + lastConversionTime))ms")
            print("GoPro: Model Inference: SKIPPED (calibration mode or processing load)")
            print("GoPro: Event-Driven Extraction-only - Actual Throughput: \(String(format: "%.1f", actualThroughputFPS))")
        }
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

// MARK: - VLC Media Player Delegate

extension GoProSource {
    nonisolated func mediaPlayerStateChanged(_ aNotification: Notification!) {
        guard let player = aNotification.object as? VLCMediaPlayer else { return }
        let playerState = player.state
        let playerVideoSize = player.videoSize
        
        Task { @MainActor in
            switch playerState {
            case .playing:
                self.goProDelegate?.goProSource(self, didUpdateStatus: .playing)
                self.startEventDrivenProcessing()
                
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
                self.stopEventDrivenProcessing()
                self.goProDelegate?.goProSource(self, didUpdateStatus: .error("Playback error"))
                
                if let completion = self.streamReadyCompletion {
                    self.streamReadyCompletion = nil
                    self.streamReadyTimer?.invalidate()
                    self.streamReadyTimer = nil
                    completion(.failure(NSError(domain: "GoProSource", code: 2, 
                        userInfo: [NSLocalizedDescriptionKey: "VLC playback error"])))
                }
                
            case .ended, .stopped:
                self.stopEventDrivenProcessing()
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

// MARK: - Model Results Handling

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
            self.handleInferenceCompletion(result: result)
        }
    }
    
    /// Handles inference completion and UI updates
    @MainActor
    private func handleInferenceCompletion(result: YOLOResult) {
        // Clear processing flag and trigger next frame extraction
        isModelProcessing = false
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.005) { [weak self] in
            self?.triggerNextFrameProcessing(source: "completion")
        }
        
        let postProcessingStartTime = CACurrentMediaTime()
        let timestamp = CACurrentMediaTime()
        
        // Update frame timestamps
        frameTimestamps.append(timestamp)
        if frameTimestamps.count > 30 {
            frameTimestamps.removeFirst()
        }
        
        // Forward results to delegate and update UI
        let uiDelegateStartTime = CACurrentMediaTime()
        if let videoCaptureDelegate = videoCaptureDelegate {
            videoCaptureDelegate.onPredict(result: result)
            updateContainerUI()
        } else if frameExtractionCount % 300 == 0 {
            print("GoPro: WARNING - videoCaptureDelegate is nil for prediction result")
        }
        
        // Calculate timing metrics
        lastUITime = (CACurrentMediaTime() - uiDelegateStartTime) * 1000
        let totalUITime = (CACurrentMediaTime() - postProcessingStartTime) * 1000
        lastTotalPipelineTime = lastExtractionTime + lastConversionTime + lastInferenceTime + totalUITime
        
        // Update completion timestamps
        pipelineCompleteTimes.append(timestamp)
        if pipelineCompleteTimes.count > 30 {
            pipelineCompleteTimes.removeFirst()
        }
        
        // Update delegate with throughput FPS
        let actualThroughputFPS = calculateThroughputFPS()
        delegate?.frameSource(self, didUpdateWithSpeed: result.speed, fps: actualThroughputFPS)
    }
    
    /// Updates container UI for responsive box rendering
    @MainActor
    private func updateContainerUI() {
        guard let containerView = containerView else { return }
        
        DispatchQueue.main.async {
            containerView.setNeedsDisplay()
            
            if let yoloView = containerView as? YOLOView {
                yoloView.setNeedsLayout()
                yoloView.layoutIfNeeded()
                
                for box in yoloView.boundingBoxViews {
                    if !box.shapeLayer.isHidden {
                        box.shapeLayer.zPosition = 1000
                        box.textLayer.zPosition = 1001
                    }
                }
            }
        }
    }
}

// MARK: - UI Integration and Coordinate Transformation

extension GoProSource {
    @MainActor
    func transformDetectionToScreenCoordinates(rect: CGRect, viewBounds: CGRect, orientation: UIDeviceOrientation) -> CGRect {
        // Convert to unified coordinate system first
        let unifiedRect = toUnifiedCoordinates(rect)
        
        // Convert from unified to screen coordinates
        return UnifiedCoordinateSystem.toScreen(unifiedRect, screenBounds: viewBounds)
    }
    
    /// Converts GoPro detection coordinates to unified coordinate system
    /// - Parameter rect: Detection rectangle from GoPro (normalized Vision coordinates)
    /// - Returns: Rectangle in unified coordinate system
    @MainActor
    func toUnifiedCoordinates(_ rect: CGRect) -> UnifiedCoordinateSystem.UnifiedRect {
        // GoPro detections come from Vision framework, so convert from Vision coordinates
        let visionRect = rect
        
        // Convert from Vision (bottom-left origin) to unified (top-left origin)
        let unifiedFromVision = UnifiedCoordinateSystem.fromVision(visionRect)
        
        // Apply GoPro-specific adjustments for webcam stream
        return UnifiedCoordinateSystem.fromGoPro(
            unifiedFromVision.cgRect,
            streamSize: lastFrameSize,
            displayBounds: containerView?.bounds ?? CGRect.zero
        )
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
        urlComponents.host = GoProNetworkConfig.goProIP
        urlComponents.port = GoProNetworkConfig.httpPort
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


