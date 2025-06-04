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

/// GoPro API endpoints for structured HTTP requests
private enum GoProEndpoint {
    case version
    case preview
    case start(resolution: Int, fov: Int)
    case stop
    case exit
    
    var path: String {
        switch self {
        case .version: return "/gopro/webcam/version"
        case .preview: return "/gopro/webcam/preview"
        case .start: return "/gopro/webcam/start"
        case .stop: return "/gopro/webcam/stop"
        case .exit: return "/gopro/webcam/exit"
        }
    }
    
    var queryItems: [URLQueryItem]? {
        switch self {
        case .start(let res, let fov):
            return [
                URLQueryItem(name: "res", value: "\(res)"),
                URLQueryItem(name: "fov", value: "\(fov)"),
                URLQueryItem(name: "protocol", value: "RTSP")
            ]
        default:
            return nil
        }
    }
}

// MARK: - Main Class

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
    var longSide: CGFloat = 1280  // Default HD resolution
    
    /// The short side dimension of the frames produced by this source.
    var shortSide: CGFloat = 720  // Default HD resolution
    
    /// Flag indicating if inference should be performed on frames
    var inferenceOK: Bool = true
    
    /// The source type identifier.
    var sourceType: FrameSourceType { return .goPro }
    
    /// Additional delegate for video capture (for compatibility with YOLOView)
    weak var videoCaptureDelegate: VideoCaptureDelegate?
    
    // MARK: - GoProSource Properties
    
    // Network configuration
    private let goProIP = "10.5.5.9"
    private let goProPort = 8080
    private let rtspPort = 554
    private let rtspPath = "/live"
    
    // VLC components
    private var videoPlayer: VLCMediaPlayer?
    var playerView: UIView?
    private weak var containerView: UIView?
    
    // Stream tracking and completion callbacks
    private var streamStartTime: Date?
    private var hasReceivedFirstFrame = false
    private var frameCount = 0
    private var lastFrameTime: Int64 = 0
    private var lastFrameSize: CGSize = .zero
    
    // Stream ready callback - FIXED: Store completion callback for proper flow
    private var streamReadyCompletion: ((Result<Void, Error>) -> Void)?
    
    // Performance metrics
    private var frameTimestamps: [CFTimeInterval] = []
    
    // Frame rate measurement with CADisplayLink
    private var displayLink: CADisplayLink?
    private var frameExtractionCount = 0
    private var frameExtractionTimestamps: [CFTimeInterval] = []
    private var lastSuccessfulExtractionTime: CFTimeInterval = 0
    
    // Delegates
    weak var goProDelegate: GoProSourceDelegate?
    
    // Add timing tracking properties
    private var lastExtractionTime: Double = 0
    private var lastConversionTime: Double = 0
    private var lastInferenceTime: Double = 0
    private var lastUITime: Double = 0  // UI delegate call time, NOT fish counting time
    private var lastTotalPipelineTime: Double = 0
    
    // Add frame pipeline timing for accurate FPS calculation
    private var pipelineStartTimes: [CFTimeInterval] = []
    private var pipelineCompleteTimes: [CFTimeInterval] = []
    
    // MARK: - Performance Optimizations
    
    // Cache the renderer to avoid creating it every frame
    private var cachedRenderer: UIGraphicsImageRenderer?
    private var lastRendererBounds: CGRect = .zero
    
    // Skip frame extraction during model processing flag
    private var isModelProcessing: Bool = false
    
    // MARK: - Initialization
    
    override init() {
        super.init()
        setupVLCPlayerAndView()
        
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
        // Note: displayLink will be automatically invalidated when the object is deallocated
    }
    
    // MARK: - Core FrameSource Protocol Implementation
    
    /// Sets up the frame source with specified configuration.
    func setUp(completion: @escaping @Sendable (Bool) -> Void) {
        // Reset any previous state
        hasReceivedFirstFrame = false
        frameCount = 0
        
        // OPTIMIZATION: Reset processing state and cached renderer
        isModelProcessing = false
        cachedRenderer = nil
        lastRendererBounds = .zero
        
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
            
            // Reset timing tracking
            self.lastExtractionTime = 0
            self.lastConversionTime = 0
            self.lastInferenceTime = 0
            self.lastUITime = 0
            self.lastTotalPipelineTime = 0
            
            // Reset pipeline timing arrays
            self.pipelineStartTimes.removeAll()
            self.pipelineCompleteTimes.removeAll()
            
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
            
            // Start frame rate measurement
            self.startFrameRateMeasurement()
        }
    }
    
    /// Stops frame acquisition from the source.
    nonisolated func stop() {
        Task { @MainActor [weak self] in
            guard let self = self else { return }
            print("GoPro: Stopping stream and cleaning up resources")
            
            // Stop frame rate measurement
            self.stopFrameRateMeasurement()
            
            // Stop the VLC player
                    self.videoPlayer?.stop()
                    
            // Clean up player view from the container
            self.cleanupPlayerView()
            
            // Reset all state variables
            self.hasReceivedFirstFrame = false
            self.frameCount = 0
            self.frameTimestamps.removeAll()
            self.streamStartTime = nil
            self.lastFrameSize = .zero
            
            // Reset timing tracking
            self.lastExtractionTime = 0
            self.lastConversionTime = 0
            self.lastInferenceTime = 0
            self.lastUITime = 0
            self.lastTotalPipelineTime = 0
            
            // Reset pipeline timing arrays
            self.pipelineStartTimes.removeAll()
            self.pipelineCompleteTimes.removeAll()
            
            // OPTIMIZATION: Reset processing flag and clear cached renderer
            self.isModelProcessing = false
            self.cachedRenderer = nil
            self.lastRendererBounds = .zero
            
            // Clear completion callback
            self.streamReadyCompletion = nil
            
            // Notify delegate
                self.goProDelegate?.goProSource(self, didUpdateStatus: .stopped)
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
        // Ensure this runs on main thread for UI updates
        Task { @MainActor in
            self.updatePlayerViewLayout()
        }
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

    // MARK: - VLC Player Management
    
    @MainActor
    private func setupVLCPlayerAndView() {
        // 1. Create and configure VLC player
        videoPlayer = VLCMediaPlayer()
        videoPlayer?.delegate = self
        
        // Disable internal logging to improve performance
        videoPlayer?.libraryInstance.debugLogging = false
        videoPlayer?.libraryInstance.debugLoggingLevel = 0
        
        // 2. Create player view with optimized settings
        playerView = UIView()
        playerView?.backgroundColor = .black
        playerView?.translatesAutoresizingMaskIntoConstraints = false
        playerView?.tag = 9876  // For easier identification
        
        // Configure layer for bounding box display
        playerView?.layer.shouldRasterize = false
        playerView?.layer.drawsAsynchronously = false
        playerView?.layer.masksToBounds = false
        playerView?.clipsToBounds = false
        playerView?.layer.name = "goProPlayerLayer"
        
        // Set default frame size
        playerView?.frame = CGRect(x: 0, y: 0, width: 1920, height: 1080)
        
        // 3. Connect player to view
        videoPlayer?.drawable = playerView
        
        // 4. Configure player for optimal performance
        videoPlayer?.videoAspectRatio = UnsafeMutablePointer<Int8>(mutating: "16:9")
        videoPlayer?.videoCropGeometry = UnsafeMutablePointer<Int8>(mutating: "")
        videoPlayer?.audio?.volume = 0  // Mute audio
        
        // 5. Setup CADisplayLink for frame processing
        setupDisplayLinkForFrameRateMeasurement()
        
        print("GoPro: VLC player and view initialized with CADisplayLink frame processing")
    }
    
    /// Configure media options for optimal RTSP streaming from GoPro
    private func configureRTSPStream(_ media: VLCMedia) {
        // Critical options for performance and reliability
        let criticalOptions = [
            ":verbose=0",                           // Disable verbose logging
            ":rtsp-tcp",                           // Force TCP for reliability
            ":network-caching=0",                  // Low latency
            ":live-caching=0",                     // Low latency for live streams
            ":file-caching=0",                     // No file caching
            ":avcodec-hw=any",                     // Hardware acceleration
            ":rtsp-frame-buffer-size=4000000",     // Larger buffer for quality
            ":avcodec-threads=4",                  // Fixed thread count
            ":clock-jitter=0",                     // Disable clock jitter
            ":clock-synchro=0",                    // Disable clock synchro
            ":no-sub-autodetect-file",             // Disable subtitles
            ":no-audio"                            // Disable audio processing
        ]
        
        // Apply options efficiently
        for option in criticalOptions {
            media.addOption(option)
        }
        
        // Clear cookies for fresh connection
        media.clearStoredCookies()
        
        print("GoPro: Media configured for optimal RTSP streaming")
    }
    
    /// Clean up player view from container
    @MainActor
    private func cleanupPlayerView() {
        guard let playerView = playerView else { return }
        
        // Make the view invisible to prevent flicker
        playerView.isHidden = true
        
        // Deactivate all constraints referencing the player view
        if let superview = playerView.superview {
            let constraintsToDeactivate = superview.constraints.filter { constraint in
                return (constraint.firstItem === playerView || constraint.secondItem === playerView)
            }
            
            if !constraintsToDeactivate.isEmpty {
                NSLayoutConstraint.deactivate(constraintsToDeactivate)
            }
        }
        
        // Deactivate player view's own constraints
        NSLayoutConstraint.deactivate(playerView.constraints)
        
        // Remove from superview
        playerView.removeFromSuperview()
        
        // Break circular references
        self.containerView = nil
        
        print("GoPro: Player view cleaned up")
    }
    
    // MARK: - Frame Rate Measurement with CADisplayLink
    
    @MainActor
    private func setupDisplayLinkForFrameRateMeasurement() {
        // Create display link that fires at 30Hz for better UI responsiveness
        displayLink = CADisplayLink(target: self, selector: #selector(displayLinkCallback))
        displayLink?.preferredFramesPerSecond = 30  // Reduced to 30 FPS for better UI responsiveness
        displayLink?.isPaused = true  // Start paused
        displayLink?.add(to: .main, forMode: .default)
        
        print("GoPro: CADisplayLink setup for frame processing pipeline at 30 FPS")
    }
    
    @objc private func displayLinkCallback() {
        // This fires at 30Hz - ensure this method is as lightweight as possible
        
        let currentTime = CACurrentMediaTime()
        
        // Only process if VLC is playing AND delegate is properly set up AND model is not processing
        guard let videoPlayer = videoPlayer,
              [.playing, .buffering].contains(videoPlayer.state),
              let playerView = playerView,
              playerView.bounds.size.width > 0,
              playerView.bounds.size.height > 0,
              // CRITICAL: Wait until videoCaptureDelegate is properly set up
              videoCaptureDelegate != nil,
              // OPTIMIZATION: Skip if model is still processing previous frame
              !isModelProcessing else {
            return
        }
        
        // Dispatch frame processing to avoid blocking UI
        Task { @MainActor in
            let processingStartTime = CACurrentMediaTime()
            let didProcessFrame = self.processCurrentFrame()
            let processingTime = (CACurrentMediaTime() - processingStartTime) * 1000 // ms
            
            // Only increment counters if we actually processed a frame
            if didProcessFrame {
                self.frameExtractionCount += 1
                self.frameExtractionTimestamps.append(currentTime)
                self.lastSuccessfulExtractionTime = currentTime
                
                // Keep only last 60 successful processing attempts for rolling average (increased for better accuracy)
                while self.frameExtractionTimestamps.count > 60 {
                    self.frameExtractionTimestamps.removeFirst()
                }
                
                // Log processing rate every 300 attempts (10 seconds at 30Hz)
                if self.frameExtractionCount % 300 == 0 {
                    let windowSeconds = self.frameExtractionTimestamps.count >= 2 ? 
                        self.frameExtractionTimestamps.last! - self.frameExtractionTimestamps.first! : 1.0
                    
                    // Calculate actual successful processing FPS from timestamps
                    let successfulProcessingFPS = self.frameExtractionTimestamps.count >= 2 ? 
                        Double(self.frameExtractionTimestamps.count - 1) / windowSeconds : 0
                    
                    // Calculate CADisplayLink attempt rate (should be ~30/second)
                    let attemptWindow = 300.0 / 30.0 // 10 seconds
                    let attemptRate = 300.0 / attemptWindow
                    
                    // Get current frame size for reporting
                    let currentFrameSize = self.lastFrameSize
                    let frameSizeStr = currentFrameSize.width > 0 ? 
                        "\(String(format: "%.0f", currentFrameSize.width))×\(String(format: "%.0f", currentFrameSize.height))" : "Unknown"
                    
                    print("GoPro: CADisplayLink Performance (Frame Size: \(frameSizeStr)) - Attempts: \(String(format: "%.1f", attemptRate))/s over \(String(format: "%.1f", attemptWindow))s, Successful Processing FPS: \(String(format: "%.1f", successfulProcessingFPS)), Average CADisplayLink Processing: \(String(format: "%.1f", processingTime))ms")
                }
            }
        }
    }
    
    @MainActor
    private func startFrameRateMeasurement() {
        frameExtractionCount = 0
        frameExtractionTimestamps.removeAll()
        displayLink?.isPaused = false
        print("GoPro: Started CADisplayLink frame processing measurement")
    }
    
    @MainActor
    private func stopFrameRateMeasurement() {
        displayLink?.isPaused = true
        print("GoPro: Stopped CADisplayLink frame processing measurement")
    }
    
    // MARK: - Frame Processing Pipeline
    
    @MainActor
    private func extractCurrentFrame() -> UIImage? {
        guard let playerView = playerView,
            playerView.bounds.size.width > 0,
            playerView.bounds.size.height > 0,
            let videoPlayer = videoPlayer,
            [.playing, .paused, .buffering].contains(videoPlayer.state) else {
            return nil
        }
        
        // Hide UI overlays for clean frame extraction
        if let yoloView = containerView as? YOLOView {
            yoloView.hideUIOverlaysForExtraction()
        }
        
        defer {
            // Restore UI overlays after frame extraction
            if let yoloView = containerView as? YOLOView {
                yoloView.restoreUIOverlaysAfterExtraction()
            }
        }
        
        // OPTIMIZATION: Cache the renderer to avoid creating it every frame
        let currentBounds = playerView.bounds
        if cachedRenderer == nil || lastRendererBounds != currentBounds {
            cachedRenderer = UIGraphicsImageRenderer(bounds: currentBounds)
            lastRendererBounds = currentBounds
        }
        
        guard let renderer = cachedRenderer else { return nil }
        
        let snapshot = renderer.image { context in
            playerView.drawHierarchy(in: currentBounds, afterScreenUpdates: false)
        }
        
        // OPTIMIZATION: Only update frame size when it actually changes
        if lastFrameSize != snapshot.size {
            lastFrameSize = snapshot.size
        }
        
        return snapshot
    }
    
    // Process the extracted frame into YOLO-compatible format
    private func preProcessExtractedFrame(_ image: UIImage) -> CMSampleBuffer? {
        // Create pixel buffer from the image using standard conversion
        guard let pixelBuffer = createStandardPixelBuffer(from: image, forSourceType: sourceType) else {
            print("GoPro: Failed to create pixel buffer from frame")
            return nil
        }
        
        // Create sample buffer using standard conversion
        guard let sampleBuffer = createStandardSampleBuffer(from: pixelBuffer) else {
            print("GoPro: Failed to create sample buffer from pixel buffer")
            return nil
        }
        
        return sampleBuffer
    }
    
    // Process the current frame for either inference or calibration
    @MainActor
    private func processCurrentFrame() -> Bool {
        // Early exit if we don't have valid player state
        guard let videoPlayer = videoPlayer,
              [.playing, .buffering].contains(videoPlayer.state),
              let playerView = playerView,
              playerView.bounds.size.width > 0,
              playerView.bounds.size.height > 0 else {
            return false
        }
        
        // Measure total pipeline time from start and record for throughput calculation
        let pipelineStartTime = CACurrentMediaTime()
        pipelineStartTimes.append(pipelineStartTime)
        if pipelineStartTimes.count > 30 {
            pipelineStartTimes.removeFirst()
        }
        
        // Step 1: Frame extraction
        let extractionStartTime = CACurrentMediaTime()
        guard let frameImage = extractCurrentFrame() else {
            // Skip this frame if extraction fails
            return false
        }
        let extractionTime = (CACurrentMediaTime() - extractionStartTime) * 1000 // in ms
        lastExtractionTime = extractionTime
        
        // Always send the frame to the delegate for display - dispatch to avoid blocking
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
        
        // Step 2: Process with predictor if inference is enabled
        if inferenceOK, let predictor = self.predictor {
            // Step 2a: Format conversion
            let conversionStartTime = CACurrentMediaTime()
            if let sampleBuffer = preProcessExtractedFrame(frameImage) {
                let conversionTime = (CACurrentMediaTime() - conversionStartTime) * 1000 // in ms
                lastConversionTime = conversionTime
                
                let preInferenceTime = (CACurrentMediaTime() - pipelineStartTime) * 1000 // in ms
                
                // Log detailed timing breakdown periodically (before inference) - ONLY for every 300th frame to avoid duplicates
                if frameExtractionCount % 300 == 0 {
                    let frameSize = frameImage.size
                    print("GoPro: Frame #\(frameExtractionCount) - Pre-Inference (Size: \(String(format: "%.0f", frameSize.width))×\(String(format: "%.0f", frameSize.height))): Extraction \(String(format: "%.1f", extractionTime))ms + Conversion \(String(format: "%.1f", conversionTime))ms = \(String(format: "%.1f", preInferenceTime))ms total")
                }
                
                // Ensure the delegate is set up correctly
                if self.videoCaptureDelegate == nil {
                    // Only log warning occasionally to avoid spam
                    if frameExtractionCount % 300 == 0 {
                    print("GoPro: Warning - videoCaptureDelegate is nil, detection results will be lost")
                    }
                }
                
                // Step 3: Run inference - dispatch to background to avoid blocking UI
                // Note: Total pipeline time will be calculated when inference completes
                DispatchQueue.global(qos: .userInteractive).async { [weak self] in
                    guard let self = self else { return }
                    
                // OPTIMIZATION: Set processing flag to prevent new frame extractions
                Task { @MainActor in
                    self.isModelProcessing = true
                }
                
                predictor.predict(
                    sampleBuffer: sampleBuffer,
                    onResultsListener: self,
                    onInferenceTime: self
                )
                }
            } else {
                print("GoPro: Failed to create sample buffer from frame image")
            }
        }
        
        return true
    }
    
    /// Calculate current FPS based on recent frame timestamps
    @MainActor
    private func calculateCurrentFPS() -> Double {
        guard frameTimestamps.count >= 2 else {
            return 0
        }
        
        // Calculate FPS based on last 5 frames for 30Hz processing (reduced from 10)
        let framesToConsider = min(5, frameTimestamps.count)
        let recentTimestamps = Array(frameTimestamps.suffix(framesToConsider))
        
        // Calculate time difference between first and last frame
        let timeInterval = recentTimestamps.last! - recentTimestamps.first!
        
        // Calculate frames per second
        return timeInterval > 0 ? Double(framesToConsider - 1) / timeInterval : 0
    }
    
    /// Calculate actual throughput FPS based on complete pipeline processing cycles
    @MainActor
    private func calculateThroughputFPS() -> Double {
        guard pipelineCompleteTimes.count >= 2 else {
            return 0
        }
        
        // Calculate throughput based on last 10 complete pipeline cycles
        let cyclesToConsider = min(10, pipelineCompleteTimes.count)
        let recentCompletions = Array(pipelineCompleteTimes.suffix(cyclesToConsider))
        
        // Calculate time difference between first and last completion
        let timeInterval = recentCompletions.last! - recentCompletions.first!
        
        // Calculate completed pipelines per second (true throughput)
        return timeInterval > 0 ? Double(cyclesToConsider - 1) / timeInterval : 0
    }
    
    // MARK: - RTSP Streaming Management
    
    /// Start RTSP stream from GoPro - FIXED: Proper completion flow with integration validation
    @MainActor
    func startRTSPStream(completion: @escaping (Result<Void, Error>) -> Void) {
        // Stop any existing stream first
        if let videoPlayer = videoPlayer, videoPlayer.state != .stopped {
            print("GoPro: Stopping existing stream before starting new one")
            videoPlayer.stop()
            // Brief delay to ensure cleanup
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
                self?.startRTSPStreamInternal(completion: completion)
            }
            return
        }
        
        startRTSPStreamInternal(completion: completion)
    }
    
    /// Internal method to start RTSP stream after ensuring no conflicts
    @MainActor
    private func startRTSPStreamInternal(completion: @escaping (Result<Void, Error>) -> Void) {
        // Validate integration before starting stream
        guard isProperlyIntegrated() else {
            let error = NSError(domain: "GoProSource", code: 3, userInfo: [NSLocalizedDescriptionKey: "GoProSource not properly integrated with YOLOView yet"])
            completion(.failure(error))
            return
        }

        // Construct RTSP URL - using the known working configuration
        let rtspURLString = "rtsp://\(goProIP):\(rtspPort)\(rtspPath)"
        print("GoPro: Connecting to RTSP stream at \(rtspURLString)")
        
        guard let url = URL(string: rtspURLString), let videoPlayer = videoPlayer else {
            let error = NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid RTSP URL or player not initialized"])
            completion(.failure(error))
            return
        }
        
        // Store completion callback to call when stream is actually ready (when frames are received)
        streamReadyCompletion = completion
        
        // Set stream status to connecting
        goProDelegate?.goProSource(self, didUpdateStatus: .connecting)
        
        // Reset state
        lastExtractionTime = 0
        lastConversionTime = 0
        lastInferenceTime = 0
        lastUITime = 0
        lastTotalPipelineTime = 0
        
        // Reset pipeline timing arrays
        pipelineStartTimes.removeAll()
        pipelineCompleteTimes.removeAll()
        
        // Make sure our player view is properly configured
        if let playerView = playerView {
            playerView.layoutIfNeeded()
            playerView.isHidden = false
            videoPlayer.drawable = playerView
        }
        
        // Configure player with optimized options
        let media = VLCMedia(url: url)
        configureRTSPStream(media)
        videoPlayer.media = media
        
        // Start playback
        videoPlayer.play()
        
        print("GoPro: VLC Player - Opening stream")
        
        // NOTE: Do not call completion(.success(())) immediately
        // Completion will be called when first frame is received in mediaPlayerTimeChanged
    }
    
    /// Check if GoProSource is properly integrated with YOLOView and ready to start streaming
    @MainActor
    private func isProperlyIntegrated() -> Bool {
        // Check if we have all required components
        guard let playerView = playerView,
              let containerView = containerView,
              let videoCaptureDelegate = videoCaptureDelegate,
              let predictor = predictor else {
            print("GoPro: Integration check failed - missing required components")
            return false
        }
        
        // Check if player view is properly added to container
        guard playerView.superview === containerView else {
            print("GoPro: Integration check failed - player view not properly added to container")
            return false
        }
        
        // Check if container has reasonable size
        guard containerView.bounds.size.width > 0 && containerView.bounds.size.height > 0 else {
            print("GoPro: Integration check failed - container view has invalid size")
            return false
        }
        
        print("GoPro: Integration check passed - ready to start streaming")
        return true
    }
    
    /// Stop RTSP stream
    @MainActor
    func stopRTSPStream() {
        print("GoPro: Stopping RTSP stream")
        videoPlayer?.stop()
        streamReadyCompletion = nil  // Clear any pending completion
        goProDelegate?.goProSource(self, didUpdateStatus: .stopped)
    }
    
    // MARK: - Layout Management
    
    @MainActor
    private func updatePlayerViewLayout() {
        guard let playerView = playerView, let containerView = containerView else {
            return
        }
        
        // Safety check for container view size
        let containerSize = containerView.bounds.size
        guard containerSize.width > 0 && containerSize.height > 0,
              playerView.superview === containerView else {
            return
        }
        
        // Force immediate layout update on main thread
        DispatchQueue.main.async {
            containerView.setNeedsLayout()
            containerView.layoutIfNeeded()
            
            // Update player view frame to match container
            playerView.frame = containerView.bounds
            
            print("GoPro: Updated player layout for size: \(containerSize)")
        }
    }
    
    // Orientation change notification handler
    @objc private func orientationDidChange() {
        // Dispatch to main thread to avoid blocking UI
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            let orientation = UIDevice.current.orientation
            if orientation.isPortrait || orientation.isLandscape {
                self.updateForOrientationChange(orientation: orientation)
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
                case .opening:
                    print("GoPro: VLC Player - Opening stream")
                case .buffering:
                    print("GoPro: VLC Player - Buffering stream")
                case .playing:
                    print("GoPro: VLC Player - Stream is playing")
                    self.goProDelegate?.goProSource(self, didUpdateStatus: .playing)
                    
                // Start frame rate measurement when actually playing
                self.startFrameRateMeasurement()
                
                    if playerVideoSize.width > 0 && playerVideoSize.height > 0 {
                        self.lastFrameSize = playerVideoSize
                        self.broadcastFrameSizeChange(playerVideoSize)
                    }
                case .error:
                    print("GoPro: VLC Player - Error streaming")
                self.stopFrameRateMeasurement()
                    self.goProDelegate?.goProSource(self, didUpdateStatus: .error("Playback error"))
                    
                // Call completion with error if we have a pending callback
                if let completion = self.streamReadyCompletion {
                    self.streamReadyCompletion = nil
                    let error = NSError(domain: "GoProSource", code: 2, userInfo: [NSLocalizedDescriptionKey: "VLC playback error"])
                    completion(.failure(error))
                }
                case .ended:
                    print("GoPro: VLC Player - Stream ended")
                self.stopFrameRateMeasurement()
                    self.goProDelegate?.goProSource(self, didUpdateStatus: .stopped)
                case .stopped:
                    print("GoPro: VLC Player - Stream stopped")
                self.stopFrameRateMeasurement()
                    self.goProDelegate?.goProSource(self, didUpdateStatus: .stopped)
                default:
                    print("GoPro: VLC Player - State: \(playerState.rawValue)")
            }
        }
    }
    
    nonisolated func mediaPlayerTimeChanged(_ aNotification: Notification!) {
        guard let player = aNotification.object as? VLCMediaPlayer else { return }
        
        let timeValue = Int64(player.time.intValue)
        let videoSize = player.videoSize
        
        Task { @MainActor in
            // Update time tracking
            self.lastFrameTime = timeValue
            
            // Handle first frame detection and stream ready notification
            if !self.hasReceivedFirstFrame && videoSize.width > 0 && videoSize.height > 0 {
                print("GoPro: VLC Player - Received first valid frame, size: \(videoSize)")
                
                    self.hasReceivedFirstFrame = true
                    
                    // Update frame dimensions for FrameSource protocol
                    self.longSide = max(videoSize.width, videoSize.height)
                    self.shortSide = min(videoSize.width, videoSize.height)
                    
                    // Store the frame size and broadcast it
                    self.lastFrameSize = videoSize
                    self.broadcastFrameSizeChange(videoSize)
                    
                // Update player view layout now that we have a valid video size - dispatch to main
                DispatchQueue.main.async { [weak self] in
                    self?.updatePlayerViewLayout()
                }
                    
                    // Notify first frame received with size
                    self.goProDelegate?.goProSource(self, didReceiveFirstFrame: videoSize)
                
                // FIXED: Call completion callback now that stream is ready with frames
                if let completion = self.streamReadyCompletion {
                    self.streamReadyCompletion = nil
                    print("GoPro: Stream is ready - calling completion callback")
                    // Dispatch completion to main thread for smooth UI transition
                    DispatchQueue.main.async {
                        completion(.success(()))
                    }
                }
            }
            
            // Always notify goProDelegate about time change - but throttle to avoid spam
            if self.frameExtractionCount % 30 == 0 {  // Only notify every 30 frames at 30Hz = once per second
            self.goProDelegate?.goProSource(self, didReceiveFrameWithTime: self.lastFrameTime)
            }
        }
    }
    
    /// Helper method to broadcast frame size changes to observers
    @MainActor
    private func broadcastFrameSizeChange(_ size: CGSize) {
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
}

// MARK: - ResultsListener and InferenceTimeListener Implementation
extension GoProSource: @preconcurrency ResultsListener, @preconcurrency InferenceTimeListener {
    nonisolated func on(inferenceTime: Double, fpsRate: Double) {
        Task { @MainActor [weak self] in
            guard let self = self else { return }
            
            // Store inference time - total pipeline time will be calculated in on(result:)
            self.lastInferenceTime = inferenceTime
            
            // Forward to both delegate types to ensure complete integration
            self.delegate?.frameSource(self, didUpdateWithSpeed: inferenceTime, fps: fpsRate)
            
            // Also forward to video capture delegate for YOLOView integration
            if let videoCaptureDelegate = self.videoCaptureDelegate {
                videoCaptureDelegate.onInferenceTime(speed: inferenceTime, fps: fpsRate)
            } else {
                // Only log warning occasionally to avoid spam
                if self.frameExtractionCount % 300 == 0 {
                    print("GoPro: Warning - videoCaptureDelegate is nil for inferenceTime update")
                }
            }
            
            // NOTE: Complete pipeline timing report is now handled in on(result:) method
            // which includes post-processing time for fish counting
        }
    }
    
    nonisolated func on(result: YOLOResult) {
        Task { @MainActor in
            // Ensure we're on the main thread for UI operations
            dispatchPrecondition(condition: .onQueue(.main))
            
            // OPTIMIZATION: Clear processing flag to allow new frame extractions
            self.isModelProcessing = false
            
            // Record the start of post-processing phase
            let postProcessingStartTime = CACurrentMediaTime()
            
            // DO NOT increment frame counter here - it should only be incremented once per frame processing cycle
            // The frame counter is already incremented in the CADisplayLink callback
            
            // Record timestamp for performance monitoring
            let timestamp = CACurrentMediaTime()
            self.frameTimestamps.append(timestamp)
            
            // Only keep the last 30 timestamps for rolling performance calculation
            if self.frameTimestamps.count > 30 {
                self.frameTimestamps.removeFirst()
            }

            // UI DELEGATE CALL - Track this time separately (NOT actual fish counting)
            let uiDelegateStartTime = CACurrentMediaTime()
            
            // Forward detection results to video capture delegate - this is just the delegate call, real fish counting happens inside TrackingDetector
            var delegateCallSuccessful = false
            if let videoCaptureDelegate = self.videoCaptureDelegate {
                // Process detection results immediately on main thread for responsive UI
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
                    
                    // Try to reestablish the delegate connection if possible
                    if let view = self.delegate as? VideoCaptureDelegate {
                        print("GoPro: Attempting to recover by connecting to delegate")
                        self.videoCaptureDelegate = view
                        view.onPredict(result: result)
                        delegateCallSuccessful = true
                    }
                }
            }
            
            // Calculate UI delegate call time (this is NOT fish counting time)
            let uiDelegateTime = (CACurrentMediaTime() - uiDelegateStartTime) * 1000 // ms
            self.lastUITime = uiDelegateTime
            
            // Calculate total UI processing time (including async UI updates)
            let totalUITime = (CACurrentMediaTime() - postProcessingStartTime) * 1000 // ms
            
            // Update total pipeline time to include UI processing
            self.lastTotalPipelineTime = self.lastExtractionTime + self.lastConversionTime + self.lastInferenceTime + totalUITime
            
            // Calculate theoretical FPS based on complete processing time
            let theoreticalFPS = self.lastTotalPipelineTime > 0 ? 1000.0 / self.lastTotalPipelineTime : 0
            
            // Calculate FPS based on frame completion timestamps for true throughput
            let currentFPS = self.calculateCurrentFPS()
            
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

    // MARK: - FrameSource Protocol Implementation for UI Integration and Coordinate Transformation

extension GoProSource {
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
        
        // Handle aspect ratio differences
        if isPortrait {
            // PORTRAIT MODE HANDLING
            if videoAspectRatio > viewAspectRatio {
                // Video is wider than container (letterboxing - black bars on top/bottom)
                let scaledHeight = containerWidth / videoAspectRatio
                let verticalOffset = (containerHeight - scaledHeight) / 2
                
                let yScale = scaledHeight / containerHeight
                let normalizedOffset = verticalOffset / containerHeight
                
                adjustedBox.origin.y = (rect.origin.y * yScale) + normalizedOffset
                adjustedBox.size.height = rect.size.height * yScale
            } else if videoAspectRatio < viewAspectRatio {
                // Video is taller than container (pillarboxing - black bars on sides)
                let scaledWidth = containerHeight * videoAspectRatio
                let horizontalOffset = (containerWidth - scaledWidth) / 2
                
                let xScale = scaledWidth / containerWidth
                let normalizedOffset = horizontalOffset / containerWidth
                
                adjustedBox.origin.x = (rect.origin.x * xScale) + normalizedOffset
                adjustedBox.size.width = rect.size.width * xScale
            }
        } else {
            // LANDSCAPE MODE HANDLING
            if videoAspectRatio > viewAspectRatio {
                // Video is wider than container (letterboxing - black bars on top/bottom)
                let scaledHeight = containerWidth / videoAspectRatio
                let verticalOffset = (containerHeight - scaledHeight) / 2
                
                let yScale = scaledHeight / containerHeight
                let normalizedOffset = verticalOffset / containerHeight
                
                adjustedBox.origin.y = (rect.origin.y * yScale) + normalizedOffset
                adjustedBox.size.height = rect.size.height * yScale
            } else if videoAspectRatio < viewAspectRatio {
                // Video is taller than container (pillarboxing - black bars on sides)
                let scaledWidth = containerHeight * videoAspectRatio
                let horizontalOffset = (containerWidth - scaledWidth) / 2
                
                let xScale = scaledWidth / containerWidth
                let normalizedOffset = horizontalOffset / containerWidth
                
                adjustedBox.origin.x = (rect.origin.x * xScale) + normalizedOffset
                adjustedBox.size.width = rect.size.width * xScale
            }
        }
        
        // Handle Y-coordinate inversion for both portrait and landscape modes
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

    // MARK: - Delegate Management (Simplified)
    
    /// Sets all required delegates to properly integrate with YOLOView
    @MainActor
    func integrateWithYOLOView(view: UIView) {
        // Type check and store all required delegates
        guard let captureDelegate = view as? VideoCaptureDelegate,
              let frameDelegate = view as? FrameSourceDelegate else {
            print("GoPro: Error - View does not conform to required delegate protocols")
            return
        }
        
        // Set all delegates in one place
        self.delegate = frameDelegate
        self.videoCaptureDelegate = captureDelegate
        
        // Set GoProSourceDelegate if available
        if let goProDelegate = view as? GoProSourceDelegate {
            self.goProDelegate = goProDelegate
        }
        
        // Setup player view in container
        setupPlayerViewInContainer(view)
        
        // Get predictor from view if needed
        if predictor == nil, let yoloView = view as? YOLOView {
            predictor = yoloView.getCurrentPredictor()
        }
        
        print("GoPro: Successfully integrated with YOLOView")
    }
    
    /// Setup player view in the container with proper constraints
    @MainActor
    private func setupPlayerViewInContainer(_ containerView: UIView) {
        self.containerView = containerView
        
        guard let playerView = playerView else {
            print("GoPro: Error - No player view available for integration")
            return
        }
        
        // Make sure we're on the main thread for UI operations
        dispatchPrecondition(condition: .onQueue(.main))
        
        // Remove from any existing hierarchy
        playerView.removeFromSuperview()
        
        // Add as subview at the bottom of the hierarchy
        if containerView.subviews.isEmpty {
            containerView.addSubview(playerView)
        } else {
            containerView.insertSubview(playerView, at: 0)
        }
        
        // Setup constraints to fill the container
        let constraints = [
            playerView.leadingAnchor.constraint(equalTo: containerView.leadingAnchor),
            playerView.trailingAnchor.constraint(equalTo: containerView.trailingAnchor),
            playerView.topAnchor.constraint(equalTo: containerView.topAnchor),
            playerView.bottomAnchor.constraint(equalTo: containerView.bottomAnchor)
        ]
        
        NSLayoutConstraint.activate(constraints)
        containerView.layoutIfNeeded()
        
        // Connect the player view to the VLC player
        videoPlayer?.drawable = playerView
        
        // Add bounding box views if this is a YOLOView
        if let yoloView = containerView as? YOLOView {
            for box in yoloView.boundingBoxViews {
                box.addToLayer(playerView.layer)
            }
            
            // Setup overlay layer
            if let overlayLayer = yoloView.layer.sublayers?.first(where: { $0.name == "overlayLayer" }) {
                playerView.layer.addSublayer(overlayLayer)
            }
        }
        
        print("GoPro: Player view integrated with container: \(containerView.bounds.size)")
    }

    // MARK: - HTTP API Methods (Consolidated)
    
    /// Perform a unified GoPro HTTP request
    private func performGoProRequest(
        endpoint: GoProEndpoint,
        completion: @escaping (Result<Data?, Error>) -> Void
    ) {
        // Construct URL
        var urlComponents = URLComponents()
        urlComponents.scheme = "http"
        urlComponents.host = goProIP
        urlComponents.port = goProPort
        urlComponents.path = endpoint.path
        urlComponents.queryItems = endpoint.queryItems
        
        guard let url = urlComponents.url else {
            let error = NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])
            DispatchQueue.main.async {
                completion(.failure(error))
            }
            return
        }
        
        // Create request with timeout
        var request = URLRequest(url: url)
        request.timeoutInterval = 5.0
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            // Handle network error
            if let error = error {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
                return
            }
            
            // Check HTTP response
            guard let httpResponse = response as? HTTPURLResponse else {
                let error = NSError(domain: "GoProSource", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid HTTP response"])
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
                return
            }
            
            // Check status code
            guard httpResponse.statusCode == 200 else {
                let error = NSError(domain: "GoProSource", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP error: \(httpResponse.statusCode)"])
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
                return
            }
            
            // Success
            DispatchQueue.main.async {
                completion(.success(data))
            }
        }
        
        task.resume()
    }
    
    /// Check if connected to a GoPro camera by requesting webcam version
    func checkConnection(completion: @escaping (Result<GoProWebcamVersion, Error>) -> Void) {
        print("GoPro: Starting connection check")
        
        performGoProRequest(endpoint: .version) { result in
            switch result {
            case .success(let data):
                guard let data = data, !data.isEmpty else {
                    let error = NSError(domain: "GoProSource", code: 3, userInfo: [NSLocalizedDescriptionKey: "No data received"])
                    completion(.failure(error))
                    return
                }
                
                // Try to parse JSON
                do {
                    // First try parsing as a dictionary to check for version field
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let versionValue = json["version"] as? Int {
                        let version = GoProWebcamVersion(
                            version: versionValue,
                            max_lens_support: json["max_lens_support"] as? Bool,
                            usb_3_1_compatible: json["usb_3_1_compatible"] as? Bool
                        )
                        print("GoPro: Connection successful")
                        completion(.success(version))
                        return
                    }
                    
                    // Try standard decoding
                    let decoder = JSONDecoder()
                    let version = try decoder.decode(GoProWebcamVersion.self, from: data)
                    print("GoPro: Connection successful (standard decoding)")
                    completion(.success(version))
                } catch {
                    // If we got ANY response data, consider it a success with default values
                    if let dataString = String(data: data, encoding: .utf8), !dataString.isEmpty {
                        let defaultVersion = GoProWebcamVersion(version: 1, max_lens_support: nil, usb_3_1_compatible: nil)
                        print("GoPro: Connection successful (with default values)")
                        completion(.success(defaultVersion))
                    } else {
                        print("GoPro: Connection failed - \(error.localizedDescription)")
                        completion(.failure(error))
                    }
                }
                
            case .failure(let error):
                print("GoPro: Connection failed - \(error.localizedDescription)")
                completion(.failure(error))
            }
        }
    }
    
    /// Enter webcam preview mode
    func enterWebcamPreview(completion: @escaping (Result<Void, Error>) -> Void) {
        print("GoPro: Entering webcam preview mode")
        
        performGoProRequest(endpoint: .preview) { result in
            switch result {
            case .success:
                print("GoPro: Preview mode enabled successfully")
                completion(.success(()))
            case .failure(let error):
                print("GoPro: Preview network error: \(error.localizedDescription)")
                completion(.failure(error))
            }
        }
    }
    
    /// Start webcam streaming
    func startWebcam(completion: @escaping (Result<Void, Error>) -> Void) {
        print("GoPro: Starting webcam stream")
        
        performGoProRequest(endpoint: .start(resolution: 7, fov: 4)) { result in
            switch result {
            case .success:
                print("GoPro: Webcam started successfully")
                completion(.success(()))
            case .failure(let error):
                print("GoPro: Start webcam network error: \(error.localizedDescription)")
                completion(.failure(error))
            }
        }
    }
    
    /// Stop webcam streaming
    func stopWebcam(completion: @escaping (Result<Void, Error>) -> Void) {
        print("GoPro: Stopping webcam stream")
        
        performGoProRequest(endpoint: .stop) { result in
            switch result {
            case .success:
                print("GoPro: Webcam stopped successfully")
                completion(.success(()))
            case .failure(let error):
                print("GoPro: Stop webcam network error: \(error.localizedDescription)")
                completion(.failure(error))
            }
        }
    }
    
    /// Exit webcam mode
    func exitWebcam(completion: @escaping (Result<Void, Error>) -> Void) {
        print("GoPro: Exiting webcam mode")
        
        performGoProRequest(endpoint: .exit) { result in
            switch result {
            case .success:
                print("GoPro: Webcam exited successfully")
                completion(.success(()))
            case .failure(let error):
                print("GoPro: Exit webcam network error: \(error.localizedDescription)")
                completion(.failure(error))
            }
        }
    }
    
    /// Gracefully exit webcam mode with both stop and exit commands
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
                        completion(.success(()))
                    case .failure(let error):
                        print("GoPro: Exit command failed: \(error.localizedDescription)")
                        completion(.failure(error))
                    }
                }
            case .failure(let error):
                print("GoPro: Stop command failed: \(error.localizedDescription)")
                // Try exit anyway as a best effort
                self.exitWebcam { exitResult in
                    switch exitResult {
                    case .success:
                        // If exit succeeds, still report the stop error
                        completion(.failure(error))
                    case .failure(let exitError):
                        // Report both errors
                        let combinedError = NSError(
                            domain: "GoProSource",
                            code: 3,
                            userInfo: [
                                NSLocalizedDescriptionKey: "Multiple errors: Stop - \(error.localizedDescription), Exit - \(exitError.localizedDescription)"
                            ]
                        )
                        completion(.failure(combinedError))
                    }
                }
            }
        }
    }
}

// MARK: - Backward Compatibility Extensions

extension VideoCaptureDelegate {
    var viewForDrawing: UIView? {
        return self as? UIView
    }
}

