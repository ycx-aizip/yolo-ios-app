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
    private var playerView: UIView?
    
    // Container view that will hold the player view
    private weak var containerView: UIView?
    
    // Current orientation of the device
    private var currentOrientation: UIDeviceOrientation = .portrait
    
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
            self.stopRTSPStream()
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
        videoPlayer?.libraryInstance.debugLogging = true
        videoPlayer?.libraryInstance.debugLoggingLevel = 3
        
        // Use a default sensible size (16:9 aspect ratio for common HD videos)
        // This prevents zero-sized frame issues during initialization
        let defaultWidth: CGFloat = 1280
        let defaultHeight: CGFloat = 720
        
        // Create a view to use as the drawable for frame extraction
        playerView = UIView(frame: CGRect(x: 0, y: 0, width: defaultWidth, height: defaultHeight))
        
        // Set critical properties for proper rendering
        playerView?.backgroundColor = .black
        playerView?.layer.masksToBounds = true
        
        // Add a visual indication that the player view exists but no content is being shown
        playerView?.layer.borderWidth = 1.0
        playerView?.layer.borderColor = UIColor.red.cgColor
        
        // This is crucial - use VLCMediaPlayer's drawable property
        videoPlayer?.drawable = playerView
        
        // Force the video player to start with a good frame size
        videoPlayer?.videoAspectRatio = UnsafeMutablePointer<Int8>(mutating: "16:9")
        videoPlayer?.videoCropGeometry = UnsafeMutablePointer<Int8>(mutating: "")
        
        print("GoPro: VLC player view initialized with size: \(playerView?.bounds.size ?? .zero)")
    }
    
    // MARK: - Frame Extraction Methods
    
    @MainActor
    private func extractCurrentFrame() -> UIImage? {
        guard let playerView = playerView else { 
            print("GoPro: Cannot extract frame - player view is nil")
            return nil 
        }
        
        // Check for valid size - prevent crash with zero-sized frame
        let viewSize = playerView.bounds.size
        if viewSize.width <= 0 || viewSize.height <= 0 {
            print("GoPro: Cannot extract frame - invalid view size: \(viewSize)")
            return nil
        }
        
        // IMPORTANT: Force main thread for UIGraphics operations
        dispatchPrecondition(condition: .onQueue(.main))
        
        // Attempt to grab the frame - use a bit larger scale for better quality
        let scale: CGFloat = 1.0
        UIGraphicsBeginImageContextWithOptions(viewSize, false, scale)
        defer { UIGraphicsEndImageContext() }
        
        if let context = UIGraphicsGetCurrentContext() {
            // Try to force the layer to update before rendering
            playerView.layer.setNeedsDisplay()
            
            // Render the layer
            playerView.layer.render(in: context)
            
            // Get the image
            if let image = UIGraphicsGetImageFromCurrentImageContext() {
                // Check if image is all black or invalid
                if isBlackImage(image) {
                    print("GoPro: Warning - Extracted all-black frame from player view")
                    // Return the image anyway, as we can use it for debugging
                }
                return image
            } else {
                print("GoPro: Failed to get image from graphics context")
            }
        } else {
            print("GoPro: Failed to get graphics context")
        }
        
        return nil
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
    
    // MARK: - Improved Delegate Management
    
    /// Sets both goProDelegate and FrameSource delegate to ensure complete integration
    func setDelegate(_ delegate: GoProSourceDelegate & FrameSourceDelegate) {
        self.goProDelegate = delegate
        self.delegate = delegate
    }
    
    /// Sets all required delegates to properly integrate with YOLOView
    @MainActor
    func integrateWithYOLOView(view: VideoCaptureDelegate & FrameSourceDelegate) {
        // Set both delegate types
        self.delegate = view
        self.videoCaptureDelegate = view
        
        // If the view also conforms to GoProSourceDelegate, set that too
        if let goProDelegate = view as? GoProSourceDelegate {
            self.goProDelegate = goProDelegate
        }
        
        // Get the container view where we'll display the video
        if let containerView = view.viewForDrawing {
            self.containerView = containerView
            
            // Make sure we have a valid player view
            guard let playerView = playerView else {
                print("GoPro: Error - No player view available for integration")
                return
            }
            
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
            
            // Ensure the player view is properly connected to the VLC player
            videoPlayer?.drawable = playerView
            
            print("GoPro: Successfully integrated player view with container: \(containerView.bounds.size)")
        } else {
            print("GoPro: Error - No container view available for integration")
        }
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
        
        NSLayoutConstraint.activate([
            leadingConstraint,
            trailingConstraint,
            topConstraint,
            bottomConstraint
        ])
        
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
        // Critical options for RTSP
        media.addOption(":rtsp-tcp")  // Force TCP for more reliable streaming
        
        // Don't set a fixed size to allow the video to match actual stream dimensions
        // media.addOption(":video-filter=crops{width=1280,height=720}")
        
        // Reduced caching for lower latency
        media.addOption(":network-caching=100")  // Lower for less latency
        media.addOption(":live-caching=100")     // Lower for less latency
        
        // Hardware acceleration (when available)
        media.addOption(":avcodec-hw=any")  // Let VLC choose best hardware option
        
        // Keep audio enabled for better synchronization
        // media.addOption(":no-audio")
        
        // Improve H.264 handling 
        media.addOption(":rtsp-frame-buffer-size=1000000")  // Smaller buffer for less latency
        
        // SPS/PPS handling (critical for H.264)
        media.addOption(":rtsp-sps-pps=true")    // Force SPS/PPS with each keyframe
        
        // Timeout settings
        media.addOption(":rtp-timeout=5000")     // 5 second timeout
        
        // Optimize for low latency
        media.addOption(":clock-jitter=0")       // Disable clock jitter detection
        media.addOption(":clock-synchro=0")      // Disable clock synchro
        
        // Force decoding threads to 4 for better performance
        media.addOption(":avcodec-threads=4")
        
        // Disable subtitles
        media.addOption(":no-sub-autodetect-file")
        
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
        
        // Capture state locally to avoid data races with self
        let state = player.state
        
        // The state changes need to be processed on the main thread
        Task { @MainActor [weak self] in
            guard let self = self else { return }
            
            switch state {
            case .opening:
                print("GoPro: VLC Player - Opening stream")
            
            case .buffering:
                print("GoPro: VLC Player - Buffering stream")
                
            case .playing:
                print("GoPro: VLC Player - Stream is playing")
                self.goProDelegate?.goProSource(self, didUpdateStatus: .playing)
                
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
                print("GoPro: VLC Player - State: \(state.rawValue)")
            }
        }
    }
    
    nonisolated func mediaPlayerTimeChanged(_ aNotification: Notification!) {
        guard let player = aNotification.object as? VLCMediaPlayer else { return }
        
        // Capture values locally to avoid data races
        let time = Int64(player.time.intValue)
        let videoSize = player.videoSize
        
        // Process frame updates on the main thread
        Task { @MainActor [weak self] in
            guard let self = self else { return }
            
            // Increment frame count
            self.frameCount += 1
            
            // Store the current media time
            self.lastFrameTime = time
            
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
                    
                    // Update player view layout now that we have a valid video size
                    self.updatePlayerViewLayout()
                    
                    // Notify first frame received with size - add extra debugging
                    print("GoPro: Calling delegate didReceiveFirstFrame with size: \(videoSize)")
                    self.goProDelegate?.goProSource(self, didReceiveFirstFrame: videoSize)
                    print("GoPro: Delegate didReceiveFirstFrame call completed")
                } else {
                    print("GoPro: Received invalid frame size: \(videoSize). Waiting for valid frame...")
                    return // Skip processing this frame
                }
            }
            
            // Extract current frame for FrameSource delegate and predictor
            guard let frameImage = self.extractCurrentFrame() else {
                print("GoPro: Failed to extract frame")
                return
            }
            
            // Send to FrameSource delegate for display and processing
            self.delegate?.frameSource(self, didOutputImage: frameImage)
            
            // Process with predictor if inference is enabled
            if self.inferenceOK, let predictor = self.predictor {
                if let pixelBuffer = self.createStandardPixelBuffer(from: frameImage, forSourceType: self.sourceType) {
                    if let sampleBuffer = self.createStandardSampleBuffer(from: pixelBuffer) {
                        predictor.predict(
                            sampleBuffer: sampleBuffer,
                            onResultsListener: self,
                            onInferenceTime: self
                        )
                    } else {
                        print("GoPro: Failed to create sample buffer")
                    }
                } else {
                    print("GoPro: Failed to create pixel buffer")
                }
            }
            
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
}

// MARK: - ResultsListener and InferenceTimeListener Implementation
extension GoProSource: @preconcurrency ResultsListener, @preconcurrency InferenceTimeListener {
    nonisolated func on(inferenceTime: Double, fpsRate: Double) {
        // Since this is nonisolated, we need to dispatch to the main actor
        Task { @MainActor [weak self] in
            guard let self = self else { return }
            
            self.delegate?.frameSource(self, didUpdateWithSpeed: inferenceTime, fps: fpsRate)
            self.videoCaptureDelegate?.onInferenceTime(speed: inferenceTime, fps: fpsRate)
        }
    }
    
    nonisolated func on(result: YOLOResult) {
        // Since this is nonisolated, we need to dispatch to the main actor
        Task { @MainActor [weak self] in
            guard let self = self else { return }
            
            self.videoCaptureDelegate?.onPredict(result: result)
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
