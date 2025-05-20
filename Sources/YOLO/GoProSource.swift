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
    
    // MARK: - Initialization
    
    override init() {
        super.init()
        
        // Since UIView creation must happen on the main thread
        DispatchQueue.main.async {
            self.setupVLCPlayer()
        }
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
        // Since the class is @MainActor, we need to use Task for isolation crossing
        Task { @MainActor in
            if self.videoPlayer?.state != .playing {
                self.startRTSPStream { _ in }
            }
        }
    }
    
    /// Stops frame acquisition from the source.
    nonisolated func stop() {
        // Since the class is @MainActor, we need to use Task for isolation crossing
        Task { @MainActor in
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
        // No specific orientation handling needed for RTSP
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
        guard let playerView = playerView else { return nil }
        
        // Check for valid size - prevent crash with zero-sized frame
        let viewSize = playerView.bounds.size
        if viewSize.width <= 0 || viewSize.height <= 0 {
            print("GoPro: Cannot extract frame - invalid view size: \(viewSize)")
            return nil
        }
        
        // Capture the current view contents
        UIGraphicsBeginImageContextWithOptions(viewSize, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        
        if let context = UIGraphicsGetCurrentContext() {
            playerView.layer.render(in: context)
            if let image = UIGraphicsGetImageFromCurrentImageContext() {
                return image
            }
        }
        
        return nil
    }
    
    // MARK: - Backward Compatibility
    
    // Add a setter for goProDelegate that also sets the FrameSource delegate
    // This ensures backward compatibility with existing code
    func setDelegate(_ delegate: GoProSourceDelegate?) {
        self.goProDelegate = delegate
        
        // If delegate is also a FrameSourceDelegate, set that too
        if let fsDelegate = delegate as? FrameSourceDelegate {
            self.delegate = fsDelegate
        }
    }
    
    /// Helper class to adapt between older delegate types and newer ones
    private class DelegateAdapter: NSObject, GoProSourceDelegate {
        private weak var frameSourceDelegate: FrameSourceDelegate?
        private weak var videoCaptureDelegate: VideoCaptureDelegate?
        
        init(frameSourceDelegate: FrameSourceDelegate? = nil, videoCaptureDelegate: VideoCaptureDelegate? = nil) {
            self.frameSourceDelegate = frameSourceDelegate
            self.videoCaptureDelegate = videoCaptureDelegate
            super.init()
        }
        
        func goProSource(_ source: GoProSource, didUpdateStatus status: GoProStreamingStatus) {
            // No direct mapping to FrameSourceDelegate
        }
        
        func goProSource(_ source: GoProSource, didReceiveFirstFrame size: CGSize) {
            // No direct mapping to FrameSourceDelegate
        }
        
        func goProSource(_ source: GoProSource, didReceiveFrameWithTime time: Int64) {
            // No direct mapping to FrameSourceDelegate
        }
    }
    
    /// Called from YOLOView to create an integration with GoPro source
    @MainActor
    func integrateWithYOLOView(view: VideoCaptureDelegate & FrameSourceDelegate) {
        // Set both delegate types
        self.delegate = view
        self.videoCaptureDelegate = view
        
        // Create adapter for backward compatibility if needed
        let adapter = DelegateAdapter(frameSourceDelegate: view, videoCaptureDelegate: view)
        self.goProDelegate = adapter
        
        // Make sure VLC player view is properly sized and added to view hierarchy
        if let playerView = playerView, let containerView = view.viewForDrawing {
            // Add player view to the container's view hierarchy for drawing
            if playerView.superview != containerView {
                playerView.removeFromSuperview()
                containerView.addSubview(playerView)
                print("GoPro: Added player view to container view")
            }
            
            // First make sure the player view is sized correctly - must be non-zero
            let containerSize = containerView.bounds.size
            if containerSize.width > 0 && containerSize.height > 0 {
                playerView.frame = CGRect(x: 0, y: 0, width: containerSize.width, height: containerSize.height)
            } else {
                // Use a fallback size if container has zero size (happens during initialization)
                playerView.frame = CGRect(x: 0, y: 0, width: 1280, height: 720)
            }
            
            // Make sure the player view stays properly sized
            playerView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
            
            // Ensure proper layer hierarchy
            if let player = videoPlayer {
                player.drawable = playerView
            }
            
            print("GoPro: Player view added to container with size: \(containerSize)")
        } else {
            print("GoPro: Warning - Could not add player view to container")
        }
    }
    
    /// Configure media options for optimal RTSP streaming from GoPro
    private func configureRTSPMediaOptions(_ media: VLCMedia) {
        // Critical options for RTSP
        media.addOption(":rtsp-tcp")  // Force TCP for more reliable streaming
        
        // Set a fixed size to ensure we get a valid frame size
        media.addOption(":video-filter=crops{width=1280,height=720}")
        
        // Reduced caching for lower latency but increased reliability
        media.addOption(":network-caching=500")  // Increased for more reliability
        media.addOption(":live-caching=300")     // Increased for more stability
        
        // Hardware acceleration (when available)
        media.addOption(":avcodec-hw=any")  // Let VLC choose best hardware option
        
        // Video only (disable audio)
        media.addOption(":no-audio")
        
        // Improve H.264 handling 
        media.addOption(":rtsp-frame-buffer-size=2000000")  // Larger buffer for reliability
        media.addOption(":h264-fps=30.0")  // Force 30 FPS for consistent timing
        
        // SPS/PPS handling (critical for H.264)
        media.addOption(":rtsp-sps-pps=true")    // Force SPS/PPS with each keyframe
        
        // Timeout settings (more generous to avoid disconnections)
        media.addOption(":rtp-timeout=10000")     // 10 second timeout
        
        // Optimize for low latency
        media.addOption(":clock-jitter=0")       // Disable clock jitter detection
        media.addOption(":clock-synchro=0")      // Disable clock synchro
        
        // Force decoding threads to 1 for stability
        media.addOption(":avcodec-threads=1")
        
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
        
        // Ensure the player view has a valid size before starting
        if let playerView = playerView, (playerView.bounds.width <= 0 || playerView.bounds.height <= 0) {
            playerView.frame = CGRect(x: 0, y: 0, width: 1280, height: 720)
            print("GoPro: Reset player view to default size: \(playerView.bounds.size)")
        }
        
        // Set stream status to connecting - for both delegate types
        goProDelegate?.goProSource(self, didUpdateStatus: .connecting)
        
        // Reset frame tracking
        hasReceivedFirstFrame = false
        frameCount = 0
        frameTimestamps.removeAll()
        streamStartTime = Date()
        
        // Configure player with optimized options
        let media = VLCMedia(url: url)
        configureRTSPMediaOptions(media)
        videoPlayer.media = media
        
        // Start playback
        videoPlayer.play()
        
        // Create a timer for timeout - increased to 15 seconds to give more time for network
        let timeoutTimer = Timer(timeInterval: 15.0, repeats: false) { [weak self] timer in
            guard let self = self else { return }
            
            Task { @MainActor in
                if !self.hasReceivedFirstFrame {
                    print("GoPro: Timeout waiting for first frame")
                    
                    self.videoPlayer?.stop()
                    
                    let error = NSError(
                        domain: "GoProSource", 
                        code: 2,
                        userInfo: [NSLocalizedDescriptionKey: "Timeout waiting for first frame from RTSP stream"]
                    )
                    self.goProDelegate?.goProSource(self, didUpdateStatus: .error("Connection timeout"))
                    
                    // We don't call completion here since we already reported success
                    // The caller needs to check for frames themselves if needed
                }
            }
            
            timer.invalidate()
        }
        
        // Add timer to the main run loop
        RunLoop.main.add(timeoutTimer, forMode: .common)
        
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
    
    // These functions are called on the main thread by VLC
    func mediaPlayerStateChanged(_ aNotification: Notification!) {
        guard let player = aNotification.object as? VLCMediaPlayer else { return }
        
        switch player.state {
        case .opening:
            print("GoPro: VLC Player - Opening stream")
        
        case .buffering:
            print("GoPro: VLC Player - Buffering stream")
            
        case .playing:
            print("GoPro: VLC Player - Stream is playing")
            goProDelegate?.goProSource(self, didUpdateStatus: .playing)
            
        case .error:
            print("GoPro: VLC Player - Error streaming")
            goProDelegate?.goProSource(self, didUpdateStatus: .error("Playback error"))
            
        case .ended:
            print("GoPro: VLC Player - Stream ended")
            goProDelegate?.goProSource(self, didUpdateStatus: .stopped)
            
        case .stopped:
            print("GoPro: VLC Player - Stream stopped")
            goProDelegate?.goProSource(self, didUpdateStatus: .stopped)
            
        default:
            print("GoPro: VLC Player - State: \(player.state.rawValue)")
        }
    }
    
    @MainActor
    func mediaPlayerTimeChanged(_ aNotification: Notification!) {
        guard let player = aNotification.object as? VLCMediaPlayer else { return }
        
        // Increment frame count
        frameCount += 1
        
        // Store the current media time
        lastFrameTime = Int64(player.time.intValue)
        
        // Track frame timestamps for FPS calculation - keep last 30 frames
        frameTimestamps.append(CACurrentMediaTime())
        while frameTimestamps.count > 30 {
            frameTimestamps.removeFirst()
        }
        
        // Calculate frame size if this is the first frame
        if !hasReceivedFirstFrame {
            // Get video size while on the main thread
            let videoSize = player.videoSize
            print("GoPro: VLC Player - Received first frame, size: \(videoSize)")
            
            // Only proceed if we have a valid frame size
            if videoSize.width > 0 && videoSize.height > 0 {
                hasReceivedFirstFrame = true
                
                // Update frame dimensions for FrameSource protocol
                longSide = max(videoSize.width, videoSize.height)
                shortSide = min(videoSize.width, videoSize.height)
                
                // Ensure we have a proper sized playerView for rendering frames
                if let playerView = playerView {
                    playerView.frame = CGRect(x: 0, y: 0, width: videoSize.width, height: videoSize.height)
                    // Force playerView to redraw with new size
                    playerView.setNeedsDisplay()
                }
                
                // Notify first frame received with size - add extra debugging
                print("GoPro: Calling delegate didReceiveFirstFrame with size: \(videoSize)")
                goProDelegate?.goProSource(self, didReceiveFirstFrame: videoSize)
                print("GoPro: Delegate didReceiveFirstFrame call completed")
            } else {
                print("GoPro: Received invalid frame size: \(videoSize). Waiting for valid frame...")
                return // Skip processing this frame
            }
        }
        
        // Extract current frame for FrameSource delegate and predictor
        if let frameImage = extractCurrentFrame() {
            // Add more debug logging to track frame flow
            print("GoPro: Extracted frame of size: \(frameImage.size)")
            
            // Send to FrameSource delegate for display and processing
            delegate?.frameSource(self, didOutputImage: frameImage)
            
            // Process with predictor if inference is enabled
            if inferenceOK, let predictor = predictor {
                if let pixelBuffer = createStandardPixelBuffer(from: frameImage, forSourceType: sourceType) {
                    if let sampleBuffer = createStandardSampleBuffer(from: pixelBuffer) {
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
            let instantFps = calculateInstantaneousFps()
            delegate?.frameSource(self, didUpdateWithSpeed: 1000.0 / instantFps, fps: instantFps)
        } else {
            print("GoPro: Failed to extract frame")
        }
        
        // Every 30 frames, report status and log detailed metrics
        if frameCount % 30 == 0 {
            // Calculate instantaneous FPS from last several frames if available
            let instantFps = calculateInstantaneousFps()
            
            // Calculate overall FPS if stream has been running for at least 1 second
            if let startTime = streamStartTime {
                let elapsed = Date().timeIntervalSince(startTime)
                let overallFps = Double(frameCount) / elapsed
                print("GoPro: VLC Player - Receiving frames at \(String(format: "%.2f", instantFps)) FPS (avg: \(String(format: "%.2f", overallFps)))")
            }
            
            // Notify about frame reception (for backward compatibility)
            goProDelegate?.goProSource(self, didReceiveFrameWithTime: lastFrameTime)
        } else {
            // Still notify goProDelegate about each frame (for backward compatibility)
            goProDelegate?.goProSource(self, didReceiveFrameWithTime: lastFrameTime)
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
            completion(.failure(NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
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
            completion(.failure(NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid preview URL"])))
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
            completion(.failure(NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid start URL"])))
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
            completion(.failure(NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid stop URL"])))
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
            completion(.failure(NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid exit URL"])))
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
        
        // Report result
        completion(success, message)
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
extension GoProSource: ResultsListener, InferenceTimeListener {
    nonisolated func on(inferenceTime: Double, fpsRate: Double) {
        // Pass inference time metrics to FrameSource delegate
        Task { @MainActor in
            self.delegate?.frameSource(self, didUpdateWithSpeed: inferenceTime, fps: fpsRate)
            self.videoCaptureDelegate?.onInferenceTime(speed: inferenceTime, fps: fpsRate)
        }
    }
    
    nonisolated func on(result: YOLOResult) {
        // Pass detection results to video capture delegate
        Task { @MainActor in
            self.videoCaptureDelegate?.onPredict(result: result)
        }
    }
}

// MARK: - FrameSource Helper Extension Implementation
extension GoProSource {
    // Helper to convert UIImage to pixel buffer suitable for YOLO processing
    private func createStandardPixelBuffer(from image: UIImage, forSourceType sourceType: FrameSourceType) -> CVPixelBuffer? {
        guard let cgImage = image.cgImage else { return nil }
        
        let width = cgImage.width
        let height = cgImage.height
        
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs,
            &pixelBuffer
        )
        
        guard let pixel = pixelBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(pixel, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(pixel, CVPixelBufferLockFlags(rawValue: 0)) }
        
        let context = CGContext(
            data: CVPixelBufferGetBaseAddress(pixel),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixel),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        )
        
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return pixel
    }
    
    // Creates a standard CMSampleBuffer from a CVPixelBuffer for use with the predictor
    private func createStandardSampleBuffer(from pixelBuffer: CVPixelBuffer) -> CMSampleBuffer? {
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
