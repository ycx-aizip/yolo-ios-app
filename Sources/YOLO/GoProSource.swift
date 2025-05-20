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
class GoProSource: NSObject, @preconcurrency VLCMediaPlayerDelegate {
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
    weak var delegate: GoProSourceDelegate?
    
    // Add a property to store the test delegate to prevent it from being deallocated
    private var testDelegate: TestDelegate?
    
    // Add frameTimestamps property to the class
    private var frameTimestamps: [CFTimeInterval] = []
    
    // Initialize VLC player
    @MainActor
    override init() {
        super.init()
        setupVLCPlayer()
    }
    
    @MainActor
    private func setupVLCPlayer() {
        videoPlayer = VLCMediaPlayer()
        videoPlayer?.delegate = self
        videoPlayer?.libraryInstance.debugLogging = true
        videoPlayer?.libraryInstance.debugLoggingLevel = 3
        
        // Set a drawable early to prevent threading issues later
        let minimalDrawable = UIView(frame: CGRect(x: 0, y: 0, width: 1, height: 1))
        videoPlayer?.drawable = minimalDrawable
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
        
        // Set stream status to connecting
        delegate?.goProSource(self, didUpdateStatus: .connecting)
        
        // Reset frame tracking
        hasReceivedFirstFrame = false
        frameCount = 0
        streamStartTime = Date()
        
        // Configure player with optimized options
        let media = VLCMedia(url: url)
        configureRTSPMediaOptions(media)
        videoPlayer.media = media
        
        // Create a timer for timeout - increased to 15 seconds to give more time for SPS/PPS
        let timeoutTimer = Timer(timeInterval: 15.0, repeats: false) { [weak self] timer in
            guard let self = self else { return }
            
            if !self.hasReceivedFirstFrame {
                print("GoPro: Timeout waiting for first frame")
                
                DispatchQueue.main.async {
                    self.videoPlayer?.stop()
                    
                    let error = NSError(
                        domain: "GoProSource",
                        code: 2,
                        userInfo: [NSLocalizedDescriptionKey: "Timeout waiting for first frame from RTSP stream"]
                    )
                    self.delegate?.goProSource(self, didUpdateStatus: .error("Connection timeout"))
                    completion(.failure(error))
                }
            }
            
            timer.invalidate()
        }
        
        // Add timer to the main run loop
        RunLoop.main.add(timeoutTimer, forMode: .common)
        
        // Start playback
        videoPlayer.play()
        
        // Success is reported via delegate when first frame is received
        completion(.success(()))
    }
    
    /// Configure media options for optimal RTSP streaming from GoPro
    private func configureRTSPMediaOptions(_ media: VLCMedia) {
        // Core essential options for better streaming
        // media.addOption("--verbose=3")
        // media.addOption(":rtsp-tcp")
        media.addOption(":rtsp-udp")
        media.addOption(":network-caching=500") // Reduced from 1000 for faster startup
        media.addOption(":live-caching=100") // Reduced from 300 for less latency
        media.addOption(":rtsp-frame-buffer-size=5000000") // Larger buffer
        
        // Critical SPS/PPS handling options
        media.addOption(":h264-fps=30.0")
        media.addOption(":codec=avcodec")
        media.addOption(":no-audio")
        media.addOption(":avcodec-threads=1")
        media.addOption(":rtsp-sps-pps=true") // Force each keyframe to contain SPS/PPS
        media.addOption(":rtp-timeout=10000") // Longer timeout (10 seconds)
        media.addOption(":avcodec-hw=videotoolbox") // Use hardware acceleration if available

        // Additional performance options for faster startup
        media.addOption(":clock-jitter=0")
        media.addOption(":clock-synchro=0")
        media.addOption(":no-skip-frames")
        media.addOption(":no-drop-late-frames")
                
        // Clear cookies to ensure fresh connection
        media.clearStoredCookies()
        
        print("GoPro: Media options configured for RTSP streaming")
    }
    
    /// Stop RTSP stream
    func stopRTSPStream() {
        print("GoPro: Stopping RTSP stream")
        videoPlayer?.stop()
        delegate?.goProSource(self, didUpdateStatus: .stopped)
    }
    
    // MARK: - VLC Media Player Delegate Methods
    
    @MainActor
    func mediaPlayerStateChanged(_ aNotification: Notification!) {
        guard let player = aNotification.object as? VLCMediaPlayer else { return }
        
        switch player.state {
        case .opening:
            print("GoPro: VLC Player - Opening stream")
        
        case .buffering:
            print("GoPro: VLC Player - Buffering stream")
            
        case .playing:
            print("GoPro: VLC Player - Stream is playing")
            delegate?.goProSource(self, didUpdateStatus: .playing)
            
        case .error:
            print("GoPro: VLC Player - Error streaming")
            delegate?.goProSource(self, didUpdateStatus: .error("Playback error"))
            
        case .ended:
            print("GoPro: VLC Player - Stream ended")
            delegate?.goProSource(self, didUpdateStatus: .stopped)
            
        case .stopped:
            print("GoPro: VLC Player - Stream stopped")
            delegate?.goProSource(self, didUpdateStatus: .stopped)
            
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
            hasReceivedFirstFrame = true
            
            // Get video size while on the main thread
            let videoSize = player.videoSize
            print("GoPro: VLC Player - Received first frame, size: \(videoSize)")
            
            // Notify first frame received with size - add extra debugging
            print("GoPro: Calling delegate didReceiveFirstFrame with size: \(videoSize)")
            delegate?.goProSource(self, didReceiveFirstFrame: videoSize)
            print("GoPro: Delegate didReceiveFirstFrame call completed")
        }
        
        // Every 30 frames, report status
        if frameCount % 30 == 0 {
            // Calculate instantaneous FPS from last several frames if available
            let instantFps = calculateInstantaneousFps()
            
            // Calculate overall FPS if stream has been running for at least 1 second
            if let startTime = streamStartTime {
                let elapsed = Date().timeIntervalSince(startTime)
                let overallFps = Double(frameCount) / elapsed
                print("GoPro: VLC Player - Receiving frames at \(String(format: "%.2f", instantFps)) FPS (avg: \(String(format: "%.2f", overallFps)))")
            }
            
            // Notify about frame reception
            delegate?.goProSource(self, didReceiveFrameWithTime: lastFrameTime)
        } else {
            // Still notify delegate about each frame, not just every 30th
            delegate?.goProSource(self, didReceiveFrameWithTime: lastFrameTime)
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
    @MainActor
    func testRTSPStream(timeout: TimeInterval = 30.0, completion: @escaping (Bool, String) -> Void) {
        var testLog = "Starting GoPro RTSP stream test (capturing 30 frames)...\n"
        let testCompleted = Atomic<Bool>(value: false)
        
        // Store original delegate to restore later
        let originalDelegate = self.delegate
        
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
        self.delegate = self.testDelegate
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
                Task { @MainActor [weak self] in
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
    @MainActor
    private func completeTest(
        originalDelegate: GoProSourceDelegate?,
        success: Bool,
        message: String,
        completion: @escaping (Bool, String) -> Void
    ) {
        // Stop the stream
        stopRTSPStream()
        
        // Restore original delegate
        self.delegate = originalDelegate
        
        // Clear the test delegate reference
        self.testDelegate = nil
        
        // Report result
        completion(success, message)
    }
    
    // Thread-safe boolean wrapper for atomic operations
    private class Atomic<T> {
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
