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
class GoProSource: NSObject, VLCMediaPlayerDelegate {
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
    
    // Initialize VLC player
    override init() {
        super.init()
        setupVLCPlayer()
    }
    
    private func setupVLCPlayer() {
        videoPlayer = VLCMediaPlayer()
        videoPlayer?.delegate = self
        videoPlayer?.libraryInstance.debugLogging = true
        videoPlayer?.libraryInstance.debugLoggingLevel = 3
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
        
        // Configure player - with improved options from the example code
        let media = VLCMedia(url: url)
        configureSimplifiedMediaOptions(media)
        videoPlayer.media = media
        
        // Ensure a drawable is set (even if it's just a placeholder)
        if videoPlayer.drawable == nil {
            // Create a minimal drawable if none provided
            // This may be necessary for proper VLC media player initialization
            let minimalDrawable = UIView(frame: CGRect(x: 0, y: 0, width: 1, height: 1))
            videoPlayer.drawable = minimalDrawable
        }
        
        // Create a timer for timeout
        let timeoutTimer = Timer(timeInterval: 10.0, repeats: false) { [weak self] timer in
            guard let self = self else { return }
            
            if !self.hasReceivedFirstFrame {
                print("GoPro: Timeout waiting for first frame")
                
                // Use main thread to stop player and update UI
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
            
            // Invalidate the timer
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
    private func configureSimplifiedMediaOptions(_ media: VLCMedia) {
        // Apply the suggested options from the example code
        media.addOption("-vv") // Verbose logging
        
        media.addOptions([
            "network-caching": 500,
            "sout-rtp-caching": 100,
            "sout-rtp-port-audio": 20000,
            "sout-rtp-port-video": 20002,
            ":rtp-timeout": 10000,
            ":rtsp-tcp": true,
            ":rtsp-frame-buffer-size": 1024,
            ":rtsp-caching": 0,
            ":live-caching": 0,
        ])
        
        // Add additional codec specifications
        media.addOption(":codec=avcodec")
        media.addOption(":vcodec=h264")
        media.addOption("--file-caching=2000")
        media.addOption("clock-jitter=0")
        media.addOption("--rtsp-tcp")
        
        // Clear cookies
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
    
    func mediaPlayerTimeChanged(_ aNotification: Notification!) {
        guard let player = aNotification.object as? VLCMediaPlayer else { return }
        
        // Increment frame count
        frameCount += 1
        
        // Store the current media time
        lastFrameTime = Int64(player.time.intValue)
        
        // Calculate frame size if this is the first frame
        if !hasReceivedFirstFrame {
            hasReceivedFirstFrame = true
            
            // Log frame details
            let videoSize = player.videoSize
            print("GoPro: VLC Player - Received first frame, size: \(videoSize)")
            
            // Notify first frame received with size
            delegate?.goProSource(self, didReceiveFirstFrame: videoSize)
        }
        
        // Every 30 frames, report status
        if frameCount % 30 == 0 {
            // Calculate FPS if stream has been running for at least 1 second
            if let startTime = streamStartTime {
                let elapsed = Date().timeIntervalSince(startTime)
                let fps = Double(frameCount) / elapsed
                print("GoPro: VLC Player - Receiving frames at \(String(format: "%.2f", fps)) FPS")
            }
            
            // Notify about frame reception
            delegate?.goProSource(self, didReceiveFrameWithTime: lastFrameTime)
        }
    }
    
    // MARK: - Simple Test Function for RTSP Stream
    
    /// Simple test function to validate RTSP streaming from GoPro
    /// Call this function to perform a quick verification of the RTSP stream
    /// - Parameters:
    ///   - timeout: Timeout in seconds to wait for frames (default 15 seconds)
    ///   - completion: Called with success/failure and detailed message
    @MainActor
    func testRTSPStream(timeout: TimeInterval = 15.0, completion: @escaping (Bool, String) -> Void) {
        var testLog = "Starting GoPro RTSP stream test...\n"
        testLog += "Initializing video player...\n"
        
        // Create a temporary view to serve as the drawable
        // This is important as some VLC functionality may depend on having a valid drawable
        let tempView = UIView(frame: CGRect(x: 0, y: 0, width: 1, height: 1))
        
        // Create a standalone player instance for testing, which won't interfere with the main player
        let testPlayer = VLCMediaPlayer()
        testPlayer.delegate = self
        testPlayer.drawable = tempView // Assign the drawable instead of nil
        
        // Strong reference to avoid premature deallocation
        var strongPlayerRef: VLCMediaPlayer? = testPlayer
        var strongView: UIView? = tempView
        
        // Initialize tracking variables
        var frameCountDuringTest = 0
        var hasReceivedFrame = false
        var testCompleted = false
        
        // Construct RTSP URL
        let rtspURLString = "rtsp://\(goProIP):\(rtspPort)\(rtspPath)"
        testLog += "Attempting to connect to RTSP stream at \(rtspURLString)\n"
        print("GoPro RTSP Test: Trying \(rtspURLString)")
        
        guard let url = URL(string: rtspURLString) else {
            testLog += "ERROR: Invalid URL format\n"
            testCompleted = true
            completion(false, "Failed to create valid RTSP URL\n\nLog:\n\(testLog)")
            return
        }
        
        // Configure media
        let media = VLCMedia(url: url)
        
        // Apply the more comprehensive options
        media.addOption("-vv") // Verbose logging
        
        media.addOptions([
            "network-caching": 500,
            "sout-rtp-caching": 100,
            "sout-rtp-port-audio": 20000,
            "sout-rtp-port-video": 20002,
            ":rtp-timeout": 10000,
            ":rtsp-tcp": true,
            ":rtsp-frame-buffer-size": 1024,
            ":rtsp-caching": 0,
            ":live-caching": 0,
        ])
        
        // Add additional codec specifications
        media.addOption(":codec=avcodec")
        media.addOption(":vcodec=h264")
        media.addOption("--file-caching=2000")
        media.addOption("clock-jitter=0")
        media.addOption("--rtsp-tcp")
        
        // Clear cookies
        media.clearStoredCookies()
        
        testLog += "Comprehensive media options configured\n"
        
        // Set up player
        testPlayer.media = media
        testPlayer.audio?.isMuted = true // Mute audio for test
        
        // Set verbosity level
        testPlayer.libraryInstance.debugLogging = true
        testPlayer.libraryInstance.debugLoggingLevel = 3
        
        // Set up observation token
        var timeObserverToken: NSObjectProtocol? = nil
        var stateObserverToken: NSObjectProtocol? = nil
        
        // Function to clean up observers
        @MainActor func cleanupObservers() {
            if let timeToken = timeObserverToken {
                NotificationCenter.default.removeObserver(timeToken)
                timeObserverToken = nil
            }
            
            if let stateToken = stateObserverToken {
                NotificationCenter.default.removeObserver(stateToken)
                stateObserverToken = nil
            }
            
            strongPlayerRef = nil
            strongView = nil
        }
        
        // Very important: Set up callbacks for frame counting and logging
        timeObserverToken = NotificationCenter.default.addObserver(
            forName: NSNotification.Name(rawValue: "VLCMediaPlayerTimeChanged"),
            object: testPlayer,
            queue: .main
        ) { notification in
            guard let player = notification.object as? VLCMediaPlayer, !testCompleted else { return }
            
            // A time change notification indicates we've received a frame
            frameCountDuringTest += 1
            
            // Calculate frame size if this is the first frame
            if !hasReceivedFrame {
                hasReceivedFrame = true
                
                // Try to get video dimensions
                let videoSize = player.videoSize
                if videoSize != .zero {
                    testLog += "First frame received! Size: \(Int(videoSize.width))×\(Int(videoSize.height))\n"
                } else {
                    testLog += "First frame received! (Size unavailable)\n"
                }
                
                // Auto-complete test after receiving at least one frame
                if !testCompleted {
                    testCompleted = true
                    
                    // Clean up
                    player.stop()
                    Task { @MainActor in
                        cleanupObservers()
                    }
                    
                    // Success!
                    testLog += "Test completed successfully: First frame received!\n"
                    completion(true, "Successfully received video frame\n\nLog:\n\(testLog)")
                }
            }
            
            // Log every 10th frame during test
            if frameCountDuringTest % 10 == 0 {
                print("GoPro RTSP Test: Received \(frameCountDuringTest) frames")
                testLog += "Received \(frameCountDuringTest) frames\n"
            }
        }
        
        // Set up callback for state changes
        stateObserverToken = NotificationCenter.default.addObserver(
            forName: NSNotification.Name(rawValue: "VLCMediaPlayerStateChanged"),
            object: testPlayer,
            queue: .main
        ) { notification in
            guard let player = notification.object as? VLCMediaPlayer, !testCompleted else { return }
            
            let state = player.state
            testLog += "Player state changed: \(state.rawValue)\n"
            
            // Handle errors
            if state == .error {
                testLog += "Player ERROR occurred\n"
                
                if !testCompleted {
                    testCompleted = true
                    
                    // Clean up
                    player.stop()
                    Task { @MainActor in
                        cleanupObservers()
                    }
                    
                    completion(false, "Stream error occurred\n\nLog:\n\(testLog)")
                }
            }
        }
        
        // Start playback
        testPlayer.play()
        testLog += "Started playback\n"
        
        // Set timeout timer
        Task {
            try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
            
            if !testCompleted {
                testCompleted = true
                
                // Clean up
                testPlayer.stop()
                await cleanupObservers()
                
                if hasReceivedFrame {
                    // Success!
                    completion(true, "Successfully received \(frameCountDuringTest) frames\n\nLog:\n\(testLog)")
                } else {
                    testLog += "Timeout: No frames received after \(Int(timeout)) seconds\n"
                    completion(false, "Failed to receive any frames\n\nLog:\n\(testLog)")
                }
            }
        }
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
}
