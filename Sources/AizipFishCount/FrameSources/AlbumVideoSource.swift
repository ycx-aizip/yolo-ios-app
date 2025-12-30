// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
//  This file is part of the Ultralytics YOLO Package, providing video file playback for inference.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The AlbumVideoSource component provides functionality to play videos from the device's
//  photo library and process the frames through YOLO models. It manages video playback,
//  frame extraction, and delivery of frames to the YOLO processing pipeline. The component
//  implements the FrameSource protocol to maintain compatibility with the existing architecture
//  and enables seamless switching between camera feed and pre-recorded videos.
//  Please note: upon testing, the current Album mode only works for 16:9 videos taken originally from the iPhone or GoPro camera.

import AVFoundation
import CoreVideo
import Photos
import UIKit
import Vision
import CoreMedia

// Extension for Notification.Name to define custom notifications
extension Notification.Name {
    /// Notification posted when video playback has ended
    static let videoPlaybackDidEnd = Notification.Name("videoPlaybackDidEnd")
}

/// Frame source implementation that plays videos from the photo library.
@preconcurrency
class AlbumVideoSource: NSObject, FrameSource, ResultsListener, InferenceTimeListener {
    /// The delegate to receive frames and performance metrics.
    weak var delegate: FrameSourceDelegate?
    
    /// The VideoCaptureDelegate to receive prediction results (same as in CameraVideoSource).
    weak var videoCaptureDelegate: VideoCaptureDelegate?
    
    /// The predictor used to process frames from this source.
    var predictor: FrameProcessor!
    
    /// The video player.
    private var player: AVPlayer?
    
    /// The video output for extracting frames.
    private var videoOutput: AVPlayerItemVideoOutput?
    
    /// Timer for controlling frame extraction rate.
    private var frameTimer: Timer?
    
    /// Display link for smooth frame extraction (replaces timer)
    private var displayLink: CADisplayLink?
    
    /// Target frame rate for playback.
    private var frameRate: Float = 30.0
    
    /// Last time a frame was processed, used for FPS calculation.
    private var lastFrameTime: TimeInterval = 0
    
    /// Frame processing times for smoothing.
    private var processingTimes: [Double] = []
    
    /// The size of the video frames.
    private var videoSize: CGSize = .zero
    
    /// The actual size of processed frames (may differ from video asset size)
    private var actualFrameSize: CGSize = .zero
    
    /// The long side of the video frame (for compatibility with CameraVideoSource).
    var longSide: CGFloat = 3
    
    /// The short side of the video frame (for compatibility with CameraVideoSource).
    var shortSide: CGFloat = 4
    
    /// Flag indicating if the frame size has been captured.
    var frameSizeCaptured = false
    
    /// The actual rect of the video content within the player layer (accounting for aspect ratio).
    private var videoContentRect: CGRect = .zero
    
    /// Scale factor for converting normalized coordinates to screen coordinates.
    private var videoToScreenScale: CGPoint = CGPoint(x: 1.0, y: 1.0)
    
    /// Offset for converting normalized coordinates to screen coordinates.
    private var videoToScreenOffset: CGPoint = .zero
    
    /// The URL of the current video.
    private(set) var videoURL: URL?
    
    /// Flag indicating if processing is active.
    private var isProcessing: Bool = false
    
    /// Flag indicating if inference should be performed on frames
    var inferenceOK: Bool = true
    
    /// The preview layer for displaying the source's visual output.
    private var _previewLayer: AVPlayerLayer?
    var previewLayer: AVCaptureVideoPreviewLayer? {
        return nil // AVPlayerLayer is used directly instead
    }
    
    /// Returns the player layer for displaying video.
    var playerLayer: AVPlayerLayer? {
        return _previewLayer
    }
    
    /// The source type identifier.
    var sourceType: FrameSourceType {
        return .videoFile
    }
    
    /// Video playback state tracking
    private var isRunning = false
    private var isProcessingFrame = false
    private var shouldForceStop = false
    
    // MARK: - Performance Metrics (similar to other frame sources)
    
    // Frame Processing State
    private var frameProcessingCount = 0
    private var frameProcessingTimestamps: [CFTimeInterval] = []
    private var isModelProcessing: Bool = false
    private var lastTriggerSource: String = "album"
    
    // Performance Timing Metrics
    private var lastFramePreparationTime: Double = 0
    private var lastConversionTime: Double = 0
    private var lastInferenceTime: Double = 0
    private var lastUITime: Double = 0
    private var lastTotalPipelineTime: Double = 0
    
    // FPS Calculation Arrays
    private var frameTimestamps: [CFTimeInterval] = []
    private var pipelineStartTimes: [CFTimeInterval] = []
    private var pipelineCompleteTimes: [CFTimeInterval] = []
    
    // MARK: - Initialization
    
    override init() {
        super.init()
        // Explicitly set inferenceOK to true to prevent automatic calibration
        inferenceOK = true
    }
    
    deinit {
        // Can't use Task in deinit, use Runtime hooks instead
        if let frameTimer = frameTimer {
            frameTimer.invalidate()
        }
    }
    
    // MARK: - FrameSource Protocol Methods
    
    /// Sets up the video source with the specified URL.
    ///
    /// - Parameters:
    ///   - url: The URL of the video to play.
    ///   - completion: Called when setup is complete, with a Boolean indicating success.
    @MainActor
    func setVideoURL(_ url: URL, completion: @escaping (Bool) -> Void) {
        cleanupResources()

        self.videoURL = url
        let asset = AVAsset(url: url)

        // ⚡ OPTIMIZATION: Load tracks asynchronously to prevent main thread blocking
        Task { @MainActor in
            do {
                // Load tracks asynchronously (non-blocking)
                let tracks = try await asset.loadTracks(withMediaType: .video)

                // Setup player for preview
                let playerItem = AVPlayerItem(asset: asset)
                self.player = AVPlayer(playerItem: playerItem)

                self._previewLayer = AVPlayerLayer(player: self.player)
                self._previewLayer?.videoGravity = .resizeAspect  // Use aspect to prevent distortion

                // Create a pixel buffer attributes dictionary
                let pixelBufferAttributes: [String: Any] = [
                    kCVPixelBufferPixelFormatTypeKey as String: FrameSourceSettings.videoSourcePixelFormat
                ]

                // Setup video output for frame extraction
                self.videoOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: pixelBufferAttributes)
                playerItem.add(self.videoOutput!)

                // Get video size and set up frame dimensions
                if let track = tracks.first {
                    // Load track properties asynchronously (non-blocking)
                    let videoTransform = try await track.load(.preferredTransform)
                    let naturalSize = try await track.load(.naturalSize)
                    let frameRateValue = try await track.load(.nominalFrameRate)

                    // Apply transform to get the correct dimensions
                    var transformedSize = naturalSize
                    if !videoTransform.isIdentity {
                        transformedSize = naturalSize.applying(videoTransform)
                    }

                    // Use absolute values as transform can make dimensions negative
                    self.videoSize = CGSize(width: abs(transformedSize.width), height: abs(transformedSize.height))

                    // Set longSide and shortSide for compatibility with camera frame handling
                    self.longSide = max(self.videoSize.width, self.videoSize.height)
                    self.shortSide = min(self.videoSize.width, self.videoSize.height)
                    self.frameSizeCaptured = true

                    // Try to get frame rate - remove 30fps cap for maximum processing speed
                    if frameRateValue > 0 {
                        self.frameRate = min(frameRateValue, 60.0) // Allow up to 60fps for optimal performance
                    }

                    // Initial setup of content rect
                    self.updateVideoContentRect()
                }

                // CRITICAL: Calculate coordinate transformation immediately
                self.updateVideoContentRect()
                print("Album: Video setup completed - size: \(self.videoSize), framerate: \(self.frameRate)")

                completion(true)

            } catch {
                print("Album: Error loading video: \(error)")
                completion(false)
            }
        }
    }
    
    /// Updates the calculated video content rect and scaling factors based on current layout
    @MainActor
    private func updateVideoContentRect() {
        guard let playerLayer = _previewLayer, videoSize.width > 0, videoSize.height > 0 else { return }
        
        let layerSize = playerLayer.bounds.size
        
        // Calculate aspect ratios
        let videoAspect = videoSize.width / videoSize.height
        let layerAspect = layerSize.width / layerSize.height
        
        // Calculate the rectangle of the video within the player layer
        var rect = CGRect.zero
        
        if videoAspect > layerAspect {
            // Video is wider than the layer (letterboxing - black bars on top and bottom)
            let height = layerSize.width / videoAspect
            let y = (layerSize.height - height) / 2
            rect = CGRect(x: 0, y: y, width: layerSize.width, height: height)
        } else {
            // Video is taller than the layer (pillarboxing - black bars on sides)
            let width = layerSize.height * videoAspect
            let x = (layerSize.width - width) / 2
            rect = CGRect(x: x, y: 0, width: width, height: layerSize.height)
        }
        
        // Store the calculated rect
        videoContentRect = rect
        
        // Calculate scale factors for converting normalized coordinates to screen
        videoToScreenScale = CGPoint(
            x: rect.width, 
            y: rect.height
        )
        
        // Calculate offset for converting normalized coordinates to screen
        videoToScreenOffset = CGPoint(
            x: rect.minX,
            y: rect.minY
        )
    }
    
    /// Configures video orientation and layout when device orientation changes
    @MainActor
    func updateForOrientationChange(orientation: UIDeviceOrientation) {
        // Update content rect based on new orientation
        updateVideoContentRect()
    }
    
    /// Sets up the frame source with default configuration.
    ///
    /// - Parameters:
    ///   - completion: Called when setup is complete, with a Boolean indicating success.
    @MainActor
    func setUp(completion: @escaping @Sendable (Bool) -> Void) {
        // Default implementation returns true since actual setup requires a video URL
        completion(true)
    }
    
    /// Begins frame acquisition from the source.
    nonisolated func start() {
        Task { @MainActor in
            // Check here instead of in the nonisolated context
            if videoURL == nil { return }
            startMainActorIsolated()
        }
    }
    
    /// Start method that runs on the MainActor
    @MainActor
    private func startMainActorIsolated() {
        // Start playback at normal speed for natural timing
        player?.play()  // Use normal 1x speed
        startFrameExtraction()
    }
    
    /// Stops frame acquisition from the source.
    nonisolated func stop() {
        Task { @MainActor in
            stopMainActorIsolated()
        }
    }
    
    /// Stop method that runs on the MainActor
    @MainActor
    private func stopMainActorIsolated() {
        player?.pause()
        stopFrameExtraction()
    }
    
    /// Sets the zoom level for the video (not supported).
    ///
    /// - Parameter ratio: The zoom ratio to apply.
    nonisolated func setZoomRatio(ratio: CGFloat) {
        // Zoom not supported for video playback
    }
    
    // MARK: - Video-specific FrameSource methods
    
    /// Implementation of the FrameSource protocol method to request photo library permission
    @MainActor
    func requestPermission(completion: @escaping (Bool) -> Void) {
        let status = PHPhotoLibrary.authorizationStatus()
        
        switch status {
        case .authorized, .limited:
            // Already authorized
            completion(true)
        case .notDetermined:
            // Request permission
            PHPhotoLibrary.requestAuthorization { newStatus in
                DispatchQueue.main.async {
                    completion(newStatus == .authorized || newStatus == .limited)
                }
            }
        case .denied, .restricted:
            // Permission denied
            completion(false)
        @unknown default:
            completion(false)
        }
    }
    
    /// Implementation of the FrameSource protocol method to show video picker
    @MainActor
    func showContentSelectionUI(from viewController: UIViewController, completion: @escaping (Bool) -> Void) {
        // First check for permission
        requestPermission { granted in
            if !granted {
                completion(false)
                return
            }
            
            // Configure and show picker
            DispatchQueue.main.async {
                let picker = UIImagePickerController()
                picker.delegate = self
                picker.sourceType = .photoLibrary
                picker.mediaTypes = ["public.movie"]
                picker.videoQuality = .typeHigh
                picker.allowsEditing = false
                
                // Store completion handler
                self.contentSelectionCompletion = completion
                
                viewController.present(picker, animated: true)
            }
        }
    }
    
    /// Restart the video from the beginning
    @MainActor
    func restartVideo() {
        stop()
        if let player = player {
            player.seek(to: CMTime.zero)
            start()
        }
    }
    
    // MARK: - ResultsListener & InferenceTimeListener Implementation
    
    func on(inferenceTime: Double, fpsRate: Double) {
        lastInferenceTime = inferenceTime
        
        // Forward to delegates on the main thread
        Task { @MainActor in
            self.videoCaptureDelegate?.onInferenceTime(speed: inferenceTime, fps: fpsRate)
            self.delegate?.frameSource(self, didUpdateWithSpeed: inferenceTime, fps: fpsRate)
        }
    }
    
    func on(result: YOLOResult) {
        let postProcessingStartTime = CACurrentMediaTime()
        let timestamp = CACurrentMediaTime()
        
        // Clear processing flag
        isModelProcessing = false
        
        // Update frame timestamps for FPS calculation
        frameTimestamps.append(timestamp)
        if frameTimestamps.count > 30 {
            frameTimestamps.removeFirst()
        }
        
        // Handle UI updates
        let uiDelegateStartTime = CACurrentMediaTime()
        
        // Forward results to VideoCaptureDelegate on the main thread
        Task { @MainActor in
            self.videoCaptureDelegate?.onPredict(result: result)
        }
        
        // Calculate timing metrics
        lastUITime = (CACurrentMediaTime() - uiDelegateStartTime) * 1000
        let totalUITime = (CACurrentMediaTime() - postProcessingStartTime) * 1000
        lastTotalPipelineTime = lastFramePreparationTime + lastConversionTime + lastInferenceTime + totalUITime
        
        // Update completion timestamps for throughput calculation
        pipelineCompleteTimes.append(timestamp)
        if pipelineCompleteTimes.count > 30 {
            pipelineCompleteTimes.removeFirst()
        }
        
        // Update delegate with throughput FPS
        let actualThroughputFPS = calculateThroughputFPS()
        Task { @MainActor in
            self.delegate?.frameSource(self, didUpdateWithSpeed: result.speed, fps: actualThroughputFPS)
        }
    }
    
    // MARK: - Private Methods
    
    /// Resets processing state to allow normal inference to resume after calibration
    @MainActor
    func resetProcessingState() {
        isProcessingFrame = false
        isModelProcessing = false
        print("Album: Processing state reset - ready for normal inference")
    }
    
    /// Completion handler for content selection UI
    private var contentSelectionCompletion: ((Bool) -> Void)?
    
    // Removed setupAssetReader - using unified synchronized approach
    
    @MainActor
    private func startFrameExtraction() {
        stopFrameExtraction() // Ensure any existing timer is invalidated
        
        isProcessing = true
        lastFrameTime = CACurrentMediaTime()
        
        // IMPROVED: Use display link for better synchronization instead of timer
        startDisplayLinkBasedExtraction()
    }
    
    @MainActor
    private func startDisplayLinkBasedExtraction() {
        // Create a display link synchronized with video playback for accuracy
        let displayLink = CADisplayLink(target: self, selector: #selector(extractFrameFromDisplayLink))
        // Use maximum refresh rate but sync with video timing
        displayLink.preferredFramesPerSecond = 0 // Maximum available, but sync with video
        displayLink.add(to: .main, forMode: .common)
        
        // Store the display link
        self.displayLink = displayLink
    }
    
    @MainActor @objc private func extractFrameFromDisplayLink() {
        guard isProcessing, let predictor = predictor, let videoOutput = videoOutput, let player = player else { 
            return 
        }
        
        let pipelineStartTime = CACurrentMediaTime()
        pipelineStartTimes.append(pipelineStartTime)
        if pipelineStartTimes.count > 30 {
            pipelineStartTimes.removeFirst()
        }
        
        frameProcessingCount += 1
        frameProcessingTimestamps.append(pipelineStartTime)
        if frameProcessingTimestamps.count > 60 {
            frameProcessingTimestamps.removeFirst()
        }
        
        // SYNCHRONIZED APPROACH: Always use video player time to ensure sync
        let playerTime = player.currentTime()
        
        // Check if video has reached the end
        guard let duration = player.currentItem?.duration, 
              playerTime < duration else {
            stopMainActorIsolated()
            NotificationCenter.default.post(name: .videoPlaybackDidEnd, object: self)
            return
        }
        
        // Get pixel buffer from video output at current playback time
        guard let pixelBuffer = videoOutput.copyPixelBuffer(forItemTime: playerTime, itemTimeForDisplay: nil) else {
            return
        }
        
        processPixelBuffer(pixelBuffer, pipelineStartTime: pipelineStartTime)
    }
    
    /// Processes a pixel buffer for both high-speed and fallback modes
    @MainActor
    private func processPixelBuffer(_ pixelBuffer: CVPixelBuffer, pipelineStartTime: CFTimeInterval) {
        guard let predictor = predictor else { return }
        
        let framePreparationStartTime = CACurrentMediaTime()
        
        // Update frame dimensions if needed (only once)
        if !frameSizeCaptured {
            let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
            longSide = max(frameWidth, frameHeight)
            shortSide = min(frameWidth, frameHeight)
            frameSizeCaptured = true
            actualFrameSize = CGSize(width: frameWidth, height: frameHeight)
            
            // CRITICAL: Ensure coordinate transformation is ready immediately
            updateVideoContentRect()
            print("Album: Frame size captured and coordinate transformation updated: \(Int(frameWidth))×\(Int(frameHeight))")
        }
        
        lastFramePreparationTime = (CACurrentMediaTime() - framePreparationStartTime) * 1000
        
        // OPTIMIZED: Create UIImage only for display, use pixelBuffer directly for inference
        let displayImage = createUIImageFromPixelBuffer(pixelBuffer)
        if let displayImage = displayImage {
            // Pass frame to delegate for display
            delegate?.frameSource(self, didOutputImage: displayImage)
        }
        
        // Handle calibration vs normal inference
        if !inferenceOK, let trackingDetector = predictor as? TrackingDetector {
            // Calibration mode - use pixel buffer directly
            trackingDetector.processFrame(pixelBuffer)
            
            if shouldLogPerformance(frameCount: frameProcessingCount) {
                logPipelineAnalysis(mode: "calibration")
            }
            return
        }
        
        // Normal inference mode - only if not already processing
        if inferenceOK && !isModelProcessing {
            let conversionStartTime = CACurrentMediaTime()
            
            // OPTIMIZED: Create sample buffer directly from pixel buffer
            if let sampleBuffer = createOptimizedSampleBuffer(from: pixelBuffer) {
                lastConversionTime = (CACurrentMediaTime() - conversionStartTime) * 1000
                
                isModelProcessing = true
                predictor.predict(sampleBuffer: sampleBuffer, onResultsListener: self, onInferenceTime: self)
            } else {
                lastConversionTime = (CACurrentMediaTime() - conversionStartTime) * 1000
            }
        }
        
        // Log performance analysis every 300 frames
        if shouldLogPerformance(frameCount: frameProcessingCount) {
            logPipelineAnalysis(mode: "inference")
        }
    }
    
    // OPTIMIZED: Direct pixel buffer to UIImage conversion
    private func createUIImageFromPixelBuffer(_ pixelBuffer: CVPixelBuffer) -> UIImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }
        
        return UIImage(cgImage: cgImage)
    }
    
    // OPTIMIZED: Direct pixel buffer to sample buffer conversion
    private func createOptimizedSampleBuffer(from pixelBuffer: CVPixelBuffer) -> CMSampleBuffer? {
        var formatDescription: CMFormatDescription?
        CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescriptionOut: &formatDescription
        )
        
        guard let formatDesc = formatDescription else { return nil }
        
        var sampleBuffer: CMSampleBuffer?
        var timingInfo = CMSampleTimingInfo()
        timingInfo.duration = CMTime.invalid
        timingInfo.presentationTimeStamp = CMTime(value: Int64(CACurrentMediaTime() * 1000), timescale: 1000)
        timingInfo.decodeTimeStamp = CMTime.invalid
        
        let status = CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: formatDesc,
            sampleTiming: &timingInfo,
            sampleBufferOut: &sampleBuffer
        )
        
        return status == noErr ? sampleBuffer : nil
    }
    
    @MainActor
    private func stopFrameExtraction() {
        frameTimer?.invalidate()
        frameTimer = nil
        displayLink?.invalidate()
        displayLink = nil
        isProcessing = false
    }
    
    @MainActor
    private func updatePerformanceMetrics(processingTime: Double, frameTime: Double) {
        // Maintain a rolling average of processing times
        processingTimes.append(processingTime)
        if processingTimes.count > 30 {
            processingTimes.removeFirst()
        }
        
        // Calculate average processing time
        let avgProcessingTime = processingTimes.reduce(0, +) / Double(processingTimes.count)
        
        // Calculate FPS based on frame time
        let fps = 1.0 / frameTime
        
        // Capture values before dispatching to main thread
        let capturedSpeed = avgProcessingTime * 1000
        let capturedFps = fps
        
        // Report metrics on the main thread
        Task { @MainActor in
            self.delegate?.frameSource(self, didUpdateWithSpeed: capturedSpeed, fps: capturedFps)
        }
    }
    
    @MainActor
    private func cleanup() {
        cleanupResources()
    }
    
    @MainActor
    private func cleanupResources() {
        // Stop playback and frame extraction
        stopMainActorIsolated()
        
        // Clean up resources
        player = nil
        videoOutput = nil
        _previewLayer = nil
        
        processingTimes.removeAll()
    }
    
    // Helper method to create CMSampleBuffer from UIImage
    private func createSampleBufferFrom(image: UIImage) -> CMSampleBuffer? {
        // Revert to original working implementation
        guard let cgImage = image.cgImage else { return nil }
        
        // Create a CVPixelBuffer to hold the image data
        var pixelBuffer: CVPixelBuffer?
        let width = cgImage.width
        let height = cgImage.height
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary
        
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &pixelBuffer)
        
        guard let pixel = pixelBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(pixel, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixel, .readOnly) }
        
        // Get the pixel data pointer
        let baseAddress = CVPixelBufferGetBaseAddress(pixel)
        
        // Setup a graphics context
        let context = CGContext(
            data: baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixel),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        )
        
        // Draw the image into the context
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Now create the sample buffer from the pixel buffer
        var formatDesc: CMFormatDescription?
        CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: pixel, formatDescriptionOut: &formatDesc)
        
        guard let format = formatDesc else { return nil }
        
        var sampleBuffer: CMSampleBuffer?
        var timingInfo = CMSampleTimingInfo()
        timingInfo.duration = CMTime.invalid
        timingInfo.presentationTimeStamp = CMTime.zero
        timingInfo.decodeTimeStamp = CMTime.invalid
        
        CMSampleBufferCreateForImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixel,
            dataReady: true,
            makeDataReadyCallback: nil,
            refcon: nil,
            formatDescription: format,
            sampleTiming: &timingInfo,
            sampleBufferOut: &sampleBuffer
        )
        
        return sampleBuffer
    }
    
    /// Create a standard pixel buffer for use with the tracking detector
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
    
    /// Converts normalized video coordinates to screen coordinates based on content rect
    @MainActor
    func convertNormalizedRectToScreenRect(_ normalizedRect: CGRect) -> CGRect {
        // For resizeAspectFill, we can use simple scaling since video fills the entire layer
        let x = normalizedRect.minX * videoToScreenScale.x + videoToScreenOffset.x
        
        // Y-coordinate with flipping for display consistency
        let y = (1.0 - normalizedRect.minY - normalizedRect.height) * videoToScreenScale.y + videoToScreenOffset.y
        
        let width = normalizedRect.width * videoToScreenScale.x
        let height = normalizedRect.height * videoToScreenScale.y
        
        return CGRect(x: x, y: y, width: width, height: height)
    }
    
    // MARK: - FrameSource Protocol Implementation for UI Integration
    
    @MainActor
    func integrateWithFishCountView(view: UIView) {
        // For video source, we need to add the player layer to the view's layer
        if let playerLayer = self._previewLayer {
            playerLayer.frame = view.bounds
            view.layer.insertSublayer(playerLayer, at: 0)
            
            // Ensure coordinate transformation is updated after integration
            // This fixes the initial box sync issue on iPad
            DispatchQueue.main.async { [weak self] in
                // Force layout update
                view.setNeedsLayout()
                view.layoutIfNeeded()
                
                // Update video content rect with proper bounds
                self?.updateVideoContentRect()
                
                // Additional layout update to ensure coordinate transformation is ready
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
                    self?.updateVideoContentRect()
                }
            }
        }
    }
    
    @MainActor
    func addOverlayLayer(_ layer: CALayer) {
        // Add the overlay layer to the player layer
        if let playerLayer = self._previewLayer {
            playerLayer.addSublayer(layer)
        }
    }
    
    @MainActor
    func addBoundingBoxViews(_ boxViews: [BoundingBoxView]) {
        // Add bounding box views to the player layer
        if let playerLayer = self._previewLayer {
            for box in boxViews {
                box.addToLayer(playerLayer)
            }
        }
    }
    
    // MARK: - Coordinate Transformation
    
    @MainActor
    func transformDetectionToScreenCoordinates(
        rect: CGRect,
        viewBounds: CGRect,
        orientation: UIDeviceOrientation
    ) -> CGRect {
        // Convert to unified coordinate system first
        let unifiedRect = toUnifiedCoordinates(rect)
        
        // Convert from unified to screen coordinates
        return UnifiedCoordinateSystem.toScreen(unifiedRect, screenBounds: viewBounds)
    }
    
    /// Converts album video detection coordinates to unified coordinate system
    /// - Parameter rect: Detection rectangle from album video (normalized Vision coordinates)
    /// - Returns: Rectangle in unified coordinate system
    @MainActor
    func toUnifiedCoordinates(_ rect: CGRect) -> UnifiedCoordinateSystem.UnifiedRect {
        // Album video detections come from Vision framework, so convert from Vision coordinates
        let visionRect = rect
        
        // Convert from Vision (bottom-left origin) to unified (top-left origin)
        let unifiedFromVision = UnifiedCoordinateSystem.fromVision(visionRect)
        
        // Get actual screen bounds from the player layer if available
        let actualDisplayBounds: CGRect
        if let playerLayer = _previewLayer {
            actualDisplayBounds = playerLayer.bounds
        } else {
            // Fallback to calculated bounds
            actualDisplayBounds = CGRect(origin: .zero, size: CGSize(width: videoToScreenScale.x, height: videoToScreenScale.y))
        }
        
        // Apply album-specific adjustments for aspect ratio and letterboxing
        return UnifiedCoordinateSystem.fromAlbum(
            unifiedFromVision.cgRect, 
            videoSize: actualFrameSize.width > 0 ? actualFrameSize : videoSize, 
            displayBounds: actualDisplayBounds
        )
    }
    
    // MARK: - Performance Analysis Methods
    
    /// Calculates actual pipeline throughput FPS based on complete processing cycles
    private func calculateThroughputFPS() -> Double {
        guard pipelineCompleteTimes.count >= 2 else { return 0 }
        let cyclesToConsider = min(10, pipelineCompleteTimes.count)
        let recentCompletions = Array(pipelineCompleteTimes.suffix(cyclesToConsider))
        let timeInterval = recentCompletions.last! - recentCompletions.first!
        return timeInterval > 0 ? Double(cyclesToConsider - 1) / timeInterval : 0
    }
    
    /// Logs comprehensive pipeline analysis every 300 frames
    private func logPipelineAnalysis(mode: String) {
        // Use unified performance logging interface
        guard shouldLogPerformance(frameCount: frameProcessingCount) else { return }
        
        // Calculate frame processing FPS from recent timestamps
        let processingFPS: Double
        if frameProcessingTimestamps.count >= 2 {
            let windowSeconds = frameProcessingTimestamps.last! - frameProcessingTimestamps.first!
            processingFPS = Double(frameProcessingTimestamps.count - 1) / windowSeconds
        } else {
            processingFPS = 0
        }
        
        // Calculate actual throughput FPS
        let actualThroughputFPS = calculateThroughputFPS()
        
        // Prepare timing data for unified logging
        let timingData: [String: Double] = [
            "preparation": lastFramePreparationTime,
            "conversion": lastConversionTime,
            "inference": lastInferenceTime,
            "ui": lastUITime,
            "total": lastTotalPipelineTime,
            "throughput": actualThroughputFPS
        ]
        
        logUnifiedPerformanceAnalysis(
            frameCount: frameProcessingCount,
            sourcePrefix: "Album",
            frameSize: actualFrameSize.width > 0 ? actualFrameSize : videoSize,
            processingFPS: processingFPS,
            mode: mode,
            timingData: timingData
        )
    }
}

// MARK: - UIImagePickerControllerDelegate
extension AlbumVideoSource: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    public func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        guard let mediaType = info[.mediaType] as? String,
              mediaType == "public.movie",
              let url = info[.mediaURL] as? URL else {
            picker.dismiss(animated: true)
            contentSelectionCompletion?(false)
            contentSelectionCompletion = nil
            return
        }

        // ⚡ OPTIMIZATION: Load video BEFORE dismissing picker to prevent black screen freeze
        // This way the picker stays visible while loading, preventing the freeze/black screen
        print("Album: Starting video setup while picker is still visible...")

        setVideoURL(url) { [weak self] success in
            guard let self = self else { return }

            // Now dismiss the picker after video is loaded
            picker.dismiss(animated: true) {
                // Call completion after dismiss animation completes
                self.contentSelectionCompletion?(success)
                self.contentSelectionCompletion = nil

                if success {
                    print("Album: Video loaded successfully, starting playback...")
                    // Start playback if setup was successful
                    self.start()
                } else {
                    print("Album: Video setup failed")
                }
            }
        }
    }
    
    public func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true)
        
        // Call completion with false to indicate cancellation
        contentSelectionCompletion?(false)
        contentSelectionCompletion = nil
    }
} 