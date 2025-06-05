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
    
    /// The asset reader for more efficient frame extraction.
    private var assetReader: AVAssetReader?
    
    /// The video track output from the asset reader.
    private var trackOutput: AVAssetReaderTrackOutput?
    
    /// Timer for controlling frame extraction rate.
    private var frameTimer: Timer?
    
    /// Target frame rate for playback.
    private var frameRate: Float = 30.0
    
    /// Last time a frame was processed, used for FPS calculation.
    private var lastFrameTime: TimeInterval = 0
    
    /// Frame processing times for smoothing.
    private var processingTimes: [Double] = []
    
    /// The size of the video frames.
    private var videoSize: CGSize = .zero
    
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
        
        // Setup player for preview
        let playerItem = AVPlayerItem(asset: asset)
        player = AVPlayer(playerItem: playerItem)
        
        _previewLayer = AVPlayerLayer(player: player)
        _previewLayer?.videoGravity = .resizeAspect  // Use aspect to prevent distortion
        
        // Create a pixel buffer attributes dictionary
        let pixelBufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: FrameSourceSettings.videoSourcePixelFormat
        ]
        
        // Setup video output for frame extraction
        videoOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: pixelBufferAttributes)
        playerItem.add(videoOutput!)
        
        // Get video size and set up frame dimensions
        if let track = asset.tracks(withMediaType: .video).first {
            // Get accurate video dimensions including the transform
            let videoTransform = track.preferredTransform
            let naturalSize = track.naturalSize
            
            // Apply transform to get the correct dimensions
            var transformedSize = naturalSize
            if !videoTransform.isIdentity {
                transformedSize = naturalSize.applying(videoTransform)
            }
            
            // Use absolute values as transform can make dimensions negative
            videoSize = CGSize(width: abs(transformedSize.width), height: abs(transformedSize.height))
            
            // Set longSide and shortSide for compatibility with camera frame handling
            longSide = max(videoSize.width, videoSize.height)
            shortSide = min(videoSize.width, videoSize.height)
            frameSizeCaptured = true
            
            // Try to get frame rate
            let frameRateValue = track.nominalFrameRate
            if frameRateValue > 0 {
                frameRate = min(frameRateValue, 30.0) // Cap at 30fps
            }
            
            // Initial setup of content rect
            updateVideoContentRect()
        }
        
        // Try to setup asset reader for more efficient extraction
        setupAssetReader(asset: asset)
        
        completion(true)
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
        player?.play()
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
        // Forward to delegates on the main thread
        Task { @MainActor in
            self.videoCaptureDelegate?.onInferenceTime(speed: inferenceTime, fps: fpsRate)
            self.delegate?.frameSource(self, didUpdateWithSpeed: inferenceTime, fps: fpsRate)
        }
    }
    
    func on(result: YOLOResult) {
        // Forward results to VideoCaptureDelegate on the main thread
        Task { @MainActor in
            self.videoCaptureDelegate?.onPredict(result: result)
        }
    }
    
    // MARK: - Private Methods
    
    /// Completion handler for content selection UI
    private var contentSelectionCompletion: ((Bool) -> Void)?
    
    @MainActor
    private func setupAssetReader(asset: AVAsset) {
        do {
            guard let videoTrack = asset.tracks(withMediaType: .video).first else { return }
            
            // Create asset reader
            assetReader = try AVAssetReader(asset: asset)
            
            // Configure reader with video track
            let outputSettings: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: FrameSourceSettings.videoSourcePixelFormat
            ]
            
            trackOutput = AVAssetReaderTrackOutput(
                track: videoTrack,
                outputSettings: outputSettings
            )
            
            if let trackOutput = trackOutput, assetReader?.canAdd(trackOutput) == true {
                assetReader?.add(trackOutput)
            }
            
            // Start reading
            assetReader?.startReading()
        } catch {
            print("Failed to setup asset reader: \(error)")
            assetReader = nil
            trackOutput = nil
        }
    }
    
    @MainActor
    private func startFrameExtraction() {
        stopFrameExtraction() // Ensure any existing timer is invalidated
        
        isProcessing = true
        lastFrameTime = CACurrentMediaTime()
        
        // Target interval between frames (1/frameRate seconds)
        let interval = 1.0 / Double(frameRate)
        
        // Create a timer that dispatches to the main actor
        frameTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            // Ensure we're on the main actor when the timer fires
            if let self = self {
                Task { @MainActor in
                    self.extractAndProcessNextFrame()
                }
            }
        }
        
        // Make sure the timer runs in common run loop modes (including when scrolling)
        frameTimer?.tolerance = interval * 0.1
        RunLoop.main.add(frameTimer!, forMode: .common)
    }
    
    @MainActor
    private func stopFrameExtraction() {
        frameTimer?.invalidate()
        frameTimer = nil
        isProcessing = false
    }
    
    @MainActor
    private func extractAndProcessNextFrame() {
        guard isProcessing, let predictor = predictor else { return }
        
        // Measure frame time for FPS calculation
        let currentTime = CACurrentMediaTime()
        let deltaTime = currentTime - lastFrameTime
        lastFrameTime = currentTime
        
        // Try to get frame from asset reader first (more efficient)
        if let frame = getNextFrameFromAssetReader() {
            // Always pass the frame to the delegate for display
            processFrame(frame)
            
            // Check if we need to handle calibration
            if !inferenceOK, let trackingDetector = predictor as? TrackingDetector,
               let pixelBuffer = createStandardPixelBuffer(from: frame, forSourceType: .videoFile) {
                
                // Already on main actor, can call directly
                trackingDetector.processFrame(pixelBuffer)
                
                // Skip regular inference during calibration
                return
            }
            // Only perform normal inference if inferenceOK is true
            else if inferenceOK, let sampleBuffer = createSampleBufferFrom(image: frame) {
                // Process with predictor - using self as the results and inference time listener
                predictor.predict(sampleBuffer: sampleBuffer, onResultsListener: self, onInferenceTime: self)
            }
            
            return
        }
        
        // Fallback to getting frame from video output if asset reader failed
        guard let videoOutput = videoOutput,
              let player = player else { return }
        
        // Get the current playback time
        let playerTime = player.currentTime()
        
        // Check if a new pixel buffer is available
        if let pixelBuffer = videoOutput.copyPixelBuffer(forItemTime: playerTime, itemTimeForDisplay: nil) {
            // Update frame dimensions if needed
            if !frameSizeCaptured {
                let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
                let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
                longSide = max(frameWidth, frameHeight)
                shortSide = min(frameWidth, frameHeight)
                frameSizeCaptured = true
            }
            
            // Convert CVPixelBuffer to UIImage for display
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let context = CIContext()
            if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
                let image = UIImage(cgImage: cgImage)
                
                // Pass the frame to delegate for display
                Task { @MainActor in
                    self.delegate?.frameSource(self, didOutputImage: image)
                }
                
                // Check if we need to handle calibration
                if !inferenceOK, let trackingDetector = predictor as? TrackingDetector {
                    // Already on main actor, can call directly
                    trackingDetector.processFrame(pixelBuffer)
                    
                    // Skip regular inference during calibration
                    return
                }
                // Only perform normal inference if inferenceOK is true
                else if inferenceOK {
                    // Create sample buffer for inference
                    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
                    let context = CIContext()
                    if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
                        let image = UIImage(cgImage: cgImage)
                        if let sampleBuffer = createSampleBufferFrom(image: image) {
                            predictor.predict(sampleBuffer: sampleBuffer, onResultsListener: self, onInferenceTime: self)
                        }
                    } else {
                        // Track and report performance metrics only if we didn't succeed with prediction
                        let processingTime = CACurrentMediaTime() - currentTime
                        updatePerformanceMetrics(processingTime: processingTime, frameTime: deltaTime)
                    }
                }
            }
        }
        
        // Check if video has reached the end
        if player.currentTime() >= player.currentItem?.duration ?? CMTime.zero {
            // Stop playback and processing when video ends
            stopMainActorIsolated()
            
            // Notify that playback has completed via a notification
            NotificationCenter.default.post(name: .videoPlaybackDidEnd, object: self)
        }
    }
    
    @MainActor
    private func getNextFrameFromAssetReader() -> UIImage? {
        guard let trackOutput = trackOutput,
              let sampleBuffer = trackOutput.copyNextSampleBuffer(),
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return nil
        }
        
        // Convert CVPixelBuffer to UIImage using original working method
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            let image = UIImage(cgImage: cgImage)
            return image
        }
        
        return nil
    }
    
    @MainActor
    private func processFrame(_ image: UIImage) {
        // Capture image before dispatching to main thread
        let capturedImage = image
        
        // Pass the frame to delegate
        Task { @MainActor in
            self.delegate?.frameSource(self, didOutputImage: capturedImage)
        }
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
        
        // Clean up asset reader resources
        assetReader?.cancelReading()
        assetReader = nil
        trackOutput = nil
        
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
        // Calculate the x coordinate - simple scaling and offset
        let x = normalizedRect.minX * videoToScreenScale.x + videoToScreenOffset.x
        
        // Calculate y-coordinate with flipping for display consistency
        // This ensures proper display orientation for bounding boxes
        let y = (1.0 - normalizedRect.minY - normalizedRect.height) * videoToScreenScale.y + videoToScreenOffset.y
        
        let width = normalizedRect.width * videoToScreenScale.x
        let height = normalizedRect.height * videoToScreenScale.y
        
        return CGRect(x: x, y: y, width: width, height: height)
    }
    
    @MainActor
    private func processVideo() {
        guard let currentFrame = getNextFrameFromAssetReader() else {
            return
        }
        
        // Note the current time before processing
        let startTime = CACurrentMediaTime()
        
        // Process flag to prevent reentrance
        var isCurrentlyProcessing = false
        
        // Avoid multiple simultaneous video processing operations
        if !isCurrentlyProcessing {
            isCurrentlyProcessing = true
            
            // Pause normal inference if in calibration mode
            if !inferenceOK, let trackingDetector = predictor as? TrackingDetector,
               let pixelBuffer = createStandardPixelBuffer(from: currentFrame, forSourceType: .videoFile) {
                
                // Clear any existing bounding boxes
                DispatchQueue.main.async { [weak self] in
                    // Explicitly call onClearBoxes to ensure all boxes are cleared 
                    self?.videoCaptureDelegate?.onClearBoxes()
                    
                    // Additionally, if this is the first frame in calibration,
                    // make sure to clear any lingering boxes in the UI
                    if trackingDetector.getCalibrationFrameCount() == 0 {
                        self?.videoCaptureDelegate?.onClearBoxes()
                    }
                }
                
                // Process frame for calibration
                trackingDetector.processFrame(pixelBuffer)
                
                // Very important: Skip further processing to ensure no boxes are shown during calibration
                isCurrentlyProcessing = false
                return
            } 
            // Only perform normal inference if inferenceOK is true
            else if inferenceOK, let sampleBuffer = createSampleBufferFrom(image: currentFrame) {
                // Process with predictor - using self as the results and inference time listener
                predictor.predict(sampleBuffer: sampleBuffer, onResultsListener: self, onInferenceTime: self)
            }
            
            // Display the frame regardless of inference state
            processFrame(currentFrame)
            
            // Calculate processing time and update metrics
            let endTime = CACurrentMediaTime()
            let processingTime = endTime - startTime
            let frameTime = processingTime  // For recorded video, frame time equals processing time
            
            updatePerformanceMetrics(processingTime: processingTime, frameTime: frameTime)
            
            isCurrentlyProcessing = false
            
            // Schedule the next frame - use a proper delay for smoother playback
            let targetFrameTime: Double = 1.0 / 30.0  // Target 30 fps
            let nextFrameDelay = max(0.001, targetFrameTime - processingTime)  // Ensure positive delay
            
            DispatchQueue.main.asyncAfter(deadline: .now() + nextFrameDelay) { [weak self] in
                self?.processVideo()
            }
        }
    }
    
    // MARK: - FrameSource Protocol Implementation for UI Integration
    
    @MainActor
    func integrateWithYOLOView(view: UIView) {
        // For video source, we need to add the player layer to the view's layer
        if let playerLayer = self._previewLayer {
            playerLayer.frame = view.bounds
            view.layer.insertSublayer(playerLayer, at: 0)
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
        // For video source, we need to use the videoContentRect to account for letterboxing/pillarboxing
        // Use our existing method that already handles this conversion
        return convertNormalizedRectToScreenRect(rect)
    }
}

// MARK: - UIImagePickerControllerDelegate
extension AlbumVideoSource: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    public func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        picker.dismiss(animated: true)
        
        guard let mediaType = info[.mediaType] as? String,
              mediaType == "public.movie",
              let url = info[.mediaURL] as? URL else {
            contentSelectionCompletion?(false)
            contentSelectionCompletion = nil
            return
        }
        
        // Setup the video source with the selected URL
        setVideoURL(url) { success in
            self.contentSelectionCompletion?(success)
            self.contentSelectionCompletion = nil
            
            if success {
                // Start playback if setup was successful
                self.start()
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