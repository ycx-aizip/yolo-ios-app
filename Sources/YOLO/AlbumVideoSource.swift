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
import UIKit
import Vision

/// Frame source implementation that plays videos from the photo library.
@preconcurrency
class AlbumVideoSource: NSObject, FrameSource {
    /// The delegate to receive frames and performance metrics.
    weak var delegate: FrameSourceDelegate?
    
    /// The predictor used to process frames from this source.
    var predictor: Predictor!
    
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
    
    /// The URL of the current video.
    private(set) var videoURL: URL?
    
    /// Flag indicating if processing is active.
    private var isProcessing: Bool = false
    
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
    
    // MARK: - Initialization
    
    override init() {
        super.init()
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
        _previewLayer?.videoGravity = .resizeAspectFill
        
        // Create a pixel buffer attributes dictionary
        let pixelBufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        
        // Setup video output for frame extraction
        videoOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: pixelBufferAttributes)
        playerItem.add(videoOutput!)
        
        // Get video size
        if let track = asset.tracks(withMediaType: .video).first {
            videoSize = track.naturalSize
            
            // Try to get frame rate
            let frameRateValue = track.nominalFrameRate
            if frameRateValue > 0 {
                frameRate = min(frameRateValue, 30.0) // Cap at 30fps
            }
        }
        
        // Try to setup asset reader for more efficient extraction
        setupAssetReader(asset: asset)
        
        completion(true)
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
    
    /// Captures a still image from the video.
    ///
    /// - Parameters:
    ///   - completion: Callback with the captured image, or nil if capture failed.
    func capturePhoto(completion: @escaping @Sendable (UIImage?) -> Void) {
        // For video, we'll just grab the current frame
        guard let videoOutput = videoOutput, let player = player else {
            completion(nil)
            return
        }
        
        let currentTime = player.currentTime()
        if let pixelBuffer = videoOutput.copyPixelBuffer(forItemTime: currentTime, itemTimeForDisplay: nil) {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let context = CIContext()
            if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
                let image = UIImage(cgImage: cgImage)
                completion(image)
                return
            }
        }
        
        completion(nil)
    }
    
    /// Sets the zoom level for the video (not supported).
    ///
    /// - Parameter ratio: The zoom ratio to apply.
    nonisolated func setZoomRatio(ratio: CGFloat) {
        // Zoom not supported for video playback
    }
    
    // MARK: - Private Methods
    
    @MainActor
    private func setupAssetReader(asset: AVAsset) {
        do {
            guard let videoTrack = asset.tracks(withMediaType: .video).first else { return }
            
            // Create asset reader
            assetReader = try AVAssetReader(asset: asset)
            
            // Configure reader with video track
            let outputSettings: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
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
            processFrame(frame)
            return
        }
        
        // Fallback to getting frame from video output if asset reader failed
        guard let videoOutput = videoOutput,
              let player = player else { return }
        
        // Get the current playback time
        let playerTime = player.currentTime()
        
        // Check if a new pixel buffer is available
        if let pixelBuffer = videoOutput.copyPixelBuffer(forItemTime: playerTime, itemTimeForDisplay: nil) {
            // Convert CVPixelBuffer to UIImage
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let context = CIContext()
            if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
                let image = UIImage(cgImage: cgImage)
                
                // Capture necessary values before dispatching to main thread
                let capturedImage = image
                let capturedTime = currentTime
                let capturedDeltaTime = deltaTime
                
                // Pass the frame to delegate on the main thread
                Task { @MainActor in
                    self.delegate?.frameSource(self, didOutputImage: capturedImage)
                    
                    // Track and report performance
                    let processingTime = CACurrentMediaTime() - capturedTime
                    updatePerformanceMetrics(processingTime: processingTime, frameTime: capturedDeltaTime)
                }
            }
        }
        
        // Check if video has reached the end
        if player.currentTime() >= player.currentItem?.duration ?? CMTime.zero {
            // Loop playback
            player.seek(to: CMTime.zero)
            player.play()
        }
    }
    
    @MainActor
    private func getNextFrameFromAssetReader() -> UIImage? {
        guard let trackOutput = trackOutput,
              let sampleBuffer = trackOutput.copyNextSampleBuffer(),
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return nil
        }
        
        // Convert to UIImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }
        
        return UIImage(cgImage: cgImage)
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
} 