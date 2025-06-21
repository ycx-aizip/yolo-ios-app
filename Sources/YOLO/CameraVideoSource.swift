// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
//  This file is part of the Ultralytics YOLO Package, managing camera capture for real-time inference.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The CameraVideoSource component manages the camera and video processing pipeline for real-time
//  object detection. It handles setting up the AVCaptureSession, managing camera devices,
//  configuring camera properties like focus and exposure, and processing video frames for
//  model inference. The class delivers capture frames to the predictor component for real-time
//  analysis and returns results through delegate callbacks. It also supports camera controls
//  such as switching between front and back cameras, zooming, and capturing still photos.

import AVFoundation
import CoreVideo
import UIKit
import Vision

/// Protocol for receiving video capture frame processing results.
@MainActor
protocol VideoCaptureDelegate: AnyObject {
  func onPredict(result: YOLOResult)
  func onInferenceTime(speed: Double, fps: Double)
  func onClearBoxes()
}

func bestCaptureDevice(position: AVCaptureDevice.Position) -> AVCaptureDevice {
  // print("USE TELEPHOTO: ")
  // print(UserDefaults.standard.bool(forKey: "use_telephoto"))

  if UserDefaults.standard.bool(forKey: "use_telephoto"),
    let device = AVCaptureDevice.default(.builtInTelephotoCamera, for: .video, position: position)
  {
    return device
  } else if let device = AVCaptureDevice.default(
    .builtInDualCamera, for: .video, position: position)
  {
    return device
  } else if let device = AVCaptureDevice.default(
    .builtInWideAngleCamera, for: .video, position: position)
  {
    return device
  } else {
    fatalError("Missing expected back camera device.")
  }
}

@preconcurrency
class CameraVideoSource: NSObject, FrameSource, @unchecked Sendable {
  var predictor: FrameProcessor!
  var previewLayer: AVCaptureVideoPreviewLayer?
  weak var videoCaptureDelegate: VideoCaptureDelegate?
  weak var frameSourceDelegate: FrameSourceDelegate?
  var captureDevice: AVCaptureDevice?
  let captureSession = AVCaptureSession()
  var videoInput: AVCaptureDeviceInput? = nil
  let videoOutput = AVCaptureVideoDataOutput()
  let cameraQueue = DispatchQueue(label: "camera-queue")
  var inferenceOK = true
  var longSide: CGFloat = 3
  var shortSide: CGFloat = 4
  var frameSizeCaptured = false
  
  // Implement FrameSource protocol property
  var delegate: FrameSourceDelegate? {
    get { return frameSourceDelegate }
    set { frameSourceDelegate = newValue }
  }
  
  // Implement FrameSource protocol property
  var sourceType: FrameSourceType { return .camera }

  private var currentBuffer: CVPixelBuffer?
  
  // MARK: - Performance Metrics (similar to GoProSource)
  
  // Frame Processing State
  private var frameProcessingCount = 0
  private var frameProcessingTimestamps: [CFTimeInterval] = []
  private var isModelProcessing: Bool = false
  private var lastTriggerSource: String = "camera"
  
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
  
  // MARK: - Device-Specific Camera Configuration
  
  /// Determines the optimal camera session preset for YOLO model performance
  /// 
  /// All devices now use HD 720p (1280×720) for optimal YOLO processing:
  /// - YOLO model uses 640×640 input, so 1280×720 provides perfect 2:1 scaling
  /// - Significantly reduces processing overhead compared to higher resolutions
  /// - Maintains excellent visual quality through resizeAspectFill display scaling
  /// - Previous performance issue was due to .photo preset using ~4M+ pixels vs 720p's ~0.9M pixels
  @MainActor
  private var optimalSessionPreset: AVCaptureSession.Preset {
    // Unified resolution for optimal YOLO performance across all devices
    return .hd1280x720  // 1280×720 resolution - perfect for 640×640 YOLO input
  }
  
  /// Gets a human-readable description of the current session preset
  @MainActor
  private var sessionPresetDescription: String {
    switch optimalSessionPreset {
    case .hd1280x720:
      return "HD 1280×720 (Optimized for YOLO)"
    case .hd1920x1080:
      return "Full HD 1920×1080"
    case .photo:
      return "Photo (4032×3024+)"
    default:
      return "Unknown"
    }
  }
  
  // Default initializer
  override init() {
    super.init()
  }

  func setUp(
    sessionPreset: AVCaptureSession.Preset? = nil,
    position: AVCaptureDevice.Position,
    orientation: UIDeviceOrientation,
    completion: @escaping @MainActor @Sendable (Bool) -> Void
  ) {
    // Get the preset on main actor if needed
    Task { @MainActor in
      let preset = sessionPreset ?? self.optimalSessionPreset
      
      // Move to background queue for camera setup
      Task.detached {
        let success = self.setUpCamera(
          sessionPreset: preset, position: position, orientation: orientation)
        
        // Call completion on main queue
        await MainActor.run {
          completion(success)
        }
      }
    }
  }
  
  // Simplified version for FrameSource protocol compatibility
  @MainActor
  func setUp(completion: @escaping @Sendable (Bool) -> Void) {
    // Capture orientation and preset on the main actor
    let orientation = UIDevice.current.orientation
    let preset = self.optimalSessionPreset
    
    // Then dispatch to background queue for camera setup
    Task.detached {
      let success = self.setUpCamera(
        sessionPreset: preset, 
        position: .back, 
        orientation: orientation)
      
      // Call completion on main actor
      await MainActor.run {
        completion(success)
      }
    }
  }

  func setUpCamera(
    sessionPreset: AVCaptureSession.Preset, position: AVCaptureDevice.Position,
    orientation: UIDeviceOrientation
  ) -> Bool {
    captureSession.beginConfiguration()
    captureSession.sessionPreset = sessionPreset
    
    // Log the device type and resolution being used for performance analysis
    Task { @MainActor in
      let deviceType = UIDevice.current.userInterfaceIdiom == .pad ? "iPad" : "iPhone"
      let presetDescription = self.sessionPresetDescription
      print("Camera: Configuring \(deviceType) with \(presetDescription) for unified YOLO optimization")
    }

    captureDevice = bestCaptureDevice(position: position)
    videoInput = try! AVCaptureDeviceInput(device: captureDevice!)

    if captureSession.canAddInput(videoInput!) {
      captureSession.addInput(videoInput!)
    }
    var videoOrientaion = AVCaptureVideoOrientation.portrait
    switch orientation {
    case .portrait:
      videoOrientaion = .portrait
    case .landscapeLeft:
      videoOrientaion = .landscapeRight
    case .landscapeRight:
      videoOrientaion = .landscapeLeft
    default:
      videoOrientaion = .portrait
    }
    let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
    previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
    previewLayer.connection?.videoOrientation = videoOrientaion
    self.previewLayer = previewLayer

    let settings: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
    ]

    videoOutput.videoSettings = settings
    videoOutput.alwaysDiscardsLateVideoFrames = true
    videoOutput.setSampleBufferDelegate(self, queue: cameraQueue)
    if captureSession.canAddOutput(videoOutput) {
      captureSession.addOutput(videoOutput)
    }

    // We want the buffers to be in portrait orientation otherwise they are
    // rotated by 90 degrees. Need to set this _after_ addOutput()!
    // let curDeviceOrientation = UIDevice.current.orientation
    let connection = videoOutput.connection(with: AVMediaType.video)
    connection?.videoOrientation = videoOrientaion
    if position == .front {
      connection?.isVideoMirrored = true
    }

    // Configure captureDevice
    do {
      try captureDevice!.lockForConfiguration()
    } catch {
      print("device configuration not working")
    }
    // captureDevice.setFocusModeLocked(lensPosition: 1.0, completionHandler: { (time) -> Void in })
    if captureDevice!.isFocusModeSupported(AVCaptureDevice.FocusMode.continuousAutoFocus),
      captureDevice!.isFocusPointOfInterestSupported
    {
      captureDevice!.focusMode = AVCaptureDevice.FocusMode.continuousAutoFocus
      captureDevice!.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
    }
    captureDevice!.exposureMode = AVCaptureDevice.ExposureMode.continuousAutoExposure
    captureDevice!.unlockForConfiguration()

    captureSession.commitConfiguration()
    return true
  }

  nonisolated func start() {
    if !captureSession.isRunning {
      DispatchQueue.global().async {
        self.captureSession.startRunning()
      }
    }
  }

  nonisolated func stop() {
    if captureSession.isRunning {
      DispatchQueue.global().async {
        self.captureSession.stopRunning()
      }
    }
  }

  nonisolated func setZoomRatio(ratio: CGFloat) {
    do {
      try captureDevice!.lockForConfiguration()
      defer {
        captureDevice!.unlockForConfiguration()
      }
      captureDevice!.videoZoomFactor = ratio
    } catch {}
  }
  
  // MARK: - State Management
  
  /// Resets processing state to allow normal inference to resume after calibration
  @MainActor
  func resetProcessingState() {
    isModelProcessing = false
    print("Camera: Processing state reset - ready for normal inference")
  }
  
  private func processFrameOnCameraQueue(sampleBuffer: CMSampleBuffer) {
    guard let predictor = predictor else {
      print("predictor is nil")
      return
    }
    
    let pipelineStartTime = CACurrentMediaTime()
    pipelineStartTimes.append(pipelineStartTime)
    if pipelineStartTimes.count > 30 {
      pipelineStartTimes.removeFirst()
    }
    
    if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
      // Update frame size if needed
      if !frameSizeCaptured {
        let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        longSide = max(frameWidth, frameHeight)
        shortSide = min(frameWidth, frameHeight)
        frameSizeCaptured = true
        
        // Log actual captured frame dimensions for performance analysis
        Task { @MainActor in
          let deviceType = UIDevice.current.userInterfaceIdiom == .pad ? "iPad" : "iPhone"
          let presetDescription = self.sessionPresetDescription
          print("Camera: \(deviceType) actual frame size captured: \(Int(frameWidth))×\(Int(frameHeight)) (unified 720p)")
        }
      }
      
      frameProcessingCount += 1
      frameProcessingTimestamps.append(CACurrentMediaTime())
      if frameProcessingTimestamps.count > 60 {
        frameProcessingTimestamps.removeFirst()
      }
      
      // For frameSourceDelegate, we need to safely pass the frame data to the main thread
      let framePreparationStartTime = CACurrentMediaTime()
      
      // Create a CIImage from the pixel buffer which can be safely passed between threads
      if let frameSourceDelegate = frameSourceDelegate {
        // Create a CIImage from the pixel buffer - this is thread-safe
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // Convert to UIImage using original working method
        let context = CIContext()
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
          let uiImage = UIImage(cgImage: cgImage)
          
          // Dispatch to main thread with the thread-safe image
          DispatchQueue.main.async {
            // Use the new delegate method that accepts UIImage
            self.frameSourceDelegate?.frameSource(self, didOutputImage: uiImage)
          }
        }
      }
      
      lastFramePreparationTime = (CACurrentMediaTime() - framePreparationStartTime) * 1000
      
      // Check if we should process for calibration
      // Phase 1 (threshold detection): Use special calibration processing (!inferenceOK)
      // Phase 2 (movement analysis): Use normal YOLO inference (inferenceOK)
      let isTrackingDetector = predictor is TrackingDetector
      let shouldProcessForThresholdCalibration = !inferenceOK && isTrackingDetector
      
      if shouldProcessForThresholdCalibration {
        let conversionStartTime = CACurrentMediaTime()
        
        // Create a UIImage from the pixel buffer to safely pass across threads
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
          let image = UIImage(cgImage: cgImage)
          lastConversionTime = (CACurrentMediaTime() - conversionStartTime) * 1000
          
          // Process on main thread for auto-calibration
          DispatchQueue.main.async { [weak self] in
            guard let self = self,
                  let trackingDetector = self.predictor as? TrackingDetector else { return }
            
            if trackingDetector.getCalibrationFrameCount() == 0 {
              self.videoCaptureDelegate?.onClearBoxes()
            }
                  
            if let pixelBufferForCalibration = self.createStandardPixelBuffer(from: image, forSourceType: self.sourceType) {
              trackingDetector.processFrame(pixelBufferForCalibration)
            }
          }
        } else {
          lastConversionTime = (CACurrentMediaTime() - conversionStartTime) * 1000
        }
        
        // Log performance analysis every 300 frames during calibration
        if frameProcessingCount % 300 == 0 {
          logPipelineAnalysis(mode: "calibration")
        }
        
        return
      }
      
      if inferenceOK && !isModelProcessing {
        isModelProcessing = true
        predictor.predict(sampleBuffer: sampleBuffer, onResultsListener: self, onInferenceTime: self)
      }
      
      // Log performance analysis every 300 frames
      if frameProcessingCount % 300 == 0 {
        logPipelineAnalysis(mode: "inference")
      }
    }
  }

  func updateVideoOrientation(orientation: AVCaptureVideoOrientation) {
    guard let connection = videoOutput.connection(with: .video) else { return }

    connection.videoOrientation = orientation
    let currentInput = self.captureSession.inputs.first as? AVCaptureDeviceInput
    if currentInput?.device.position == .front {
      connection.isVideoMirrored = true
    } else {
      connection.isVideoMirrored = false
    }
    
    // No need to store the orientation in a variable
    self.previewLayer?.connection?.videoOrientation = connection.videoOrientation
  }
  
  // MARK: - Camera-specific FrameSource methods
  
  /// Implementation of the FrameSource protocol method to request camera permission
  @MainActor
  func requestPermission(completion: @escaping (Bool) -> Void) {
    switch AVCaptureDevice.authorizationStatus(for: .video) {
    case .authorized:
      // Already authorized
      completion(true)
    case .notDetermined:
      // Request permission
      AVCaptureDevice.requestAccess(for: .video) { granted in
        DispatchQueue.main.async {
          completion(granted)
        }
      }
    case .denied, .restricted:
      // Permission denied
      completion(false)
    @unknown default:
      completion(false)
    }
  }
  
  /// Implementation of the FrameSource protocol method to handle orientation changes
  @MainActor
  func updateForOrientationChange(orientation: UIDeviceOrientation) {
    var videoOrientation: AVCaptureVideoOrientation = .portrait
    
    switch orientation {
    case .portrait:
      videoOrientation = .portrait
    case .portraitUpsideDown:
      videoOrientation = .portraitUpsideDown
    case .landscapeLeft:
      videoOrientation = .landscapeRight
    case .landscapeRight:
      videoOrientation = .landscapeLeft
    default:
      return
    }
    
    updateVideoOrientation(orientation: videoOrientation)
  }
  
  /// Switch between front and back cameras
  @MainActor
  func switchCamera() {
    self.captureSession.beginConfiguration()
    let currentInput = self.captureSession.inputs.first as? AVCaptureDeviceInput
    self.captureSession.removeInput(currentInput!)
    guard let currentPosition = currentInput?.device.position else { return }

    let nextCameraPosition: AVCaptureDevice.Position = currentPosition == .back ? .front : .back

    let newCameraDevice = bestCaptureDevice(position: nextCameraPosition)

    guard let videoInput1 = try? AVCaptureDeviceInput(device: newCameraDevice) else {
      return
    }

    self.captureSession.addInput(videoInput1)
    
    // Get current orientation
    var orientation: AVCaptureVideoOrientation = .portrait
    switch UIDevice.current.orientation {
    case .portrait:
      orientation = .portrait
    case .portraitUpsideDown:
      orientation = .portraitUpsideDown
    case .landscapeRight:
      orientation = .landscapeLeft
    case .landscapeLeft:
      orientation = .landscapeRight
    default:
      orientation = .portrait
    }
    
    self.updateVideoOrientation(orientation: orientation)
    self.captureSession.commitConfiguration()
  }
  
  // MARK: - FrameSource Protocol Implementation for UI Integration
  
  @MainActor
  func integrateWithYOLOView(view: UIView) {
    // For camera source, we need to add the preview layer to the view's layer
    if let previewLayer = self.previewLayer {
      view.layer.insertSublayer(previewLayer, at: 0)
      previewLayer.frame = view.bounds
    }
  }
  
  @MainActor
  func addOverlayLayer(_ layer: CALayer) {
    // Add the overlay layer to the preview layer
    if let previewLayer = self.previewLayer {
      previewLayer.addSublayer(layer)
    }
  }
  
  @MainActor
  func addBoundingBoxViews(_ boxViews: [BoundingBoxView]) {
    // Add bounding box views to the preview layer
    if let previewLayer = self.previewLayer {
      for box in boxViews {
        box.addToLayer(previewLayer)
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
  
  /// Converts camera detection coordinates to unified coordinate system
  /// - Parameter rect: Detection rectangle from camera (normalized Vision coordinates)
  /// - Returns: Rectangle in unified coordinate system
  @MainActor
  func toUnifiedCoordinates(_ rect: CGRect) -> UnifiedCoordinateSystem.UnifiedRect {
    // Camera detections come from Vision framework, so convert from Vision coordinates
    let visionRect = rect
    
    // Convert from Vision (bottom-left origin) to unified (top-left origin)
    let unifiedFromVision = UnifiedCoordinateSystem.fromVision(visionRect)
    
    // For camera with resizeAspectFill, we need to account for cropping
    if let previewLayer = self.previewLayer {
      let cameraSize = CGSize(width: longSide, height: shortSide) // Actual camera frame size
      let displayBounds = previewLayer.bounds
      
      // Use specialized camera transformation that handles resizeAspectFill cropping
      return UnifiedCoordinateSystem.fromCameraWithAspectFill(
        unifiedFromVision.cgRect, 
        cameraSize: cameraSize, 
        displayBounds: displayBounds
      )
    } else {
      // Fallback to basic camera transformation if preview layer not available
      return UnifiedCoordinateSystem.fromCamera(unifiedFromVision.cgRect, sessionPreset: captureSession.sessionPreset)
    }
  }
}

extension CameraVideoSource: AVCaptureVideoDataOutputSampleBufferDelegate {
  func captureOutput(
    _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    // Always process frames, regardless of inferenceOK state
    // inferenceOK will control whether to run normal inference or calibration
    processFrameOnCameraQueue(sampleBuffer: sampleBuffer)
  }
}

extension CameraVideoSource: ResultsListener, InferenceTimeListener {
  func on(inferenceTime: Double, fpsRate: Double) {
    lastInferenceTime = inferenceTime
    
    DispatchQueue.main.async {
      self.videoCaptureDelegate?.onInferenceTime(speed: inferenceTime, fps: fpsRate)
      self.frameSourceDelegate?.frameSource(self, didUpdateWithSpeed: inferenceTime, fps: fpsRate)
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
    
    DispatchQueue.main.async {
      self.videoCaptureDelegate?.onPredict(result: result)
    }
    
    // Calculate timing metrics
    lastUITime = (CACurrentMediaTime() - uiDelegateStartTime) * 1000
    let totalUITime = (CACurrentMediaTime() - postProcessingStartTime) * 1000
    lastTotalPipelineTime = lastFramePreparationTime + lastInferenceTime + totalUITime
    
    // Update completion timestamps for throughput calculation
    pipelineCompleteTimes.append(timestamp)
    if pipelineCompleteTimes.count > 30 {
      pipelineCompleteTimes.removeFirst()
    }
    
    // Update delegate with throughput FPS
    let actualThroughputFPS = calculateThroughputFPS()
    DispatchQueue.main.async {
      self.frameSourceDelegate?.frameSource(self, didUpdateWithSpeed: result.speed, fps: actualThroughputFPS)
    }
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
  
  // MARK: - Performance Analysis Methods
  
  /// Logs comprehensive pipeline analysis every 300 frames
  private func logPipelineAnalysis(mode: String) {
    let frameSizeStr = frameSizeCaptured ? 
      "\(String(format: "%.0f", longSide))×\(String(format: "%.0f", shortSide))" : "Unknown"
    
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
    
    print("Camera: === CAMERA SOURCE PIPELINE ANALYSIS (Frame #\(frameProcessingCount)) ===")
    print("Camera: Frame Size: \(frameSizeStr) | Processing FPS: \(String(format: "%.1f", processingFPS)) | Mode: \(mode)")
    
    if mode == "inference" && lastInferenceTime > 0 && lastTotalPipelineTime > 0 {
      // Full pipeline data available
      print("Camera: Frame Preparation: \(String(format: "%.1f", lastFramePreparationTime))ms")
      print("Camera: Model Inference: \(String(format: "%.1f", lastInferenceTime))ms (includes fish counting inside TrackingDetector)")
      print("Camera: UI Delegate Call: \(String(format: "%.1f", lastUITime))ms | Total Pipeline: \(String(format: "%.1f", lastTotalPipelineTime))ms")
      
      let theoreticalFPS = lastTotalPipelineTime > 0 ? 1000.0 / lastTotalPipelineTime : 0
      print("Camera: Theoretical FPS: \(String(format: "%.1f", theoreticalFPS)) | Actual Throughput: \(String(format: "%.1f", actualThroughputFPS))")
      
      // Calculate breakdown percentages
      let preparationPct = (lastFramePreparationTime / lastTotalPipelineTime) * 100
      let inferencePct = (lastInferenceTime / lastTotalPipelineTime) * 100
      let uiPct = (lastUITime / lastTotalPipelineTime) * 100
      
      print("Camera: Breakdown - Preparation: \(String(format: "%.1f", preparationPct))% | Inference+FishCount: \(String(format: "%.1f", inferencePct))% | UI: \(String(format: "%.1f", uiPct))%")
    } else if mode == "calibration" {
      // Calibration mode data
      print("Camera: Frame Preparation: \(String(format: "%.1f", lastFramePreparationTime))ms + Conversion: \(String(format: "%.1f", lastConversionTime))ms = \(String(format: "%.1f", lastFramePreparationTime + lastConversionTime))ms")
      print("Camera: Model Inference: CALIBRATION MODE - Actual Throughput: \(String(format: "%.1f", actualThroughputFPS))")
    } else {
      // Limited data available
      print("Camera: Frame Preparation: \(String(format: "%.1f", lastFramePreparationTime))ms")
      print("Camera: Model Inference: PENDING - Actual Throughput: \(String(format: "%.1f", actualThroughputFPS))")
    }
  }
  
  /// Calculates actual pipeline throughput FPS based on complete processing cycles
  private func calculateThroughputFPS() -> Double {
    guard pipelineCompleteTimes.count >= 2 else { return 0 }
    let cyclesToConsider = min(10, pipelineCompleteTimes.count)
    let recentCompletions = Array(pipelineCompleteTimes.suffix(cyclesToConsider))
    let timeInterval = recentCompletions.last! - recentCompletions.first!
    return timeInterval > 0 ? Double(cyclesToConsider - 1) / timeInterval : 0
  }
}
