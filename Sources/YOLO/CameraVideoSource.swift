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
  
  // Default initializer
  override init() {
    super.init()
  }

  func setUp(
    sessionPreset: AVCaptureSession.Preset = .hd1280x720,
    position: AVCaptureDevice.Position,
    orientation: UIDeviceOrientation,
    completion: @escaping (Bool) -> Void
  ) {
    cameraQueue.async {
      let success = self.setUpCamera(
        sessionPreset: sessionPreset, position: position, orientation: orientation)
      DispatchQueue.main.async {
        completion(success)
      }
    }
  }
  
  // Simplified version for FrameSource protocol compatibility
  @MainActor
  func setUp(completion: @escaping @Sendable (Bool) -> Void) {
    // Capture orientation on the main actor
    let orientation = UIDevice.current.orientation
    
    // Then dispatch to background queue for camera setup
    cameraQueue.async {
      let success = self.setUpCamera(
        sessionPreset: .hd1280x720, 
        position: .back, 
        orientation: orientation)
      
      DispatchQueue.main.async {
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
  
  private func processFrameOnCameraQueue(sampleBuffer: CMSampleBuffer) {
    guard let predictor = predictor else {
      print("predictor is nil")
      return
    }
    
    if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
      // Update frame size if needed
      if !frameSizeCaptured {
        let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        longSide = max(frameWidth, frameHeight)
        shortSide = min(frameWidth, frameHeight)
        frameSizeCaptured = true
      }
      
      // For frameSourceDelegate, we need to safely pass the frame data to the main thread
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
      
      // Check if we should process for calibration
      let shouldProcessForCalibration = !inferenceOK && predictor is TrackingDetector
      
      if shouldProcessForCalibration {
        // Create a UIImage from the pixel buffer to safely pass across threads
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
          let image = UIImage(cgImage: cgImage)
          
          // Process on main thread for auto-calibration
          DispatchQueue.main.async { [weak self] in
            guard let self = self,
                  let trackingDetector = self.predictor as? TrackingDetector else { return }
            
            // Clear boxes when starting calibration (first frame)
            if trackingDetector.getCalibrationFrameCount() == 0 {
              self.videoCaptureDelegate?.onClearBoxes()
            }
                  
            // Convert UIImage back to CVPixelBuffer
            if let pixelBufferForCalibration = self.createStandardPixelBuffer(from: image, forSourceType: self.sourceType) {
              // Process the frame for calibration
              trackingDetector.processFrame(pixelBufferForCalibration)
            }
          }
        }
        
        // Skip regular inference during calibration
        return
      }
      
      // Only run regular inference when inferenceOK is true and we're not calibrating
      if inferenceOK {
        // Process with predictor - this happens on camera queue
        predictor.predict(sampleBuffer: sampleBuffer, onResultsListener: self, onInferenceTime: self)
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
    let width = viewBounds.width
    let height = viewBounds.height
    
    // Start with the original normalized rect
    var displayRect = rect
    
    // Handle orientation-specific transformations
    switch orientation {
    case .portraitUpsideDown:
      displayRect = CGRect(
        x: 1.0 - rect.origin.x - rect.width,
        y: 1.0 - rect.origin.y - rect.height,
        width: rect.width,
        height: rect.height)
    case .landscapeLeft, .landscapeRight:
      // In landscape mode, no additional transformation needed here
      break
    case .unknown:
      print("The device orientation is unknown, the predictions may be affected")
      fallthrough
    default:
      break
    }
    
    // For portrait orientation
    if orientation == .portrait || orientation == .portraitUpsideDown || orientation == .unknown {
      var ratio: CGFloat = 1.0
      
      if captureSession.sessionPreset == .photo {
        ratio = (height / width) / (4.0 / 3.0)
      } else {
        ratio = (height / width) / (16.0 / 9.0)
      }
      
      if ratio >= 1 {
        let offset = (1 - ratio) * (0.5 - displayRect.minX)
        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
        displayRect = displayRect.applying(transform)
        displayRect.size.width *= ratio
      } else {
        let offset = (ratio - 1) * (0.5 - displayRect.maxY)
        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
        displayRect = displayRect.applying(transform)
        ratio = (height / width) / (3.0 / 4.0)
        displayRect.size.height /= ratio
      }
      
      // Convert normalized coordinates to screen coordinates
      return VNImageRectForNormalizedRect(displayRect, Int(width), Int(height))
    } else {
      // Landscape mode
      let frameAspectRatio = longSide / shortSide
      let viewAspectRatio = width / height
      var scaleX: CGFloat = 1.0
      var scaleY: CGFloat = 1.0
      var offsetX: CGFloat = 0.0
      var offsetY: CGFloat = 0.0
      
      if frameAspectRatio > viewAspectRatio {
        scaleY = height / shortSide
        scaleX = scaleY
        offsetX = (longSide * scaleX - width) / 2
      } else {
        scaleX = width / longSide
        scaleY = scaleX
        offsetY = (shortSide * scaleY - height) / 2
      }
      
      // Transform rectangle to screen coordinates
      let screenRect = CGRect(
        x: displayRect.origin.x * longSide * scaleX - offsetX,
        y: height - (displayRect.origin.y * shortSide * scaleY - offsetY + displayRect.size.height * shortSide * scaleY),
        width: displayRect.size.width * longSide * scaleX,
        height: displayRect.size.height * shortSide * scaleY
      )
      
      return screenRect
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
    DispatchQueue.main.async {
      self.videoCaptureDelegate?.onInferenceTime(speed: inferenceTime, fps: fpsRate)
      self.frameSourceDelegate?.frameSource(self, didUpdateWithSpeed: inferenceTime, fps: fpsRate)
    }
  }

  func on(result: YOLOResult) {
    DispatchQueue.main.async {
      self.videoCaptureDelegate?.onPredict(result: result)
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
}
