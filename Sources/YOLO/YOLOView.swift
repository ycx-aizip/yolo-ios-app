// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
//  This file is part of the Ultralytics YOLO Package, providing the core UI component for real-time object detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The YOLOView class is the primary UI component for displaying real-time YOLO model results.
//  It handles camera setup, model loading, video frame processing, rendering of detection results,
//  and user interactions such as pinch-to-zoom. The view can display bounding boxes, masks for segmentation,
//  pose estimation keypoints, and oriented bounding boxes depending on the active task. It includes
//  UI elements for controlling inference settings such as confidence threshold and IoU threshold,
//  and provides functionality for capturing photos with detection results overlaid.

import AVFoundation
import UIKit
import Vision

/// Protocol for communicating user actions from YOLOView to its container
@MainActor
public protocol YOLOViewActionDelegate: AnyObject {
  /// Called when user taps the models button
  func didTapModelsButton()
}

/// A UIView component that provides real-time object detection, segmentation, and pose estimation capabilities.
@MainActor
public class YOLOView: UIView, VideoCaptureDelegate, FrameSourceDelegate {
  func onInferenceTime(speed: Double, fps: Double) {
    DispatchQueue.main.async {
      self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, speed)  // t2 seconds to ms
    }
  }

  func onPredict(result: YOLOResult) {
    // Skip showing boxes if we're in calibration mode
    if !isCalibrating {
      // Use the standard showBoxes method for all tasks including fish counting
      showBoxes(predictions: result)
      onDetection?(result)

      if task == .segment {
        DispatchQueue.main.async {
          if let maskImage = result.masks?.combinedMask {
            guard let maskLayer = self.maskLayer else { return }
            maskLayer.isHidden = false
            maskLayer.frame = self.overlayLayer.bounds
            maskLayer.contents = maskImage
            self.videoCapture.predictor.isUpdating = false
          } else {
            self.videoCapture.predictor.isUpdating = false
          }
        }
      } else if task == .classify {
        self.overlayYOLOClassificationsCALayer(on: self, result: result)
      } else if task == .pose {
        self.removeAllSubLayers(parentLayer: poseLayer)
        var keypointList = [[(x: Float, y: Float)]]()
        var confsList = [[Float]]()

        for keypoint in result.keypointsList {
          keypointList.append(keypoint.xyn)
          confsList.append(keypoint.conf)
        }
        guard let poseLayer = poseLayer else { return }
        drawKeypoints(
          keypointsList: keypointList, confsList: confsList, boundingBoxes: result.boxes,
          on: poseLayer, imageViewSize: overlayLayer.frame.size, originalImageSize: result.orig_shape)
      } else if task == .obb {
        guard let obbLayer = self.obbLayer else { return }
        let obbDetections = result.obb
        self.obbRenderer.drawObbDetectionsWithReuse(
          obbDetections: obbDetections,
          on: obbLayer,
          imageViewSize: self.overlayLayer.frame.size,
          originalImageSize: result.orig_shape,
          lineWidth: 3
        )
      }
    }
  }
  
  func onClearBoxes() {
    // Clear all bounding boxes when requested
    boundingBoxViews.forEach { box in
      box.hide()
    }
  }

  // Implement FrameSourceDelegate methods
  func frameSource(_ source: FrameSource, didOutputImage image: UIImage) {
    // We can add frame handling here in the future when needed
    // For now, we're relying on the predictor to handle frames and onPredict for results
  }
  
  func frameSource(_ source: FrameSource, didUpdateWithSpeed speed: Double, fps: Double) {
    // This is already handled by onInferenceTime, but we keep this for protocol compliance
    // No need to duplicate the UI updates
  }

  var onDetection: ((YOLOResult) -> Void)?
  private var videoCapture: CameraVideoSource
  private var albumVideoSource: AlbumVideoSource?
  private var currentFrameSource: FrameSource
  private var busy = false
  private var currentBuffer: CVPixelBuffer?
  var framesDone = 0
  var t0 = 0.0  // inference start
  var t1 = 0.0  // inference dt
  var t2 = 0.0  // inference dt smoothed
  var t3 = CACurrentMediaTime()  // FPS start
  var t4 = 0.0  // FPS dt smoothed
  var task = YOLOTask.detect
  var colors: [String: UIColor] = [:]
  var modelName: String = ""
  var classes: [String] = []
  let maxBoundingBoxViews = 100
  var boundingBoxViews = [BoundingBoxView]()
  public var sliderNumItems = UISlider()
  public var labelSliderNumItems = UILabel()
  public var sliderConf = UISlider()
  public var labelSliderConf = UILabel()
  public var sliderIoU = UISlider()
  public var labelSliderIoU = UILabel()
  
  // Fish counting thresholds
  public var threshold1Layer: CAShapeLayer?
  public var threshold2Layer: CAShapeLayer?
  public var threshold1Slider = UISlider()
  public var threshold2Slider = UISlider()
  public var labelThreshold1 = UILabel()
  public var labelThreshold2 = UILabel()
  public var threshold1: CGFloat = 0.3 // Add threshold1 property with default value
  public var threshold2: CGFloat = 0.5 // Add threshold2 property with default value
  
  // Auto calibration button
  public var autoCalibrationButton = UIButton()
  public var isCalibrating = false
  
  // Fish Count display and reset
  public var labelFishCount = UILabel()
  public var resetButton = UIButton()
  public var fishCount: Int = 0
  
  public var labelName = UILabel()
  public var labelFPS = UILabel()
  public var labelZoom = UILabel()
  public var activityIndicator = UIActivityIndicatorView()
  public var playButton = UIButton()
  public var pauseButton = UIButton()
  public var switchCameraButton = UIButton()
  public var toolbar = UIView()
  
  // Add new properties for frame source switching
  public var switchSourceButton = UIButton()
  private var frameSourceType: FrameSourceType = .camera
  
  // Add new property for models selection button
  public var modelsButton = UIButton()
  
  // Add properties for direction selection button
  public var directionButton = UIButton()
  private var countingDirection: CountingDirection = .topToBottom
  
  /// Action delegate to communicate with ViewController
  public weak var actionDelegate: YOLOViewActionDelegate?
  
  let selection = UISelectionFeedbackGenerator()
  private var overlayLayer = CALayer()
  private var maskLayer: CALayer?
  private var poseLayer: CALayer?
  private var obbLayer: CALayer?

  let obbRenderer = OBBRenderer()

  private let minimumZoom: CGFloat = 1.0
  private let maximumZoom: CGFloat = 10.0
  private var lastZoomFactor: CGFloat = 1.0

  @MainActor private var longPressDetected = false
  @MainActor private var isPinching = false

  // Add property for GoPro stream test reference
  private var tempGoProSource: GoProSource?

  // Add property to store last frame size for GoPro source
  internal var goProLastFrameSize: CGSize = CGSize(width: 1920, height: 1080)
  
  // Add property to store reference to GoPro source
  private var goProSource: GoProSource?

  public init(
    frame: CGRect,
    modelPathOrName: String,
    task: YOLOTask
  ) {
    self.videoCapture = CameraVideoSource()
    self.currentFrameSource = self.videoCapture
    super.init(frame: frame)
    setModel(modelPathOrName: modelPathOrName, task: task)
    setUpOrientationChangeNotification()
    self.setUpBoundingBoxViews()
    self.setupUI()
    self.videoCapture.videoCaptureDelegate = self
    self.videoCapture.frameSourceDelegate = self
    start(position: .back)
    setupOverlayLayer()
  }

  required init?(coder: NSCoder) {
    self.videoCapture = CameraVideoSource()
    self.currentFrameSource = self.videoCapture
    super.init(coder: coder)
  }

  public override func awakeFromNib() {
    super.awakeFromNib()
    Task { @MainActor in
      setUpOrientationChangeNotification()
      setUpBoundingBoxViews()
      setupUI()
      videoCapture.videoCaptureDelegate = self
      videoCapture.frameSourceDelegate = self
      start(position: .back)
      setupOverlayLayer()
    }
  }

  public func setModel(
    modelPathOrName: String,
    task: YOLOTask,
    completion: ((Result<Void, Error>) -> Void)? = nil
  ) {
    activityIndicator.startAnimating()
    boundingBoxViews.forEach { box in
      box.hide()
    }
    removeClassificationLayers()

    self.task = task
    setupSublayers()

    var modelURL: URL?
    let lowercasedPath = modelPathOrName.lowercased()
    let fileManager = FileManager.default

    // Determine model URL
    if lowercasedPath.hasSuffix(".mlmodel") || lowercasedPath.hasSuffix(".mlpackage")
      || lowercasedPath.hasSuffix(".mlmodelc")
    {
      let possibleURL = URL(fileURLWithPath: modelPathOrName)
      if fileManager.fileExists(atPath: possibleURL.path) {
        modelURL = possibleURL
      }
    } else {
      if let compiledURL = Bundle.main.url(forResource: modelPathOrName, withExtension: "mlmodelc")
      {
        modelURL = compiledURL
      } else if let packageURL = Bundle.main.url(
        forResource: modelPathOrName, withExtension: "mlpackage")
      {
        modelURL = packageURL
      }
    }

    guard let unwrappedModelURL = modelURL else {
      let error = PredictorError.modelFileNotFound
      fatalError(error.localizedDescription)
    }

    modelName = unwrappedModelURL.deletingPathExtension().lastPathComponent

    // Common success handling for all tasks
    func handleSuccess(predictor: Predictor) {
      self.videoCapture.predictor = predictor
      self.activityIndicator.stopAnimating()
      self.labelName.text = modelName
      
      // Apply the initial threshold values to the model
      if let detector = predictor as? ObjectDetector {
        // Get values from the sliders
        let conf = Double(round(100 * sliderConf.value)) / 100
        let iou = Double(round(100 * sliderIoU.value)) / 100
        
        // Apply thresholds to the model
        detector.setConfidenceThreshold(confidence: conf)
        detector.setIouThreshold(iou: iou)
        detector.setNumItemsThreshold(numItems: Int(sliderNumItems.value))
        
        print("Initial thresholds applied - Confidence: \(conf), IoU: \(iou), Max Items: \(Int(sliderNumItems.value))")
        
        // Initialize the fish counting threshold lines
        if task == .fishCount {
          // Setup the threshold layers
          setupThresholdLayers()
          
          // The threshold values will be used by fish counting logic later
          print("Fish counting thresholds set - Threshold 1: \(threshold1Slider.value), Threshold 2: \(threshold2Slider.value)")
        }
      }
      
      completion?(.success(()))
    }

    // Common failure handling for all tasks
    func handleFailure(_ error: Error) {
      print("Failed to load model with error: \(error)")
      self.activityIndicator.stopAnimating()
      completion?(.failure(error))
    }

    switch task {
    case .classify:
      Classifier.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true) {
        [weak self] result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .segment:
      Segmenter.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true) {
        [weak self] result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .pose:
      PoseEstimater.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true) {
        [weak self] result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .obb:
      ObbDetector.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true) {
        [weak self] result in
        switch result {
        case .success(let predictor):
          self?.obbLayer?.isHidden = false

          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .fishCount:
      TrackingDetector.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true) {
        [weak self] result in
        switch result {
        case .success(let predictor):
          // Configure tracking detector with current thresholds
          if let trackingDetector = predictor as? TrackingDetector,
             let slf = self {
            trackingDetector.setThresholds([CGFloat(slf.threshold1Slider.value), 
                                            CGFloat(slf.threshold2Slider.value)])
          }
          
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    default:
      // Handle the .detect case using ObjectDetector
      ObjectDetector.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true) {
        [weak self] result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }
    }
  }

  private func start(position: AVCaptureDevice.Position) {
    if !busy {
      busy = true
      let orientation = UIDevice.current.orientation
      videoCapture.setUp(sessionPreset: .photo, position: position, orientation: orientation) {
        success in
        // .hd4K3840x2160 or .photo (4032x3024)  Warning: 4k may not work on all devices i.e. 2019 iPod
        if success {
          // Add the video preview into the UI.
          if let previewLayer = self.videoCapture.previewLayer {
            self.layer.insertSublayer(previewLayer, at: 0)
            self.videoCapture.previewLayer?.frame = self.bounds  // resize preview layer
            for box in self.boundingBoxViews {
              box.addToLayer(previewLayer)
            }
          }
          self.videoCapture.previewLayer?.addSublayer(self.overlayLayer)
          // Once everything is set up, we can start capturing live video.
          self.videoCapture.start()

          self.busy = false
        }
      }
    }
  }

  public func stop() {
    currentFrameSource.stop()
  }

  public func resume() {
    currentFrameSource.start()
  }

  func setUpBoundingBoxViews() {
    // Ensure all bounding box views are initialized up to the maximum allowed.
    while boundingBoxViews.count < maxBoundingBoxViews {
      boundingBoxViews.append(BoundingBoxView())
    }

  }

  func setupOverlayLayer() {
    let width = self.bounds.width
    let height = self.bounds.height

    var ratio: CGFloat = 1.0
    if videoCapture.captureSession.sessionPreset == .photo {
      ratio = (4.0 / 3.0)
    } else {
      ratio = (16.0 / 9.0)
    }
    var offSet = CGFloat.zero
    var margin = CGFloat.zero
    if self.bounds.width < self.bounds.height {
      offSet = height / ratio
      margin = (offSet - self.bounds.width) / 2
      self.overlayLayer.frame = CGRect(
        x: -margin, y: 0, width: offSet, height: self.bounds.height)
    } else {
      offSet = width / ratio
      margin = (offSet - self.bounds.height) / 2
      self.overlayLayer.frame = CGRect(
        x: 0, y: -margin, width: self.bounds.width, height: offSet)
    }
  }

  func setupMaskLayerIfNeeded() {
    if maskLayer == nil {
      let layer = CALayer()
      layer.frame = self.overlayLayer.bounds
      layer.opacity = 0.5
      layer.name = "maskLayer"
      // Specify contentsGravity or backgroundColor as needed
      // layer.contentsGravity = .resizeAspectFill
      // layer.backgroundColor = UIColor.clear.cgColor

      self.overlayLayer.addSublayer(layer)
      self.maskLayer = layer
    }
  }

  func setupPoseLayerIfNeeded() {
    if poseLayer == nil {
      let layer = CALayer()
      layer.frame = self.overlayLayer.bounds
      layer.opacity = 0.5
      self.overlayLayer.addSublayer(layer)
      self.poseLayer = layer
    }
  }

  func setupObbLayerIfNeeded() {
    if obbLayer == nil {
      let layer = CALayer()
      layer.frame = self.overlayLayer.bounds
      layer.opacity = 0.5
      self.overlayLayer.addSublayer(layer)
      self.obbLayer = layer
    }
  }

  public func resetLayers() {
    removeAllSubLayers(parentLayer: maskLayer)
    removeAllSubLayers(parentLayer: poseLayer)
    removeAllSubLayers(parentLayer: overlayLayer)

    maskLayer = nil
    poseLayer = nil
    obbLayer?.isHidden = true
  }

  func setupSublayers() {
    resetLayers()

    switch task {
    case .segment:
      setupMaskLayerIfNeeded()
    case .pose:
      setupPoseLayerIfNeeded()
    case .obb:
      setupObbLayerIfNeeded()
      overlayLayer.addSublayer(obbLayer!)
      obbLayer?.isHidden = false
    default: break
    }
  }

  func removeAllSubLayers(parentLayer: CALayer?) {
    guard let parentLayer = parentLayer else { return }
    parentLayer.sublayers?.forEach { layer in
      layer.removeFromSuperlayer()
    }
    parentLayer.sublayers = nil
    parentLayer.contents = nil
  }

  func addMaskSubLayers() {
    guard let maskLayer = maskLayer else { return }
    self.overlayLayer.addSublayer(maskLayer)
  }

  func showBoxes(predictions: YOLOResult) {
    let width = self.bounds.width
    let height = self.bounds.height
    var resultCount = 0

    resultCount = predictions.boxes.count
    
    // CRITICAL FIX: First hide all boxes, then only show the ones that are active
    // This ensures boxes are cleared when no fish are present
    boundingBoxViews.forEach { box in
      box.hide()
    }
    
    // If there are no boxes to show, return early
    if resultCount == 0 {
      return
    }

    if UIDevice.current.orientation == .portrait {
      var ratio: CGFloat = 1.0

      // Use the session preset from the active frame source (only CameraVideoSource has this)
      if currentFrameSource.sourceType == .camera && videoCapture.captureSession.sessionPreset == .photo {
        ratio = (height / width) / (4.0 / 3.0)
      } else if currentFrameSource.sourceType == .goPro {
        // For GoPro source, use a special handling
        ratio = (height / width) / (16.0 / 9.0) // Most GoPros use 16:9
      } else {
        ratio = (height / width) / (16.0 / 9.0)
      }

      self.labelSliderNumItems.text =
        String(resultCount) + " items (max " + String(Int(sliderNumItems.value)) + ")"
      for i in 0..<boundingBoxViews.count {
        if i < (resultCount) && i < 50 {
          var rect = CGRect.zero
          var label = ""
          var boxColor: UIColor = .white
          var confidence: CGFloat = 0
          var alpha: CGFloat = 0.9
          var bestClass = ""

          switch task {
          case .detect:
            let prediction = predictions.boxes[i]
            rect = CGRect(
              x: prediction.xywhn.minX, y: 1 - prediction.xywhn.maxY, width: prediction.xywhn.width,
              height: prediction.xywhn.height)
            bestClass = prediction.cls
            confidence = CGFloat(prediction.conf)
            let colorIndex = prediction.index % ultralyticsColors.count
            boxColor = ultralyticsColors[colorIndex]
            label = String(format: "%@ %.1f", bestClass, confidence * 100)
            alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
          case .fishCount:
            // For fish count task, use custom colors based on tracking status
            let prediction = predictions.boxes[i]
            rect = CGRect(
              x: prediction.xywhn.minX, y: 1 - prediction.xywhn.maxY, width: prediction.xywhn.width,
              height: prediction.xywhn.height)
            bestClass = prediction.cls
            confidence = CGFloat(prediction.conf)
            
            // Check tracking status to determine color - need to ensure this works with current frame source
            if let trackingDetector = currentFrameSource.predictor as? TrackingDetector {
              let isTracked = trackingDetector.isObjectTracked(box: prediction)
              let isCounted = trackingDetector.isObjectCounted(box: prediction)
              
              // Color scheme:
              // Green: Counted fish
              // Light blue: Tracked but not counted fish
              // Dark blue: Newly tracked fish
              if isCounted {
                boxColor = .green 
              } else if isTracked {
                boxColor = UIColor(red: 0.4, green: 0.7, blue: 1.0, alpha: 1.0) // Light blue
              } else {
                boxColor = UIColor(red: 0.0, green: 0.0, blue: 0.8, alpha: 1.0) // Dark blue
              }
              
              alpha = isTracked ? 0.7 : 0.5 // More transparent if newly tracked
              
              // Display tracking ID for tracked fish, empty label for untracked
              if isTracked {
                // Get the tracking ID and display it
                if let trackInfo = trackingDetector.getTrackInfo(for: prediction) {
                  label = "#\(trackInfo.trackId)"
                } else {
                  label = "#?"
                }
              } else {
                // No label for untracked fish
                label = ""
              }
            } else {
              // Fallback to standard coloring if not using tracking detector
              let colorIndex = prediction.index % ultralyticsColors.count
              boxColor = ultralyticsColors[colorIndex]
              alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
              label = bestClass
            }
          default:
            let prediction = predictions.boxes[i]
            rect = prediction.xywhn
            bestClass = prediction.cls
            confidence = CGFloat(prediction.conf)
            label = String(format: "%@ %.1f", bestClass, confidence * 100)
            let colorIndex = prediction.index % ultralyticsColors.count
            boxColor = ultralyticsColors[colorIndex]
            alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
          }
          var displayRect = rect
          switch UIDevice.current.orientation {
          case .portraitUpsideDown:
            displayRect = CGRect(
              x: 1.0 - rect.origin.x - rect.width,
              y: 1.0 - rect.origin.y - rect.height,
              width: rect.width,
              height: rect.height)
          case .landscapeLeft:
            displayRect = CGRect(
              x: rect.origin.x,
              y: rect.origin.y,
              width: rect.width,
              height: rect.height)
          case .landscapeRight:
            displayRect = CGRect(
              x: rect.origin.x,
              y: rect.origin.y,
              width: rect.width,
              height: rect.height)
          case .unknown:
            print("The device orientation is unknown, the predictions may be affected")
            fallthrough
          default: break
          }
          
          // For video source, use the special conversion to handle letterboxing/pillarboxing
          if frameSourceType == .videoFile, let albumSource = albumVideoSource {
            // Make sure we're using the right coordinates for video source
            // For video sources, we need to invert the y-axis transformation
            // since it's already been inverted in the construction of displayRect
            let normalizedRect = CGRect(
              x: displayRect.minX,
              y: displayRect.minY,
              width: displayRect.width,
              height: displayRect.height
            )
            
            // Convert normalized coordinates to screen coordinates based on video content rect
            let screenRect = albumSource.convertNormalizedRectToScreenRect(normalizedRect)
            
            // Set the box with the converted rect
            boundingBoxViews[i].show(
              frame: screenRect, label: label, color: boxColor, alpha: alpha)
          } else if frameSourceType == .goPro {
            // NOTE: We no longer need this special handling in YOLOView 
            // because GoProSource now handles the coordinate transformation properly
            // We'll keep this code as a fallback, but it should rarely be used
            
            // Get the transformed coordinates from the GoPro source directly
            // The coordinates should already be properly transformed by GoProSource.transformResultCoordinates
            let displayRect = VNImageRectForNormalizedRect(rect, Int(width), Int(height))
            
            boundingBoxViews[i].show(
                frame: displayRect, label: label, color: boxColor, alpha: alpha)
          } else {
            // Original camera frame handling
          if ratio >= 1 {
            let offset = (1 - ratio) * (0.5 - displayRect.minX)
            if task == .detect || task == .fishCount {
              let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
              displayRect = displayRect.applying(transform)
            } else {
              let transform = CGAffineTransform(translationX: offset, y: 0)
              displayRect = displayRect.applying(transform)
            }
            displayRect.size.width *= ratio
          } else {
            if task == .detect || task == .fishCount {
              let offset = (ratio - 1) * (0.5 - displayRect.maxY)

              let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
              displayRect = displayRect.applying(transform)
            } else {
              let offset = (ratio - 1) * (0.5 - displayRect.minY)
              let transform = CGAffineTransform(translationX: 0, y: offset)
              displayRect = displayRect.applying(transform)
            }
            ratio = (height / width) / (3.0 / 4.0)
            displayRect.size.height /= ratio
          }
          displayRect = VNImageRectForNormalizedRect(displayRect, Int(width), Int(height))

          boundingBoxViews[i].show(
            frame: displayRect, label: label, color: boxColor, alpha: alpha)
          }
        }
      }
    } else {
      // Landscape mode
      resultCount = predictions.boxes.count
      self.labelSliderNumItems.text =
        String(resultCount) + " items (max " + String(Int(sliderNumItems.value)) + ")"

      // Use longSide and shortSide from the current frame source (important fix)
      let frameAspectRatio = currentFrameSource.longSide / currentFrameSource.shortSide
      let viewAspectRatio = width / height
      var scaleX: CGFloat = 1.0
      var scaleY: CGFloat = 1.0
      var offsetX: CGFloat = 0.0
      var offsetY: CGFloat = 0.0

      if frameAspectRatio > viewAspectRatio {
        scaleY = height / currentFrameSource.shortSide
        scaleX = scaleY
        offsetX = (currentFrameSource.longSide * scaleX - width) / 2
      } else {
        scaleX = width / currentFrameSource.longSide
        scaleY = scaleX
        offsetY = (currentFrameSource.shortSide * scaleY - height) / 2
      }

      // Then show only the active boxes
      for i in 0..<resultCount {
        if i < 50 { // Limit to maximum 50 boxes
          var rect = CGRect.zero
          var label = ""
          var boxColor: UIColor = .white
          var confidence: CGFloat = 0
          var alpha: CGFloat = 0.9
          var bestClass = ""

          switch task {
          case .detect:
            let prediction = predictions.boxes[i]
            // For the detect task, invert y using "1 - maxY" as before
            rect = CGRect(
              x: prediction.xywhn.minX,
              y: 1 - prediction.xywhn.maxY,
              width: prediction.xywhn.width,
              height: prediction.xywhn.height
            )
            bestClass = prediction.cls
            confidence = CGFloat(prediction.conf)
            let colorIndex = prediction.index % ultralyticsColors.count
            boxColor = ultralyticsColors[colorIndex]
            
          case .fishCount:
            // For fish count task, use custom colors based on tracking status
            let prediction = predictions.boxes[i]
            rect = CGRect(
              x: prediction.xywhn.minX,
              y: 1 - prediction.xywhn.maxY,
              width: prediction.xywhn.width,
              height: prediction.xywhn.height
            )
            bestClass = prediction.cls
            confidence = CGFloat(prediction.conf)
            
            // Check tracking status to determine color - need to ensure this works with current frame source
            if let trackingDetector = currentFrameSource.predictor as? TrackingDetector {
              let isTracked = trackingDetector.isObjectTracked(box: prediction)
              let isCounted = trackingDetector.isObjectCounted(box: prediction)
              
              // Color scheme:
              // Green: Counted fish
              // Light blue: Tracked but not counted fish
              // Dark blue: Newly tracked fish
              if isCounted {
                boxColor = .green 
              } else if isTracked {
                boxColor = UIColor(red: 0.4, green: 0.7, blue: 1.0, alpha: 1.0) // Light blue
              } else {
                boxColor = UIColor(red: 0.0, green: 0.0, blue: 0.8, alpha: 1.0) // Dark blue
              }
              
              alpha = isTracked ? 0.7 : 0.5 // More transparent if newly tracked
              
              // Display tracking ID for tracked fish, empty label for untracked
              if isTracked {
                // Get the tracking ID and display it
                if let trackInfo = trackingDetector.getTrackInfo(for: prediction) {
                  label = "#\(trackInfo.trackId)"
                } else {
                  label = "#?"
                }
              } else {
                // No label for untracked fish
                label = ""
              }
            } else {
              let colorIndex = prediction.index % ultralyticsColors.count
              boxColor = ultralyticsColors[colorIndex]
              label = bestClass
            }

          default:
            let prediction = predictions.boxes[i]
            rect = CGRect(
              x: prediction.xywhn.minX,
              y: 1 - prediction.xywhn.maxY,
              width: prediction.xywhn.width,
              height: prediction.xywhn.height
            )
            bestClass = prediction.cls
            confidence = CGFloat(prediction.conf)
            let colorIndex = prediction.index % ultralyticsColors.count
            boxColor = ultralyticsColors[colorIndex]
          }

          // For non-fishCount tasks, use standard label format
          if task != .fishCount {
          label = String(format: "%@ %.1f", bestClass, confidence * 100)
          alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
          }
          
          // For video source, use the special conversion to handle letterboxing/pillarboxing
          if frameSourceType == .videoFile, let albumSource = albumVideoSource {
            // Make sure we're using the right coordinates for video source
            // For video sources, we need to invert the y-axis transformation
            // since it's already been inverted in the construction of displayRect
            let normalizedRect = CGRect(
              x: rect.minX,
              y: rect.minY,
              width: rect.width,
              height: rect.height
            )
            
            // Convert normalized coordinates to screen coordinates based on video content rect
            let screenRect = albumSource.convertNormalizedRectToScreenRect(normalizedRect)
            
            // Set the box with the converted rect
            boundingBoxViews[i].show(
              frame: screenRect, label: label, color: boxColor, alpha: alpha)
          } else if frameSourceType == .goPro {
            // NOTE: We no longer need this special handling in YOLOView for landscape mode
            // because GoProSource now handles the coordinate transformation properly
            // We'll keep this code as a fallback, but it should rarely be used
            
            // Get the transformed coordinates from the GoPro source directly
            // The coordinates should already be properly transformed by GoProSource.transformResultCoordinates
            let displayRect = VNImageRectForNormalizedRect(rect, Int(width), Int(height))
            
            if i < 3 { // Only log first few boxes to avoid spam
              print("YOLOView(Landscape): GoPro box \(i) - Using simplified coordinate transformation")
              print("YOLOView(Landscape): Original rect: \(rect), Display rect: \(displayRect)")
            }
            
            boundingBoxViews[i].show(
              frame: displayRect, label: label, color: boxColor, alpha: alpha
            )
          } else {
            // Original camera frame handling
            // Transform rectangle to screen coordinates - use currentFrameSource
            rect.origin.x = rect.origin.x * currentFrameSource.longSide * scaleX - offsetX
          rect.origin.y =
            height
              - (rect.origin.y * currentFrameSource.shortSide * scaleY
              - offsetY
                + rect.size.height * currentFrameSource.shortSide * scaleY)
            rect.size.width *= currentFrameSource.longSide * scaleX
            rect.size.height *= currentFrameSource.shortSide * scaleY

          boundingBoxViews[i].show(
            frame: rect,
            label: label,
            color: boxColor,
            alpha: alpha
          )
          }
        }
      }
    }

    // Update fish count display if we're in fish count mode
    if task == .fishCount, let trackingDetector = currentFrameSource.predictor as? TrackingDetector {
      let currentCount = trackingDetector.getCount()
      labelFishCount.text = "Fish Count: \(currentCount)"
    }
  }

  func removeClassificationLayers() {
    if let sublayers = self.layer.sublayers {
      for layer in sublayers where layer.name == "YOLOOverlayLayer" {
        layer.removeFromSuperlayer()
      }
    }
  }

  func overlayYOLOClassificationsCALayer(on view: UIView, result: YOLOResult) {
    removeClassificationLayers()

    let overlayLayer = CALayer()
    overlayLayer.frame = view.bounds
    overlayLayer.name = "YOLOOverlayLayer"

    guard let top1 = result.probs?.top1,
      let top1Conf = result.probs?.top1Conf
    else {
      return
    }

    var colorIndex = 0
    if let index = result.names.firstIndex(of: top1) {
      colorIndex = index % ultralyticsColors.count
    }
    let color = ultralyticsColors[colorIndex]

    let confidencePercent = round(top1Conf * 1000) / 10
    let labelText = " \(top1) \(confidencePercent)% "

    let textLayer = CATextLayer()
    textLayer.contentsScale = UIScreen.main.scale  // Retina対応
    textLayer.alignmentMode = .left
    let fontSize = self.bounds.height * 0.02
    textLayer.font = UIFont.systemFont(ofSize: fontSize, weight: .semibold)
    textLayer.fontSize = fontSize
    textLayer.foregroundColor = UIColor.white.cgColor
    textLayer.backgroundColor = color.cgColor
    textLayer.cornerRadius = 4
    textLayer.masksToBounds = true

    textLayer.string = labelText
    let textAttributes: [NSAttributedString.Key: Any] = [
      .font: UIFont.systemFont(ofSize: fontSize, weight: .semibold)
    ]
    let textSize = (labelText as NSString).size(withAttributes: textAttributes)
    let width: CGFloat = textSize.width + 10
    let x: CGFloat = self.center.x - (width / 2)
    let y: CGFloat = self.center.y - textSize.height
    let height: CGFloat = textSize.height + 4

    textLayer.frame = CGRect(x: x, y: y, width: width, height: height)

    overlayLayer.addSublayer(textLayer)

    view.layer.addSublayer(overlayLayer)
  }

  private func setupUI() {
    labelName.text = modelName
    labelName.textAlignment = .center
    labelName.font = UIFont.systemFont(ofSize: 24, weight: .medium)
    labelName.textColor = .black
    labelName.font = UIFont.preferredFont(forTextStyle: .title1)
    labelName.isHidden = true
    self.addSubview(labelName)

    labelFPS.text = String(format: "%.1f FPS - %.1f ms", 0.0, 0.0)
    labelFPS.textAlignment = .center
    labelFPS.textColor = .black
    labelFPS.font = UIFont.systemFont(ofSize: 12)
    self.addSubview(labelFPS)

    labelSliderNumItems.text = "0 items (max 30)"
    labelSliderNumItems.textAlignment = .left
    labelSliderNumItems.textColor = .black
    labelSliderNumItems.font = UIFont.preferredFont(forTextStyle: .subheadline)
    labelSliderNumItems.isHidden = true
    self.addSubview(labelSliderNumItems)

    sliderNumItems.minimumValue = 0
    sliderNumItems.maximumValue = 100
    sliderNumItems.value = 100
    sliderNumItems.minimumTrackTintColor = .darkGray
    sliderNumItems.maximumTrackTintColor = .systemGray.withAlphaComponent(0.7)
    sliderNumItems.addTarget(self, action: #selector(sliderChanged), for: .valueChanged)
    sliderNumItems.isHidden = true
    self.addSubview(sliderNumItems)

    labelSliderConf.text = "Conf: 0.75"
    labelSliderConf.textAlignment = .left
    labelSliderConf.textColor = .white
    labelSliderConf.font = UIFont.systemFont(ofSize: 16, weight: .medium)
    self.addSubview(labelSliderConf)

    sliderConf.minimumValue = 0
    sliderConf.maximumValue = 1
    sliderConf.value = 0.75
    sliderConf.minimumTrackTintColor = .white
    sliderConf.maximumTrackTintColor = .lightGray
    sliderConf.addTarget(self, action: #selector(sliderChanged), for: .valueChanged)
    self.addSubview(sliderConf)

    labelSliderIoU.text = "IoU: 0.5"
    labelSliderIoU.textAlignment = .right
    labelSliderIoU.textColor = .white
    labelSliderIoU.font = UIFont.systemFont(ofSize: 16, weight: .medium)
    self.addSubview(labelSliderIoU)

    sliderIoU.minimumValue = 0
    sliderIoU.maximumValue = 1
    sliderIoU.value = 0.5
    sliderIoU.minimumTrackTintColor = .white
    sliderIoU.maximumTrackTintColor = .lightGray
    sliderIoU.addTarget(self, action: #selector(sliderChanged), for: .valueChanged)
    self.addSubview(sliderIoU)

    self.labelSliderNumItems.text = "0 items (max " + String(Int(sliderNumItems.value)) + ")"
    self.labelSliderConf.text = "Conf: " + String(Double(round(100 * sliderConf.value)) / 100)
    self.labelSliderIoU.text = "IoU: " + String(Double(round(100 * sliderIoU.value)) / 100)

    // Initialize fish counting threshold UI elements
    let threshold1Value = String(format: "%.2f", 0.3)
    labelThreshold1.text = "Threshold 1: " + threshold1Value
    labelThreshold1.textAlignment = .left
    labelThreshold1.textColor = UIColor.red
    labelThreshold1.font = UIFont.systemFont(ofSize: 16, weight: .medium)
    self.addSubview(labelThreshold1)
    
    threshold1Slider.minimumValue = 0
    threshold1Slider.maximumValue = 1
    threshold1Slider.value = 0.3 // Initial value
    threshold1Slider.minimumTrackTintColor = UIColor.red
    threshold1Slider.maximumTrackTintColor = UIColor.lightGray
    threshold1Slider.addTarget(self, action: #selector(threshold1Changed), for: .valueChanged)
    self.addSubview(threshold1Slider)
    
    let threshold2Value = String(format: "%.2f", 0.5)
    labelThreshold2.text = "Threshold 2: " + threshold2Value
    labelThreshold2.textAlignment = .right
    labelThreshold2.textColor = UIColor.yellow
    labelThreshold2.font = UIFont.systemFont(ofSize: 16, weight: .medium)
    self.addSubview(labelThreshold2)
    
    threshold2Slider.minimumValue = 0
    threshold2Slider.maximumValue = 1
    threshold2Slider.value = 0.5 // Initial value
    threshold2Slider.minimumTrackTintColor = UIColor.yellow
    threshold2Slider.maximumTrackTintColor = UIColor.lightGray
    threshold2Slider.addTarget(self, action: #selector(threshold2Changed), for: .valueChanged)
    self.addSubview(threshold2Slider)

    // Initialize auto-calibration button
    // Use attributed string for split-colored text
    let autoAttributedText = NSMutableAttributedString()
    let auAttributes: [NSAttributedString.Key: Any] = [
      .foregroundColor: UIColor.red.withAlphaComponent(0.5),
      .font: UIFont.systemFont(ofSize: 14, weight: .bold)
    ]
    let toAttributes: [NSAttributedString.Key: Any] = [
      .foregroundColor: UIColor.yellow.withAlphaComponent(0.5),
      .font: UIFont.systemFont(ofSize: 14, weight: .bold)
    ]
    autoAttributedText.append(NSAttributedString(string: "AU", attributes: auAttributes))
    autoAttributedText.append(NSAttributedString(string: "TO", attributes: toAttributes))
    
    autoCalibrationButton.setAttributedTitle(autoAttributedText, for: .normal)
    autoCalibrationButton.backgroundColor = UIColor.darkGray.withAlphaComponent(0.1)
    autoCalibrationButton.layer.cornerRadius = 12
    autoCalibrationButton.layer.masksToBounds = true
    autoCalibrationButton.addTarget(self, action: #selector(toggleAutoCalibration), for: .touchUpInside)
    self.addSubview(autoCalibrationButton)

    // Initialize Fish Count display and Reset button
    labelFishCount.text = "Fish Count: 0"
    labelFishCount.textAlignment = .center
    labelFishCount.textColor = UIColor.white
    labelFishCount.font = UIFont.systemFont(ofSize: 16, weight: .bold)
    labelFishCount.backgroundColor = UIColor.darkGray.withAlphaComponent(0.7)
    labelFishCount.layer.cornerRadius = 12
    labelFishCount.layer.masksToBounds = true
    self.addSubview(labelFishCount)
    
    resetButton.setTitle("Reset", for: .normal)
    resetButton.setTitleColor(UIColor.white, for: .normal)
    resetButton.titleLabel?.font = UIFont.systemFont(ofSize: 16, weight: .bold)
    resetButton.backgroundColor = UIColor.darkGray.withAlphaComponent(0.7)
    resetButton.layer.cornerRadius = 12
    resetButton.layer.masksToBounds = true
    resetButton.addTarget(self, action: #selector(resetFishCount), for: .touchUpInside)
    self.addSubview(resetButton)

    labelZoom.text = "1.00x"
    labelZoom.textColor = .black
    labelZoom.font = UIFont.systemFont(ofSize: 14)
    labelZoom.textAlignment = .center
    labelZoom.font = UIFont.preferredFont(forTextStyle: .body)
    labelZoom.isHidden = true
    self.addSubview(labelZoom)

    let config = UIImage.SymbolConfiguration(pointSize: 16, weight: .regular, scale: .default)

    // Set up toolbar with consistent styling
    toolbar.backgroundColor = .clear
    toolbar.layer.cornerRadius = 10
    
    // Setup play button with consistent styling
    playButton.setImage(UIImage(systemName: "play.fill", withConfiguration: config), for: .normal)
    playButton.tintColor = .white
    playButton.backgroundColor = .clear
    playButton.isEnabled = false
    playButton.addTarget(self, action: #selector(playTapped), for: .touchUpInside)
    
    // Setup pause button with consistent styling
    pauseButton.setImage(UIImage(systemName: "pause.fill", withConfiguration: config), for: .normal)
    pauseButton.tintColor = .white
    pauseButton.backgroundColor = .clear
    pauseButton.isEnabled = true
    pauseButton.addTarget(self, action: #selector(pauseTapped), for: .touchUpInside)
    
    // Create switch source button with consistent styling
    switchSourceButton.setImage(UIImage(systemName: "rectangle.on.rectangle", withConfiguration: config), for: .normal)
    switchSourceButton.tintColor = .white
    switchSourceButton.backgroundColor = .clear
    switchSourceButton.addTarget(self, action: #selector(switchSourceButtonTapped), for: .touchUpInside)
    
    // Create models button with consistent styling
    modelsButton.setImage(UIImage(systemName: "square.stack.3d.up", withConfiguration: config), for: .normal)
    modelsButton.tintColor = .white
    modelsButton.backgroundColor = .clear
    modelsButton.addTarget(self, action: #selector(modelsButtonTapped), for: .touchUpInside)
    
    // Create direction button with consistent styling
    directionButton.setImage(UIImage(systemName: "arrow.triangle.2.circlepath", withConfiguration: config), for: .normal)
    directionButton.tintColor = .white
    directionButton.backgroundColor = .clear
    directionButton.addTarget(self, action: #selector(directionButtonTapped), for: .touchUpInside)
    
    // Add buttons to toolbar
    self.addSubview(toolbar)
    toolbar.addSubview(playButton)
    toolbar.addSubview(pauseButton)
    toolbar.addSubview(switchSourceButton)
    toolbar.addSubview(modelsButton)
    toolbar.addSubview(directionButton)
    
    self.addGestureRecognizer(UIPinchGestureRecognizer(target: self, action: #selector(pinch)))
  }

  public override func layoutSubviews() {
    setupOverlayLayer()
    let isLandscape = bounds.width > bounds.height
    activityIndicator.frame = CGRect(x: center.x - 50, y: center.y - 50, width: 100, height: 100)
    
    // Setup threshold lines if needed
    setupThresholdLayers()
    
    if isLandscape {
      // Toolbar background should be completely transparent
      // toolbar.backgroundColor = .black.withAlphaComponent(0.4)
      
      let width = bounds.width
      let height = bounds.height

      // Move the model name label even higher, closer to top edge
      let titleLabelHeight: CGFloat = height * 0.02
      labelName.frame = CGRect(
        x: 0,
        y: height * 0.01, // Position at 1% from top (moved much closer to top)
        width: width,
        height: titleLabelHeight
      )
      
      // Position FPS label at center bottom
      let toolBarHeight: CGFloat = 50
      let subLabelHeight: CGFloat = height * 0.03
      labelFPS.frame = CGRect(
        x: width * 0.35,
        y: height - toolBarHeight - subLabelHeight - 5,
        width: width * 0.3,
        height: subLabelHeight
      )
      
      // Move all controls much higher up to be well above the toolbar
      // Start positioning from upper part of the screen
      let topControlY = height * 0.5 // Start controls at 50% from top
      
      // Calculate slider dimensions and spacing
      let sliderWidth = width * 0.22
      let sliderLabelHeight: CGFloat = 20
      let sliderHeight: CGFloat = height * 0.05
      
      // First row - Fish Count and Reset (moved much higher)
      let fishCountWidth = width * 0.18
      let resetButtonWidth = fishCountWidth * 0.7
      let controlHeight: CGFloat = 36
      
      // Fish count on left side 
      labelFishCount.frame = CGRect(
        x: width * 0.05,
        y: topControlY,
        width: fishCountWidth,
        height: controlHeight
      )
      
      // Reset button on right side
      resetButton.frame = CGRect(
        x: width - width * 0.05 - resetButtonWidth,
        y: topControlY,
        width: resetButtonWidth,
        height: controlHeight
      )
      
      // Left side - Threshold 1 (second row left)
      let secondRowY = topControlY + controlHeight + height * 0.02
      
      labelThreshold1.frame = CGRect(
        x: width * 0.05,
        y: secondRowY,
        width: sliderWidth,
        height: sliderLabelHeight
      )
      
      threshold1Slider.frame = CGRect(
        x: width * 0.05,
        y: secondRowY + sliderLabelHeight + 2,
        width: sliderWidth,
        height: sliderHeight
      )
      
      // AUTO button (positioned at the same height as threshold labels) - LANDSCAPE
      let autoButtonWidthLandscape = width * 0.08
      let autoButtonHeightLandscape: CGFloat = 24
      autoCalibrationButton.frame = CGRect(
        x: width * 0.5 - autoButtonWidthLandscape / 2,
        y: secondRowY,
        width: autoButtonWidthLandscape,
        height: autoButtonHeightLandscape
      )
      
      // Right side - Threshold 2 (second row right)
      labelThreshold2.frame = CGRect(
        x: width - width * 0.05 - sliderWidth,
        y: secondRowY,
        width: sliderWidth,
        height: sliderLabelHeight
      )
      labelThreshold2.textAlignment = .right
      
      threshold2Slider.frame = CGRect(
        x: width - width * 0.05 - sliderWidth,
        y: secondRowY + sliderLabelHeight + 2,
        width: sliderWidth,
        height: sliderHeight
      )
      
      // Left side - Confidence (third row left)
      let thirdRowY = secondRowY + sliderLabelHeight + sliderHeight + height * 0.02
      
      labelSliderConf.frame = CGRect(
        x: width * 0.05,
        y: thirdRowY,
        width: sliderWidth,
        height: sliderLabelHeight
      )
      
      sliderConf.frame = CGRect(
        x: width * 0.05,
        y: thirdRowY + sliderLabelHeight + 2,
        width: sliderWidth,
        height: sliderHeight
      )
      
      // Right side - IoU (third row right, properly right-aligned)
      labelSliderIoU.frame = CGRect(
        x: width - width * 0.05 - sliderWidth,
        y: thirdRowY,
        width: sliderWidth,
        height: sliderLabelHeight
      )
      labelSliderIoU.textAlignment = .right
      
      sliderIoU.frame = CGRect(
        x: width - width * 0.05 - sliderWidth,
        y: thirdRowY + sliderLabelHeight + 2,
        width: sliderWidth,
        height: sliderHeight
      )
      
      // Update threshold line positions
      updateThresholdLayer(threshold1Layer, position: CGFloat(threshold1Slider.value))
      updateThresholdLayer(threshold2Layer, position: CGFloat(threshold2Slider.value))
      
      // Items slider - center in the screen
      let numItemsSliderWidth: CGFloat = width * 0.25
      let numItemsSliderHeight: CGFloat = height * 0.02
      
      // Center the model dropdown list
      sliderNumItems.frame = CGRect(
        x: (width - numItemsSliderWidth) / 2,
        y: height * 0.3, // Position in center area vertically
        width: numItemsSliderWidth,
        height: numItemsSliderHeight
      )

      // Position label for model dropdown
      labelSliderNumItems.frame = CGRect(
        x: (width - numItemsSliderWidth) / 2,
        y: height * 0.27, // Just above the slider
        width: numItemsSliderWidth,
        height: height * 0.03
      )
      labelSliderNumItems.textAlignment = .center // Center text

      // Zoom indicator position
      let zoomLabelWidth: CGFloat = width * 0.1
      labelZoom.frame = CGRect(
        x: width - zoomLabelWidth - 10,
        y: height * 0.08,
        width: zoomLabelWidth,
        height: height * 0.03
      )

      // Position toolbar at bottom
      toolbar.frame = CGRect(
        x: 0, 
        y: height - toolBarHeight, 
        width: width, 
        height: toolBarHeight
      )
      
      // For landscape, adjust toolbar button spacing 
      let buttonWidth: CGFloat = 50
      let spacing = (width - 5 * buttonWidth) / 6 // 5 buttons (removed switchCameraButton)
      
      playButton.frame = CGRect(
        x: spacing, y: 0, width: buttonWidth, height: toolBarHeight)
      pauseButton.frame = CGRect(
        x: 2 * spacing + buttonWidth, y: 0, width: buttonWidth, height: toolBarHeight)
      switchSourceButton.frame = CGRect(
        x: 3 * spacing + 2 * buttonWidth, y: 0, width: buttonWidth, height: toolBarHeight)
      modelsButton.frame = CGRect(
        x: 4 * spacing + 3 * buttonWidth, y: 0, width: buttonWidth, height: toolBarHeight)
      directionButton.frame = CGRect(
        x: 5 * spacing + 4 * buttonWidth, y: 0, width: buttonWidth, height: toolBarHeight)
    } else {
      // Toolbar background should be completely transparent
      // toolbar.backgroundColor = .black.withAlphaComponent(0.4)
      
      let width = bounds.width
      let height = bounds.height

      let topMargin: CGFloat = height * 0.02

      let titleLabelHeight: CGFloat = height * 0.1
      labelName.frame = CGRect(
        x: 0,
        y: topMargin,
        width: width,
        height: titleLabelHeight
      )
      
      // Position FPS label just above the toolbar
      let toolBarHeight: CGFloat = 66
      let subLabelHeight: CGFloat = height * 0.03
      labelFPS.frame = CGRect(
        x: 0,
        y: height - toolBarHeight - subLabelHeight - 5,
        width: width,
        height: subLabelHeight
      )
      
      // Layout for confidence and IoU sliders in the style shown in the image
      let sliderWidth = width * 0.4
      let sliderHeight: CGFloat = height * 0.02
      let sliderLabelHeight: CGFloat = 20
      let sliderY = height * 0.85 // Position near bottom of screen
      
      // Confidence slider and label (left side)
      labelSliderConf.frame = CGRect(
        x: width * 0.05,
        y: sliderY - sliderLabelHeight - 5,
        width: sliderWidth * 0.5,
        height: sliderLabelHeight
      )
      
      sliderConf.frame = CGRect(
        x: width * 0.05,
        y: sliderY,
        width: sliderWidth,
        height: sliderHeight
      )
      
      // IoU slider and label (right side)
      labelSliderIoU.frame = CGRect(
        x: width * 0.55,
        y: sliderY - sliderLabelHeight - 5,
        width: sliderWidth,
        height: sliderLabelHeight
      )
      labelSliderIoU.textAlignment = .right
      
      sliderIoU.frame = CGRect(
        x: width * 0.55,
        y: sliderY,
        width: sliderWidth,
        height: sliderHeight
      )
      
      // Position fish threshold sliders - closer to confidence/IoU sliders
      let thresholdY = sliderY - sliderHeight - 40 // Position just above the confidence/IoU sliders
      
      // Position Fish Count display and Reset button above threshold togglers
      let fishCountWidth = width * 0.4
      let fishCountHeight: CGFloat = 40
      let fishCountY = thresholdY - sliderHeight - fishCountHeight - 15 // Position above threshold sliders
      let resetButtonWidth = fishCountWidth * 0.5
      
      labelFishCount.frame = CGRect(
        x: width * 0.05,
        y: fishCountY,
        width: fishCountWidth,
        height: fishCountHeight
      )
      
      // Position the Reset button to align with the right edge of threshold2Slider
      resetButton.frame = CGRect(
        x: width * 0.55 + sliderWidth - resetButtonWidth,
        y: fishCountY,
        width: resetButtonWidth,
        height: fishCountHeight
      )
      
      labelThreshold1.frame = CGRect(
        x: width * 0.05,
        y: thresholdY - sliderLabelHeight - 5,
        width: sliderWidth,
        height: sliderLabelHeight
      )
      
      threshold1Slider.frame = CGRect(
        x: width * 0.05,
        y: thresholdY,
        width: sliderWidth,
        height: sliderHeight
      )
      
      // AUTO button (positioned at the same height as threshold labels) - PORTRAIT
      let autoButtonWidthPortrait = width * 0.18
      let autoButtonHeightPortrait: CGFloat = 24
      autoCalibrationButton.frame = CGRect(
        x: (width - autoButtonWidthPortrait) / 2,
        y: thresholdY - sliderLabelHeight,
        width: autoButtonWidthPortrait,
        height: autoButtonHeightPortrait
      )
      
      labelThreshold2.frame = CGRect(
        x: width * 0.55,
        y: thresholdY - sliderLabelHeight - 5,
        width: sliderWidth,
        height: sliderLabelHeight
      )
      labelThreshold2.textAlignment = .right
      
      threshold2Slider.frame = CGRect(
        x: width * 0.55,
        y: thresholdY,
        width: sliderWidth,
        height: sliderHeight
      )
      
      // Update threshold line positions
      updateThresholdLayer(threshold1Layer, position: CGFloat(threshold1Slider.value))
      updateThresholdLayer(threshold2Layer, position: CGFloat(threshold2Slider.value))
      
      // Number of items slider
      let numItemsSliderWidth: CGFloat = width * 0.46
      let numItemsSliderHeight: CGFloat = height * 0.02

      sliderNumItems.frame = CGRect(
        x: width * 0.01,
        y: center.y - numItemsSliderHeight - height * 0.24,
        width: numItemsSliderWidth,
        height: numItemsSliderHeight
      )

      labelSliderNumItems.frame = CGRect(
        x: width * 0.01,
        y: sliderNumItems.frame.minY - numItemsSliderHeight - 10,
        width: numItemsSliderWidth,
        height: numItemsSliderHeight
      )

      let zoomLabelWidth: CGFloat = width * 0.2
      labelZoom.frame = CGRect(
        x: center.x - zoomLabelWidth / 2,
        y: self.bounds.maxY - 120,
        width: zoomLabelWidth,
        height: height * 0.03
      )

      let buttonHeight: CGFloat = toolBarHeight * 0.75
      toolbar.frame = CGRect(x: 0, y: height - toolBarHeight, width: width, height: toolBarHeight)
      
      playButton.frame = CGRect(x: 0, y: 0, width: buttonHeight, height: buttonHeight)
      pauseButton.frame = CGRect(
        x: playButton.frame.maxX, y: 0, width: buttonHeight, height: buttonHeight)
      // switchCameraButton.frame = CGRect(
      //   x: pauseButton.frame.maxX, y: 0, width: buttonHeight, height: buttonHeight)

      // Position switch source button directly after pause button
      switchSourceButton.frame = CGRect(
        x: pauseButton.frame.maxX, 
        y: 0, 
        width: buttonHeight, 
        height: buttonHeight
      )
      
      // Position models button after switch source button
      modelsButton.frame = CGRect(
        x: switchSourceButton.frame.maxX, 
        y: 0, 
        width: buttonHeight, 
        height: buttonHeight
      )

      // Position direction button after models button
      directionButton.frame = CGRect(
        x: modelsButton.frame.maxX, 
        y: 0, 
        width: buttonHeight, 
        height: buttonHeight
      )
    }

    self.videoCapture.previewLayer?.frame = self.bounds

    // Update layout for player layer if using video source
    if frameSourceType == .videoFile, let playerLayer = albumVideoSource?.playerLayer {
      // Set frame to full bounds for proper display
      playerLayer.frame = self.bounds
      
      // Force update of AlbumVideoSource configuration to adapt to new layout
      albumVideoSource?.updateForOrientationChange(orientation: UIDevice.current.orientation)
      
      // Ensure overlay is properly updated
      setupOverlayLayer()
    }
  }

  private func setUpOrientationChangeNotification() {
    NotificationCenter.default.addObserver(
      self, selector: #selector(orientationDidChange),
      name: UIDevice.orientationDidChangeNotification, object: nil)
  }

  @objc func orientationDidChange() {
    let orientation = UIDevice.current.orientation
    
    // Update the current frame source's orientation
    currentFrameSource.updateForOrientationChange(orientation: orientation)
    
    // Update overlay layer
    setupOverlayLayer()
  }

  @objc func sliderChanged(_ sender: Any) {
    if sender as? UISlider === sliderNumItems {
      if let detector = videoCapture.predictor as? ObjectDetector {
        let numItems = Int(sliderNumItems.value)
        detector.setNumItemsThreshold(numItems: numItems)
      }
    }
    let conf = Double(round(100 * sliderConf.value)) / 100
    let iou = Double(round(100 * sliderIoU.value)) / 100
    self.labelSliderConf.text = "Conf: " + String(conf)
    self.labelSliderIoU.text = "IoU: " + String(iou)
    if let detector = videoCapture.predictor as? ObjectDetector {
      detector.setIouThreshold(iou: iou)
      detector.setConfidenceThreshold(confidence: conf)
    }
  }

  @objc func pinch(_ pinch: UIPinchGestureRecognizer) {
    guard let device = videoCapture.captureDevice else { return }

    // Return zoom value between the minimum and maximum zoom values
    func minMaxZoom(_ factor: CGFloat) -> CGFloat {
      return min(min(max(factor, minimumZoom), maximumZoom), device.activeFormat.videoMaxZoomFactor)
    }

    func update(scale factor: CGFloat) {
      do {
        try device.lockForConfiguration()
        defer {
          device.unlockForConfiguration()
        }
        device.videoZoomFactor = factor
      } catch {
        print("\(error.localizedDescription)")
      }
    }

    let newScaleFactor = minMaxZoom(pinch.scale * lastZoomFactor)
    switch pinch.state {
    case .began, .changed:
      update(scale: newScaleFactor)
      self.labelZoom.text = String(format: "%.2fx", newScaleFactor)
      self.labelZoom.font = UIFont.preferredFont(forTextStyle: .title2)
    case .ended:
      lastZoomFactor = minMaxZoom(newScaleFactor)
      update(scale: lastZoomFactor)
      self.labelZoom.font = UIFont.preferredFont(forTextStyle: .body)
    default: break
    }
  }

  @objc func playTapped() {
    selection.selectionChanged()
    
    // Ensure we're in normal inference mode when resuming playback
    currentFrameSource.inferenceOK = true
    
    if frameSourceType == .videoFile, let albumSource = albumVideoSource {
      // For video source, handle special case of restarting
      if !self.pauseButton.isEnabled {
        // If paused or ended, restart from beginning
        albumSource.stop()
        Task { @MainActor in
          // Seek to beginning and start
          if let player = albumSource.playerLayer?.player {
            player.seek(to: CMTime.zero)
            albumSource.start()
          }
        }
      } else {
        // Normal resume
        albumSource.start()
      }
    } else {
      // Camera source - standard behavior
      self.videoCapture.start()
    }
    
    playButton.isEnabled = false
    pauseButton.isEnabled = true
  }

  @objc func pauseTapped() {
    selection.selectionChanged()
    currentFrameSource.stop()
    playButton.isEnabled = true
    pauseButton.isEnabled = false
  }

  public func capturePhoto(completion: @escaping (UIImage?) -> Void) {
    // No-op implementation since photo capture is removed
    completion(nil)
  }

  public func setInferenceFlag(ok: Bool) {
    videoCapture.inferenceOK = ok
  }

  // Setup threshold layers for fish counting
  private func setupThresholdLayers() {
    if threshold1Layer == nil {
      let layer = CAShapeLayer()
      layer.strokeColor = UIColor.red.cgColor
      layer.lineWidth = 3.0
      layer.lineDashPattern = [5, 5] // Creates a dashed line
      layer.zPosition = 999 // Ensure it's on top of other layers
      layer.opacity = 0.5 // 50% transparency
      self.layer.addSublayer(layer)
      threshold1Layer = layer
    }
    
    if threshold2Layer == nil {
      let layer = CAShapeLayer()
      layer.strokeColor = UIColor.yellow.cgColor
      layer.lineWidth = 3.0
      layer.lineDashPattern = [5, 5] // Creates a dashed line
      layer.zPosition = 999 // Ensure it's on top of other layers
      layer.opacity = 0.5 // 50% transparency
      self.layer.addSublayer(layer)
      threshold2Layer = layer
    }
    
    // Force update the threshold lines immediately
    DispatchQueue.main.async {
      self.updateThresholdLayer(self.threshold1Layer, position: CGFloat(self.threshold1Slider.value))
      self.updateThresholdLayer(self.threshold2Layer, position: CGFloat(self.threshold2Slider.value))
    }
  }
  
  // Update the position of a threshold line
  private func updateThresholdLayer(_ layer: CAShapeLayer?, position: CGFloat) {
    guard let layer = layer else { return }
    
    // Position is a value between 0 and 1 representing the position along the relevant axis
    let path = UIBezierPath()
    
    // Calculate the line position based on the current counting direction
    switch countingDirection {
    case .topToBottom:
      // For top to bottom, normal order
    let height = self.bounds.height
    let yPosition = height * position
    
    // Draw a horizontal line across the width of the view
    path.move(to: CGPoint(x: 0, y: yPosition))
    path.addLine(to: CGPoint(x: self.bounds.width, y: yPosition))
      
    case .bottomToTop:
      // For bottom to top, flip the position (1 - position)
      let height = self.bounds.height
      let yPosition = height * (1 - position)
      
      // Draw a horizontal line across the width of the view
      path.move(to: CGPoint(x: 0, y: yPosition))
      path.addLine(to: CGPoint(x: self.bounds.width, y: yPosition))
      
    case .leftToRight:
      // For left to right, normal order
      let width = self.bounds.width
      let xPosition = width * position
      
      // Draw a vertical line across the height of the view
      path.move(to: CGPoint(x: xPosition, y: 0))
      path.addLine(to: CGPoint(x: xPosition, y: self.bounds.height))
      
    case .rightToLeft:
      // For right to left, flip the position (1 - position)
      let width = self.bounds.width
      let xPosition = width * (1 - position)
      
      // Draw a vertical line across the height of the view
      path.move(to: CGPoint(x: xPosition, y: 0))
      path.addLine(to: CGPoint(x: xPosition, y: self.bounds.height))
    }
    
    layer.path = path.cgPath
    layer.isHidden = false
  }
  
  // Update threshold lines for the current direction
  private func updateThresholdLinesForDirection(_ direction: CountingDirection) {
    // Update the slider labels consistently (always use "Threshold 1" and "Threshold 2")
    labelThreshold1.text = "Threshold 1: " + String(format: "%.2f", threshold1Slider.value)
    labelThreshold2.text = "Threshold 2: " + String(format: "%.2f", threshold2Slider.value)
    
    // Update the threshold lines with the current values
    updateThresholdLayer(threshold1Layer, position: threshold1)
    updateThresholdLayer(threshold2Layer, position: threshold2)
    
    // Update tracking detector thresholds
    if task == .fishCount, let trackingDetector = currentFrameSource.predictor as? TrackingDetector {
      trackingDetector.setThresholds([threshold1, threshold2])
    }
  }
  
  // Threshold 1 slider changed
  @objc func threshold1Changed(_ sender: UISlider) {
    let value = CGFloat(sender.value)
    threshold1 = value
    labelThreshold1.text = "Threshold 1: " + String(format: "%.2f", value)
    updateThresholdLayer(threshold1Layer, position: value)
    
    // Update the thresholds in the tracking detector
    if task == .fishCount, let trackingDetector = currentFrameSource.predictor as? TrackingDetector {
      trackingDetector.setThresholds([value, threshold2])
    }
  }
  
  // Threshold 2 slider changed
  @objc func threshold2Changed(_ sender: UISlider) {
    let value = CGFloat(sender.value)
    threshold2 = value
    labelThreshold2.text = "Threshold 2: " + String(format: "%.2f", value)
    updateThresholdLayer(threshold2Layer, position: value)
    
    // Update the thresholds in the tracking detector
    if task == .fishCount, let trackingDetector = currentFrameSource.predictor as? TrackingDetector {
      trackingDetector.setThresholds([threshold1, value])
    }
  }

  @objc func resetFishCount() {
    // Reset the fish count in tracking detector
    if task == .fishCount, let trackingDetector = videoCapture.predictor as? TrackingDetector {
      trackingDetector.resetCount()
      labelFishCount.text = "Fish Count: 0"
    }
  }

  // Toggle auto calibration
  @objc func toggleAutoCalibration() {
    if isCalibrating {
      // If currently calibrating, cancel it
      if task == .fishCount, let trackingDetector = currentFrameSource.predictor as? TrackingDetector {
        trackingDetector.setAutoCalibration(enabled: false)
      }
      
      // Resume video processing by setting inferenceOK to true
      currentFrameSource.inferenceOK = true
      
      // Reset calibration state flag
      isCalibrating = false
      
      // Restore the AUTO split-color button text
      let autoAttributedText = NSMutableAttributedString()
      let auAttributes: [NSAttributedString.Key: Any] = [
        .foregroundColor: UIColor.red.withAlphaComponent(0.5),
        .font: UIFont.systemFont(ofSize: 14, weight: .bold)
      ]
      let toAttributes: [NSAttributedString.Key: Any] = [
        .foregroundColor: UIColor.yellow.withAlphaComponent(0.5),
        .font: UIFont.systemFont(ofSize: 14, weight: .bold)
      ]
      autoAttributedText.append(NSAttributedString(string: "AU", attributes: auAttributes))
      autoAttributedText.append(NSAttributedString(string: "TO", attributes: toAttributes))
      
      autoCalibrationButton.setAttributedTitle(autoAttributedText, for: .normal)
    } else {
      // Not currently calibrating, so start calibration
      isCalibrating = true
      
      // Clear all bounding boxes to avoid lingering boxes during calibration
      // This ensures the bound box views are hidden in the UI
      boundingBoxViews.forEach { box in
        box.hide()
      }
      
      // Explicitly call onClearBoxes on the video capture delegate to ensure boxes are cleared
      // This ensures that sources respond to the clearing request
      if let cameraSource = currentFrameSource as? CameraVideoSource {
          cameraSource.videoCaptureDelegate?.onClearBoxes()
      } else if let albumSource = currentFrameSource as? AlbumVideoSource {
          albumSource.videoCaptureDelegate?.onClearBoxes()
      }
      
      // Show initial progress percentage
      let progressText = NSAttributedString(
        string: "0%",
        attributes: [
          .foregroundColor: UIColor.white.withAlphaComponent(0.5),
          .font: UIFont.systemFont(ofSize: 14, weight: .bold)
        ]
      )
      
      autoCalibrationButton.setAttributedTitle(progressText, for: .normal)
      
      // Set up callbacks for calibration progress and completion
      if task == .fishCount, let trackingDetector = currentFrameSource.predictor as? TrackingDetector {
        // Set up progress callback
        trackingDetector.onCalibrationProgress = { [weak self] progress, total in
          guard let self = self else { return }
          
          // Calculate percentage
          let percentage = Int(Double(progress) / Double(total) * 100.0)
          
          // Update button text with progress percentage
          DispatchQueue.main.async {
            let progressText = NSAttributedString(
              string: "\(percentage)%",
              attributes: [
                .foregroundColor: UIColor.white.withAlphaComponent(0.5),
                .font: UIFont.systemFont(ofSize: 14, weight: .bold)
              ]
            )
            
            self.autoCalibrationButton.setAttributedTitle(progressText, for: .normal)
          }
        }
        
        // Set up completion callback
        trackingDetector.onCalibrationComplete = { [weak self] thresholds in
          guard let self = self else { return }
          
          // Update UI on the main thread
          DispatchQueue.main.async {
            // Update threshold sliders with new values
            if thresholds.count >= 2 {
              self.threshold1Slider.value = Float(thresholds[0])
              self.threshold2Slider.value = Float(thresholds[1])
              
              // Update threshold labels
              self.labelThreshold1.text = "Threshold 1: " + String(format: "%.2f", thresholds[0])
              self.labelThreshold2.text = "Threshold 2: " + String(format: "%.2f", thresholds[1])
              
              // Update threshold lines
              self.updateThresholdLayer(self.threshold1Layer, position: thresholds[0])
              self.updateThresholdLayer(self.threshold2Layer, position: thresholds[1])
              
              // Store the new threshold values
              self.threshold1 = thresholds[0]
              self.threshold2 = thresholds[1]
            }
            
            // Reset calibration state
            self.isCalibrating = false
            
            // Restore the AUTO button text
            let autoAttributedText = NSMutableAttributedString()
            let auAttributes: [NSAttributedString.Key: Any] = [
              .foregroundColor: UIColor.red.withAlphaComponent(0.5),
              .font: UIFont.systemFont(ofSize: 14, weight: .bold)
            ]
            let toAttributes: [NSAttributedString.Key: Any] = [
              .foregroundColor: UIColor.yellow.withAlphaComponent(0.5),
              .font: UIFont.systemFont(ofSize: 14, weight: .bold)
            ]
            autoAttributedText.append(NSAttributedString(string: "AU", attributes: auAttributes))
            autoAttributedText.append(NSAttributedString(string: "TO", attributes: toAttributes))
            
            self.autoCalibrationButton.setAttributedTitle(autoAttributedText, for: .normal)
            
            // IMPORTANT: Resume normal inference
            self.currentFrameSource.inferenceOK = true
          }
        }
        
        // Start auto-calibration
        trackingDetector.setAutoCalibration(enabled: true)
        
        // Pause normal inference during calibration
        currentFrameSource.inferenceOK = false
      }
    }
  }

  // Method to switch between frame sources
  public func switchToFrameSource(_ sourceType: FrameSourceType) {
    // Already using this source type
    if frameSourceType == sourceType {
      return
    }
    
    // Stop current frame source
    currentFrameSource.stop()
    
    // Clear any existing bounding boxes
    boundingBoxViews.forEach { box in
      box.hide()
    }
    
    // Clear any existing layers that might show outdated content
    resetLayers()
    
    // Reset calibration state if in progress
    if isCalibrating {
      isCalibrating = false
      
      // Restore the AUTO button text
      let autoAttributedText = NSMutableAttributedString()
      let auAttributes: [NSAttributedString.Key: Any] = [
        .foregroundColor: UIColor.red.withAlphaComponent(0.5),
        .font: UIFont.systemFont(ofSize: 14, weight: .bold)
      ]
      let toAttributes: [NSAttributedString.Key: Any] = [
        .foregroundColor: UIColor.yellow.withAlphaComponent(0.5),
        .font: UIFont.systemFont(ofSize: 14, weight: .bold)
      ]
      autoAttributedText.append(NSAttributedString(string: "AU", attributes: auAttributes))
      autoAttributedText.append(NSAttributedString(string: "TO", attributes: toAttributes))
      
      autoCalibrationButton.setAttributedTitle(autoAttributedText, for: .normal)
    }
    
    switch sourceType {
    case .camera:
      // Make sure camera permissions are granted
      videoCapture.requestPermission { [weak self] granted in
        guard let self = self else { return }
        
        if !granted {
          // Show alert about camera permission if denied
          self.showPermissionAlert(for: .camera)
          return
        }
        
        // Remove any existing video player layer
        if let albumSource = self.albumVideoSource, let playerLayer = albumSource.playerLayer {
          playerLayer.removeFromSuperlayer()
        }
        
        // Show camera preview layer
        if let previewLayer = self.videoCapture.previewLayer {
          previewLayer.isHidden = false
        }
        
        // Use camera as current frame source
        self.currentFrameSource = self.videoCapture
        self.frameSourceType = .camera
        
        // Ensure inferenceOK is set to true for the new source
        self.currentFrameSource.inferenceOK = true
        
        // Start camera capture
        self.start(position: .back)
      }
      
    case .videoFile:
      // Create album video source if needed
      if albumVideoSource == nil {
        albumVideoSource = AlbumVideoSource()
        albumVideoSource?.predictor = videoCapture.predictor
        albumVideoSource?.delegate = self
        albumVideoSource?.videoCaptureDelegate = self
        
        // Register for video playback end notification
        NotificationCenter.default.addObserver(
          self,
          selector: #selector(handleVideoPlaybackEnd),
          name: .videoPlaybackDidEnd,
          object: nil
        )
      }
      
      // Hide camera preview layer
      if let previewLayer = videoCapture.previewLayer {
        previewLayer.isHidden = true
      }
      
      // Find the current view controller to present the picker
      var topViewController = UIApplication.shared.windows.first?.rootViewController
      while let presentedViewController = topViewController?.presentedViewController {
        topViewController = presentedViewController
      }
      
      guard let viewController = topViewController else {
        print("Could not find a view controller to present the video picker")
        return
      }
      
      // Show content selection UI
      albumVideoSource?.showContentSelectionUI(from: viewController) { [weak self] success in
        guard let self = self else { return }
        
        if !success {
          // If selection was cancelled or failed, switch back to camera
          self.switchToFrameSource(.camera)
          return
        }
        
        // If selection was successful, set up the player layer
        if let playerLayer = self.albumVideoSource?.playerLayer {
          playerLayer.frame = self.bounds
          
          // Insert player layer at index 0 (same as camera preview layer)
          self.layer.insertSublayer(playerLayer, at: 0)
          
          // Add overlay layer to player layer
          playerLayer.addSublayer(self.overlayLayer)
          
          // Add bounding box views to the overlay
          for box in self.boundingBoxViews {
            box.addToLayer(playerLayer)
          }
          
          // Update the overlay layer frame to match the view bounds
          self.setupOverlayLayer()
        }
        
        // Set as current frame source
        self.currentFrameSource = self.albumVideoSource!
        self.frameSourceType = .videoFile
        
        // Ensure inferenceOK is set to true for the new source
        self.currentFrameSource.inferenceOK = true
      }
      
    case .goPro:
      // Find the current view controller to present alerts
      var topViewController = UIApplication.shared.windows.first?.rootViewController
      while let presentedViewController = topViewController?.presentedViewController {
        topViewController = presentedViewController
      }
      
      guard let viewController = topViewController else {
        print("Could not find a view controller to present alerts")
        return
      }
      
      // Show GoPro connection prompt - this will handle the full setup flow
      self.showGoProConnectionPrompt(viewController: viewController)
      
    default:
      // For other future source types
      break
    }
  }
  
  private func showPermissionAlert(for sourceType: FrameSourceType) {
    // Find the current view controller to present the alert
    var topViewController = UIApplication.shared.windows.first?.rootViewController
    while let presentedViewController = topViewController?.presentedViewController {
      topViewController = presentedViewController
    }
    
    guard let viewController = topViewController else { return }
    
    let title: String
    let message: String
    
    switch sourceType {
    case .camera:
      title = "Camera Access Required"
      message = "Please enable camera access in Settings to use this feature."
    case .videoFile:
      title = "Photo Library Access Required"
      message = "Please enable photo library access in Settings to use this feature."
    default:
      return
    }
    
    let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
    alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
    alert.addAction(UIAlertAction(title: "Settings", style: .default) { _ in
      if let url = URL(string: UIApplication.openSettingsURLString) {
        UIApplication.shared.open(url)
      }
    })
    
    viewController.present(alert, animated: true)
  }

  @objc func switchSourceButtonTapped() {
    // Find the current view controller to present the alert
    var topViewController = UIApplication.shared.windows.first?.rootViewController
    while let presentedViewController = topViewController?.presentedViewController {
      topViewController = presentedViewController
    }
    
    guard let viewController = topViewController else { return }
    
    // Create an alert controller for source selection
    let alert = UIAlertController(title: "Select Frame Source", message: nil, preferredStyle: .actionSheet)
    
    // Add action for each source type
    // Camera source
    let cameraAction = UIAlertAction(title: "Camera", style: .default) { [weak self] _ in
      guard let self = self else { return }
      if self.frameSourceType != .camera {
        self.switchToFrameSource(.camera)
      }
    }
    // Add checkmark to current source
    if frameSourceType == .camera {
      cameraAction.setValue(true, forKey: "checked")
    }
    alert.addAction(cameraAction)
    
    // Video file source - renamed to "Album"
    let albumAction = UIAlertAction(title: "Album", style: .default) { [weak self] _ in
      guard let self = self else { return }
      if self.frameSourceType != .videoFile {
        self.switchToFrameSource(.videoFile)
      }
    }
    // Add checkmark to current source
    if frameSourceType == .videoFile {
      albumAction.setValue(true, forKey: "checked")
    }
    alert.addAction(albumAction)
    
    // GoPro Hero action (now enabled)
    let goProAction = UIAlertAction(title: "GoPro Hero", style: .default) { [weak self] _ in
      guard let self = self, let viewController = topViewController else { return }
      
      // Show connection instructions alert with Back and Next buttons
      self.showGoProConnectionPrompt(viewController: viewController)
    }
    
    alert.addAction(goProAction)
    
    // Cancel action
    alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))
    
    // For iPad support
    if let popoverController = alert.popoverPresentationController {
      popoverController.sourceView = switchSourceButton
      popoverController.sourceRect = switchSourceButton.bounds
    }
    
    // Present the alert
    viewController.present(alert, animated: true, completion: nil)
  }

  // New method to show GoPro connection prompt with Back and Next buttons
  private func showGoProConnectionPrompt(viewController: UIViewController) {
    // Create alert with instruction text
    let connectionAlert = UIAlertController(
        title: "GoPro Connection Required",
        message: "Please connect to GoPro WiFi via GoPro Quik",
        preferredStyle: .alert
    )
    
    // Add Back button (cancel)
    connectionAlert.addAction(UIAlertAction(title: "Back", style: .cancel))
    
    // Add Next button to check connection
    connectionAlert.addAction(UIAlertAction(title: "Next", style: .default) { [weak self] _ in
        guard let self = self else { return }
        
        // Show activity indicator
        let loadingAlert = UIAlertController(
            title: "Checking Connection",
            message: "Connecting to GoPro...",
            preferredStyle: .alert
        )
        viewController.present(loadingAlert, animated: true)
        
        // Create GoPro source instance to check connection
        let goProSource = GoProSource()
        
        // Set timeout for the request
        let taskGroup = DispatchGroup()
        taskGroup.enter()
        
        var connectionResult: Result<GoProWebcamVersion, Error>?
        
        // Timeout handling
        let timeoutTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: false) { _ in
            if connectionResult == nil {
                // Create timeout error
                let timeoutError = NSError(
                    domain: "GoProSource",
                    code: NSURLErrorTimedOut,
                    userInfo: [NSLocalizedDescriptionKey: "Connection timed out. Please verify you are connected to the GoPro WiFi network."]
                )
                connectionResult = .failure(timeoutError)
                taskGroup.leave()
            }
        }
        
        // Check GoPro connection
        goProSource.checkConnection { result in
            // Only process result if we haven't timed out
            if connectionResult == nil {
                connectionResult = result
                timeoutTimer.invalidate()
                taskGroup.leave()
            }
        }
        
        // After completion (success or failure)
        taskGroup.notify(queue: .main) {
            // Always invalidate timer
            timeoutTimer.invalidate()
            
            loadingAlert.dismiss(animated: true) {
                // Handle any unexpected errors gracefully
                guard let result = connectionResult else {
                    // This should never happen, but just in case
                    self.showConnectionError(
                        viewController: viewController,
                        message: "An unexpected error occurred. Please try again."
                    )
                    return
                }
                
                switch result {
                case .success(_):
                    print("GoPro: Showing connection success dialog")
                    // Connection successful - show enable webcam prompt
                    let successAlert = UIAlertController(
                        title: "GoPro Connected",
                        message: "Connection to GoPro was successful. Enable Webcam mode?",
                        preferredStyle: .alert
                    )
                    
                    // Add Cancel button
                    successAlert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
                    
                    // Add Enable button for webcam initialization
                    successAlert.addAction(UIAlertAction(title: "Enable", style: .default) { [weak self] _ in
                        guard let self = self else { return }
                        
                        // Show loading indicator during initialization
                        let loadingAlert = UIAlertController(
                            title: "Initializing Webcam",
                            message: "Setting up GoPro webcam mode...",
                            preferredStyle: .alert
                        )
                        viewController.present(loadingAlert, animated: true)
                        
                        // Create GoPro source for initialization
                        let goProSource = GoProSource()
                        
                        // Step 1: Enter preview mode
                        goProSource.enterWebcamPreview { result in
                            switch result {
                            case .success:
                                // Step 2: Start webcam
                                goProSource.startWebcam { startResult in
                                    // Dismiss loading indicator
                                    loadingAlert.dismiss(animated: true) {
                                        switch startResult {
                                        case .success:
                                            // Show success dialog with stream option
                                            let startedAlert = UIAlertController(
                                                title: "Webcam Started",
                                                message: "GoPro webcam is ready. Start streaming?",
                                                preferredStyle: .alert
                                            )
                                            
                                            // Add Cancel button with graceful exit
                                            startedAlert.addAction(UIAlertAction(title: "Cancel", style: .cancel) { [weak self] _ in
                                                guard let self = self else { return }
                                                
                                                // Show loading indicator during exit
                                                let exitingAlert = UIAlertController(
                                                    title: "Exiting Webcam",
                                                    message: "Closing GoPro webcam mode...",
                                                    preferredStyle: .alert
                                                )
                                                viewController.present(exitingAlert, animated: true)
                                                
                                                // Perform graceful exit
                                                goProSource.gracefulWebcamExit { result in
                                                    // Dismiss loading indicator
                                                    exitingAlert.dismiss(animated: true) {
                                                        switch result {
                                                        case .success:
                                                            // Successfully exited - switch back to camera source
                                                            self.switchToFrameSource(.camera)
                                                            
                                                        case .failure(let error):
                                                            // Show error with retry option
                                                            let errorAlert = UIAlertController(
                                                                title: "Exit Failed",
                                                                message: "Failed to exit webcam mode: \(error.localizedDescription)\nRetry exit?",
                                                                preferredStyle: .alert
                                                            )
                                                            
                                                            // Add Retry button
                                                            errorAlert.addAction(UIAlertAction(title: "Retry", style: .default) { [weak self] _ in
                                                                guard let self = self else { return }
                                                                // Retry the exit process
                                                                viewController.present(exitingAlert, animated: true)
                                                                goProSource.gracefulWebcamExit { retryResult in
                                                                    exitingAlert.dismiss(animated: true) {
                                                                        switch retryResult {
                                                                        case .success:
                                                                            // Successfully exited on retry
                                                                            self.switchToFrameSource(.camera)
                                                                        case .failure:
                                                                            // If retry fails, force switch to camera
                                                                            print("GoPro: Exit retry failed, forcing camera switch")
                                                                            self.switchToFrameSource(.camera)
                                                                        }
                                                                    }
                                                                }
                                                            })
                                                            
                                                            // Add Force Exit button
                                                            errorAlert.addAction(UIAlertAction(title: "Force Exit", style: .destructive) { [weak self] _ in
                                                                guard let self = self else { return }
                                                                // Force switch to camera source
                                                                print("GoPro: Forcing camera switch after exit failure")
                                                                self.switchToFrameSource(.camera)
                                                            })
                                                            
                                                            viewController.present(errorAlert, animated: true)
                                                        }
                                                    }
                                                }
                                            })
                                            
                                            // Add Stream button that will start real fish counting with GoPro source
                                            startedAlert.addAction(UIAlertAction(title: "Stream", style: .default) { [weak self] _ in
                                                guard let self = self else { return }
                                                // Use our new optimized method to initialize GoPro fish counting
                                                self.initializeGoProFishCounting(viewController: viewController)
                                            })
                                            
                                            viewController.present(startedAlert, animated: true)
                                            
                                        case .failure(let error):
                                            // Show error with retry option
                                            self.showConnectionError(
                                                viewController: viewController,
                                                message: "Failed to start webcam: \(error.localizedDescription)"
                                            )
                                        }
                                    }
                                }
                                
                            case .failure(let error):
                                // Dismiss loading indicator and show error
                                loadingAlert.dismiss(animated: true) {
                                    self.showConnectionError(
                                        viewController: viewController,
                                        message: "Failed to enter preview mode: \(error.localizedDescription)"
                                    )
                                }
                            }
                        }
                    })
                    
                    viewController.present(successAlert, animated: true)
                    
                case .failure(let error):
                    print("GoPro: Connection failed - \(error.localizedDescription)")
                    
                    // Show error with retry option
                    self.showConnectionError(
                        viewController: viewController,
                        message: error.localizedDescription
                    )
                }
            }
        }
    })
    
    viewController.present(connectionAlert, animated: true)
  }

  // Add GoPro RTSP stream test function
  private func testGoProRTSPStream(viewController: UIViewController) {
    // Create loading alert
    let loadingAlert = UIAlertController(
      title: "Testing RTSP Stream",
      message: "Connecting to GoPro RTSP stream...",
      preferredStyle: .alert
    )
    
    // Show cancel button
    loadingAlert.addAction(UIAlertAction(title: "Cancel", style: .cancel) { _ in
      // Create a GoPro source just to stop any streams
      let goProSource = GoProSource()
      goProSource.stopRTSPStream()
    })
    
    viewController.present(loadingAlert, animated: true)
    
    // Create GoPro source and test RTSP streaming
    let goProSource = GoProSource()
    
    // Save strong reference to prevent premature deallocation
    self.tempGoProSource = goProSource
    
    // Run a simple test with shorter timeout
    goProSource.testRTSPStream(timeout: 8.0) { success, log in
      // Update alert on main thread
      DispatchQueue.main.async {
        loadingAlert.dismiss(animated: true) {
          // Show result
          self.showStreamTestResults(
            viewController: viewController,
            success: success,
            message: success ? "Successfully connected to GoPro RTSP stream!" : "Could not connect to RTSP stream",
            log: log
          )
          
          // Clean up
          goProSource.stopRTSPStream()
          self.tempGoProSource = nil
        }
      }
    }
  }
  
  // Improved method for handling GoPro streams
  @MainActor
  func optimizeForGoProSource(_ goProSource: GoProSource) {
    // Ensure the view is prepared for GoPro-specific display
    print("YOLOView: Optimizing for GoPro source")
    
    // Stop current source first
    if frameSourceType != .goPro {
      print("YOLOView: Stopping previous source: \(frameSourceType)")
      currentFrameSource.stop()
      
      // Remove any existing album video player layer
      if let albumSource = albumVideoSource, let playerLayer = albumSource.playerLayer {
        playerLayer.removeFromSuperlayer()
      }
      
      // Hide camera preview layer if visible
      if let previewLayer = videoCapture.previewLayer {
        previewLayer.isHidden = true
      }
    }
    
    // Update frameSourceType
    frameSourceType = .goPro
    currentFrameSource = goProSource
    
    // Get the predictor from the previously active source, if available
    let previousPredictor = getCurrentPredictor()
    
    // Store reference to source
    self.goProSource = goProSource
    
    // Share the predictor between sources for consistent detection
    if let existingPredictor = previousPredictor {
      print("YOLOView: Shared predictor of type \(type(of: existingPredictor)) with GoPro source")
      goProSource.predictor = existingPredictor
    }
    
    // Configure for fish counting if needed
    if task == .fishCount, let trackingDetector = goProSource.predictor as? TrackingDetector {
      print("YOLOView: Configuring TrackingDetector for GoPro source")
      // Set detection thresholds - use thresholds array format
      let minThreshold = CGFloat(threshold1Slider.value)
      let maxThreshold = CGFloat(threshold2Slider.value)
      trackingDetector.setThresholds([minThreshold, maxThreshold])
      
      // Set counting direction
      trackingDetector.setCountingDirection(countingDirection)
      print("YOLOView: TrackingDetector configured with thresholds [\(minThreshold), \(maxThreshold)], direction: \(countingDirection)")
    }
    
    // Set up integration with proper delegates
    goProSource.integrateWithYOLOView(view: self)
    
    // CRITICAL FIX: Ensure bounding box views are properly set up for GoPro source
    print("YOLOView: Setting up bounding box views for GoPro source")
    
    // Reset all bounding box views
    boundingBoxViews.forEach { box in
      box.hide()
    }
    
    // Setup overlay layer
    setupOverlayLayer()
    
    // Ensure all bounding box views are added to the right layer
    if let goProPlayerView = goProSource.playerView {
      boundingBoxViews.forEach { box in
        box.addToLayer(goProPlayerView.layer)
      }
      
      // Add overlay layer to the player view
      goProPlayerView.layer.addSublayer(overlayLayer)
    }
    
    // Register to receive frame size updates
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(updateGoProFrameSize(_:)),
      name: NSNotification.Name("GoProFrameSizeChanged"),
      object: nil
    )
    
    // Start streaming
    goProSource.start()
  }
  
  // Method to receive frame size updates from GoPro source
  @objc func updateGoProFrameSize(_ notification: Notification) {
    if let frameSize = notification.userInfo?["frameSize"] as? CGSize {
      print("YOLOView: Received GoPro frame size update: \(frameSize)")
      self.goProLastFrameSize = frameSize
      
      // Update layout if needed to ensure proper coordinate transformation
      DispatchQueue.main.async {
        self.setNeedsLayout()
        self.layoutIfNeeded()
        
        // Force redraw of bounding boxes to ensure they're properly positioned
        for box in self.boundingBoxViews where !box.shapeLayer.isHidden {
          box.shapeLayer.setNeedsDisplay()
        }
      }
    }
  }
  
  // Method to initialize GoProSource for fish counting
  private func initializeGoProFishCounting(viewController: UIViewController) {
    // Create loading alert
    let loadingAlert = UIAlertController(
      title: "Starting GoPro Stream",
      message: "Connecting to GoPro RTSP stream for fish counting...",
      preferredStyle: .alert
    )
    
    // Show cancel button
    loadingAlert.addAction(UIAlertAction(title: "Cancel", style: .cancel) { _ in
      // Create a GoPro source just to stop any streams
      let goProSource = GoProSource()
      goProSource.stopRTSPStream()
    })
    
    viewController.present(loadingAlert, animated: true)
    
    // Create a fresh GoPro source for fish counting
    let goProSource = GoProSource()
    
    // Set up the GoProSource
    goProSource.predictor = videoCapture.predictor
    goProSource.setUp { success in
      if !success {
        DispatchQueue.main.async {
          loadingAlert.dismiss(animated: true) {
            // Show error
            let errorAlert = UIAlertController(
              title: "Setup Failed",
              message: "Failed to set up GoPro stream.",
              preferredStyle: .alert
            )
            errorAlert.addAction(UIAlertAction(title: "OK", style: .default))
            viewController.present(errorAlert, animated: true)
          }
        }
        return
      }
      
      // Setup successful, now start the RTSP stream
      Task {
        do {
          // Create a callback Task to handle the result asynchronously
          let streamResultTask = Task<Result<Void, Error>, Never> { 
            return await withCheckedContinuation { continuation in
              // Need to call via MainActor.run since goProSource.startRTSPStream is @MainActor isolated
              Task { @MainActor in
                goProSource.startRTSPStream { result in
                  continuation.resume(returning: result)
                }
              }
            }
          }
          
          // Wait for the result with a timeout
          let result = await streamResultTask.value
          
          // Now we can safely run UI updates on the main actor
          await MainActor.run {
            switch result {
            case .success:
              // Stream started successfully - use our optimized method
              self.optimizeForGoProSource(goProSource)
              
              // Dismiss loading indicator
              loadingAlert.dismiss(animated: true) {
                // Show success message
                let successAlert = UIAlertController(
                  title: "GoPro Stream Active",
                  message: "GoPro RTSP stream is now connected and ready for fish counting!",
                  preferredStyle: .alert
                )
                successAlert.addAction(UIAlertAction(title: "OK", style: .default))
                viewController.present(successAlert, animated: true)
              }
              
              // Save reference to prevent deallocation
              self.tempGoProSource = goProSource
              
            case .failure(let error):
              // Stream failed to start
              loadingAlert.dismiss(animated: true) {
                // Show error
                let errorAlert = UIAlertController(
                  title: "Stream Failed",
                  message: "Failed to start GoPro stream: \(error.localizedDescription)",
                  preferredStyle: .alert
                )
                
                // Add option to retry
                errorAlert.addAction(UIAlertAction(title: "Retry", style: .default) { [weak self] _ in
                  self?.initializeGoProFishCounting(viewController: viewController)
                })
                
                errorAlert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
                viewController.present(errorAlert, animated: true)
              }
            }
          }
        } catch {
          await MainActor.run {
            loadingAlert.dismiss(animated: true) {
              // Show error for any exceptions
              let errorAlert = UIAlertController(
                title: "Error",
                message: "An unexpected error occurred: \(error.localizedDescription)",
                preferredStyle: .alert
              )
              errorAlert.addAction(UIAlertAction(title: "OK", style: .default))
              viewController.present(errorAlert, animated: true)
            }
          }
        }
      }
    }
  }
  
  // Show RTSP stream test results
  private func showStreamTestResults(viewController: UIViewController, success: Bool, message: String, log: String) {
    let title = success ? "Stream Test Successful" : "Stream Test Failed"
    let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
    
    // Add action to view detailed log
    alert.addAction(UIAlertAction(title: "View Details", style: .default) { _ in
      let logAlert = UIAlertController(title: "Stream Test Log", message: log, preferredStyle: .alert)
      logAlert.addAction(UIAlertAction(title: "Close", style: .cancel))
      viewController.present(logAlert, animated: true)
    })
    
    // Add OK button
    alert.addAction(UIAlertAction(title: "OK", style: .default))
    
    viewController.present(alert, animated: true)
  }

  // Helper method to show connection errors with consistent UI
  private func showConnectionError(viewController: UIViewController, message: String) {
    let failureAlert = UIAlertController(
        title: "Connection Failed",
        message: message,
        preferredStyle: .alert
    )
    
    // Add Try Again button
    failureAlert.addAction(UIAlertAction(title: "Try Again", style: .default) { [weak self] _ in
        guard let self = self else { return }
        self.showGoProConnectionPrompt(viewController: viewController)
    })
    
    // Add Cancel button
    failureAlert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
    
    viewController.present(failureAlert, animated: true)
  }

  @objc func handleVideoPlaybackEnd(_ notification: Notification) {
    // When video playback ends, provide visual feedback
    DispatchQueue.main.async {
      // Enable the play button and disable the pause button
      self.playButton.isEnabled = true
      self.pauseButton.isEnabled = false
    }
  }

  // Add a method to handle the models button tap
  @objc func modelsButtonTapped() {
    selection.selectionChanged()
    // Notify the ViewController that the models button was tapped
    actionDelegate?.didTapModelsButton()
  }

  // Add method to handle direction button tap
  @objc func directionButtonTapped() {
    selection.selectionChanged()
    
    // Find the current view controller to present the alert
    var topViewController = UIApplication.shared.windows.first?.rootViewController
    while let presentedViewController = topViewController?.presentedViewController {
        topViewController = presentedViewController
    }
    
    guard let viewController = topViewController else { return }
    
    // Create an alert controller for direction selection
    let alert = UIAlertController(title: "Select Counting Direction", message: nil, preferredStyle: .actionSheet)
    
    // Add action for each direction
    // Top to Bottom
    let topToBottomAction = UIAlertAction(title: "Top to Bottom", style: .default) { [weak self] _ in
        guard let self = self else { return }
        if self.countingDirection != .topToBottom {
            self.switchCountingDirection(.topToBottom)
        }
    }
    // Add checkmark to current direction
    if countingDirection == .topToBottom {
        topToBottomAction.setValue(true, forKey: "checked")
    }
    alert.addAction(topToBottomAction)
    
    // Bottom to Top
    let bottomToTopAction = UIAlertAction(title: "Bottom to Top", style: .default) { [weak self] _ in
        guard let self = self else { return }
        if self.countingDirection != .bottomToTop {
            self.switchCountingDirection(.bottomToTop)
        }
    }
    if countingDirection == .bottomToTop {
        bottomToTopAction.setValue(true, forKey: "checked")
    }
    alert.addAction(bottomToTopAction)
    
    // Left to Right
    let leftToRightAction = UIAlertAction(title: "Left to Right", style: .default) { [weak self] _ in
        guard let self = self else { return }
        if self.countingDirection != .leftToRight {
            self.switchCountingDirection(.leftToRight)
        }
    }
    if countingDirection == .leftToRight {
        leftToRightAction.setValue(true, forKey: "checked")
    }
    alert.addAction(leftToRightAction)
    
    // Right to Left
    let rightToLeftAction = UIAlertAction(title: "Right to Left", style: .default) { [weak self] _ in
        guard let self = self else { return }
        if self.countingDirection != .rightToLeft {
            self.switchCountingDirection(.rightToLeft)
        }
    }
    if countingDirection == .rightToLeft {
        rightToLeftAction.setValue(true, forKey: "checked")
    }
    alert.addAction(rightToLeftAction)
    
    // Cancel action
    alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))
    
    // For iPad support
    if let popoverController = alert.popoverPresentationController {
        popoverController.sourceView = directionButton
        popoverController.sourceRect = directionButton.bounds
    }
    
    // Present the alert
    viewController.present(alert, animated: true, completion: nil)
  }

  // Add method to switch counting direction
  private func switchCountingDirection(_ direction: CountingDirection) {
    // Store the new direction
    countingDirection = direction
    
    // Update the tracking detector's direction
    if task == .fishCount, let trackingDetector = currentFrameSource.predictor as? TrackingDetector {
        trackingDetector.setCountingDirection(direction)
    }
    
    // Re-draw the threshold lines for the new direction
    updateThresholdLinesForDirection(direction)
  }

  /// Get the current predictor from the active frame source
  func getCurrentPredictor() -> Predictor? {
    return currentFrameSource.predictor
  }

  // Helper method to update the fish count display
  private func updateFishCountDisplay() {
    // Update the fish count label with the current count
    labelFishCount.text = "Fish Count: \(fishCount)"
  }
}

// Empty implementation to maintain compatibility
extension YOLOView: AVCapturePhotoCaptureDelegate {
  // Photo capture functionality has been removed
}
