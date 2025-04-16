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

/// A UIView component that provides real-time object detection, segmentation, and pose estimation capabilities.
@MainActor
public class YOLOView: UIView, VideoCaptureDelegate, FrameSourceDelegate {
  func onInferenceTime(speed: Double, fps: Double) {
    DispatchQueue.main.async {
      self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, speed)  // t2 seconds to ms
    }
  }

  func onPredict(result: YOLOResult) {
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
  private var videoCapture: VideoCapture
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

  let selection = UISelectionFeedbackGenerator()
  private var overlayLayer = CALayer()
  private var maskLayer: CALayer?
  private var poseLayer: CALayer?
  private var obbLayer: CALayer?

  let obbRenderer = OBBRenderer()

  private let minimumZoom: CGFloat = 1.0
  private let maximumZoom: CGFloat = 10.0
  private var lastZoomFactor: CGFloat = 1.0

  public var capturedImage: UIImage?
  private var photoCaptureCompletion: ((UIImage?) -> Void)?

  public init(
    frame: CGRect,
    modelPathOrName: String,
    task: YOLOTask
  ) {
    self.videoCapture = VideoCapture()
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
    self.videoCapture = VideoCapture()
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

    if UIDevice.current.orientation == .portrait {
      var ratio: CGFloat = 1.0

      // Use the session preset from the active frame source (only VideoCapture has this)
      if currentFrameSource.sourceType == .camera && videoCapture.captureSession.sessionPreset == .photo {
        ratio = (height / width) / (4.0 / 3.0)
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
            // Convert normalized coordinates to screen coordinates based on video content rect
            let screenRect = albumSource.convertNormalizedRectToScreenRect(displayRect)
            
            // Set the box with the converted rect
            boundingBoxViews[i].show(
              frame: screenRect, label: label, color: boxColor, alpha: alpha)
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
        } else {
          boundingBoxViews[i].hide()
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

      // Important: First hide all boxes, then show only the active ones
      for i in 0..<boundingBoxViews.count {
        boundingBoxViews[i].hide()
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
            // Convert normalized coordinates to screen coordinates based on video content rect
            let screenRect = albumSource.convertNormalizedRectToScreenRect(rect)
            
            // Set the box with the converted rect
            boundingBoxViews[i].show(
              frame: screenRect,
              label: label,
              color: boxColor,
              alpha: alpha
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
    labelSliderIoU.textAlignment = .left
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
    labelThreshold2.textAlignment = .left
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

    let config = UIImage.SymbolConfiguration(pointSize: 20, weight: .regular, scale: .default)

    playButton.setImage(UIImage(systemName: "play.fill", withConfiguration: config), for: .normal)
    playButton.tintColor = .systemGray
    pauseButton.setImage(UIImage(systemName: "pause.fill", withConfiguration: config), for: .normal)
    pauseButton.tintColor = .systemGray
    switchCameraButton = UIButton()
    switchCameraButton.setImage(
      UIImage(systemName: "camera.rotate", withConfiguration: config), for: .normal)
    switchCameraButton.tintColor = .systemGray
    playButton.isEnabled = false
    pauseButton.isEnabled = true
    playButton.addTarget(self, action: #selector(playTapped), for: .touchUpInside)
    pauseButton.addTarget(self, action: #selector(pauseTapped), for: .touchUpInside)
    switchCameraButton.addTarget(self, action: #selector(switchCameraTapped), for: .touchUpInside)
    toolbar.backgroundColor = .darkGray.withAlphaComponent(0.7)
    self.addSubview(toolbar)
    toolbar.addSubview(playButton)
    toolbar.addSubview(pauseButton)
    toolbar.addSubview(switchCameraButton)

    // Create switch source button with matching style
    switchSourceButton.setImage(UIImage(systemName: "photo.on.rectangle", withConfiguration: config), for: .normal)
    switchSourceButton.tintColor = .systemGray // Match other buttons' tint color
    switchSourceButton.backgroundColor = .clear // Remove background color to match other buttons
    switchSourceButton.addTarget(self, action: #selector(switchSourceButtonTapped), for: .touchUpInside)
    toolbar.addSubview(switchSourceButton)

    self.addGestureRecognizer(UIPinchGestureRecognizer(target: self, action: #selector(pinch)))
  }

  public override func layoutSubviews() {
    setupOverlayLayer()
    let isLandscape = bounds.width > bounds.height
    activityIndicator.frame = CGRect(x: center.x - 50, y: center.y - 50, width: 100, height: 100)
    
    // Setup threshold lines if needed
    setupThresholdLayers()
    
    if isLandscape {
      toolbar.backgroundColor = .clear
      playButton.tintColor = .darkGray
      pauseButton.tintColor = .darkGray
      switchCameraButton.tintColor = .darkGray

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
      
      // Right side - Threshold 2 (second row right)
      labelThreshold2.frame = CGRect(
        x: width - width * 0.05 - sliderWidth,
        y: secondRowY,
        width: sliderWidth,
        height: sliderLabelHeight
      )
      
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
      
      // Right side - IoU (third row right)
      labelSliderIoU.frame = CGRect(
        x: width - width * 0.05 - sliderWidth,
        y: thirdRowY,
        width: sliderWidth,
        height: sliderLabelHeight
      )
      
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
      
      let buttonSize: CGFloat = 40
      playButton.frame = CGRect(
        x: 10, 
        y: (toolBarHeight - buttonSize) / 2, 
        width: buttonSize, 
        height: buttonSize
      )
      pauseButton.frame = CGRect(
        x: playButton.frame.maxX + 10, 
        y: (toolBarHeight - buttonSize) / 2, 
        width: buttonSize, 
        height: buttonSize
      )
      switchCameraButton.frame = CGRect(
        x: pauseButton.frame.maxX + 10, 
        y: (toolBarHeight - buttonSize) / 2, 
        width: buttonSize, 
        height: buttonSize
      )

      // Position switch source button in landscape mode
      switchSourceButton.frame = CGRect(
        x: switchCameraButton.frame.maxX + 10, 
        y: (toolBarHeight - buttonSize) / 2, 
        width: buttonSize, 
        height: buttonSize
      )
    } else {
      toolbar.backgroundColor = .darkGray.withAlphaComponent(0.7)
      playButton.tintColor = .systemGray
      pauseButton.tintColor = .systemGray
      switchCameraButton.tintColor = .systemGray

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
        width: sliderWidth * 0.5,
        height: sliderLabelHeight
      )
      
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
      
      labelThreshold2.frame = CGRect(
        x: width * 0.55,
        y: thresholdY - sliderLabelHeight - 5,
        width: sliderWidth,
        height: sliderLabelHeight
      )
      
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
      switchCameraButton.frame = CGRect(
        x: pauseButton.frame.maxX, y: 0, width: buttonHeight, height: buttonHeight)

      // Position switch source button in portrait mode
      switchSourceButton.frame = CGRect(
        x: switchCameraButton.frame.maxX, 
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
      albumVideoSource?.updateForOrientationChange()
      
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
      return
    }
    
    // Update camera orientation if using camera source
    if frameSourceType == .camera {
      videoCapture.updateVideoOrientation(orientation: orientation)
    }
    
    // Update player layer when orientation changes
    if frameSourceType == .videoFile, let albumSource = albumVideoSource {
      // First update the player layer frame
      if let playerLayer = albumSource.playerLayer {
        playerLayer.frame = self.bounds
      }
      
      // Then update the album source's orientation handling
      albumSource.updateForOrientationChange()
      
      // Update overlay layer
      setupOverlayLayer()
    }
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

  @objc func switchCameraTapped() {
    self.videoCapture.captureSession.beginConfiguration()
    let currentInput = self.videoCapture.captureSession.inputs.first as? AVCaptureDeviceInput
    self.videoCapture.captureSession.removeInput(currentInput!)
    guard let currentPosition = currentInput?.device.position else { return }

    let nextCameraPosition: AVCaptureDevice.Position = currentPosition == .back ? .front : .back

    let newCameraDevice = bestCaptureDevice(position: nextCameraPosition)

    guard let videoInput1 = try? AVCaptureDeviceInput(device: newCameraDevice) else {
      return
    }

    self.videoCapture.captureSession.addInput(videoInput1)
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
      return
    }
    self.videoCapture.updateVideoOrientation(orientation: orientation)

    self.videoCapture.captureSession.commitConfiguration()
  }

  public func capturePhoto(completion: @escaping (UIImage?) -> Void) {
    self.photoCaptureCompletion = completion
    let settings = AVCapturePhotoSettings()
    usleep(20_000)  // short 10 ms delay to allow camera to focus
    self.videoCapture.photoOutput.capturePhoto(
      with: settings, delegate: self as AVCapturePhotoCaptureDelegate
    )
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
    
    // Position is a value between 0 and 1 representing the vertical position
    let path = UIBezierPath()
    
    // Calculate the y-position (vertical placement) based on the threshold value
    let height = self.bounds.height
    let yPosition = height * position
    
    // Draw a horizontal line across the width of the view
    path.move(to: CGPoint(x: 0, y: yPosition))
    path.addLine(to: CGPoint(x: self.bounds.width, y: yPosition))
    
    layer.path = path.cgPath
    layer.isHidden = false
  }
  
  // Threshold 1 slider changed
  @objc func threshold1Changed(_ sender: UISlider) {
    threshold1 = CGFloat(sender.value)
    updateThresholdLayer(threshold1Layer, position: threshold1)
    labelThreshold1.text = "Threshold 1: " + String(format: "%.2f", sender.value)
    
    // Update tracking detector threshold if we're in fish count mode
    if task == .fishCount, let trackingDetector = videoCapture.predictor as? TrackingDetector {
      trackingDetector.setThresholds([threshold1, threshold2])
    }
  }
  
  // Threshold 2 slider changed
  @objc func threshold2Changed(_ sender: UISlider) {
    threshold2 = CGFloat(sender.value)
    updateThresholdLayer(threshold2Layer, position: threshold2)
    labelThreshold2.text = "Threshold 2: " + String(format: "%.2f", sender.value)
    
    // Update tracking detector threshold if we're in fish count mode
    if task == .fishCount, let trackingDetector = videoCapture.predictor as? TrackingDetector {
      trackingDetector.setThresholds([threshold1, threshold2])
    }
  }

  @objc func resetFishCount() {
    // Reset the fish count in tracking detector
    if task == .fishCount, let trackingDetector = videoCapture.predictor as? TrackingDetector {
      trackingDetector.resetCount()
      labelFishCount.text = "Fish Count: 0"
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
    
    // Switch to new frame source
    switch sourceType {
    case .camera:
      // Remove any existing video player layer
      if let albumSource = albumVideoSource, let playerLayer = albumSource.playerLayer {
        playerLayer.removeFromSuperlayer()
      }
      
      // Show camera preview layer
      if let previewLayer = videoCapture.previewLayer {
        previewLayer.isHidden = false
      }
      
      // Use camera as current frame source
      currentFrameSource = videoCapture
      frameSourceType = .camera
      
      // Start camera capture
      start(position: .back)
      
    case .videoFile:
      // Hide camera preview layer
      if let previewLayer = videoCapture.previewLayer {
        previewLayer.isHidden = true
      }
      
      // Show video selection picker
      showVideoPickerAlert()
      
    default:
      // For future source types
      break
    }
  }
  
  // Shows UI for selecting a video from the photo library
  private func showVideoPickerAlert() {
    // This is called from the main thread because it's triggered by UI interactions
    let picker = UIImagePickerController()
    picker.delegate = self
    picker.sourceType = .photoLibrary
    picker.mediaTypes = ["public.movie"]
    picker.videoQuality = .typeHigh
    picker.allowsEditing = false
    
    // Find the current view controller to present the picker
    var topViewController = UIApplication.shared.windows.first?.rootViewController
    while let presentedViewController = topViewController?.presentedViewController {
      topViewController = presentedViewController
    }
    
    topViewController?.present(picker, animated: true)
  }
  
  // Method to set up the video source with a selected URL
  private func setupVideoSource(with url: URL) {
    // Clear any existing bounding boxes first
    boundingBoxViews.forEach { box in
      box.hide()
    }
    
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
    
    // Configure the video source
    albumVideoSource?.setVideoURL(url) { success in
      if success {
        // Set as current frame source
        self.currentFrameSource = self.albumVideoSource!
        self.frameSourceType = .videoFile
        
        // Configure player layer
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
        
        // Start playback
        self.albumVideoSource?.start()
      } else {
        // If video setup failed, switch back to camera
        self.switchToFrameSource(.camera)
      }
    }
  }

  @objc func switchSourceButtonTapped() {
    // Toggle between camera and video source
    if frameSourceType == .camera {
      switchToFrameSource(.videoFile)
    } else {
      switchToFrameSource(.camera)
    }
  }

  @objc func handleVideoPlaybackEnd(_ notification: Notification) {
    // When video playback ends, provide visual feedback
    DispatchQueue.main.async {
      // Enable the play button and disable the pause button
      self.playButton.isEnabled = true
      self.pauseButton.isEnabled = false
      
      // Optionally show a message or UI change to indicate playback ended
      // For example, we could show a toast or change button appearance
    }
  }
}

// MARK: - UIImagePickerControllerDelegate
extension YOLOView: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
  public func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
    picker.dismiss(animated: true)
    
    guard let mediaType = info[.mediaType] as? String,
          mediaType == "public.movie",
          let url = info[.mediaURL] as? URL else {
      return
    }
    
    // Setup the video source with the selected URL
    setupVideoSource(with: url)
  }
  
  public func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
    picker.dismiss(animated: true)
    
    // Switch back to camera if video selection was cancelled
    if frameSourceType == .videoFile && albumVideoSource?.videoURL == nil {
      switchToFrameSource(.camera)
    }
  }
}

// MARK: - AVCapturePhotoCaptureDelegate
extension YOLOView: AVCapturePhotoCaptureDelegate {
  public nonisolated func photoOutput(
    _ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?
  ) {
    if let error = error {
      print("error occurred : \(error.localizedDescription)")
      return
    }
    
    guard let dataImage = photo.fileDataRepresentation() else {
      print("AVCapturePhotoCaptureDelegate Error: No image data")
      return
    }
    
    let dataProvider = CGDataProvider(data: dataImage as CFData)
    guard let cgImageRef = CGImage(
      jpegDataProviderSource: dataProvider!, decode: nil, shouldInterpolate: true,
      intent: .defaultIntent) else {
      print("AVCapturePhotoCaptureDelegate Error: Cannot create CGImage")
      return
    }
    
    // Create the initial UIImage
    let initialImage = UIImage(cgImage: cgImageRef, scale: 0.5, orientation: .right)
    
    // Process on the main actor
    Task { @MainActor in
      var isCameraFront = false
      if let currentInput = self.videoCapture.captureSession.inputs.first as? AVCaptureDeviceInput,
        currentInput.device.position == .front
      {
        isCameraFront = true
      }
      
      var orientation: CGImagePropertyOrientation = isCameraFront ? .leftMirrored : .right
      switch UIDevice.current.orientation {
      case .landscapeLeft:
        orientation = isCameraFront ? .downMirrored : .up
      case .landscapeRight:
        orientation = isCameraFront ? .upMirrored : .down
      default:
        break
      }
      
      // Process the image with the correct orientation
      var image = initialImage
      if let orientedCIImage = CIImage(image: image)?.oriented(orientation),
        let cgImage = CIContext().createCGImage(orientedCIImage, from: orientedCIImage.extent)
      {
        image = UIImage(cgImage: cgImage)
      }
      
      // Create and add the image view layer
      let imageView = UIImageView(image: image)
      imageView.contentMode = .scaleAspectFill
      imageView.frame = self.frame
      let imageLayer = imageView.layer
      self.layer.insertSublayer(imageLayer, above: videoCapture.previewLayer)

      // Add bounding boxes
      var tempViews = [UIView]()
      let boundingBoxInfos = makeBoundingBoxInfos(from: boundingBoxViews)
      for info in boundingBoxInfos where !info.isHidden {
        let boxView = createBoxView(from: info)
        boxView.frame = info.rect
        self.addSubview(boxView)
        tempViews.append(boxView)
      }
      
      // Capture the resulting view as an image
      let bounds = UIScreen.main.bounds
      UIGraphicsBeginImageContextWithOptions(bounds.size, true, 0.0)
      self.drawHierarchy(in: bounds, afterScreenUpdates: true)
      let img = UIGraphicsGetImageFromCurrentImageContext()
      UIGraphicsEndImageContext()
      
      // Clean up the temporary views
      imageLayer.removeFromSuperlayer()
      for v in tempViews {
        v.removeFromSuperview()
      }
      
      // Call the completion with the captured image
      photoCaptureCompletion?(img)
      photoCaptureCompletion = nil
    }
  }
}
