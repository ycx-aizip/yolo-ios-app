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

/// Camera angle configuration for fish counting
public enum CameraAngle: String, CaseIterable {
    case front = "Front"
    case top = "Top"
    case bottom = "Bottom"
    case left = "Left"
    case right = "Right"
}

/// A UIView component that provides real-time object detection, segmentation, and pose estimation capabilities.
@MainActor
public class YOLOView: UIView, VideoCaptureDelegate {
  func onInferenceTime(speed: Double, fps: Double) {
    DispatchQueue.main.async {
      self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, speed)  // t2 seconds to ms
    }
  }

  func onPredict(result: YOLOResult) {

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
      //            self.setupObbLayerIfNeeded()
      guard let obbLayer = self.obbLayer else { return }
      let obbDetections = result.obb
      self.obbRenderer.drawObbDetectionsWithReuse(
        obbDetections: obbDetections,
        on: obbLayer,
        imageViewSize: self.overlayLayer.frame.size,
        originalImageSize: result.orig_shape,  // 例
        lineWidth: 3
      )
    }
  }

  var onDetection: ((YOLOResult) -> Void)?
  private var videoCapture: VideoCapture
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
  
  // MARK: - New Threshold Sliders for Fish Counting
  public var sliderThreshold1 = UISlider()
  public var labelSliderThreshold1 = UILabel()
  public var sliderThreshold2 = UISlider()
  public var labelSliderThreshold2 = UILabel()
  
  public var labelName = UILabel()
  public var labelFPS = UILabel()
  public var labelZoom = UILabel()
  public var activityIndicator = UIActivityIndicatorView()
  public var playButton = UIButton()
  public var pauseButton = UIButton()
  public var switchCameraButton = UIButton()
  public var toolbar = UIView()
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
  
  // MARK: - Fish Counting Properties
  
  /// Layers for threshold line visualization
  private var thresholdLayers: [CAShapeLayer] = []
  
  /// Label for displaying fish count
  private var fishCountLabel: UILabel?
  
  /// Reset count button
  private var resetCountButton: UIButton?
  
  // Comment out camera angle selection components as they're no longer needed
  /*
  /// Label for camera angle selection
  private var cameraAngleLabel: UILabel?
  
  /// Segmented control for camera angle selection
  private var cameraAngleControl: UISegmentedControl?
  
  /// Current camera angle
  private var currentCameraAngle: CameraAngle = .front
  */
  
  /// Current fish count
  private var fishCount: Int = 0
  
  /// Current threshold values (normalized 0-1) for fish counting
  private var thresholdValues: [CGFloat] = [0.3, 0.5]

  public init(
    frame: CGRect,
    modelPathOrName: String,
    task: YOLOTask
  ) {
    self.videoCapture = VideoCapture()
    super.init(frame: frame)
    setModel(modelPathOrName: modelPathOrName, task: task)
    setUpOrientationChangeNotification()
    self.setUpBoundingBoxViews()
    self.setupUI()
    self.videoCapture.delegate = self
    start(position: .back)
    setupOverlayLayer()
  }

  required init?(coder: NSCoder) {
    self.videoCapture = VideoCapture()
    super.init(coder: coder)
  }

  public override func awakeFromNib() {
    super.awakeFromNib()
    Task { @MainActor in
      setUpOrientationChangeNotification()
      setUpBoundingBoxViews()
      setupUI()
      videoCapture.delegate = self
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
      // Use TrackingDetector instead of ObjectDetector for fish counting
      // Create a TrackingDetector instance using the provided model URL
      TrackingDetector.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true) {
        [weak self] result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
          // Setup fish counting UI components after the model is loaded
          DispatchQueue.main.async {
            self?.setupFishCountingUI()
          }
        case .failure(let error):
          handleFailure(error)
        }
      }

    default:
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
    videoCapture.stop()
  }

  public func resume() {
    videoCapture.start()
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
    // For fish counting mode, handle bounding boxes differently
    if task == .fishCount {
      let resultCount = predictions.boxes.count
      
      // Process detections
      for i in 0..<boundingBoxViews.count {
        if i < resultCount && i < 50 {
          let prediction = predictions.boxes[i]
          let confidence = CGFloat(prediction.conf)
          
          // Skip boxes with confidence below threshold
          if confidence < CGFloat(sliderConf.value) {
            boundingBoxViews[i].hide()
            continue
          }
          
          // Direct conversion from normalized coordinates to screen coordinates
          let viewWidth = self.bounds.width
          let viewHeight = self.bounds.height
          
          let rect = CGRect(
            x: prediction.xywhn.minX,
            y: 1 - prediction.xywhn.maxY,
            width: prediction.xywhn.width,
            height: prediction.xywhn.height
          )
          
          // Convert normalized coordinates to screen coordinates
          let displayRect = CGRect(
            x: rect.minX * viewWidth,
            y: rect.minY * viewHeight,
            width: rect.width * viewWidth,
            height: rect.height * viewHeight
          )
          
          // Show box with label
          let colorIndex = prediction.index % ultralyticsColors.count
          let boxColor = ultralyticsColors[colorIndex]
          let label = String(format: "%@ %.1f", prediction.cls, confidence * 100)
          let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
          
          boundingBoxViews[i].show(
            frame: displayRect,
            label: label,
            color: boxColor,
            alpha: alpha
          )
        } else {
          boundingBoxViews[i].hide()
        }
      }
      return
    }
    
    // Original code for other tasks
    let width = self.bounds.width
    let height = self.bounds.height
    var resultCount = 0

    resultCount = predictions.boxes.count

    if UIDevice.current.orientation == .portrait {
      var ratio: CGFloat = 1.0

      if videoCapture.captureSession.sessionPreset == .photo {
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
          default:
            let prediction = predictions.boxes[i]
            let clsIndex = prediction.index
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
          if ratio >= 1 {
            let offset = (1 - ratio) * (0.5 - displayRect.minX)
            if task == .detect {
              let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
              displayRect = displayRect.applying(transform)
            } else {
              let transform = CGAffineTransform(translationX: offset, y: 0)
              displayRect = displayRect.applying(transform)
            }
            displayRect.size.width *= ratio
          } else {
            if task == .detect {
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

        } else {
          boundingBoxViews[i].hide()
        }
      }
    } else {
      resultCount = predictions.boxes.count
      self.labelSliderNumItems.text =
        String(resultCount) + " items (max " + String(Int(sliderNumItems.value)) + ")"

      let frameAspectRatio = videoCapture.longSide / videoCapture.shortSide
      let viewAspectRatio = width / height
      var scaleX: CGFloat = 1.0
      var scaleY: CGFloat = 1.0
      var offsetX: CGFloat = 0.0
      var offsetY: CGFloat = 0.0

      if frameAspectRatio > viewAspectRatio {
        scaleY = height / videoCapture.shortSide
        scaleX = scaleY
        offsetX = (videoCapture.longSide * scaleX - width) / 2
      } else {
        scaleX = width / videoCapture.longSide
        scaleY = scaleX
        offsetY = (videoCapture.shortSide * scaleY - height) / 2
      }

      for i in 0..<boundingBoxViews.count {
        if i < resultCount && i < 50 {
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
          }

          let colorIndex = predictions.boxes[i].index % ultralyticsColors.count
          boxColor = ultralyticsColors[colorIndex]
          label = String(format: "%@ %.1f", bestClass, confidence * 100)
          alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)

          rect.origin.x = rect.origin.x * videoCapture.longSide * scaleX - offsetX
          rect.origin.y =
            height
            - (rect.origin.y * videoCapture.shortSide * scaleY
              - offsetY
              + rect.size.height * videoCapture.shortSide * scaleY)
          rect.size.width *= videoCapture.longSide * scaleX
          rect.size.height *= videoCapture.shortSide * scaleY

          boundingBoxViews[i].show(
            frame: rect,
            label: label,
            color: boxColor,
            alpha: alpha
          )
        } else {
          boundingBoxViews[i].hide()
        }
      }
    }
  }

  func removeClassificationLayers() {
    if let sublayers = self.layer.sublayers {
      for layer in sublayers where layer.name == "YOLOOverlayLayer" {
        layer.removeFromSuperlayer()
      }
    }
  }

  private func setupFishCountingUI() {
    // Create threshold layers for fish counting
    setupThresholdLayers()
    
    // Setup fish count display
    setupCountDisplay()
    
    // Setup reset button
    setupResetButton()
    
    // Configure UI elements for fish counting
    configureUIForFishCounting()
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
    // labelName.text = modelName
    // labelName.textAlignment = .center
    // labelName.font = UIFont.systemFont(ofSize: 24, weight: .medium)
    // labelName.textColor = .black
    // labelName.font = UIFont.preferredFont(forTextStyle: .title1)
    // self.addSubview(labelName)
    labelName.isHidden = true
    
    labelFPS.text = String(format: "%.1f FPS - %.1f ms", 0.0, 0.0)
    labelFPS.textAlignment = .center
    labelFPS.textColor = .black
    labelFPS.font = UIFont.preferredFont(forTextStyle: .body)
    self.addSubview(labelFPS)

    // Hide items slider for fish counting
    labelSliderNumItems.text = "0 items (max 30)"
    labelSliderNumItems.textAlignment = .left
    labelSliderNumItems.textColor = .black
    labelSliderNumItems.font = UIFont.preferredFont(forTextStyle: .subheadline)
    self.addSubview(labelSliderNumItems)
    labelSliderNumItems.isHidden = true

    sliderNumItems.minimumValue = 0
    sliderNumItems.maximumValue = 100
    sliderNumItems.value = 30
    sliderNumItems.minimumTrackTintColor = .darkGray
    sliderNumItems.maximumTrackTintColor = .systemGray.withAlphaComponent(0.7)
    sliderNumItems.addTarget(self, action: #selector(sliderChanged), for: .valueChanged)
    self.addSubview(sliderNumItems)
    sliderNumItems.isHidden = true
    
    // Configure confidence slider
    labelSliderConf.text = String(format: "Conf: %.2f", 0.25)
    labelSliderConf.textAlignment = .left
    labelSliderConf.textColor = .white
    labelSliderConf.font = UIFont.preferredFont(forTextStyle: .subheadline)
    self.addSubview(labelSliderConf)

    sliderConf.minimumValue = 0
    sliderConf.maximumValue = 1
    sliderConf.value = 0.25
    sliderConf.minimumTrackTintColor = .white
    sliderConf.maximumTrackTintColor = .systemGray.withAlphaComponent(0.7)
    sliderConf.addTarget(self, action: #selector(sliderChanged), for: .valueChanged)
    self.addSubview(sliderConf)

    // Configure IoU slider
    labelSliderIoU.text = String(format: "IoU: %.2f", 0.45)
    labelSliderIoU.textAlignment = .left
    labelSliderIoU.textColor = .white
    labelSliderIoU.font = UIFont.preferredFont(forTextStyle: .subheadline)
    self.addSubview(labelSliderIoU)

    sliderIoU.minimumValue = 0
    sliderIoU.maximumValue = 1
    sliderIoU.value = 0.45
    sliderIoU.minimumTrackTintColor = .white
    sliderIoU.maximumTrackTintColor = .systemGray.withAlphaComponent(0.7)
    sliderIoU.addTarget(self, action: #selector(sliderChanged), for: .valueChanged)
    self.addSubview(sliderIoU)

    // Hide zoom label
    labelZoom.isHidden = true

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

    self.addGestureRecognizer(UIPinchGestureRecognizer(target: self, action: #selector(pinch)))
  }

  public override func layoutSubviews() {
    super.layoutSubviews()
    setupOverlayLayer()
    let isLandscape = bounds.width > bounds.height
    activityIndicator.frame = CGRect(x: center.x - 50, y: center.y - 50, width: 100, height: 100)
    
    // Update threshold positions if in fish counting mode
    if task == .fishCount {
      updateThresholdPositions()
      
      // Position fish counting UI elements
      positionFishCountingUI()
    }
    
    // Position other UI elements (hiding the ones we don't need)
    labelName.isHidden = true  // Hide the model name
    labelZoom.isHidden = true  // Hide the zoom label
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
    videoCapture.updateVideoOrientation(orientation: orientation)

    //      frameSizeCaptured = false
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
    
    // Update formatted text for slider labels
    self.labelSliderConf.text = String(format: "Conf: %.2f", conf)
    self.labelSliderIoU.text = String(format: "IoU: %.2f", iou)
    
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
    self.videoCapture.start()
    playButton.isEnabled = false
    pauseButton.isEnabled = true
  }

  @objc func pauseTapped() {
    selection.selectionChanged()
    self.videoCapture.stop()
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
  
  // MARK: - Fish Counting Methods
  
  /// Create and configure threshold layers for fish counting
  private func setupThresholdLayers() {
    // Clear any existing layers
    for layer in thresholdLayers {
        layer.removeFromSuperlayer()
    }
    thresholdLayers.removeAll()
    
    // Create horizontal threshold layers - one for each threshold
    for (index, _) in thresholdValues.enumerated() {
        let thresholdLayer = CAShapeLayer()
        thresholdLayer.strokeColor = index == 0 ? UIColor.red.cgColor : UIColor.green.cgColor
        thresholdLayer.lineWidth = 2.0
        thresholdLayer.lineDashPattern = [6, 3] // Dashed line
        thresholdLayer.opacity = 0.8
        thresholdLayer.zPosition = 10 // Ensure it appears above other layers
        
        // Add layers to the preview layer for proper z-ordering
        if let previewLayer = videoCapture.previewLayer {
            previewLayer.addSublayer(thresholdLayer)
        } else {
            // Fall back to adding to self.layer if preview layer isn't available yet
            self.layer.addSublayer(thresholdLayer)
        }
        
        thresholdLayers.append(thresholdLayer)
    }
    
    // Update threshold positions
    updateThresholdPositions()
  }

  /// Setup threshold sliders
  private func setupThresholdSliders() {
    // Configure the threshold sliders
    
    // First threshold slider
    sliderThreshold1.minimumValue = 0.0
    sliderThreshold1.maximumValue = 1.0
    sliderThreshold1.value = Float(thresholdValues[0])
    sliderThreshold1.addTarget(self, action: #selector(thresholdSliderChanged(_:)), for: .valueChanged)
    sliderThreshold1.frame = CGRect(x: 20, y: self.bounds.height - 80, width: self.bounds.width / 2 - 30, height: 20)
    sliderThreshold1.minimumTrackTintColor = UIColor.red
    
    labelSliderThreshold1.text = "Threshold 1: \(thresholdValues[0])"
    labelSliderThreshold1.frame = CGRect(x: 20, y: self.bounds.height - 100, width: self.bounds.width / 2 - 30, height: 20)
    labelSliderThreshold1.font = UIFont.systemFont(ofSize: 12)
    labelSliderThreshold1.textColor = UIColor.red
    
    // Second threshold slider
    sliderThreshold2.minimumValue = 0.0
    sliderThreshold2.maximumValue = 1.0
    sliderThreshold2.value = Float(thresholdValues[1])
    sliderThreshold2.addTarget(self, action: #selector(thresholdSliderChanged(_:)), for: .valueChanged)
    sliderThreshold2.frame = CGRect(x: self.bounds.width / 2 + 10, y: self.bounds.height - 80, width: self.bounds.width / 2 - 30, height: 20)
    sliderThreshold2.minimumTrackTintColor = UIColor.green
    
    labelSliderThreshold2.text = "Threshold 2: \(thresholdValues[1])"
    labelSliderThreshold2.frame = CGRect(x: self.bounds.width / 2 + 10, y: self.bounds.height - 100, width: self.bounds.width / 2 - 30, height: 20)
    labelSliderThreshold2.font = UIFont.systemFont(ofSize: 12)
    labelSliderThreshold2.textColor = UIColor.green
    
    // Add to view
    self.addSubview(sliderThreshold1)
    self.addSubview(labelSliderThreshold1)
    self.addSubview(sliderThreshold2)
    self.addSubview(labelSliderThreshold2)
  }

  /// Update threshold positions based on current values
  private func updateThresholdPositions() {
    guard let previewLayer = videoCapture.previewLayer else { return }
    
    let bounds = previewLayer.bounds
    
    // Update each threshold line
    for (index, threshold) in thresholdValues.enumerated() {
        guard index < thresholdLayers.count else { continue }
        
        let thresholdLayer = thresholdLayers[index]
        
        // Create a path for the horizontal line
        let path = CGMutablePath()
        let yPosition = bounds.height * threshold
        path.move(to: CGPoint(x: 0, y: yPosition))
        path.addLine(to: CGPoint(x: bounds.width, y: yPosition))
        
        thresholdLayer.path = path
        thresholdLayer.isHidden = false
    }
  }

  /// Update fish count display with current value
  private func updateCountDisplay() {
    DispatchQueue.main.async {
        self.fishCountLabel?.text = "Fish Count: \(self.fishCount)"
    }
  }

  /// Reset the fish count
  public func resetFishCount() {
    fishCount = 0
    updateCountDisplay()
    
    // Also reset count in TrackingDetector
    if let trackingDetector = videoCapture.predictor as? TrackingDetector {
        trackingDetector.resetCount()
    }
  }

  /// Update the fish count (should be called by TrackingDetector)
  public func updateFishCount(_ newCount: Int) {
    fishCount = newCount
    updateCountDisplay()
  }

  /// Setup count display label
  private func setupCountDisplay() {
    // Create fish count label if needed
    if fishCountLabel == nil {
        fishCountLabel = UILabel(frame: CGRect(x: 20, y: 120, width: 150, height: 30))  // Make smaller
        fishCountLabel?.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        fishCountLabel?.layer.cornerRadius = 8  // Adjust corner radius for smaller label
        fishCountLabel?.layer.masksToBounds = true
        fishCountLabel?.textColor = UIColor.white
        fishCountLabel?.textAlignment = .center
        fishCountLabel?.font = UIFont.systemFont(ofSize: 16, weight: .bold)  // Smaller font
        self.addSubview(fishCountLabel!)
    }
    
    // Set initial count text
    updateCountDisplay()
  }

  /// Setup reset count button
  private func setupResetButton() {
    if resetCountButton == nil {
        let buttonWidth: CGFloat = 80
        let buttonHeight: CGFloat = 30
        let rightMargin: CGFloat = 20
        
        resetCountButton = UIButton(frame: CGRect(
            x: self.bounds.width - buttonWidth - rightMargin, 
            y: 120, 
            width: buttonWidth, 
            height: buttonHeight))
            
        resetCountButton?.setTitle("Reset", for: .normal)
        resetCountButton?.backgroundColor = UIColor.darkGray
        resetCountButton?.layer.cornerRadius = 8
        resetCountButton?.addTarget(self, action: #selector(resetButtonTapped), for: .touchUpInside)
        resetCountButton?.titleLabel?.font = UIFont.systemFont(ofSize: 14, weight: .medium)
        self.addSubview(resetCountButton!)
    }
  }

  /// Handle reset button tap
  @objc private func resetButtonTapped() {
    resetFishCount()
  }

  /// Handle threshold slider changes
  @objc private func thresholdSliderChanged(_ sender: UISlider) {
    if sender === sliderThreshold1 {
        thresholdValues[0] = CGFloat(sender.value)
        labelSliderThreshold1.text = String(format: "Threshold 1: %.2f", thresholdValues[0])
    } else if sender === sliderThreshold2 {
        thresholdValues[1] = CGFloat(sender.value)
        labelSliderThreshold2.text = String(format: "Threshold 2: %.2f", thresholdValues[1])
    }
    
    // Update threshold positions
    updateThresholdPositions()
    
    // Update thresholds in TrackingDetector
    if let trackingDetector = videoCapture.predictor as? TrackingDetector {
        trackingDetector.updateThresholds(thresholds: thresholdValues)
    }
  }

  /// Configure UI elements based on fish counting mode
  private func configureUIForFishCounting() {
    // Hide unnecessary sliders
    sliderNumItems.isHidden = true
    labelSliderNumItems.isHidden = true
    
    // Keep confidence slider for fish detection threshold
    sliderConf.isHidden = false
    labelSliderConf.isHidden = false
    
    // Keep IoU slider as requested
    sliderIoU.isHidden = false
    labelSliderIoU.isHidden = false
    
    // Show threshold sliders
    setupThresholdSliders()
    
    // Show fish counting UI elements
    fishCountLabel?.isHidden = false
    resetCountButton?.isHidden = false
    
    // Setup threshold layers
    setupThresholdLayers()
    
    // Connect to the TrackingDetector for count updates
    connectToTrackingDetector()
  }

  /// Connect to the TrackingDetector to get count updates
  private func connectToTrackingDetector() {
    guard let predictor = videoCapture.predictor,
          let trackingDetector = predictor as? TrackingDetector else {
      print("ERROR: Could not connect to TrackingDetector")
      return
    }
    
    // Set up callback to update count
    trackingDetector.onCountChanged = { [weak self] newCount in
      self?.updateFishCount(newCount)
    }
    
    // Configure initial thresholds
    trackingDetector.updateThresholds(thresholds: thresholdValues)
  }

  /// Position UI elements for fish counting
  private func positionFishCountingUI() {
    // Position fish count label
    fishCountLabel?.frame = CGRect(
        x: 20,
        y: 120,
        width: 150,
        height: 30)
    
    // Position reset button
    let buttonWidth: CGFloat = 80
    let buttonHeight: CGFloat = 30
    let rightMargin: CGFloat = 20
    
    resetCountButton?.frame = CGRect(
        x: self.bounds.width - buttonWidth - rightMargin,
        y: 120,
        width: buttonWidth,
        height: buttonHeight)
    
    // Position confidence and IoU sliders side by side in one row
    // Confidence slider (left half)
    labelSliderConf.textColor = UIColor.white
    labelSliderConf.frame = CGRect(x: 20, y: self.bounds.height - 150, width: self.bounds.width / 2 - 30, height: 20)
    sliderConf.frame = CGRect(x: 20, y: self.bounds.height - 130, width: self.bounds.width / 2 - 30, height: 20)
    sliderConf.minimumTrackTintColor = UIColor.white
    
    // IoU slider (right half)
    labelSliderIoU.textColor = UIColor.white
    labelSliderIoU.frame = CGRect(x: self.bounds.width / 2 + 10, y: self.bounds.height - 150, width: self.bounds.width / 2 - 30, height: 20)
    sliderIoU.frame = CGRect(x: self.bounds.width / 2 + 10, y: self.bounds.height - 130, width: self.bounds.width / 2 - 30, height: 20)
    sliderIoU.minimumTrackTintColor = UIColor.white
    
    // Position threshold sliders
    sliderThreshold1.frame = CGRect(x: 20, y: self.bounds.height - 80, width: self.bounds.width / 2 - 30, height: 20)
    labelSliderThreshold1.frame = CGRect(x: 20, y: self.bounds.height - 100, width: self.bounds.width / 2 - 30, height: 20)
    
    sliderThreshold2.frame = CGRect(x: self.bounds.width / 2 + 10, y: self.bounds.height - 80, width: self.bounds.width / 2 - 30, height: 20)
    labelSliderThreshold2.frame = CGRect(x: self.bounds.width / 2 + 10, y: self.bounds.height - 100, width: self.bounds.width / 2 - 30, height: 20)
  }

  /// Show or hide UI controls for fish counting
  public func setFishCountUIVisible(_ isVisible: Bool) {
    // Fish count display
    fishCountLabel?.isHidden = !isVisible
    resetCountButton?.isHidden = !isVisible
    
    // Threshold sliders for counting lines
    sliderThreshold1.isHidden = !isVisible
    labelSliderThreshold1.isHidden = !isVisible
    sliderThreshold2.isHidden = !isVisible
    labelSliderThreshold2.isHidden = !isVisible
    
    // Update threshold layers visibility
    for layer in thresholdLayers {
      layer.isHidden = !isVisible
    }
    
    // Note: Confidence and IoU sliders remain visible as they're needed for detection quality
  }

  /// Get the current predictor (internal use only)
  func getPredictor() -> Predictor? {
    return videoCapture.predictor
  }
  
  /// Update the threshold values for fish counting (can be called from outside the module)
  public func updateFishCountingThresholds(thresholds: [CGFloat]) {
    // Store the new threshold values
    self.thresholdValues = thresholds
    
    // Update threshold positions in UI
    updateThresholdPositions()
    
    // Pass to the tracking detector if available
    if let predictor = videoCapture.predictor,
       let trackingDetector = predictor as? TrackingDetector {
      trackingDetector.updateThresholds(thresholds: thresholdValues)
    }
  }
}

extension YOLOView: @preconcurrency AVCapturePhotoCaptureDelegate {
  public func photoOutput(
    _ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?
  ) {
    if let error = error {
      print("error occurred : \(error.localizedDescription)")
    }
    if let dataImage = photo.fileDataRepresentation() {
      let dataProvider = CGDataProvider(data: dataImage as CFData)
      let cgImageRef: CGImage! = CGImage(
        jpegDataProviderSource: dataProvider!, decode: nil, shouldInterpolate: true,
        intent: .defaultIntent)
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
      var image = UIImage(cgImage: cgImageRef, scale: 0.5, orientation: .right)
      if let orientedCIImage = CIImage(image: image)?.oriented(orientation),
        let cgImage = CIContext().createCGImage(orientedCIImage, from: orientedCIImage.extent)
      {
        image = UIImage(cgImage: cgImage)
      }
      let imageView = UIImageView(image: image)
      imageView.contentMode = .scaleAspectFill
      imageView.frame = self.frame
      let imageLayer = imageView.layer
      self.layer.insertSublayer(imageLayer, above: videoCapture.previewLayer)

      var tempViews = [UIView]()
      let boundingBoxInfos = makeBoundingBoxInfos(from: boundingBoxViews)
      for info in boundingBoxInfos where !info.isHidden {
        let boxView = createBoxView(from: info)
        boxView.frame = info.rect

        self.addSubview(boxView)
        tempViews.append(boxView)
      }
      let bounds = UIScreen.main.bounds
      UIGraphicsBeginImageContextWithOptions(bounds.size, true, 0.0)
      self.drawHierarchy(in: bounds, afterScreenUpdates: true)
      let img = UIGraphicsGetImageFromCurrentImageContext()
      UIGraphicsEndImageContext()
      imageLayer.removeFromSuperlayer()
      for v in tempViews {
        v.removeFromSuperview()
      }
      photoCaptureCompletion?(img)
      photoCaptureCompletion = nil
    } else {
      print("AVCapturePhotoCaptureDelegate Error")
    }
  }
}
