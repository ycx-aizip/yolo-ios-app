// from Aizip
import AVFoundation
import UIKit
import Vision

// MARK: - Protocols
@MainActor
public protocol FishCountViewActionDelegate: AnyObject {
  func didTapModelsButton()
}
// MARK: - FishCountView Main Class
/// FishCountView - Primary UI component for real-time fish counting and object detection
///
/// This class provides a comprehensive interface for real-time YOLO model inference with specialized
/// fish counting capabilities. It handles camera setup, model loading, video frame processing,
/// rendering of detection results, and user interactions such as pinch-to-zoom.
///
/// Key Features:
/// - Multiple frame sources: Camera, Video Files, GoPro, UVC External Cameras
/// - Fish counting with ByteTracker integration
/// - Configurable detection thresholds and counting directions
/// - Auto-calibration for optimal threshold detection
/// - Real-time bounding box rendering with tracking status
/// - Support for detection, segmentation, pose estimation, and oriented bounding boxes
@MainActor
public class FishCountView: UIView, VideoCaptureDelegate, FrameSourceDelegate {
  // MARK: - Public Properties
  /// Callback for detection results
  var onDetection: ((YOLOResult) -> Void)?
  /// Delegate for handling model selection actions
  public weak var actionDelegate: FishCountViewActionDelegate?
  // MARK: - Frame Source Properties
  /// Primary camera video source
  private var videoCapture: CameraVideoSource
  /// Album video source for playing video files
  private var albumVideoSource: AlbumVideoSource?
  /// Current active frame source
  private var currentFrameSource: FrameSource
  /// Type of current frame source
  private var frameSourceType: FrameSourceType = .camera
  /// UVC video source for external USB cameras
  private var uvcVideoSource: UVCVideoSource?
  // MARK: - Model and Task Properties
  /// Current YOLO task type (detect, classify, segment, etc.)
  var task = YOLOTask.detect
  /// Model name for display
  var modelName: String = ""
  // MARK: - Detection and Tracking Properties
  /// Maximum number of bounding box views to create
  let maxBoundingBoxViews = 100
  /// Array of bounding box views for rendering detections
  var boundingBoxViews = [BoundingBoxView]()
  // MARK: - UI Control Properties
  /// Slider for controlling maximum number of detections
  public var sliderNumItems = UISlider()
  /// Label for number of items slider
  public var labelSliderNumItems = UILabel()
  /// Slider for confidence threshold
  public var sliderConf = UISlider()
  /// Label for confidence slider
  public var labelSliderConf = UILabel()
  /// Slider for IoU threshold
  public var sliderIoU = UISlider()
  /// Label for IoU slider
  public var labelSliderIoU = UILabel()
  // MARK: - Fish Counting Properties
  /// First threshold line layer for fish counting
  public var threshold1Layer: CAShapeLayer?
  /// Second threshold line layer for fish counting
  public var threshold2Layer: CAShapeLayer?
  /// Slider for first threshold position
  public var threshold1Slider = UISlider()
  /// Slider for second threshold position
  public var threshold2Slider = UISlider()
  /// Label for first threshold
  public var labelThreshold1 = UILabel()
  /// Label for second threshold
  public var labelThreshold2 = UILabel()
  /// First threshold value in display coordinates
  public var threshold1: CGFloat = ThresholdCounter.defaultThresholds.first ?? 0.2
  /// Second threshold value in display coordinates
  public var threshold2: CGFloat =
    ThresholdCounter.defaultThresholds.count > 1 ? ThresholdCounter.defaultThresholds[1] : 0.4
  /// Auto-calibration button
  public var autoCalibrationButton = UIButton()
  /// Flag indicating if calibration is in progress
  public var isCalibrating = false
  /// Label showing current fish count
  public var labelFishCount = UILabel()
  /// Button to reset fish count
  public var resetButton = UIButton()
  /// Current fish count (for UI display)
  public var fishCount: Int = 0
  /// Current counting direction
  private var countingDirection: CountingDirection = ThresholdCounter.defaultCountingDirection
  // MARK: - UI Display Properties
  /// Label for model name
  public var labelName = UILabel()
  /// Label for FPS display
  public var labelFPS = UILabel()
  /// Label for zoom level
  public var labelZoom = UILabel()
  /// Loading indicator
  public var activityIndicator = UIActivityIndicatorView()
  /// Play button for video controls
  public var playButton = UIButton()
  /// Pause button for video controls
  public var pauseButton = UIButton()
  /// Button to switch camera (front/back)
  public var switchCameraButton = UIButton()
  /// Toolbar containing control buttons
  public var toolbar = UIView()
  /// Button to switch frame source
  public var switchSourceButton = UIButton()
  /// Button to open model selection
  public var modelsButton = UIButton()
  /// Button to change counting direction
  public var directionButton = UIButton()
  // MARK: - Interaction Properties
  /// Haptic feedback generator for button interactions
  let selection = UISelectionFeedbackGenerator()
  // MARK: - Rendering Properties
  /// Main overlay layer for all detection visualizations (fish counting only)
  private var overlayLayer = CALayer()
  // MARK: - Zoom Properties
  /// Minimum zoom level
  private let minimumZoom: CGFloat = 1.0
  /// Maximum zoom level
  private let maximumZoom: CGFloat = 10.0
  /// Last zoom factor for gesture handling
  private var lastZoomFactor: CGFloat = 1.0
  // MARK: - GoPro Properties
  // /// Last known frame size from GoPro for coordinate transformation
  // internal var goProLastFrameSize: CGSize = CGSize(width: 1920, height: 1080)
  // MARK: - Device Type Helpers
  /// Returns true if running on iPad
  private var isIPad: Bool {
    return UIDevice.current.userInterfaceIdiom == .pad
  }
  /// Returns true if running on iPhone
  private var isIPhone: Bool {
    return UIDevice.current.userInterfaceIdiom == .phone
  }
  /// Returns true if running on large iPad
  private var isLargeIPad: Bool {
    return isIPad && (bounds.width > 1000 || bounds.height > 1000)
  }
  private var busy = false
  private var currentBuffer: CVPixelBuffer?
  var framesDone = 0
  var t0 = 0.0
  var t1 = 0.0
  var t2 = 0.0
  var t3 = CACurrentMediaTime()
  var t4 = 0.0
  // MARK: - Initialization
  /**
   * Initializes FishCountView with a specific model and task
   *
   * This initializer sets up the complete FishCountView with camera source,
   * loads the specified model, configures UI, and starts video capture.
   */
  public init(frame: CGRect, modelPathOrName: String, task: YOLOTask) {
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
  /**
   * Required initializer for Interface Builder
   */
  required init?(coder: NSCoder) {
    self.videoCapture = CameraVideoSource()
    self.currentFrameSource = self.videoCapture
    super.init(coder: coder)
  }
  /**
   * Called when view is loaded from Interface Builder
   * Sets up all components for storyboard-based initialization
   */
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
  // MARK: - VideoCaptureDelegate Implementation
  /**
   * Called when inference time information is available
   * Updates the FPS display label with current performance metrics
   */
  func onInferenceTime(speed: Double, fps: Double) {
    DispatchQueue.main.async {
      self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, speed)
    }
  }
  /**
   * Called when YOLO prediction results are available
   * For fish counting, displays bounding boxes with tracking information
   */
  func onPredict(result: YOLOResult) {
    if !isCalibrating {
      showBoxes(predictions: result)
      onDetection?(result)
      // Fish counting only - no additional task-specific rendering needed
    }
  }
  /**
   * Called to clear all bounding boxes from display
   */
  func onClearBoxes() {
    boundingBoxViews.forEach { box in
      box.hide()
    }
  }
  // MARK: - FrameSourceDelegate Implementation
  /**
   * Called when a new frame image is available from the frame source
   */
  func frameSource(_ source: FrameSource, didOutputImage image: UIImage) {
    // Currently not used, but required by protocol
  }
  /**
   * Called when frame processing performance metrics are updated
   */
  func frameSource(_ source: FrameSource, didUpdateWithSpeed speed: Double, fps: Double) {
    // Currently not used, but required by protocol
  }
  // MARK: - Model Management
  /**
   * Loads and configures a YOLO model for the specified task
   *
   * This method handles loading different types of YOLO models from file paths or bundle resources.
   * It supports detection, classification, segmentation, pose estimation, OBB detection, and fish counting tasks.
   *
   * - Parameters:
   *   - modelPathOrName: File path to .mlmodel/.mlpackage/.mlmodelc or bundle resource name
   *   - task: The YOLO task type to perform with this model
   *   - completion: Optional callback with loading result
   */
  public func setModel(
    modelPathOrName: String, task: YOLOTask, completion: ((Result<Void, Error>) -> Void)? = nil
  ) {
    activityIndicator.startAnimating()
    boundingBoxViews.forEach { box in
      box.hide()
    }
    // Classification layers removed - not needed for fish counting
    self.task = task
    setupSublayers()
    var modelURL: URL?
    let lowercasedPath = modelPathOrName.lowercased()
    let fileManager = FileManager.default
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
    func handleSuccess(predictor: Predictor) {
      self.videoCapture.predictor = predictor
      self.activityIndicator.stopAnimating()
      self.labelName.text = modelName
      // Fish counting setup
      if let detector = predictor as? ObjectDetector {
        let conf = Double(round(100 * sliderConf.value)) / 100
        let iou = Double(round(100 * sliderIoU.value)) / 100
        detector.setConfidenceThreshold(confidence: conf)
        detector.setIouThreshold(iou: iou)
        detector.setNumItemsThreshold(numItems: Int(sliderNumItems.value))
        print(
          "Initial thresholds applied - Confidence: \(conf), IoU: \(iou), Max Items: \(Int(sliderNumItems.value))"
        )
        // Always setup threshold layers for fish counting
        setupThresholdLayers()
        print(
          "Fish counting thresholds set - Threshold 1: \(threshold1Slider.value), Threshold 2: \(threshold2Slider.value)"
        )
      }
      completion?(.success(()))
    }
    func handleFailure(_ error: Error) {
      print("Failed to load model with error: \(error)")
      self.activityIndicator.stopAnimating()
      completion?(.failure(error))
    }
    // Fish counting only - hardcoded to use TrackingDetector
    TrackingDetector.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true) {
      [weak self] result in
      switch result {
      case .success(let predictor):
        if let trackingDetector = predictor as? TrackingDetector,
          let slf = self
        {
          Task { @MainActor in
            trackingDetector.applySharedConfiguration()
            trackingDetector.setThresholds([
              CGFloat(slf.threshold1Slider.value),
              CGFloat(slf.threshold2Slider.value),
            ])
            trackingDetector.setCountingDirection(slf.countingDirection)
          }
        }
        handleSuccess(predictor: predictor)
      case .failure(let error):
        handleFailure(error)
      }
    }
  }
  // MARK: - Frame Source Management
  /**
   * Starts camera capture with the specified position
   * Configures video capture session and integrates with FishCountView
   */
  private func start(position: AVCaptureDevice.Position) {
    if !busy {
      busy = true
      let orientation = UIDevice.current.orientation
      videoCapture.setUp(sessionPreset: nil, position: position, orientation: orientation) {
        @MainActor success in
        if success {
          self.videoCapture.integrateWithFishCountView(view: self)
          self.videoCapture.addBoundingBoxViews(self.boundingBoxViews)
          self.videoCapture.addOverlayLayer(self.overlayLayer)
          self.videoCapture.start()
          self.busy = false
        }
      }
    }
  }
  /**
   * Stops the current frame source
   */
  public func stop() {
    currentFrameSource.stop()
  }
  /**
   * Resumes the current frame source
   */
  public func resume() {
    currentFrameSource.start()
  }
  /**
   * Enables or disables inference processing
   * Used to pause inference during calibration or other operations
   */
  public func setInferenceFlag(ok: Bool) {
    videoCapture.inferenceOK = ok
  }
  // MARK: - Detection Visualization Setup
  /**
   * Creates the maximum number of bounding box views for detection rendering
   * Pre-creates all bounding box views to avoid runtime allocation
   */
  func setUpBoundingBoxViews() {
    while boundingBoxViews.count < maxBoundingBoxViews {
      boundingBoxViews.append(BoundingBoxView())
    }
  }
  /**
   * Sets up the main overlay layer for all detection visualizations
   * Calculates proper aspect ratio and positioning based on video format
   */
  func setupOverlayLayer() {
    let width = self.bounds.width
    let height = self.bounds.height
    var ratio: CGFloat = 1.0
    switch videoCapture.captureSession.sessionPreset {
    case .photo:
      ratio = (4.0 / 3.0)
    case .hd1280x720:
      ratio = (16.0 / 9.0)
    case .hd1920x1080:
      ratio = (16.0 / 9.0)
    default:
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
  /**
   * Resets all visualization layers (fish counting only needs overlay layer)
   */
  public func resetLayers() {
    removeAllSubLayers(parentLayer: overlayLayer)
  }
  /**
   * Sets up sublayers for fish counting (no task-specific layers needed)
   */
  func setupSublayers() {
    resetLayers()
    // Fish counting only - no additional layers needed
  }
  /**
   * Removes all sublayers from the specified parent layer
   */
  func removeAllSubLayers(parentLayer: CALayer?) {
    guard let parentLayer = parentLayer else { return }
    parentLayer.sublayers?.forEach { layer in
      layer.removeFromSuperlayer()
    }
    parentLayer.sublayers = nil
    parentLayer.contents = nil
  }
  // MARK: - Detection Rendering
  /**
   * Renders detection results as bounding boxes on the overlay
   * Handles different task types and tracking states for fish counting
   */
  func showBoxes(predictions: YOLOResult) {
    let width = self.bounds.width
    let height = self.bounds.height
    var resultCount = 0
    resultCount = predictions.boxes.count
    boundingBoxViews.forEach { box in
      box.hide()
    }
    if resultCount == 0 {
      return
    }
    let orientation = UIDevice.current.orientation
    self.labelSliderNumItems.text =
      String(resultCount) + " items (max " + String(Int(sliderNumItems.value)) + ")"
    for i in 0..<boundingBoxViews.count {
      // Limit to 30 boxes for better performance in dense scenes
      if i < resultCount && i < 30 {
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
          let prediction = predictions.boxes[i]
          rect = CGRect(
            x: prediction.xywhn.minX, y: 1 - prediction.xywhn.maxY, width: prediction.xywhn.width,
            height: prediction.xywhn.height)
          bestClass = prediction.cls
          confidence = CGFloat(prediction.conf)
          if let trackingDetector = currentFrameSource.predictor as? TrackingDetector {
            let isTracked = trackingDetector.isObjectTracked(box: prediction)
            let isCounted = trackingDetector.isObjectCounted(box: prediction)
            if isCounted {
              boxColor = .green
            } else if isTracked {
              boxColor = UIColor(red: 0.4, green: 0.7, blue: 1.0, alpha: 1.0)
            } else {
              boxColor = UIColor(red: 0.0, green: 0.0, blue: 0.8, alpha: 1.0)
            }
            alpha = isTracked ? 0.7 : 0.5
            if isTracked {
              if let trackInfo = trackingDetector.getTrackInfo(for: prediction) {
                label = "#\(trackInfo.trackId)"
              } else {
                label = "#?"
              }
            } else {
              label = ""
            }
          } else {
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
        let displayRect = currentFrameSource.transformDetectionToScreenCoordinates(
          rect: rect,
          viewBounds: self.bounds,
          orientation: orientation
        )
        boundingBoxViews[i].show(
          frame: displayRect, label: label, color: boxColor, alpha: alpha)
      }
    }
    if task == .fishCount, let trackingDetector = currentFrameSource.predictor as? TrackingDetector
    {
      let currentCount = trackingDetector.getCount()
      updateFishCountDisplay(count: currentCount)
    }
  }
  /**
   * Removes classification overlay layers from the view
   */
  // Classification methods removed - not needed for fish counting
  // MARK: - UI Setup
  /**
   * Sets up all user interface elements and controls
   * Configures labels, sliders, buttons, and their initial states
   */
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
    labelSliderConf.text = "Conf: 0.5"
    labelSliderConf.textAlignment = .left
    labelSliderConf.textColor = .white
    labelSliderConf.font = UIFont.systemFont(ofSize: 16, weight: .medium)
    labelSliderConf.isHidden = true
    self.addSubview(labelSliderConf)
    sliderConf.minimumValue = 0
    sliderConf.maximumValue = 1
    sliderConf.value = TrackingDetectorConfig.shared.defaultConfidenceThreshold
    sliderConf.minimumTrackTintColor = .white
    sliderConf.maximumTrackTintColor = .lightGray
    sliderConf.addTarget(self, action: #selector(sliderChanged), for: .valueChanged)
    sliderConf.isHidden = true
    self.addSubview(sliderConf)
    labelSliderIoU.text = "IoU: \(TrackingDetectorConfig.shared.defaultIoUThreshold)"
    labelSliderIoU.textAlignment = .right
    labelSliderIoU.textColor = .white
    labelSliderIoU.font = UIFont.systemFont(ofSize: 16, weight: .medium)
    labelSliderIoU.isHidden = true
    self.addSubview(labelSliderIoU)
    sliderIoU.minimumValue = 0
    sliderIoU.maximumValue = 1
    sliderIoU.value = TrackingDetectorConfig.shared.defaultIoUThreshold
    sliderIoU.minimumTrackTintColor = .white
    sliderIoU.maximumTrackTintColor = .lightGray
    sliderIoU.addTarget(self, action: #selector(sliderChanged), for: .valueChanged)
    sliderIoU.isHidden = true
    self.addSubview(sliderIoU)
    self.labelSliderNumItems.text = "0 items (max " + String(Int(sliderNumItems.value)) + ")"
    self.labelSliderConf.text =
      "Conf: " + String(TrackingDetectorConfig.shared.defaultConfidenceThreshold)
    self.labelSliderIoU.text = "IoU: " + String(TrackingDetectorConfig.shared.defaultIoUThreshold)
    let thresholds = ThresholdCounter.defaultThresholds
    let threshold1Value = String(format: "%.2f", thresholds.first ?? 0.2)
    labelThreshold1.text = "Threshold 1: " + threshold1Value
    labelThreshold1.textAlignment = .left
    labelThreshold1.textColor = UIColor.red
    labelThreshold1.font = UIFont.systemFont(ofSize: 16, weight: .medium)
    self.addSubview(labelThreshold1)
    threshold1Slider.minimumValue = 0
    threshold1Slider.maximumValue = 1
    threshold1Slider.value = Float(thresholds.first ?? 0.2)
    threshold1Slider.minimumTrackTintColor = UIColor.red
    threshold1Slider.maximumTrackTintColor = UIColor.lightGray
    threshold1Slider.addTarget(self, action: #selector(threshold1Changed), for: .valueChanged)
    self.addSubview(threshold1Slider)
    let threshold2Value = String(format: "%.2f", thresholds.count > 1 ? thresholds[1] : 0.4)
    labelThreshold2.text = "Threshold 2: " + threshold2Value
    labelThreshold2.textAlignment = .right
    labelThreshold2.textColor = UIColor.yellow
    labelThreshold2.font = UIFont.systemFont(ofSize: 16, weight: .medium)
    self.addSubview(labelThreshold2)
    threshold2Slider.minimumValue = 0
    threshold2Slider.maximumValue = 1
    threshold2Slider.value = Float(thresholds.count > 1 ? thresholds[1] : 0.4)
    threshold2Slider.minimumTrackTintColor = UIColor.yellow
    threshold2Slider.maximumTrackTintColor = UIColor.lightGray
    threshold2Slider.addTarget(self, action: #selector(threshold2Changed), for: .valueChanged)
    self.addSubview(threshold2Slider)
    let autoAttributedText = NSMutableAttributedString()
    let auAttributes: [NSAttributedString.Key: Any] = [
      .foregroundColor: UIColor.red.withAlphaComponent(0.5),
      .font: UIFont.systemFont(ofSize: 14, weight: .bold),
    ]
    let toAttributes: [NSAttributedString.Key: Any] = [
      .foregroundColor: UIColor.yellow.withAlphaComponent(0.5),
      .font: UIFont.systemFont(ofSize: 14, weight: .bold),
    ]
    autoAttributedText.append(NSAttributedString(string: "AU", attributes: auAttributes))
    autoAttributedText.append(NSAttributedString(string: "TO", attributes: toAttributes))
    autoCalibrationButton.setAttributedTitle(autoAttributedText, for: .normal)
    autoCalibrationButton.backgroundColor = UIColor.darkGray.withAlphaComponent(0.1)
    autoCalibrationButton.layer.cornerRadius = 12
    autoCalibrationButton.layer.masksToBounds = true
    autoCalibrationButton.addTarget(
      self, action: #selector(toggleAutoCalibration), for: .touchUpInside)
    self.addSubview(autoCalibrationButton)
    let fishCountAttributedText = NSMutableAttributedString()
    let labelAttributes: [NSAttributedString.Key: Any] = [
      .foregroundColor: UIColor.white,
      .font: UIFont.systemFont(ofSize: 16, weight: .bold),
    ]
    let numberAttributes: [NSAttributedString.Key: Any] = [
      .foregroundColor: UIColor.white,
      .font: UIFont.systemFont(ofSize: 48, weight: .bold),
    ]
    fishCountAttributedText.append(
      NSAttributedString(string: "Fish Count: ", attributes: labelAttributes))
    fishCountAttributedText.append(NSAttributedString(string: "0", attributes: numberAttributes))
    labelFishCount.attributedText = fishCountAttributedText
    labelFishCount.textAlignment = .center
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
    toolbar.backgroundColor = .clear
    toolbar.layer.cornerRadius = 10
    playButton.setImage(UIImage(systemName: "play.fill", withConfiguration: config), for: .normal)
    playButton.tintColor = .white
    playButton.backgroundColor = .clear
    playButton.isEnabled = false
    playButton.addTarget(self, action: #selector(playTapped), for: .touchUpInside)
    pauseButton.setImage(UIImage(systemName: "pause.fill", withConfiguration: config), for: .normal)
    pauseButton.tintColor = .white
    pauseButton.backgroundColor = .clear
    pauseButton.isEnabled = true
    pauseButton.addTarget(self, action: #selector(pauseTapped), for: .touchUpInside)
    switchSourceButton.setImage(
      UIImage(systemName: "rectangle.on.rectangle", withConfiguration: config), for: .normal)
    switchSourceButton.tintColor = .white
    switchSourceButton.backgroundColor = .clear
    switchSourceButton.addTarget(
      self, action: #selector(switchSourceButtonTapped), for: .touchUpInside)
    modelsButton.setImage(
      UIImage(systemName: "square.stack.3d.up", withConfiguration: config), for: .normal)
    modelsButton.tintColor = .white
    modelsButton.backgroundColor = .clear
    modelsButton.addTarget(self, action: #selector(modelsButtonTapped), for: .touchUpInside)
    directionButton.setImage(
      UIImage(systemName: "arrow.triangle.2.circlepath", withConfiguration: config), for: .normal)
    directionButton.tintColor = .white
    directionButton.backgroundColor = .clear
    directionButton.addTarget(self, action: #selector(directionButtonTapped), for: .touchUpInside)
    self.addSubview(toolbar)
    toolbar.addSubview(playButton)
    toolbar.addSubview(pauseButton)
    toolbar.addSubview(switchSourceButton)
    toolbar.addSubview(modelsButton)
    toolbar.addSubview(directionButton)
    self.addGestureRecognizer(UIPinchGestureRecognizer(target: self, action: #selector(pinch)))
  }
  // MARK: - Layout Management
  /**
   * Lays out all subviews with responsive design for different orientations and device types
   * Handles landscape and portrait orientations with optimized layouts for iPhone and iPad
   */
  public override func layoutSubviews() {
    setupOverlayLayer()
    let isLandscape = bounds.width > bounds.height
    activityIndicator.frame = CGRect(x: center.x - 50, y: center.y - 50, width: 100, height: 100)
    setupThresholdLayers()
    if isLandscape {
      let width = bounds.width
      let height = bounds.height
      let titleLabelHeight: CGFloat = height * 0.02
      labelName.frame = CGRect(
        x: 0,
        y: height * 0.01,
        width: width,
        height: titleLabelHeight
      )
      let toolBarHeight: CGFloat = 50
      let subLabelHeight: CGFloat = height * 0.03
      let topControlY: CGFloat
      if isLargeIPad {
        topControlY = height * 0.7
      } else {
        topControlY = height * 0.5
      }
      let sliderWidth = width * 0.22
      let sliderLabelHeight: CGFloat = 20
      let sliderHeight: CGFloat = height * 0.05
      let fishCountWidth = sliderWidth
      let resetButtonWidth = fishCountWidth * 0.6
      let controlHeight: CGFloat = 60
      labelFishCount.frame = CGRect(
        x: width * 0.05,
        y: topControlY,
        width: fishCountWidth,
        height: controlHeight
      )
      resetButton.frame = CGRect(
        x: width - width * 0.05 - resetButtonWidth,
        y: topControlY,
        width: resetButtonWidth,
        height: controlHeight
      )
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
      let autoButtonWidthLandscape = width * 0.08
      let autoButtonHeightLandscape: CGFloat = 24
      autoCalibrationButton.frame = CGRect(
        x: width * 0.5 - autoButtonWidthLandscape / 2,
        y: secondRowY,
        width: autoButtonWidthLandscape,
        height: autoButtonHeightLandscape
      )
      if isLargeIPad {
        labelFPS.frame = CGRect(
          x: width * 0.35,
          y: secondRowY + autoButtonHeightLandscape + 8,
          width: width * 0.3,
          height: subLabelHeight
        )
      } else {
        labelFPS.frame = CGRect(
          x: width * 0.35,
          y: height - toolBarHeight - subLabelHeight - 5,
          width: width * 0.3,
          height: subLabelHeight
        )
      }
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
      let thirdRowY = secondRowY + sliderLabelHeight + sliderHeight + height * 0.02
      labelSliderConf.frame = CGRect(
        x: -1000,
        y: thirdRowY,
        width: sliderWidth,
        height: sliderLabelHeight
      )
      sliderConf.frame = CGRect(
        x: -1000,
        y: thirdRowY + sliderLabelHeight + 2,
        width: sliderWidth,
        height: sliderHeight
      )
      labelSliderIoU.frame = CGRect(
        x: -1000,
        y: thirdRowY,
        width: sliderWidth,
        height: sliderLabelHeight
      )
      labelSliderIoU.textAlignment = .right
      sliderIoU.frame = CGRect(
        x: -1000,
        y: thirdRowY + sliderLabelHeight + 2,
        width: sliderWidth,
        height: sliderHeight
      )
      updateThresholdLayer(threshold1Layer, position: CGFloat(threshold1Slider.value))
      updateThresholdLayer(threshold2Layer, position: CGFloat(threshold2Slider.value))
      let numItemsSliderWidth: CGFloat = width * 0.25
      let numItemsSliderHeight: CGFloat = height * 0.02
      sliderNumItems.frame = CGRect(
        x: (width - numItemsSliderWidth) / 2,
        y: height * 0.3,
        width: numItemsSliderWidth,
        height: numItemsSliderHeight
      )
      labelSliderNumItems.frame = CGRect(
        x: (width - numItemsSliderWidth) / 2,
        y: height * 0.27,
        width: numItemsSliderWidth,
        height: height * 0.03
      )
      labelSliderNumItems.textAlignment = .center
      let zoomLabelWidth: CGFloat = width * 0.1
      labelZoom.frame = CGRect(
        x: width - zoomLabelWidth - 10,
        y: height * 0.08,
        width: zoomLabelWidth,
        height: height * 0.03
      )
      toolbar.frame = CGRect(
        x: 0,
        y: height - toolBarHeight,
        width: width,
        height: toolBarHeight
      )
      let buttonWidth: CGFloat = 50
      let spacing = (width - 5 * buttonWidth) / 6
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
      let toolBarHeight: CGFloat = 66
      let subLabelHeight: CGFloat = height * 0.03
      let sliderWidth = width * 0.4
      let sliderHeight: CGFloat = height * 0.02
      let sliderLabelHeight: CGFloat = 20
      let bottomControlsY: CGFloat
      if isLargeIPad {
        bottomControlsY = height * 0.88
      } else {
        bottomControlsY = height * 0.85
      }
      labelSliderConf.frame = CGRect(
        x: -1000,
        y: bottomControlsY - sliderLabelHeight - 5,
        width: sliderWidth * 0.5,
        height: sliderLabelHeight
      )
      sliderConf.frame = CGRect(
        x: -1000,
        y: bottomControlsY,
        width: sliderWidth,
        height: sliderHeight
      )
      labelSliderIoU.frame = CGRect(
        x: -1000,
        y: bottomControlsY - sliderLabelHeight - 5,
        width: sliderWidth,
        height: sliderLabelHeight
      )
      labelSliderIoU.textAlignment = .right
      sliderIoU.frame = CGRect(
        x: -1000,
        y: bottomControlsY,
        width: sliderWidth,
        height: sliderHeight
      )
      let thresholdY = bottomControlsY
      let fishCountWidth = sliderWidth
      let fishCountHeight: CGFloat = 70
      let fishCountY = thresholdY - sliderHeight - fishCountHeight - 15
      let resetButtonWidth = fishCountWidth * 0.4
      labelFishCount.frame = CGRect(
        x: width * 0.05,
        y: fishCountY,
        width: fishCountWidth,
        height: fishCountHeight
      )
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
      let autoButtonWidthPortrait = width * 0.18
      let autoButtonHeightPortrait: CGFloat = 24
      autoCalibrationButton.frame = CGRect(
        x: (width - autoButtonWidthPortrait) / 2,
        y: thresholdY - sliderLabelHeight,
        width: autoButtonWidthPortrait,
        height: autoButtonHeightPortrait
      )
      if isLargeIPad {
        labelFPS.frame = CGRect(
          x: 0,
          y: thresholdY - sliderLabelHeight + autoButtonHeightPortrait + 8,
          width: width,
          height: subLabelHeight
        )
      } else {
        labelFPS.frame = CGRect(
          x: 0,
          y: height - toolBarHeight - subLabelHeight - 5,
          width: width,
          height: subLabelHeight
        )
      }
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
      updateThresholdLayer(threshold1Layer, position: CGFloat(threshold1Slider.value))
      updateThresholdLayer(threshold2Layer, position: CGFloat(threshold2Slider.value))
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
      switchSourceButton.frame = CGRect(
        x: pauseButton.frame.maxX,
        y: 0,
        width: buttonHeight,
        height: buttonHeight
      )
      modelsButton.frame = CGRect(
        x: switchSourceButton.frame.maxX,
        y: 0,
        width: buttonHeight,
        height: buttonHeight
      )
      directionButton.frame = CGRect(
        x: modelsButton.frame.maxX,
        y: 0,
        width: buttonHeight,
        height: buttonHeight
      )
    }
    self.videoCapture.previewLayer?.frame = self.bounds
    if frameSourceType == .videoFile, let playerLayer = albumVideoSource?.playerLayer {
      playerLayer.frame = self.bounds
      albumVideoSource?.updateForOrientationChange(orientation: UIDevice.current.orientation)
      setupOverlayLayer()
    }
    if frameSourceType == .uvc, let uvcSource = uvcVideoSource {
      uvcSource.updateForOrientationChange(orientation: UIDevice.current.orientation)
      if let previewLayer = uvcSource.previewLayer {
        previewLayer.frame = self.bounds
        print("FishCountView: Updated UVC preview layer frame to \(self.bounds)")
      }
      setupOverlayLayer()
    }
  }
  // MARK: - Orientation Handling
  /**
   * Sets up notification observer for device orientation changes
   */
  private func setUpOrientationChangeNotification() {
    NotificationCenter.default.addObserver(
      self, selector: #selector(orientationDidChange),
      name: UIDevice.orientationDidChangeNotification, object: nil)
  }
  /**
   * Handles device orientation changes
   * Updates frame source and overlay layer for new orientation
   */
  @objc func orientationDidChange() {
    let orientation = UIDevice.current.orientation
    currentFrameSource.updateForOrientationChange(orientation: orientation)
    setupOverlayLayer()
  }
  // MARK: - User Interaction Handlers
  /**
   * Handles changes to detection threshold sliders
   * Updates confidence, IoU, and max items thresholds
   */
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
    TrackingDetectorConfig.shared.updateDefaults(
      confidenceThreshold: Float(conf), iouThreshold: Float(iou))
    if let detector = videoCapture.predictor as? ObjectDetector {
      detector.setIouThreshold(iou: iou)
      detector.setConfidenceThreshold(confidence: conf)
    }
  }
  /**
   * Handles pinch gestures for camera zoom
   * Adjusts camera zoom factor within min/max bounds
   */
  @objc func pinch(_ pinch: UIPinchGestureRecognizer) {
    guard let device = videoCapture.captureDevice else { return }
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
  // MARK: - Video Controls
  /**
   * Handles play button tap
   * Starts or resumes video playback for different frame sources
   */
  @objc func playTapped() {
    selection.selectionChanged()
    currentFrameSource.inferenceOK = true
    if frameSourceType == .videoFile, let albumSource = albumVideoSource {
      if !self.pauseButton.isEnabled {
        albumSource.stop()
        Task { @MainActor in
          if let player = albumSource.playerLayer?.player {
            player.seek(to: CMTime.zero)
            albumSource.start()
          }
        }
      } else {
        albumSource.start()
      }
    } else if frameSourceType == .uvc, let uvcSource = uvcVideoSource {
      uvcSource.start()
    } else {
      self.videoCapture.start()
    }
    playButton.isEnabled = false
    pauseButton.isEnabled = true
  }
  /**
   * Handles pause button tap
   * Pauses video playback and enables play button
   */
  @objc func pauseTapped() {
    selection.selectionChanged()
    currentFrameSource.stop()
    playButton.isEnabled = true
    pauseButton.isEnabled = false
  }
  // MARK: - Fish Counting Features
  /**
   * Sets up threshold line layers for fish counting visualization
   * Creates red and yellow dashed lines for counting zone boundaries
   */
  private func setupThresholdLayers() {
    if threshold1Layer == nil {
      let layer = CAShapeLayer()
      layer.strokeColor = UIColor.red.cgColor
      layer.lineWidth = 3.0
      layer.lineDashPattern = [5, 5]
      layer.zPosition = 999
      layer.opacity = 0.5
      self.layer.addSublayer(layer)
      threshold1Layer = layer
    }
    if threshold2Layer == nil {
      let layer = CAShapeLayer()
      layer.strokeColor = UIColor.yellow.cgColor
      layer.lineWidth = 3.0
      layer.lineDashPattern = [5, 5]
      layer.zPosition = 999
      layer.opacity = 0.5
      self.layer.addSublayer(layer)
      threshold2Layer = layer
    }
    DispatchQueue.main.async {
      self.updateThresholdLayer(
        self.threshold1Layer, position: CGFloat(self.threshold1Slider.value))
      self.updateThresholdLayer(
        self.threshold2Layer, position: CGFloat(self.threshold2Slider.value))
    }
  }
  /**
   * Updates a threshold line layer to the specified position
   * Converts threshold position to screen coordinates based on counting direction
   */
  private func updateThresholdLayer(_ layer: CAShapeLayer?, position: CGFloat) {
    guard let layer = layer else { return }
    let thresholdLines = UnifiedCoordinateSystem.thresholdsToScreen(
      [position],
      countingDirection: countingDirection,
      screenBounds: self.bounds
    )
    guard let thresholdLine = thresholdLines.first else { return }
    let path = UIBezierPath()
    switch countingDirection {
    case .topToBottom, .bottomToTop:
      path.move(to: CGPoint(x: thresholdLine.minX, y: thresholdLine.minY))
      path.addLine(to: CGPoint(x: thresholdLine.maxX, y: thresholdLine.minY))
    case .leftToRight, .rightToLeft:
      path.move(to: CGPoint(x: thresholdLine.minX, y: thresholdLine.minY))
      path.addLine(to: CGPoint(x: thresholdLine.minX, y: thresholdLine.maxY))
    }
    layer.path = path.cgPath
  }
  /**
   * Updates threshold lines when counting direction changes
   * Recalculates positions and updates tracking detector
   */
  private func updateThresholdLinesForDirection(_ direction: CountingDirection) {
    labelThreshold1.text = "Threshold 1: " + String(format: "%.2f", threshold1Slider.value)
    labelThreshold2.text = "Threshold 2: " + String(format: "%.2f", threshold2Slider.value)
    updateThresholdLayer(threshold1Layer, position: threshold1)
    updateThresholdLayer(threshold2Layer, position: threshold2)
    if task == .fishCount, let trackingDetector = currentFrameSource.predictor as? TrackingDetector
    {
      let countingThresholds = UnifiedCoordinateSystem.displayToCounting(
        [threshold1, threshold2],
        countingDirection: direction
      )
      trackingDetector.setThresholds(countingThresholds)
    }
  }
  /**
   * Handles first threshold slider changes
   * Updates threshold position and tracking detector configuration
   */
  @objc func threshold1Changed(_ sender: UISlider) {
    let value = CGFloat(sender.value)
    threshold1 = value
    labelThreshold1.text = "Threshold 1: " + String(format: "%.2f", value)
    updateThresholdLayer(threshold1Layer, position: value)
    // Update ThresholdCounter default thresholds (single source of truth)
    ThresholdCounter.defaultThresholds = [value, threshold2]
    if task == .fishCount, let trackingDetector = currentFrameSource.predictor as? TrackingDetector
    {
      let countingThresholds = UnifiedCoordinateSystem.displayToCounting(
        [value, threshold2],
        countingDirection: countingDirection
      )
      trackingDetector.setThresholds(countingThresholds)
    }
  }
  /**
   * Handles second threshold slider changes
   * Updates threshold position and tracking detector configuration
   */
  @objc func threshold2Changed(_ sender: UISlider) {
    let value = CGFloat(sender.value)
    threshold2 = value
    labelThreshold2.text = "Threshold 2: " + String(format: "%.2f", value)
    updateThresholdLayer(threshold2Layer, position: value)
    // Update ThresholdCounter default thresholds (single source of truth)
    ThresholdCounter.defaultThresholds = [threshold1, value]
    if task == .fishCount, let trackingDetector = currentFrameSource.predictor as? TrackingDetector
    {
      let countingThresholds = UnifiedCoordinateSystem.displayToCounting(
        [threshold1, value],
        countingDirection: countingDirection
      )
      trackingDetector.setThresholds(countingThresholds)
    }
  }
  /**
   * Resets the fish count to zero
   * Clears tracking detector count and updates display
   */
  @objc func resetFishCount() {
    if task == .fishCount, let trackingDetector = currentFrameSource.predictor as? TrackingDetector
    {
      trackingDetector.resetCount()
      updateFishCountDisplay(count: 0)
    }
  }
  /**
   * Toggles auto-calibration mode on/off
   * Handles the complete calibration workflow including progress tracking
   */
  @objc func toggleAutoCalibration() {
    guard task == .fishCount,
      let trackingDetector = currentFrameSource.predictor as? TrackingDetector
    else {
      return
    }
    let isCurrentlyCalibrating = trackingDetector.getAutoCalibrationEnabled()
    if isCurrentlyCalibrating {
      trackingDetector.setAutoCalibration(enabled: false)
      currentFrameSource.inferenceOK = true
      isCalibrating = false
      let autoAttributedText = NSMutableAttributedString()
      let auAttributes: [NSAttributedString.Key: Any] = [
        .foregroundColor: UIColor.red.withAlphaComponent(0.5),
        .font: UIFont.systemFont(ofSize: 14, weight: .bold),
      ]
      let toAttributes: [NSAttributedString.Key: Any] = [
        .foregroundColor: UIColor.yellow.withAlphaComponent(0.5),
        .font: UIFont.systemFont(ofSize: 14, weight: .bold),
      ]
      autoAttributedText.append(NSAttributedString(string: "AU", attributes: auAttributes))
      autoAttributedText.append(NSAttributedString(string: "TO", attributes: toAttributes))
      autoCalibrationButton.setAttributedTitle(autoAttributedText, for: .normal)
    } else {
      isCalibrating = true
      boundingBoxViews.forEach { box in
        box.hide()
      }
      onClearBoxes()
      let progressText = NSAttributedString(
        string: "0%",
        attributes: [
          .foregroundColor: UIColor.white.withAlphaComponent(0.5),
          .font: UIFont.systemFont(ofSize: 14, weight: .bold),
        ]
      )
      autoCalibrationButton.setAttributedTitle(progressText, for: .normal)
      trackingDetector.onCalibrationProgress = { [weak self] progress, total in
        guard let self = self else { return }
        let percentage = Int(Double(progress) / Double(total) * 100.0)
        DispatchQueue.main.async {
          let progressText = NSAttributedString(
            string: "\(percentage)%",
            attributes: [
              .foregroundColor: UIColor.white.withAlphaComponent(0.5),
              .font: UIFont.systemFont(ofSize: 14, weight: .bold),
            ]
          )
          self.autoCalibrationButton.setAttributedTitle(progressText, for: .normal)
        }
      }
      trackingDetector.onCalibrationComplete = { [weak self] thresholds in
        guard let self = self else { return }
        DispatchQueue.main.async {
          if thresholds.count >= 2 {
            let uiThresholds = UnifiedCoordinateSystem.countingToDisplay(
              thresholds,
              countingDirection: self.countingDirection
            )
            let uiThreshold1 = uiThresholds[0]
            let uiThreshold2 = uiThresholds[1]
            self.threshold1Slider.value = Float(uiThreshold1)
            self.threshold2Slider.value = Float(uiThreshold2)
            self.labelThreshold1.text = "Threshold 1: " + String(format: "%.2f", uiThreshold1)
            self.labelThreshold2.text = "Threshold 2: " + String(format: "%.2f", uiThreshold2)
            self.updateThresholdLayer(self.threshold1Layer, position: uiThreshold1)
            self.updateThresholdLayer(self.threshold2Layer, position: uiThreshold2)
            self.threshold1 = uiThreshold1
            self.threshold2 = uiThreshold2
          }
          let config = AutoCalibrationConfig.shared
          if config.isDirectionCalibrationEnabled {
            self.currentFrameSource.inferenceOK = true
            self.currentFrameSource.resetProcessingState()
          }
        }
      }
      trackingDetector.onDirectionDetected = { [weak self] detectedDirection in
        guard let self = self else { return }
        DispatchQueue.main.async {
          self.countingDirection = detectedDirection
          // Update counting direction in ThresholdCounter (single source of truth)
          ThresholdCounter.defaultCountingDirection = detectedDirection
          self.updateThresholdLinesForDirection(detectedDirection)
        }
      }
      trackingDetector.onCalibrationSummary = { [weak self] summary in
        guard let self = self else { return }
        DispatchQueue.main.async {
          self.isCalibrating = false
          let autoAttributedText = NSMutableAttributedString()
          let auAttributes: [NSAttributedString.Key: Any] = [
            .foregroundColor: UIColor.red.withAlphaComponent(0.5),
            .font: UIFont.systemFont(ofSize: 14, weight: .bold),
          ]
          let toAttributes: [NSAttributedString.Key: Any] = [
            .foregroundColor: UIColor.yellow.withAlphaComponent(0.5),
            .font: UIFont.systemFont(ofSize: 14, weight: .bold),
          ]
          autoAttributedText.append(NSAttributedString(string: "AU", attributes: auAttributes))
          autoAttributedText.append(NSAttributedString(string: "TO", attributes: toAttributes))
          self.autoCalibrationButton.setAttributedTitle(autoAttributedText, for: .normal)
          self.currentFrameSource.inferenceOK = true
          self.currentFrameSource.resetProcessingState()
          self.showCalibrationSummary(summary)
        }
      }
      let currentUIThresholds = [threshold1, threshold2]
      let currentCountingThresholds = UnifiedCoordinateSystem.displayToCounting(
        currentUIThresholds,
        countingDirection: countingDirection
      )
      trackingDetector.setThresholds(
        currentCountingThresholds, originalDisplayValues: currentUIThresholds)
      print(
        "AutoCalibration: Using current threshold values: UI=\(currentUIThresholds), Counting=\(currentCountingThresholds)"
      )
      trackingDetector.setAutoCalibration(enabled: true)
      currentFrameSource.inferenceOK = false
      DispatchQueue.main.asyncAfter(deadline: .now() + 30.0) { [weak self] in
        guard let self = self, self.isCalibrating else { return }
        self.isCalibrating = false
        self.currentFrameSource.inferenceOK = true
        let autoAttributedText = NSMutableAttributedString()
        let auAttributes: [NSAttributedString.Key: Any] = [
          .foregroundColor: UIColor.red.withAlphaComponent(0.5),
          .font: UIFont.systemFont(ofSize: 14, weight: .bold),
        ]
        let toAttributes: [NSAttributedString.Key: Any] = [
          .foregroundColor: UIColor.yellow.withAlphaComponent(0.5),
          .font: UIFont.systemFont(ofSize: 14, weight: .bold),
        ]
        autoAttributedText.append(NSAttributedString(string: "AU", attributes: auAttributes))
        autoAttributedText.append(NSAttributedString(string: "TO", attributes: toAttributes))
        self.autoCalibrationButton.setAttributedTitle(autoAttributedText, for: .normal)
      }
    }
  }
  // MARK: - Frame Source Switching
  /**
   * Switches to a different frame source (camera, video file, GoPro, UVC)
   * Handles cleanup of current source and setup of new source
   */
  public func switchToFrameSource(_ sourceType: FrameSourceType) {
    // Early return for camera source when already using camera
    if frameSourceType == sourceType && sourceType == .camera {
      return
    }
    // Handle UVC source selection with iPad check and prompt
    if sourceType == .uvc {
      guard isIPad else {
        print("UVC source is only supported on iPad")
        return
      }
      var topViewController = UIApplication.shared.windows.first?.rootViewController
      while let presentedViewController = topViewController?.presentedViewController {
        topViewController = presentedViewController
      }
      guard let viewController = topViewController else {
        print("Could not find a view controller to present UVC alerts")
        return
      }
      self.showUVCConnectionPrompt(viewController: viewController)
      return
    }
    if frameSourceType == .uvc && sourceType != .uvc {
      print("FishCountView: Clearing UVCVideoSource reference when switching away from UVC")
      if let uvcPreviewLayer = uvcVideoSource?.previewLayer {
        uvcPreviewLayer.removeFromSuperlayer()
        print("FishCountView: Removed UVC preview layer")
      }
      uvcVideoSource = nil
    }
    currentFrameSource.stop()
    boundingBoxViews.forEach { box in
      box.hide()
    }
    resetLayers()
    if isCalibrating {
      isCalibrating = false
      let autoAttributedText = NSMutableAttributedString()
      let auAttributes: [NSAttributedString.Key: Any] = [
        .foregroundColor: UIColor.red.withAlphaComponent(0.5),
        .font: UIFont.systemFont(ofSize: 14, weight: .bold),
      ]
      let toAttributes: [NSAttributedString.Key: Any] = [
        .foregroundColor: UIColor.yellow.withAlphaComponent(0.5),
        .font: UIFont.systemFont(ofSize: 14, weight: .bold),
      ]
      autoAttributedText.append(NSAttributedString(string: "AU", attributes: auAttributes))
      autoAttributedText.append(NSAttributedString(string: "TO", attributes: toAttributes))
      autoCalibrationButton.setAttributedTitle(autoAttributedText, for: .normal)
    }
    switch sourceType {
    case .camera:
      videoCapture.requestPermission { [weak self] granted in
        guard let self = self else { return }
        if !granted {
          self.showPermissionAlert(for: .camera)
          return
        }
        if let albumSource = self.albumVideoSource, let playerLayer = albumSource.playerLayer {
          playerLayer.removeFromSuperlayer()
        }
        if let previewLayer = self.videoCapture.previewLayer {
          previewLayer.isHidden = false
        }
        self.currentFrameSource = self.videoCapture
        self.frameSourceType = .camera
        self.currentFrameSource.inferenceOK = true
        self.start(position: .back)
      }
    case .videoFile:
      if let albumSource = albumVideoSource, let playerLayer = albumSource.playerLayer {
        playerLayer.removeFromSuperlayer()
        print("FishCountView: Removed existing album player layer for video reselection")
      }
      if albumVideoSource == nil {
        albumVideoSource = AlbumVideoSource()
        albumVideoSource?.predictor = videoCapture.predictor
        albumVideoSource?.delegate = self
        albumVideoSource?.videoCaptureDelegate = self
        NotificationCenter.default.addObserver(
          self,
          selector: #selector(handleVideoPlaybackEnd),
          name: .videoPlaybackDidEnd,
          object: nil
        )
      }
      if let previewLayer = videoCapture.previewLayer {
        previewLayer.isHidden = true
      }
      var topViewController = UIApplication.shared.windows.first?.rootViewController
      while let presentedViewController = topViewController?.presentedViewController {
        topViewController = presentedViewController
      }
      guard let viewController = topViewController else {
        print("Could not find a view controller to present the video picker")
        return
      }
      albumVideoSource?.showContentSelectionUI(from: viewController) { [weak self] success in
        guard let self = self else { return }
        if !success {
          self.switchToFrameSource(.camera)
          return
        }
        if let playerLayer = self.albumVideoSource?.playerLayer {
          playerLayer.frame = self.bounds
          self.layer.insertSublayer(playerLayer, at: 0)
          playerLayer.addSublayer(self.overlayLayer)
          for box in self.boundingBoxViews {
            box.addToLayer(playerLayer)
          }
          self.setupOverlayLayer()
        }
        self.currentFrameSource = self.albumVideoSource!
        self.frameSourceType = .videoFile
        self.currentFrameSource.inferenceOK = true
      }
    default:
      break
    }
  }
  private func performActualFrameSourceSwitch(to sourceType: FrameSourceType) {
    print("FishCountView: Performing actual frame source switch to \(sourceType)")
    if frameSourceType == .uvc && sourceType != .uvc {
      print("FishCountView: Clearing UVCVideoSource reference when switching away from UVC")
      if let uvcPreviewLayer = uvcVideoSource?.previewLayer {
        uvcPreviewLayer.removeFromSuperlayer()
        print("FishCountView: Removed UVC preview layer")
      }
      uvcVideoSource = nil
    }
    currentFrameSource.stop()
    boundingBoxViews.forEach { box in
      box.hide()
    }
    resetLayers()
    if isCalibrating {
      isCalibrating = false
      let autoAttributedText = NSMutableAttributedString()
      let auAttributes: [NSAttributedString.Key: Any] = [
        .foregroundColor: UIColor.red.withAlphaComponent(0.5),
        .font: UIFont.systemFont(ofSize: 14, weight: .bold),
      ]
      let toAttributes: [NSAttributedString.Key: Any] = [
        .foregroundColor: UIColor.yellow.withAlphaComponent(0.5),
        .font: UIFont.systemFont(ofSize: 14, weight: .bold),
      ]
      autoAttributedText.append(NSAttributedString(string: "AU", attributes: auAttributes))
      autoAttributedText.append(NSAttributedString(string: "TO", attributes: toAttributes))
      autoCalibrationButton.setAttributedTitle(autoAttributedText, for: .normal)
    }
    frameSourceType = sourceType
    print("FishCountView: Frame source switch to \(sourceType) completed")
  }
  private func showPermissionAlert(for sourceType: FrameSourceType) {
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
    alert.addAction(
      UIAlertAction(title: "Settings", style: .default) { _ in
        if let url = URL(string: UIApplication.openSettingsURLString) {
          UIApplication.shared.open(url)
        }
      })
    viewController.present(alert, animated: true)
  }
  @objc func switchSourceButtonTapped() {
    var topViewController = UIApplication.shared.windows.first?.rootViewController
    while let presentedViewController = topViewController?.presentedViewController {
      topViewController = presentedViewController
    }
    guard let viewController = topViewController else { return }
    let alert = UIAlertController(
      title: "Select Frame Source", message: nil, preferredStyle: .actionSheet)
    let cameraAction = UIAlertAction(title: "Camera", style: .default) { [weak self] _ in
      guard let self = self else { return }
      if self.frameSourceType != .camera {
        self.switchToFrameSource(.camera)
      }
    }
    if frameSourceType == .camera {
      cameraAction.setValue(true, forKey: "checked")
    }
    alert.addAction(cameraAction)
    let albumAction = UIAlertAction(title: "Album", style: .default) { [weak self] _ in
      guard let self = self else { return }
      self.switchToFrameSource(.videoFile)
    }
    if frameSourceType == .videoFile {
      albumAction.setValue(true, forKey: "checked")
    }
    alert.addAction(albumAction)
    if isIPad {
      let uvcAction = UIAlertAction(title: "External Camera (UVC)", style: .default) {
        [weak self] _ in
        guard let self = self else { return }
        self.switchToFrameSource(.uvc)
      }
      if frameSourceType == .uvc {
        uvcAction.setValue(true, forKey: "checked")
      }
      alert.addAction(uvcAction)
    }
    alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))
    if let popoverController = alert.popoverPresentationController {
      popoverController.sourceView = switchSourceButton
      popoverController.sourceRect = switchSourceButton.bounds
    }
    viewController.present(alert, animated: true, completion: nil)
  }
  // MARK: - Connection Prompt and Setup Helpers
  /**
   * Shows GoPro connection prompt to user
   * Handles different connection scenarios and options
   */
  /**
     * Shows UVC camera connection prompt to user
     * Provides options for connecting external cameras
     */
  private func showUVCConnectionPrompt(viewController: UIViewController) {
    let connectionAlert = UIAlertController(
      title: "UVC External Camera",
      message: "Please connect your UVC camera to the iPad's USB-C port before continuing.",
      preferredStyle: .alert
    )
    connectionAlert.addAction(
      UIAlertAction(title: "Cancel", style: .cancel) { [weak self] _ in
        print("UVC: Connection setup cancelled by user")
      })
    connectionAlert.addAction(
      UIAlertAction(title: "Continue", style: .default) { [weak self] _ in
        guard let self = self else { return }
        self.setupUVCSource(viewController: viewController)
      })
    viewController.present(connectionAlert, animated: true)
  }
  private func setupUVCSource(viewController: UIViewController) {
    print("FishCountView: Starting UVC source setup (reconnection)")
    if let existingUVCSource = uvcVideoSource {
      print("FishCountView: Cleaning up existing UVC source")
      existingUVCSource.stop()
      if let existingPreviewLayer = existingUVCSource.previewLayer {
        existingPreviewLayer.removeFromSuperlayer()
        print("FishCountView: Removed existing UVC preview layer")
      }
      uvcVideoSource = nil
    }
    videoCapture.previewLayer?.isHidden = true
    albumVideoSource?.playerLayer?.removeFromSuperlayer()
    print("FishCountView: Creating fresh UVCVideoSource")
    uvcVideoSource = UVCVideoSource()
    uvcVideoSource?.predictor = videoCapture.predictor
    uvcVideoSource?.delegate = self
    uvcVideoSource?.videoCaptureDelegate = self
    uvcVideoSource?.setUp { [weak self] success in
      Task { @MainActor in
        guard let self = self else { return }
        if success {
          print("FishCountView: UVC setup successful - starting integration")
          self.performActualFrameSourceSwitch(to: .uvc)
          self.currentFrameSource = self.uvcVideoSource!
          self.currentFrameSource.inferenceOK = true
          self.boundingBoxViews.forEach { box in
            box.hide()
          }
          self.uvcVideoSource?.integrateWithFishCountView(view: self)
          self.uvcVideoSource?.addBoundingBoxViews(self.boundingBoxViews)
          self.uvcVideoSource?.addOverlayLayer(self.overlayLayer)
          self.uvcVideoSource?.start()
          print("FishCountView: UVC source reconnection completed successfully")
        } else {
          print("FishCountView: UVC setup failed - no camera detected")
          self.performActualFrameSourceSwitch(to: .uvc)
          if let uvcSource = self.uvcVideoSource {
            self.currentFrameSource = uvcSource
            self.currentFrameSource.inferenceOK = true
          }
        }
        self.showUVCStatus(viewController: viewController)
      }
    }
  }
  private func showUVCStatus(viewController: UIViewController) {
    let statusMessage: String
    if let uvcSource = uvcVideoSource {
      statusMessage = uvcSource.getDetailedStatus()
    } else {
      statusMessage = "No UVC camera detected"
    }
    let statusAlert = UIAlertController(
      title: "UVC Camera Configuration",
      message: statusMessage,
      preferredStyle: .alert
    )
    statusAlert.addAction(UIAlertAction(title: "OK", style: .default))
    viewController.present(statusAlert, animated: true)
  }
  @MainActor
  @objc func handleVideoPlaybackEnd(_ notification: Notification) {
    DispatchQueue.main.async {
      self.playButton.isEnabled = true
      self.pauseButton.isEnabled = false
    }
  }
  @objc func modelsButtonTapped() {
    selection.selectionChanged()
    actionDelegate?.didTapModelsButton()
  }
  @objc func directionButtonTapped() {
    selection.selectionChanged()
    var topViewController = UIApplication.shared.windows.first?.rootViewController
    while let presentedViewController = topViewController?.presentedViewController {
      topViewController = presentedViewController
    }
    guard let viewController = topViewController else { return }
    let alert = UIAlertController(
      title: "Select Counting Direction", message: nil, preferredStyle: .actionSheet)
    let bottomToTopAction = UIAlertAction(title: "Bottom to Top", style: .default) {
      [weak self] _ in
      guard let self = self else { return }
      if self.countingDirection != .bottomToTop {
        self.switchCountingDirection(.bottomToTop)
      }
    }
    if countingDirection == .bottomToTop {
      bottomToTopAction.setValue(true, forKey: "checked")
    }
    alert.addAction(bottomToTopAction)
    let topToBottomAction = UIAlertAction(title: "Top to Bottom", style: .default) {
      [weak self] _ in
      guard let self = self else { return }
      if self.countingDirection != .topToBottom {
        self.switchCountingDirection(.topToBottom)
      }
    }
    if countingDirection == .topToBottom {
      topToBottomAction.setValue(true, forKey: "checked")
    }
    alert.addAction(topToBottomAction)
    let leftToRightAction = UIAlertAction(title: "Left to Right", style: .default) {
      [weak self] _ in
      guard let self = self else { return }
      if self.countingDirection != .leftToRight {
        self.switchCountingDirection(.leftToRight)
      }
    }
    if countingDirection == .leftToRight {
      leftToRightAction.setValue(true, forKey: "checked")
    }
    alert.addAction(leftToRightAction)
    let rightToLeftAction = UIAlertAction(title: "Right to Left", style: .default) {
      [weak self] _ in
      guard let self = self else { return }
      if self.countingDirection != .rightToLeft {
        self.switchCountingDirection(.rightToLeft)
      }
    }
    if countingDirection == .rightToLeft {
      rightToLeftAction.setValue(true, forKey: "checked")
    }
    alert.addAction(rightToLeftAction)
    alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))
    if let popoverController = alert.popoverPresentationController {
      popoverController.sourceView = directionButton
      popoverController.sourceRect = directionButton.bounds
    }
    viewController.present(alert, animated: true, completion: nil)
  }
  // MARK: - Fish Counting Utility Methods
  /**
   * Changes the fish counting direction and updates UI accordingly
   * Reconfigures threshold lines and tracking detector
   */
  private func switchCountingDirection(_ direction: CountingDirection) {
    countingDirection = direction
    // Update counting direction in ThresholdCounter (single source of truth)
    ThresholdCounter.defaultCountingDirection = direction
    if task == .fishCount, let trackingDetector = currentFrameSource.predictor as? TrackingDetector
    {
      trackingDetector.setCountingDirection(direction)
    }
    updateThresholdLinesForDirection(direction)
  }
  func getCurrentPredictor() -> Predictor? {
    return currentFrameSource.predictor
  }
  /**
     * Updates the fish count display with current count
     * Retrieves count from tracking detector and updates UI label
     */
  private func updateFishCountDisplay() {
    updateFishCountDisplay(count: fishCount)
  }
  /**
     * Updates the fish count display with specified count
     * Sets the UI label to show the fish count
     */
  private func updateFishCountDisplay(count: Int) {
    let fishCountAttributedText = NSMutableAttributedString()
    let labelAttributes: [NSAttributedString.Key: Any] = [
      .foregroundColor: UIColor.white,
      .font: UIFont.systemFont(ofSize: 16, weight: .bold),
    ]
    let numberAttributes: [NSAttributedString.Key: Any] = [
      .foregroundColor: UIColor.white,
      .font: UIFont.systemFont(ofSize: 48, weight: .bold),
    ]
    fishCountAttributedText.append(
      NSAttributedString(string: "Fish Count: ", attributes: labelAttributes))
    fishCountAttributedText.append(
      NSAttributedString(string: "\(count)", attributes: numberAttributes))
    labelFishCount.attributedText = fishCountAttributedText
  }
  /**
     * Displays calibration results summary to the user
     * Shows threshold values and completion status
     */
  private func showCalibrationSummary(_ summary: CalibrationSummary) {
    var topViewController = UIApplication.shared.windows.first?.rootViewController
    while let presentedViewController = topViewController?.presentedViewController {
      topViewController = presentedViewController
    }
    guard let viewController = topViewController else {
      print("Could not find a view controller to present calibration summary")
      return
    }
    var message = "Auto-calibration completed!\n\n"
    if summary.thresholdCalibrationEnabled {
      message += "✅ Phase 1: Threshold Detection\n"
      message +=
        "Thresholds: \(String(format: "%.2f", summary.thresholds[0])), \(String(format: "%.2f", summary.thresholds[1]))\n\n"
    } else {
      message += "⏭️ Phase 1: Bypassed (using current thresholds)\n"
      message +=
        "Thresholds: \(String(format: "%.2f", summary.thresholds[0])), \(String(format: "%.2f", summary.thresholds[1]))\n\n"
    }
    if summary.directionCalibrationEnabled {
      message += "✅ Phase 2: Movement Analysis\n"
      if summary.movementAnalysisSuccess, let detectedDirection = summary.detectedDirection {
        let directionName: String
        switch detectedDirection {
        case .topToBottom: directionName = "Top to Bottom"
        case .bottomToTop: directionName = "Bottom to Top"
        case .leftToRight: directionName = "Left to Right"
        case .rightToLeft: directionName = "Right to Left"
        }
        message += "Direction detected: \(directionName)\n"
        message += "Analyzed tracks: \(summary.qualifiedTracksCount)\n"
        if detectedDirection != summary.originalDirection {
          message += "⚠️ Direction changed from original\n"
        }
      } else {
        message += "❌ Direction analysis failed\n"
        message += "Keeping original direction\n"
        message += "Analyzed tracks: \(summary.qualifiedTracksCount)\n"
      }
      if !summary.warnings.isEmpty {
        message += "\n⚠️ Warnings:\n"
        for warning in summary.warnings {
          message += "• \(warning)\n"
        }
      }
    } else {
      message += "⏭️ Phase 2: Disabled (keeping original direction)\n"
    }
    let alert = UIAlertController(
      title: "Auto-Calibration Results",
      message: message,
      preferredStyle: .alert
    )
    alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
    DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) { [weak alert] in
      if let alertController = alert, alertController.presentingViewController != nil {
        alertController.dismiss(animated: true, completion: nil)
      }
    }
    viewController.present(alert, animated: true, completion: nil)
  }
}
