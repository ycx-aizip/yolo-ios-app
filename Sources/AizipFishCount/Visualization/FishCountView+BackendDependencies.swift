//
//  FishCountView+BackendDependencies.swift
//  AizipFishCount
//
//  Backend-dependent extension for FishCountView
//  This file contains all methods that directly depend on AizipFishCount backend types
//
//  ⚠️ IMPORTANT: This extension will need refactoring when moving UI.swift outside AizipFishCount
//  All methods here use internal backend types that won't be available in the xcframework:
//  - TrackingDetector
//  - ObjectDetector
//  - Predictor
//  - ThresholdCounter
//  - UnifiedCoordinateSystem
//
//  Future Plan: Replace these with PublicAPI protocol calls (FishCountSession)
//

import AVFoundation
import UIKit
import Vision

// MARK: - FishCountView Extension: Backend Dependencies
extension FishCountView {
  
  // MARK: - Model Management (*** BACKEND DEPENDENCIES ***)
  /**
   * Loads and configures a YOLO model for the specified task
   *
   * This method handles loading different types of YOLO models from file paths or bundle resources.
   * It supports detection, classification, segmentation, pose estimation, OBB detection, and fish counting tasks.
   *
   * Backend Dependencies:
   * - TrackingDetector.create() - model loading
   * - ObjectDetector type checking
   * - Predictor protocol
   * - PredictorError
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
      // Fish counting setup - thresholds configured via TrackingDetectorConfig
      if predictor is ObjectDetector {
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


  // MARK: - Detection Rendering (*** BACKEND DEPENDENCIES ***)
  /**
   * Renders detection results as bounding boxes on the overlay
   * Handles different task types and tracking states for fish counting
   * 
   * Backend Dependencies:
   * - TrackingDetector.isObjectTracked() - check tracking status
   * - TrackingDetector.isObjectCounted() - check counting status
   * - TrackingDetector.getTrackInfo() - get track ID
   * - TrackingDetector.getCount() - get current count
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

  // MARK: - Fish Counting Features (*** BACKEND DEPENDENCIES ***)
  /**
   * Sets up threshold line layers for fish counting visualization
   * Creates red and yellow dashed lines for counting zone boundaries
   */
  func setupThresholdLayers() {
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
   * 
   * Backend Dependency: UnifiedCoordinateSystem.thresholdsToScreen()
   */
  func updateThresholdLayer(_ layer: CAShapeLayer?, position: CGFloat) {
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
   * 
   * Backend Dependencies:
   * - UnifiedCoordinateSystem.displayToCounting() - coordinate conversion
   * - TrackingDetector.setThresholds() - apply thresholds
   */
  func updateThresholdLinesForDirection(_ direction: CountingDirection) {
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
   * 
   * Backend Dependencies:
   * - ThresholdCounter.defaultThresholds - update default values
   * - UnifiedCoordinateSystem.displayToCounting() - coordinate conversion
   * - TrackingDetector.setThresholds() - apply thresholds
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
   * 
   * Backend Dependencies:
   * - ThresholdCounter.defaultThresholds - update default values
   * - UnifiedCoordinateSystem.displayToCounting() - coordinate conversion
   * - TrackingDetector.setThresholds() - apply thresholds
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
   * 
   * Backend Dependency: TrackingDetector.resetCount()
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
   * 
   * Backend Dependencies:
   * - TrackingDetector.getAutoCalibrationEnabled() - check calibration state
   * - TrackingDetector.setAutoCalibration() - enable/disable calibration
   * - TrackingDetector.setThresholds() - set initial thresholds
   * - TrackingDetector.onCalibrationProgress - progress callback
   * - TrackingDetector.onCalibrationComplete - completion callback
   * - TrackingDetector.onDirectionDetected - direction detection callback
   * - TrackingDetector.onCalibrationSummary - summary callback
   * - UnifiedCoordinateSystem coordinate conversions
   * - AutoCalibrationConfig
   * - ThresholdCounter.defaultCountingDirection
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
      resetCalibrationUI()
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
          self.resetCalibrationUI()
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
        self.resetCalibrationUI()
        self.currentFrameSource.inferenceOK = true
      }
    }
  }

  // MARK: - Counting Direction Management (*** BACKEND DEPENDENCIES ***)
  /**
   * Changes the fish counting direction and updates UI accordingly
   * Reconfigures threshold lines and tracking detector
   * 
   * Backend Dependencies:
   * - ThresholdCounter.defaultCountingDirection - update default direction
   * - TrackingDetector.setCountingDirection() - apply direction
   */
  func switchCountingDirection(_ direction: CountingDirection) {
    countingDirection = direction
    // Update counting direction in ThresholdCounter (single source of truth)
    ThresholdCounter.defaultCountingDirection = direction
    if task == .fishCount, let trackingDetector = currentFrameSource.predictor as? TrackingDetector
    {
      trackingDetector.setCountingDirection(direction)
    }
    updateThresholdLinesForDirection(direction)
  }
  
  /**
   * Gets the current predictor instance
   * 
   * Backend Dependency: Returns Predictor (backend type)
   */
  func getCurrentPredictor() -> Predictor? {
    return currentFrameSource.predictor
  }
}
