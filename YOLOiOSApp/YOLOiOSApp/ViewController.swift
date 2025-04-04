// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
//  This file is part of the Ultralytics YOLO app, providing the main user interface for model selection and visualization.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ViewController serves as the primary interface for users to interact with YOLO models.
//  It provides the ability to select different models, tasks (detection, segmentation, classification, etc.),
//  and visualize results in real-time. The controller manages the loading of local and remote models,
//  handles UI updates during model loading and inference, and provides functionality for capturing
//  and sharing detection results. Advanced features include screen recording, model download progress
//  tracking, and adaptive UI layout for different device orientations.

import AVFoundation
import AudioToolbox
import CoreML
import CoreMedia
import ReplayKit
import UIKit
import YOLO

/// The main view controller for the YOLO iOS application, handling model selection and visualization.
class ViewController: UIViewController {

  @IBOutlet weak var yoloView: YOLOView!
  @IBOutlet var View0: UIView!
  @IBOutlet var segmentedControl: UISegmentedControl!
  @IBOutlet weak var labelName: UILabel!
  @IBOutlet weak var labelFPS: UILabel!
  @IBOutlet weak var labelVersion: UILabel!
  @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
  @IBOutlet weak var forcus: UIImageView!
  @IBOutlet weak var logoImage: UIImageView!

  var shareButton = UIButton()
  var recordButton = UIButton()
  let selection = UISelectionFeedbackGenerator()
  var firstLoad = true

  private let downloadProgressView: UIProgressView = {
    let pv = UIProgressView(progressViewStyle: .default)
    pv.progress = 0.0
    pv.isHidden = true
    return pv
  }()

  private let downloadProgressLabel: UILabel = {
    let label = UILabel()
    label.text = ""
    label.textAlignment = .center
    label.textColor = .systemGray
    label.font = UIFont.systemFont(ofSize: 14)
    label.isHidden = true
    return label
  }()

  private var loadingOverlayView: UIView?

  func showLoadingOverlay() {
    guard loadingOverlayView == nil else { return }
    let overlay = UIView(frame: view.bounds)
    overlay.backgroundColor = UIColor.black.withAlphaComponent(0.5)

    view.addSubview(overlay)
    loadingOverlayView = overlay
    view.bringSubviewToFront(downloadProgressView)
    view.bringSubviewToFront(downloadProgressLabel)

    view.isUserInteractionEnabled = false
  }

  func hideLoadingOverlay() {
    loadingOverlayView?.removeFromSuperview()
    loadingOverlayView = nil
    view.isUserInteractionEnabled = true
  }

  private let tasks: [(name: String, folder: String)] = [
    // ("Classify", "ClassifyModels"),  // index 0
    // ("Segment", "SegmentModels"),  // index 1
    // ("Detect", "DetectModels"),  // index 2
    // ("Pose", "PoseModels"),  // index 3
    // ("Obb", "ObbModels"),  // index 4
    ("Fish Count", "FishCountModels") // index 0
  ]

  private var modelsForTask: [String: [String]] = [:]
  
  // Add empty remote models info since we're only using local models
  private let remoteModelsInfo: [String: [(modelName: String, downloadURL: URL)]] = [:]

  private var currentModels: [ModelEntry] = []

  private var currentTask: String = ""
  private var currentModelName: String = ""

  private var isLoadingModel = false

  private let modelTableView: UITableView = {
    let table = UITableView()
    table.isHidden = true
    table.layer.cornerRadius = 8
    table.clipsToBounds = true
    return table
  }()

  private let tableViewBGView = UIView()

  private var selectedIndexPath: IndexPath?

  private struct ModelEntry {
    let name: String
    let isLocal: Bool
    
    init(name: String, isLocal: Bool) {
      self.name = name
      self.isLocal = isLocal
    }
  }

  override func viewDidLoad() {
    super.viewDidLoad()

    // Hide the segmented control since we only use Fish Count
    segmentedControl.isHidden = true
    
    // Check for model directory
    checkForModelDirectory()
    
    // Create a container view for Fish Count Models header with transparent background and white text
    let modelSelectorContainer = UIView(frame: CGRect(x: 20, y: 40, width: self.view.frame.width - 40, height: 50))
    modelSelectorContainer.backgroundColor = UIColor.black.withAlphaComponent(0.5) // More transparent, dark background
    modelSelectorContainer.layer.cornerRadius = 10
    modelSelectorContainer.layer.borderWidth = 1
    modelSelectorContainer.layer.borderColor = UIColor.white.withAlphaComponent(0.3).cgColor
    self.view.addSubview(modelSelectorContainer)
    
    // Add Fish Count Models header with dropdown arrow - white text
    let fishCountLabel = UILabel()
    fishCountLabel.text = "Fish Count Models ▼"
    fishCountLabel.textAlignment = .center
    fishCountLabel.font = UIFont.systemFont(ofSize: 18, weight: .semibold)
    fishCountLabel.textColor = UIColor.white // White text for better visibility
    fishCountLabel.frame = CGRect(x: 0, y: 0, width: modelSelectorContainer.frame.width, height: 50)
    fishCountLabel.isUserInteractionEnabled = true // Enable for tap gesture
    let tapGesture = UITapGestureRecognizer(target: self, action: #selector(toggleModelSelection))
    fishCountLabel.addGestureRecognizer(tapGesture)
    modelSelectorContainer.addSubview(fishCountLabel)

    loadModelsForAllTasks()

    // Select Fish Count task by default (index 0)
    currentTask = tasks[0].name
    reloadModelEntriesAndLoadFirst(for: currentTask)

    setupTableView()
    setupButtons()
    
    // Hide the logo and version label
    logoImage.isHidden = true
    labelVersion.isHidden = true
    
    // Hide model table view by default
    modelTableView.isHidden = true
    tableViewBGView.isHidden = true

    downloadProgressView.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(downloadProgressView)

    downloadProgressLabel.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(downloadProgressLabel)

    NSLayoutConstraint.activate([
      downloadProgressView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
      downloadProgressView.topAnchor.constraint(
        equalTo: activityIndicator.bottomAnchor, constant: 8),
      downloadProgressView.widthAnchor.constraint(equalToConstant: 200),
      downloadProgressView.heightAnchor.constraint(equalToConstant: 2),

      downloadProgressLabel.centerXAnchor.constraint(equalTo: downloadProgressView.centerXAnchor),
      downloadProgressLabel.topAnchor.constraint(
        equalTo: downloadProgressView.bottomAnchor, constant: 8),
    ])

    ModelDownloadManager.shared.progressHandler = { [weak self] progress in
      guard let self = self else { return }
      DispatchQueue.main.async {
        self.downloadProgressView.progress = Float(progress)
        self.downloadProgressLabel.isHidden = false
        let percentage = Int(progress * 100)
        self.downloadProgressLabel.text = "Downloading \(percentage)%"
      }
    }
  }

  private func checkForModelDirectory() {
    // Check if FishCountModels folder exists in bundle
    guard let resourcePath = Bundle.main.resourcePath else {
      print("Cannot access resource path")
      return
    }
    
    let fishCountModelsDirPath = resourcePath + "/FishCountModels"
    print("Checking if directory exists: \(fishCountModelsDirPath)")
    
    let fileManager = FileManager.default
    var isDir: ObjCBool = false
    
    if !fileManager.fileExists(atPath: fishCountModelsDirPath, isDirectory: &isDir) {
      // Directory doesn't exist, try to create it
      print("FishCountModels directory doesn't exist, attempting to create it")
      do {
        try fileManager.createDirectory(atPath: fishCountModelsDirPath, withIntermediateDirectories: true)
        print("Created FishCountModels directory")
      } catch {
        print("Failed to create FishCountModels directory: \(error)")
      }
    } else if !isDir.boolValue {
      print("FishCountModels exists but is not a directory")
    } else {
      print("FishCountModels directory exists")
    }
  }

  private func loadModelsForAllTasks() {
    for taskInfo in tasks {
      let taskName = taskInfo.name
      let folderName = taskInfo.folder
      let modelFiles = getModelFiles(in: folderName)
      modelsForTask[taskName] = modelFiles
    }
  }

  private func getModelFiles(in folderName: String) -> [String] {
    guard let folderURL = Bundle.main.url(forResource: folderName, withExtension: nil) else {
      print("Error: Cannot find folder: \(folderName)")
      return []
    }
    
    do {
      print("Looking for models in: \(folderURL.path)")
      
      let fileURLs = try FileManager.default.contentsOfDirectory(
        at: folderURL,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
      )
      
      // Print all files found for debugging
      print("All files in directory: \(fileURLs.map { $0.lastPathComponent })")
      
      // Get all model files in the directory
      let modelFiles =
        fileURLs
        .filter { $0.pathExtension == "mlmodel" || $0.pathExtension == "mlpackage" }
        .map { $0.lastPathComponent }
      
      print("Found model files: \(modelFiles)")
      
      // Filter models based on folder
      if folderName == "FishCountModels" {
        // For Fish Count task, show all FishCount_v* models
        let fishCountModels = modelFiles.filter { $0.contains("FishCount") }.sorted()
        print("Found Fish Count models: \(fishCountModels)")
        return fishCountModels
      } else {
        // For other tasks (not visible in our UI), return all models
        return modelFiles.sorted()
      }
    } catch {
      print("Error reading contents of folder \(folderName): \(error)")
      return []
    }
  }

  private func reorderDetectionModels(_ fileNames: [String]) -> [String] {
    let officialOrder: [Character: Int] = ["n": 0, "m": 1, "s": 2, "l": 3, "x": 4]

    var customModels: [String] = []
    var officialModels: [String] = []

    for fileName in fileNames {
      let baseName = (fileName as NSString).deletingPathExtension.lowercased()

      if baseName.hasPrefix("yolo"),
        let lastChar = baseName.last,
        officialOrder.keys.contains(lastChar)
      {
        officialModels.append(fileName)
      } else {
        customModels.append(fileName)
      }
    }

    customModels.sort { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }

    officialModels.sort { fileA, fileB in
      let baseA = (fileA as NSString).deletingPathExtension.lowercased()
      let baseB = (fileB as NSString).deletingPathExtension.lowercased()
      guard let lastA = baseA.last, let lastB = baseB.last,
        let indexA = officialOrder[lastA], let indexB = officialOrder[lastB]
      else {
        return baseA < baseB
      }
      return indexA < indexB
    }

    return customModels + officialModels
  }

  private func reloadModelEntriesAndLoadFirst(for taskName: String) {
    currentModels = makeModelEntries(for: taskName)

    if !currentModels.isEmpty {
      modelTableView.isHidden = false
      modelTableView.reloadData()

      DispatchQueue.main.async { [self] in
        let firstIndex = IndexPath(row: 0, section: 0)
        self.modelTableView.selectRow(at: firstIndex, animated: false, scrollPosition: .none)
        self.selectedIndexPath = firstIndex
        let firstModel = self.currentModels[0]
        self.loadModel(entry: firstModel, forTask: self.currentTask)
      }
    } else {
      print("No models found for task: \(taskName)")
      
      // Display a message if no models found
      let message = "No Fish Count models found in the bundle. Please include models in the FishCountModels folder."
      self.showAlert(title: "No Models Found", message: message)
      
      // Create a placeholder model
      let placeholderModel = ModelEntry(name: "No Models Available", isLocal: true)
      currentModels = [placeholderModel]
      
      modelTableView.reloadData()
      modelTableView.isHidden = true
    }
  }

  private func makeModelEntries(for taskName: String) -> [ModelEntry] {
    var models = [ModelEntry]()

    if let modelFiles = modelsForTask[taskName] {
      for modelFile in modelFiles {
        let modelName = (modelFile as NSString).deletingPathExtension
        let isLocal = true
        let entry = ModelEntry(name: modelName, isLocal: isLocal)
        models.append(entry)
      }
    }

    // Sort models to have the latest version first
    models.sort { (model1, model2) -> Bool in
      // For FishCount models, sort by version number (descending)
      if model1.name.contains("FishCount") && model2.name.contains("FishCount") {
        // Extract version numbers (e.g., "v1" from "FishCount_v1")
        let version1 = model1.name.components(separatedBy: "_v").last ?? "0"
        let version2 = model2.name.components(separatedBy: "_v").last ?? "0"
        
        // Convert to numeric values for comparison
        let num1 = Int(version1) ?? 0
        let num2 = Int(version2) ?? 0
        
        // Sort in descending order (latest version first)
        return num1 > num2
      }
      
      // For other models, sort alphabetically
      return model1.name < model2.name
    }

    return models
  }

  private func loadModel(entry: ModelEntry, forTask taskName: YOLOTask) {
    isLoadingModel = true
    
    // Show loading indicator
    activityIndicator.startAnimating()
    
    // Reset UI
    downloadProgressView.isHidden = false
    downloadProgressLabel.isHidden = false
    downloadProgressLabel.text = "Loading \(entry.name) model..."
    
    // Find the full path to the model file
    guard let folderURL = Bundle.main.url(forResource: "FishCountModels", withExtension: nil) else {
      print("Error: Cannot find FishCountModels folder")
      activityIndicator.stopAnimating()
      isLoadingModel = false
      self.showAlert(title: "Error Loading Model", message: "Cannot find FishCountModels folder")
      return
    }
    
    // Look for both .mlmodel and .mlpackage extensions
    let modelName = entry.name
    var modelPath: String? = nil
    
    // First try .mlpackage
    let mlpackagePath = folderURL.appendingPathComponent(modelName + ".mlpackage").path
    if FileManager.default.fileExists(atPath: mlpackagePath) {
      modelPath = mlpackagePath
    } else {
      // Then try .mlmodel
      let mlmodelPath = folderURL.appendingPathComponent(modelName + ".mlmodel").path
      if FileManager.default.fileExists(atPath: mlmodelPath) {
        modelPath = mlmodelPath
      }
    }
    
    guard let modelPath = modelPath else {
      print("Error: Cannot find model file for \(modelName)")
      activityIndicator.stopAnimating()
      isLoadingModel = false
      downloadProgressLabel.isHidden = true
      downloadProgressView.isHidden = true
      self.showAlert(title: "Error Loading Model", message: "Cannot find model file for \(modelName)")
      return
    }
    
    // Load the selected model with the full path
    yoloView.setModel(modelPathOrName: modelPath, task: taskName) { [weak self] result in
      guard let self = self else { return }
      
      DispatchQueue.main.async {
        self.activityIndicator.stopAnimating()
        self.downloadProgressView.isHidden = true
        self.isLoadingModel = false
        
        switch result {
        case .success(_):
          print("Model loaded successfully: \(entry.name)")
          self.downloadProgressLabel.text = "Model loaded successfully!"
          
          // Initialize fish counting with top-down view configuration
          // This replaces the previous camera angle selection with fixed top-down mode
          if taskName == .fishCount {
            // Configure default thresholds for top-down view: [0.3, 0.5]
            self.yoloView.updateFishCountingThresholds(thresholds: [0.3, 0.5])
            print("DEBUG: Initialized fish counting with top-down thresholds [0.3, 0.5]")
          }
          
          // Keep success message visible for 2 seconds, then hide
          DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
            self?.downloadProgressLabel.isHidden = true
          }
        case .failure(let error):
          print("Failed to load model: \(error)")
          self.downloadProgressLabel.isHidden = true
          self.showAlert(title: "Error Loading Model", message: error.localizedDescription)
        }
      }
    }
  }
  
  private func loadModel(entry: ModelEntry, forTask taskName: String) {
    guard let task = taskToEnum(taskName) else {
      print("Unknown task: \(taskName)")
      return
    }
    
    loadModel(entry: entry, forTask: task)
  }
  
  private func taskToEnum(_ taskName: String) -> YOLOTask? {
    switch taskName {
    case "Detect":
      return .detect
    case "Fish Count":
      return .fishCount
    case "Segment":
      return .segment
    case "Classify":
      return .classify
    case "Pose":
      return .pose
    case "Obb":
      return .obb
    default:
      return nil
    }
  }

  @IBAction func vibrate(_ sender: Any) {
    selection.selectionChanged()
  }

  @IBAction func indexChanged(_ sender: UISegmentedControl) {
    selection.selectionChanged()

    let index = sender.selectedSegmentIndex
    guard tasks.indices.contains(index) else { return }

    let newTask = tasks[index].name

    if (modelsForTask[newTask]?.isEmpty ?? true) && (remoteModelsInfo[newTask]?.isEmpty ?? true) {
      let alert = UIAlertController(
        title: "\(newTask) Models not found",
        message: "Please add or define models for \(newTask).",
        preferredStyle: .alert
      )
      alert.addAction(
        UIAlertAction(
          title: "OK", style: .cancel,
          handler: { _ in
            alert.dismiss(animated: true)
          }))
      self.present(alert, animated: true)

      if let oldIndex = tasks.firstIndex(where: { $0.name == currentTask }) {
        sender.selectedSegmentIndex = oldIndex
      }
      return
    }

    currentTask = newTask
    selectedIndexPath = nil

    reloadModelEntriesAndLoadFirst(for: currentTask)

    tableViewBGView.frame = CGRect(
      x: modelTableView.frame.minX - 1,
      y: modelTableView.frame.minY - 1,
      width: modelTableView.frame.width + 2,
      height: CGFloat(currentModels.count * 30 + 2)
    )
  }

  @objc func logoButton() {
    selection.selectionChanged()
    if let link = URL(string: "https://www.ultralytics.com") {
      UIApplication.shared.open(link)
    }
  }

  private func setupTableView() {
    // Set up a background view with lighter semi-transparent background
    tableViewBGView.backgroundColor = UIColor.lightGray.withAlphaComponent(0.7) // Lighter background
    tableViewBGView.layer.cornerRadius = 10
    tableViewBGView.clipsToBounds = true
    tableViewBGView.layer.borderColor = UIColor.white.withAlphaComponent(0.5).cgColor
    tableViewBGView.layer.borderWidth = 1
    tableViewBGView.layer.shadowColor = UIColor.black.cgColor
    tableViewBGView.layer.shadowOffset = CGSize(width: 0, height: 3)
    tableViewBGView.layer.shadowOpacity = 0.3
    tableViewBGView.layer.shadowRadius = 4
    view.addSubview(tableViewBGView)

    // Configure table view for better cell display
    modelTableView.delegate = self
    modelTableView.dataSource = self
    modelTableView.register(UITableViewCell.self, forCellReuseIdentifier: "ModelCell")
    modelTableView.backgroundColor = .clear
    modelTableView.separatorStyle = .singleLine
    modelTableView.separatorColor = UIColor.white.withAlphaComponent(0.5)
    modelTableView.isScrollEnabled = true
    modelTableView.showsVerticalScrollIndicator = true
    modelTableView.rowHeight = 44 // Taller rows for better readability
    view.addSubview(modelTableView)
    
    // Initially hide the model selection UI
    tableViewBGView.isHidden = true
    modelTableView.isHidden = true

    // Use Auto Layout for precise positioning
    tableViewBGView.translatesAutoresizingMaskIntoConstraints = false
    modelTableView.translatesAutoresizingMaskIntoConstraints = false

    NSLayoutConstraint.activate([
      // Position tableViewBGView below the fish count display (which is around y=120-150)
      tableViewBGView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 155),
      tableViewBGView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
      tableViewBGView.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.85),
      tableViewBGView.heightAnchor.constraint(equalToConstant: 180),

      // Align modelTableView with its background view
      modelTableView.topAnchor.constraint(equalTo: tableViewBGView.topAnchor, constant: 0),
      modelTableView.leadingAnchor.constraint(equalTo: tableViewBGView.leadingAnchor, constant: 0),
      modelTableView.trailingAnchor.constraint(equalTo: tableViewBGView.trailingAnchor, constant: 0),
      modelTableView.bottomAnchor.constraint(equalTo: tableViewBGView.bottomAnchor, constant: 0),
    ])
  }

  private func setupButtons() {
    let config = UIImage.SymbolConfiguration(pointSize: 20, weight: .regular, scale: .default)
    shareButton.setImage(
      UIImage(systemName: "square.and.arrow.up", withConfiguration: config), for: .normal)
    shareButton.addGestureRecognizer(
      UITapGestureRecognizer(target: self, action: #selector(shareButtonTapped)))
    view.addSubview(shareButton)

    recordButton.setImage(UIImage(systemName: "video", withConfiguration: config), for: .normal)
    recordButton.addGestureRecognizer(
      UITapGestureRecognizer(target: self, action: #selector(recordScreen)))
    view.addSubview(recordButton)

    logoImage.isUserInteractionEnabled = true
    logoImage.addGestureRecognizer(
      UITapGestureRecognizer(target: self, action: #selector(logoButton)))
  }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()

    // Configure share and record buttons
    if view.bounds.width > view.bounds.height {
      // Landscape orientation
      shareButton.tintColor = .darkGray
      recordButton.tintColor = .darkGray
    } else {
      // Portrait orientation
      shareButton.tintColor = .systemGray
      recordButton.tintColor = .systemGray
    }

    // Position share and record buttons at the bottom of the screen
    shareButton.frame = CGRect(
      x: view.bounds.maxX - 49.5,
      y: view.bounds.maxY - 66,
      width: 49.5,
      height: 49.5
    )
    recordButton.frame = CGRect(
      x: shareButton.frame.minX - 49.5,
      y: view.bounds.maxY - 66,
      width: 49.5,
      height: 49.5
    )
  }

  @objc func shareButtonTapped() {
    selection.selectionChanged()
    yoloView.capturePhoto { [weak self] captured in
      guard let self = self else { return }
      if let image = captured {
        DispatchQueue.main.async {
          let activityViewController = UIActivityViewController(
            activityItems: [image], applicationActivities: nil
          )
          activityViewController.popoverPresentationController?.sourceView = self.View0
          self.present(activityViewController, animated: true, completion: nil)
        }
      } else {
        print("error capturing photo")
      }
    }
  }

  @objc func recordScreen() {
    let recorder = RPScreenRecorder.shared()
    recorder.isMicrophoneEnabled = true

    if !recorder.isRecording {
      AudioServicesPlaySystemSound(1117)
      recordButton.tintColor = .red
      recorder.startRecording { [weak self] error in
        if let error = error {
          print("Screen recording start error: \(error)")
        } else {
          print("Started screen recording.")
        }
      }
    } else {
      AudioServicesPlaySystemSound(1118)
      if view.bounds.width > view.bounds.height {
        recordButton.tintColor = .darkGray
      } else {
        recordButton.tintColor = .systemGray
      }
      recorder.stopRecording { [weak self] previewVC, error in
        guard let self = self else { return }
        if let error = error {
          print("Stop recording error: \(error)")
        }
        if let previewVC = previewVC {
          previewVC.previewControllerDelegate = self
          self.present(previewVC, animated: true, completion: nil)
        }
      }
    }
  }

  @objc func toggleModelSelection() {
    selection.selectionChanged()
    
    // Toggle visibility with animation
    UIView.animate(withDuration: 0.3) {
      let isHidden = self.modelTableView.isHidden
      self.modelTableView.isHidden = !isHidden
      self.tableViewBGView.isHidden = !isHidden
    }
  }
  
  // Auto-hide after selection
  func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
    if tableView == modelTableView {
      selectedIndexPath = indexPath
      let model = currentModels[indexPath.row]
      loadModel(entry: model, forTask: self.currentTask)
      
      // Hide the model table after selection
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [self] in
        self.modelTableView.isHidden = true
        self.tableViewBGView.isHidden = true
      }
    }
  }

  private func showAlert(title: String, message: String) {
    let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
    alert.addAction(UIAlertAction(title: "OK", style: .default))
    self.present(alert, animated: true)
  }
}

// MARK: - UITableViewDataSource, UITableViewDelegate
extension ViewController: UITableViewDataSource, UITableViewDelegate {

  func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
    return currentModels.count
  }

  func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
    return 44 // Taller cells for better text display
  }

  func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    let cell = tableView.dequeueReusableCell(
      withIdentifier: "ModelCell", for: indexPath)
    let entry = currentModels[indexPath.row]

    var config = cell.defaultContentConfiguration()
    config.text = entry.name
    config.textProperties.font = UIFont.systemFont(ofSize: 16, weight: .medium) // Larger font
    config.textProperties.color = .white // White text for better visibility
    
    if entry.isLocal {
      config.secondaryText = "Local model"
      config.secondaryTextProperties.font = UIFont.systemFont(ofSize: 12)
      config.secondaryTextProperties.color = .white
    } else {
      config.secondaryText = "Remote model"
      config.secondaryTextProperties.font = UIFont.systemFont(ofSize: 12)
      config.secondaryTextProperties.color = .white
    }
    
    cell.contentConfiguration = config
    cell.backgroundColor = UIColor.clear // Make cells transparent
    
    // Create a better selected background
    let selectedBGView = UIView()
    selectedBGView.backgroundColor = UIColor.white.withAlphaComponent(0.3)
    cell.selectedBackgroundView = selectedBGView
    
    if selectedIndexPath == indexPath {
      cell.setSelected(true, animated: false)
    }
    
    return cell
  }

  func tableView(
    _ tableView: UITableView,
    willDisplay cell: UITableViewCell,
    forRowAt indexPath: IndexPath
  ) {
    if let selectedBGView = cell.selectedBackgroundView {
      let insetRect = cell.bounds.insetBy(dx: 4, dy: 4)
      selectedBGView.frame = insetRect
    }
  }
}

extension ViewController: RPPreviewViewControllerDelegate {
  func previewControllerDidFinish(_ previewController: RPPreviewViewController) {
    previewController.dismiss(animated: true)
  }
}
