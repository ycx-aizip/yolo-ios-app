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
// import ReplayKit
import UIKit
import YOLO

/// The main view controller for the YOLO iOS application, handling model selection and visualization.
class ViewController: UIViewController, YOLOViewActionDelegate {

  @IBOutlet weak var yoloView: YOLOView!
  @IBOutlet var View0: UIView!
  @IBOutlet var segmentedControl: UISegmentedControl! // Will be hidden, but keeping the outlet for compatibility
  @IBOutlet weak var labelName: UILabel!
  @IBOutlet weak var labelFPS: UILabel!
  @IBOutlet weak var labelVersion: UILabel!
  @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
  @IBOutlet weak var forcus: UIImageView!
  @IBOutlet weak var logoImage: UIImageView!

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
    ("Classify", "ClassifyModels"),  // index 0
    ("Segment", "SegmentModels"),  // index 1
    ("Detect", "DetectModels"),  // index 2
    ("Pose", "PoseModels"),  // index 3
    ("Obb", "ObbModels"),  // index 4
    ("FishCount", "FishCountModels"),  // index 5
  ]

  private var modelsForTask: [String: [String]] = [:]

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

  override func viewDidLoad() {
    super.viewDidLoad()

    // Hide the segmented control as we'll use a toolbar button instead
    segmentedControl.isHidden = true
    
    loadModelsForAllTasks()

    // Always select Fish Count task (index 0 now)
    currentTask = "FishCount"
    reloadModelEntriesAndLoadFirst(for: currentTask)
    
    // Hide the model name label
    labelName.isHidden = true
    
    // Hide the zoom label
    yoloView.labelZoom.isHidden = true
    
    // Hide the model table view by default
    modelTableView.isHidden = true
    tableViewBGView.isHidden = true

    setupTableView()
    
    // Set up the YOLOView action delegate
    yoloView.actionDelegate = self
    
    // Setup logo tap gesture
    logoImage.isUserInteractionEnabled = true
    logoImage.addGestureRecognizer(
      UITapGestureRecognizer(target: self, action: #selector(logoButton)))

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
      return []
    }
    do {
      let fileURLs = try FileManager.default.contentsOfDirectory(
        at: folderURL,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
      )
      let modelFiles =
        fileURLs
        .filter { $0.pathExtension == "mlmodel" || $0.pathExtension == "mlpackage" }
        .map { $0.lastPathComponent }

      if folderName == "DetectModels" {
        return reorderDetectionModels(modelFiles)
      } else if folderName == "FishCountModels" {
        // Sort FishCount models in reverse alphabetical order (latest on top)
        return modelFiles.sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedDescending }
      } else {
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
      // Don't show the table view automatically, just reload it
      modelTableView.reloadData()
      
      // Set the table height based on number of models
      let modelCount = currentModels.count
      let rowHeight: CGFloat = 30
      let tableHeight = CGFloat(modelCount * Int(rowHeight))
      
      // If in landscape mode, make sure height doesn't exceed maximum
      if view.bounds.width > view.bounds.height {
        let maxHeight: CGFloat = 300
        let adjustedHeight = min(tableHeight, maxHeight)
        
        let tableWidth = view.bounds.width * 0.3
        modelTableView.frame = CGRect(
          x: (view.bounds.width - tableWidth) / 2,
          y: segmentedControl.frame.maxY + 20,
          width: tableWidth,
          height: adjustedHeight
        )
      } else {
        // Portrait mode
        let tableWidth = view.bounds.width * 0.6
        modelTableView.frame = CGRect(
          x: (view.bounds.width - tableWidth) / 2,
          y: segmentedControl.frame.maxY + 5,
          width: tableWidth,
          height: tableHeight
        )
      }
      
      // Update the background view size to match
      tableViewBGView.frame = CGRect(
        x: modelTableView.frame.minX - 1,
        y: modelTableView.frame.minY - 1,
        width: modelTableView.frame.width + 2,
        height: modelTableView.frame.height + 2
      )

      DispatchQueue.main.async {
        let firstIndex = IndexPath(row: 0, section: 0)
        self.modelTableView.selectRow(at: firstIndex, animated: false, scrollPosition: .none)
        self.selectedIndexPath = firstIndex
        let firstModel = self.currentModels[0]
        self.loadModel(entry: firstModel, forTask: taskName)
      }
    } else {
      print("No models found for task: \(taskName)")
      modelTableView.isHidden = true
      tableViewBGView.isHidden = true
    }
  }

  private func makeModelEntries(for taskName: String) -> [ModelEntry] {
    let localFileNames = modelsForTask[taskName] ?? []
    let localEntries = localFileNames.map { fileName -> ModelEntry in
      let display = (fileName as NSString).deletingPathExtension
      return ModelEntry(
        displayName: display,
        identifier: fileName,
        isLocalBundle: true,
        isRemote: false,
        remoteURL: nil
      )
    }

    let remoteList = remoteModelsInfo[taskName] ?? []
    let remoteEntries = remoteList.map { (modelName, url) -> ModelEntry in
      ModelEntry(
        displayName: modelName,
        identifier: modelName,
        isLocalBundle: false,
        isRemote: true,
        remoteURL: url
      )
    }

    return localEntries + remoteEntries
  }

  private func loadModel(entry: ModelEntry, forTask task: String) {
    guard !isLoadingModel else {
      print("Model is already loading. Please wait.")
      return
    }
    isLoadingModel = true
    yoloView.resetLayers()
    if !firstLoad {
      showLoadingOverlay()
      yoloView.setInferenceFlag(ok: false)
    } else {
      firstLoad = false
    }

    self.activityIndicator.startAnimating()
    self.downloadProgressView.progress = 0.0
    self.downloadProgressView.isHidden = true
    self.downloadProgressLabel.isHidden = true
    self.view.isUserInteractionEnabled = false
    self.modelTableView.isUserInteractionEnabled = false

    print("Start loading model: \(entry.displayName)")

    if entry.isLocalBundle {
      DispatchQueue.global().async { [weak self] in
        guard let self = self else { return }
        let yoloTask = self.convertTaskNameToYOLOTask(task)

        guard let folderURL = self.tasks.first(where: { $0.name == task })?.folder,
          let folderPathURL = Bundle.main.url(forResource: folderURL, withExtension: nil)
        else {
          DispatchQueue.main.async {
            self.finishLoadingModel(success: false, modelName: entry.displayName)
          }
          return
        }

        let modelURL = folderPathURL.appendingPathComponent(entry.identifier)
        DispatchQueue.main.async {
          self.downloadProgressLabel.isHidden = false
          self.downloadProgressLabel.text = "Loading \(entry.displayName)"
          self.yoloView.setModel(modelPathOrName: modelURL.path, task: yoloTask) { result in
            switch result {
            case .success():
              self.finishLoadingModel(success: true, modelName: entry.displayName)
            case .failure(let error):
              print(error)
              self.finishLoadingModel(success: false, modelName: entry.displayName)
            }
          }
        }
      }
    } else {
      let yoloTask = self.convertTaskNameToYOLOTask(task)

      let key = entry.identifier  // "yolov8n", "yolov8m-seg", etc.

      if ModelCacheManager.shared.isModelDownloaded(key: key) {
        loadCachedModelAndSetToYOLOView(
          key: key, yoloTask: yoloTask, displayName: entry.displayName)
      } else {
        guard let remoteURL = entry.remoteURL else {
          self.finishLoadingModel(success: false, modelName: entry.displayName)
          return
        }

        self.downloadProgressView.progress = 0.0
        self.downloadProgressView.isHidden = false
        self.downloadProgressLabel.isHidden = false

        let localZipFileName = remoteURL.lastPathComponent  // ex. "yolov8n.mlpackage.zip"

        ModelCacheManager.shared.loadModel(
          from: localZipFileName,
          remoteURL: remoteURL,
          key: key
        ) { [weak self] mlModel, loadedKey in
          guard let self = self else { return }
          if mlModel == nil {
            self.finishLoadingModel(success: false, modelName: entry.displayName)
            return
          }
          self.loadCachedModelAndSetToYOLOView(
            key: loadedKey,
            yoloTask: yoloTask,
            displayName: entry.displayName)
        }
      }
    }
  }

  private func loadCachedModelAndSetToYOLOView(key: String, yoloTask: YOLOTask, displayName: String)
  {
    let localModelURL = ModelCacheManager.shared.getDocumentsDirectory()
      .appendingPathComponent(key)
      .appendingPathExtension("mlmodelc")

    DispatchQueue.main.async {
      self.downloadProgressLabel.isHidden = false
      self.downloadProgressLabel.text = "Loading \(displayName)"
      self.yoloView.setModel(modelPathOrName: localModelURL.path, task: yoloTask) { result in
        switch result {
        case .success():
          self.finishLoadingModel(success: true, modelName: displayName)
        case .failure(let error):
          print(error)
          self.finishLoadingModel(success: false, modelName: displayName)
        }
      }
    }
  }

  private func finishLoadingModel(success: Bool, modelName: String) {
    DispatchQueue.main.async {
      self.activityIndicator.stopAnimating()
      self.downloadProgressView.isHidden = true

      self.downloadProgressLabel.isHidden = true
      //            self.downloadProgressLabel.isHidden = false
      //            self.downloadProgressLabel.text = "Loading \(modelName)"

      self.view.isUserInteractionEnabled = true
      self.modelTableView.isUserInteractionEnabled = true
      self.isLoadingModel = false

      self.modelTableView.reloadData()

      if let ip = self.selectedIndexPath {
        self.modelTableView.selectRow(at: ip, animated: false, scrollPosition: .none)
      }
      if !self.firstLoad {
        self.hideLoadingOverlay()
      }
      self.yoloView.setInferenceFlag(ok: true)

      if success {
        print("Finished loading model: \(modelName)")
        self.currentModelName = modelName
        self.downloadProgressLabel.text = "Finished loading model \(modelName)"
        self.downloadProgressLabel.isHidden = false
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
          self.downloadProgressLabel.isHidden = true
          self.downloadProgressLabel.text = ""
        }

      } else {
        print("Failed to load model: \(modelName)")
      }
    }
  }

  private func convertTaskNameToYOLOTask(_ task: String) -> YOLOTask {
    switch task {
    case "Detect": return .detect
    case "Segment": return .segment
    case "Classify": return .classify
    case "Pose": return .pose
    case "Obb": return .obb
    case "FishCount": return .fishCount
    default: return .detect
    }
  }

  @IBAction func vibrate(_ sender: Any) {
    selection.selectionChanged()
  }

  @objc func logoButton() {
    selection.selectionChanged()
    if let link = URL(string: "https://www.ultralytics.com") {
      UIApplication.shared.open(link)
    }
  }

  private func setupTableView() {
    modelTableView.register(UITableViewCell.self, forCellReuseIdentifier: "ModelCell")
    modelTableView.delegate = self
    modelTableView.dataSource = self
    modelTableView.bounces = false
    modelTableView.backgroundColor = UIColor.darkGray.withAlphaComponent(0.5)
    modelTableView.separatorColor = UIColor.white.withAlphaComponent(0.3)
    modelTableView.showsVerticalScrollIndicator = true
    modelTableView.indicatorStyle = .white
    modelTableView.layer.cornerRadius = 8
    modelTableView.clipsToBounds = true
    
    // Apply a border to make the dropdown more visible
    modelTableView.layer.borderWidth = 1.0
    modelTableView.layer.borderColor = UIColor.white.withAlphaComponent(0.5).cgColor
    
    tableViewBGView.backgroundColor = .darkGray.withAlphaComponent(0.3)
    tableViewBGView.layer.cornerRadius = 8
    tableViewBGView.clipsToBounds = true

    view.addSubview(tableViewBGView)
    view.addSubview(modelTableView)

    modelTableView.translatesAutoresizingMaskIntoConstraints = false
    tableViewBGView.frame = CGRect(
      x: modelTableView.frame.minX - 1,
      y: modelTableView.frame.minY - 1,
      width: modelTableView.frame.width + 2,
      height: modelTableView.frame.height + 2
    )
  }

  override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()

    // Adjust segmented control appearance to match the image
    let screenWidth = view.bounds.width
    let statusBarHeight: CGFloat = 45 // Approximate height for status bar
    segmentedControl.frame = CGRect(
      x: screenWidth * 0.15,
      y: statusBarHeight,
      width: screenWidth * 0.7,
      height: 40
    )
    segmentedControl.layer.cornerRadius = 12
    segmentedControl.layer.masksToBounds = true
    
    if view.bounds.width > view.bounds.height {
      // Landscape mode
      // Calculate appropriate table dimensions
      let tableViewWidth = view.bounds.width * 0.3 // Wider table for visibility
      let tableViewHeight = min(CGFloat(currentModels.count * 30), 300) // Height based on models count with max limit
      
      // Position below the segmented control
      modelTableView.frame = CGRect(
        x: (view.bounds.width - tableViewWidth) / 2,
        y: segmentedControl.frame.maxY + 20, // Position below segmented control with spacing
        width: tableViewWidth,
        height: tableViewHeight
      )
    } else {
      // Portrait mode (unchanged)
      let tableViewWidth = view.bounds.width * 0.6
      modelTableView.frame = CGRect(
        x: (view.bounds.width - tableViewWidth) / 2,
        y: segmentedControl.frame.maxY + 5,
        width: tableViewWidth,
        height: CGFloat(currentModels.count * 30)
      )
    }

    // Update background view to match table size
    tableViewBGView.frame = CGRect(
      x: modelTableView.frame.minX - 1,
      y: modelTableView.frame.minY - 1,
      width: modelTableView.frame.width + 2,
      height: modelTableView.frame.height + 2
    )
  }

  // YOLOViewActionDelegate implementation
  func didTapModelsButton() {
    // Show/hide the model selection table
    let isCurrentlyVisible = !modelTableView.isHidden
    
    // Before showing the dropdown, update its position based on current orientation
    let screenWidth = view.bounds.width
    let screenHeight = view.bounds.height
    
    if screenWidth > screenHeight {
      // Landscape mode
      let tableViewWidth = screenWidth * 0.3
      let tableViewHeight = min(CGFloat(currentModels.count * 30), 300)
      
      // Position the table centered at the top of the screen
      modelTableView.frame = CGRect(
        x: (screenWidth - tableViewWidth) / 2,
        y: 65, // Position below the status bar with some spacing
        width: tableViewWidth,
        height: tableViewHeight
      )
    } else {
      // Portrait mode
      let tableViewWidth = screenWidth * 0.6
      let tableViewHeight = CGFloat(currentModels.count * 30)
      
      // Position centered at the top of the screen
      modelTableView.frame = CGRect(
        x: (screenWidth - tableViewWidth) / 2,
        y: 65, // Position below the status bar with some spacing
        width: tableViewWidth,
        height: tableViewHeight
      )
    }
    
    // Update background view to match table size
    tableViewBGView.frame = CGRect(
      x: modelTableView.frame.minX - 1,
      y: modelTableView.frame.minY - 1,
      width: modelTableView.frame.width + 2,
      height: modelTableView.frame.height + 2
    )
    
    // Toggle the visibility of the model selection dropdown
    modelTableView.isHidden = isCurrentlyVisible
    tableViewBGView.isHidden = isCurrentlyVisible
  }

  // Replace the segmentedControlTapped method with a stub since it's no longer needed
  @objc func segmentedControlTapped() {
    // This method is kept for compatibility but no longer used
    // The functionality is now handled by didTapModelsButton
  }
}

// MARK: - UITableViewDataSource, UITableViewDelegate
extension ViewController: UITableViewDataSource, UITableViewDelegate {

  func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
    return currentModels.count
  }

  func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
    return 30
  }

  func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {

    let cell = tableView.dequeueReusableCell(withIdentifier: "ModelCell", for: indexPath)
    let entry = currentModels[indexPath.row]

    cell.textLabel?.textAlignment = .center
    cell.textLabel?.text = entry.displayName
    cell.textLabel?.font = UIFont.systemFont(ofSize: 14, weight: .medium)
    cell.backgroundColor = .clear

    if entry.isRemote {
      let isDownloaded = ModelCacheManager.shared.isModelDownloaded(key: entry.identifier)
      if !isDownloaded {
        cell.accessoryView = UIImageView(image: UIImage(systemName: "icloud.and.arrow.down"))
      } else {
        cell.accessoryView = nil
      }
    } else {
      cell.accessoryView = nil
    }

    let selectedBGView = UIView()
    selectedBGView.backgroundColor = UIColor(white: 1.0, alpha: 0.3)
    selectedBGView.layer.cornerRadius = 8
    selectedBGView.layer.masksToBounds = true
    cell.selectedBackgroundView = selectedBGView

    cell.selectionStyle = .default
    return cell
  }

  func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
    selection.selectionChanged()

    selectedIndexPath = indexPath
    let selectedEntry = currentModels[indexPath.row]

    loadModel(entry: selectedEntry, forTask: currentTask)
    
    // Hide the dropdown after selection
    modelTableView.isHidden = true
    tableViewBGView.isHidden = true
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
