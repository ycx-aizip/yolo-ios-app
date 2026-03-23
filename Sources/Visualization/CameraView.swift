//  The AZFishCountCamera component provides a SwiftUI view for real-time object detection using device cameras.
//  It wraps the underlying FishCountView component to provide a clean SwiftUI interface for camera feed processing,
//  model inference, and result display. The component automatically handles camera setup, frame capture,
//  model loading, and inference processing, making it simple to add real-time object detection capabilities
//  to SwiftUI applications with minimal code. Results are exposed through a callback for custom handling
//  of detection results.

import AVFoundation
import SwiftUI
import AizipFishCount

/// A SwiftUI view that provides real-time camera-based object detection using AZFishCount models.
public struct AZFishCountCamera: View {
  @State private var AZFishCountResult: AZFishCountResult?

  public let modelPathOrName: String
  public let task: AZFishCountTask
  public let cameraPosition: AVCaptureDevice.Position

  public init(
    modelPathOrName: String,
    task: AZFishCountTask = .detect,
    cameraPosition: AVCaptureDevice.Position = .back
  ) {
    self.modelPathOrName = modelPathOrName
    self.task = task
    self.cameraPosition = cameraPosition
  }

  public var body: some View {
    FishCountViewRepresentable(
      modelPathOrName: modelPathOrName,
      task: task,
      cameraPosition: cameraPosition
    ) { result in
      self.AZFishCountResult = result
    }
  }
}

struct FishCountViewRepresentable: UIViewRepresentable {
  let modelPathOrName: String
  let task: AZFishCountTask
  let cameraPosition: AVCaptureDevice.Position
  let onDetection: ((AZFishCountResult) -> Void)?

  func makeUIView(context: Context) -> FishCountView {
    let AZFishCountView = FishCountView(
      frame: .zero,
      modelPathOrName: modelPathOrName,
      task: task
    )
    return AZFishCountView
  }

  func updateUIView(_ uiView: FishCountView, context: Context) {
    uiView.onDetection = onDetection
  }
}
