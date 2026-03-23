//  The AZFishCount class serves as the primary interface for loading and using AZFishCount machine learning models.
//  It supports a variety of input formats including UIImage, CIImage, CGImage, and resource files.
//  The class handles model loading, format conversion, and inference execution, offering a simple yet
//  powerful API through Swift's callable object pattern. Users can load models from local bundles or
//  file paths and perform inference with a single function call syntax, making integration into iOS
//  applications straightforward.

import Foundation
import SwiftUI
import UIKit

/// The primary interface for working with AZFishCount models, supporting multiple input types and inference methods.
public class AZFishCount {
  var predictor: Predictor!

  public init(
    _ modelPathOrName: String, task: AZFishCountTask, completion: ((Result<AZFishCount, Error>) -> Void)? = nil
  ) {
    var modelURL: URL?

    let lowercasedPath = modelPathOrName.lowercased()
    let fileManager = FileManager.default

    if lowercasedPath.hasSuffix(".mlmodel") || lowercasedPath.hasSuffix(".mlpackage") {
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
      completion?(.failure(PredictorError.modelFileNotFound))
      return
      //            fatalError(PredictorError.modelFileNotFound.localizedDescription)
    }

    func handleSuccess(predictor: Predictor) {
      self.predictor = predictor
      completion?(.success(self))
    }

    // Common failure handling for all tasks
    func handleFailure(_ error: Error) {
      print("Failed to load model with error: \(error)")
      completion?(.failure(error))
    }

    switch task {
    case .classify:
      Classifier.create(unwrappedModelURL: unwrappedModelURL) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .segment:
      Segmenter.create(unwrappedModelURL: unwrappedModelURL) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .pose:
      PoseEstimater.create(unwrappedModelURL: unwrappedModelURL) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .obb:
      ObbDetector.create(unwrappedModelURL: unwrappedModelURL) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    default:
      ObjectDetector.create(unwrappedModelURL: unwrappedModelURL) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }
    }
  }

  public func callAsFunction(_ uiImage: UIImage, returnAnnotatedImage: Bool = true) -> AZFishCountResult {
    let ciImage = CIImage(image: uiImage)!
    var result = predictor.predictOnImage(image: ciImage)
    //        if returnAnnotatedImage {
    //            let annotatedImage = drawAZFishCountDetections(on: ciImage, result: result)
    //            result.annotatedImage = annotatedImage
    //        }
    return result
  }

  public func callAsFunction(_ ciImage: CIImage, returnAnnotatedImage: Bool = true) -> AZFishCountResult {
    var result = predictor.predictOnImage(image: ciImage)
    if returnAnnotatedImage {
      let annotatedImage = drawAZFishCountDetections(on: ciImage, result: result)
      result.annotatedImage = annotatedImage
    }
    return result
  }

  public func callAsFunction(_ cgImage: CGImage, returnAnnotatedImage: Bool = true) -> AZFishCountResult {
    let ciImage = CIImage(cgImage: cgImage)
    var result = predictor.predictOnImage(image: ciImage)
    if returnAnnotatedImage {
      let annotatedImage = drawAZFishCountDetections(on: ciImage, result: result)
      result.annotatedImage = annotatedImage
    }
    return result
  }

  public func callAsFunction(
    _ resourceName: String,
    withExtension ext: String? = nil,
    returnAnnotatedImage: Bool = true
  ) -> AZFishCountResult {
    guard let url = Bundle.main.url(forResource: resourceName, withExtension: ext),
      let data = try? Data(contentsOf: url),
      let uiImage = UIImage(data: data)
    else {
      return AZFishCountResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }
    return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
  }

  public func callAsFunction(
    _ remoteURL: URL?,
    returnAnnotatedImage: Bool = true
  ) -> AZFishCountResult {
    guard let remoteURL = remoteURL,
      let data = try? Data(contentsOf: remoteURL),
      let uiImage = UIImage(data: data)
    else {
      return AZFishCountResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }
    return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
  }

  public func callAsFunction(
    _ localPath: String,
    returnAnnotatedImage: Bool = true
  ) -> AZFishCountResult {
    let fileURL = URL(fileURLWithPath: localPath)
    guard let data = try? Data(contentsOf: fileURL),
      let uiImage = UIImage(data: data)
    else {
      return AZFishCountResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }
    return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
  }

  @MainActor @available(iOS 16.0, *)
  public func callAsFunction(
    _ swiftUIImage: SwiftUI.Image,
    returnAnnotatedImage: Bool = true
  ) -> AZFishCountResult {
    let renderer = ImageRenderer(content: swiftUIImage)
    guard let uiImage = renderer.uiImage else {
      return AZFishCountResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }
    return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
  }
}
