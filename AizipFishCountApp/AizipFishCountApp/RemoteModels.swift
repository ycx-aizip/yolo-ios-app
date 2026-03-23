//  The RemoteModels file defines a registry of downloadable AZFishCount models available for each task type.
//  It provides a structured mapping between task categories (detection, segmentation, classification, etc.)
//  and the available models with their remote download URLs. These models are presented to users in the
//  application interface, allowing them to download and use additional models beyond those bundled with
//  the application. The dictionary structure enables easy filtering of models by task type and provides
//  all necessary information for the ModelDownloadManager to retrieve and install the models.

import Foundation

/// A dictionary mapping task names to available remote models with their download URLs.
public let remoteModelsInfo: [String: [(modelName: String, downloadURL: URL)]] = [
  "Detect": [
    //        ("AZFishCount11n",  URL(string: "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11n.mlpackage.zip")!),
    //        ("AZFishCount11s",  URL(string: "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11s.mlpackage.zip")!),
    //        ("AZFishCount11m",  URL(string: "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11m.mlpackage.zip")!),
    (
      "AZFishCount11l",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11l.mlpackage.zip"
      )!
    ),
    (
      "AZFishCount11x",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11x.mlpackage.zip"
      )!
    ),
  ],
  "Segment": [
    //        ("AZFishCount11n-seg",  URL(string: "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11n-seg.mlpackage.zip")!),
    (
      "AZFishCount11s-seg",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11s-seg.mlpackage.zip"
      )!
    ),
    (
      "AZFishCount11m-seg",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11m-seg.mlpackage.zip"
      )!
    ),
    (
      "AZFishCount11l-seg",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11l-seg.mlpackage.zip"
      )!
    ),
    (
      "AZFishCount11x-seg",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11x-seg.mlpackage.zip"
      )!
    ),
  ],
  "Classify": [
    //        ("AZFishCount11n-cls", URL(string: "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11n-clswasq5.mlpackage.zip")!),
    (
      "AZFishCount11s-cls",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11s-cls.mlpackage.zip"
      )!
    ),
    (
      "AZFishCount11m-cls",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11m-cls.mlpackage.zip"
      )!
    ),
    (
      "AZFishCount11l-cls",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11l-cls.mlpackage.zip"
      )!
    ),
    (
      "AZFishCount11x-cls",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11x-cls.mlpackage.zip"
      )!
    ),
  ],
  "Pose": [
    //        ("AZFishCount11n-pose", URL(string: "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11n-pose.mlpackage.zip")!),
    (
      "AZFishCount11s-pose",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11s-pose.mlpackage.zip"
      )!
    ),
    (
      "AZFishCount11m-pose",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11m-pose.mlpackage.zip"
      )!
    ),
    (
      "AZFishCount11l-pose",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11l-pose.mlpackage.zip"
      )!
    ),
    (
      "AZFishCount11x-pose",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11x-pose.mlpackage.zip"
      )!
    ),
  ],
  "Obb": [
    //        ("AZFishCount11n-obb",  URL(string: "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11n-obb.mlpackage.zip")!),
    (
      "AZFishCount11s-obb",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11s-obb.mlpackage.zip"
      )!
    ),
    (
      "AZFishCount11m-obb",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11m-obb.mlpackage.zip"
      )!
    ),
    (
      "AZFishCount11l-obb",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11l-obb.mlpackage.zip"
      )!
    ),
    (
      "AZFishCount11x-obb",
      URL(
        string:
          "https://github.com/ultralytics/AZFishCount-ios-app/releases/download/v8.3.0/AZFishCount11x-obb.mlpackage.zip"
      )!
    ),
  ],
]
