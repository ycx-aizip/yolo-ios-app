#!/bin/bash
set -e

# Package Fish Counting SDK for source distribution
# Copies all source code directly to the release repository instead of a compiled xcframework.
# The customer receives full Swift source and can build/modify everything.
#
# Difference from release.sh:
#   release.sh     → distributes AizipFishCount as binary xcframework (IP protected)
#   release_source → distributes AizipFishCount as Swift source (full access)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_REPO="${PROJECT_ROOT}/../Aizip_softbank_fishcount_ipad"

echo "📦 Packaging Fish Counting SDK for source distribution..."
echo "Development repo: ${PROJECT_ROOT}"
echo "Release repo:     ${RELEASE_REPO}"
echo ""

# ── Preflight checks ──────────────────────────────────────────────────────────

if [ ! -d "${RELEASE_REPO}" ]; then
    echo "❌ Error: Release repository not found at: ${RELEASE_REPO}"
    exit 1
fi

if [ ! -d "${PROJECT_ROOT}/Sources/AizipFishCount" ]; then
    echo "❌ Error: AizipFishCount source not found at: ${PROJECT_ROOT}/Sources/AizipFishCount"
    exit 1
fi

if [ ! -f "${PROJECT_ROOT}/Package.swift" ]; then
    echo "❌ Error: Package.swift not found at: ${PROJECT_ROOT}/Package.swift"
    exit 1
fi

DEV_XCODEPROJ="${PROJECT_ROOT}/AizipFishCountApp/AizipFishCountApp.xcodeproj"
if [ ! -d "${DEV_XCODEPROJ}" ]; then
    echo "❌ Error: Dev Xcode project not found at: ${DEV_XCODEPROJ}"
    exit 1
fi

# ── Clean release repo ────────────────────────────────────────────────────────

echo "🧹 Cleaning release repository..."

# Remove xcframework binary (replaced by source)
if [ -d "${RELEASE_REPO}/Packages/AizipFishCount.xcframework" ]; then
    rm -rf "${RELEASE_REPO}/Packages/AizipFishCount.xcframework"
    echo "   Removed Packages/AizipFishCount.xcframework"
fi

# Remove old Sources (will be replaced)
rm -rf "${RELEASE_REPO}/Sources/AizipFishCount"
rm -rf "${RELEASE_REPO}/Sources/Visualization"

# Remove old Package.swift if present
rm -f "${RELEASE_REPO}/Package.swift"

# Remove old opencv2 (will be replaced)
rm -rf "${RELEASE_REPO}/Packages/opencv2"

echo "✅ Clean complete"
echo ""

# ── Copy sources ──────────────────────────────────────────────────────────────

echo "📄 Copying AizipFishCount source..."
mkdir -p "${RELEASE_REPO}/Sources"
cp -R "${PROJECT_ROOT}/Sources/AizipFishCount" "${RELEASE_REPO}/Sources/"
echo "✅ AizipFishCount source copied ($(find "${RELEASE_REPO}/Sources/AizipFishCount" -name "*.swift" | wc -l | tr -d ' ') Swift files)"

echo "📄 Copying Visualization source..."
cp -R "${PROJECT_ROOT}/Sources/Visualization" "${RELEASE_REPO}/Sources/"
echo "✅ Visualization source copied ($(find "${RELEASE_REPO}/Sources/Visualization" -name "*.swift" | wc -l | tr -d ' ') Swift files)"

echo "📄 Copying Package.swift..."
cp "${PROJECT_ROOT}/Package.swift" "${RELEASE_REPO}/Package.swift"
echo "✅ Package.swift copied"
echo ""

# ── Copy Xcode project (SPM-based dev project) ────────────────────────────────
# The dev project uses Swift Package Manager (local package) to import AizipFishCount
# and Visualization from Sources/. The release project uses the xcframework instead.
# We replace the release xcodeproj with the dev (SPM-based) one, and fix the
# local package relativePath from "../../yolo-ios-app" to ".." so it resolves to
# Aizip_softbank_fishcount_ipad/ where Package.swift now lives.

echo "📁 Copying Xcode project (SPM-based)..."
RELEASE_XCODEPROJ="${RELEASE_REPO}/AizipFishCountApp/AizipFishCountApp.xcodeproj"
mkdir -p "${RELEASE_REPO}/AizipFishCountApp"
rm -rf "${RELEASE_XCODEPROJ}"
cp -R "${DEV_XCODEPROJ}" "${RELEASE_XCODEPROJ}"

PBXPROJ="${RELEASE_XCODEPROJ}/project.pbxproj"

# Fix SPM local package path:
#   Dev:     relativePath = "../../yolo-ios-app"  (from yolo-ios-app/AizipFishCountApp/)
#   Release: relativePath = ".."                  (from Aizip_softbank_fishcount_ipad/AizipFishCountApp/)
sed -i '' 's|relativePath = "../../yolo-ios-app"|relativePath = ".."|g' "${PBXPROJ}"

# Also fix the comment string that Xcode mirrors in the key
sed -i '' 's|XCLocalSwiftPackageReference "../../yolo-ios-app"|XCLocalSwiftPackageReference ".."|g' "${PBXPROJ}"

echo "✅ Xcode project copied and SPM path updated (../../yolo-ios-app → ..)"
echo ""

# ── Copy app source files ─────────────────────────────────────────────────────
# The Xcode project references these files but they are not part of the SPM
# packages — they are the app target's own source (ViewController, AppDelegate,
# storyboards, assets, logos, bridging header, etc.).

echo "📄 Copying app source files..."
APP_SRC="${PROJECT_ROOT}/AizipFishCountApp/AizipFishCountApp"
APP_DEST="${RELEASE_REPO}/AizipFishCountApp/AizipFishCountApp"
rm -rf "${APP_DEST}"
cp -R "${APP_SRC}" "${APP_DEST}"
echo "✅ App source files copied"

echo "📄 Copying bridging header..."
cp "${PROJECT_ROOT}/AizipFishCountApp/AizipFIshCountApp-Bridging-Header.h" \
   "${RELEASE_REPO}/AizipFishCountApp/AizipFIshCountApp-Bridging-Header.h"
echo "✅ Bridging header copied"
echo ""

# ── Copy opencv2 package ──────────────────────────────────────────────────────
# With source distribution there is no BUILD_LIBRARY_FOR_DISTRIBUTION restriction,
# so opencv2 can be included and OpenCV-based features (e.g. Phase 1 calibration)
# are available to the customer.

OPENCV_SRC="${PROJECT_ROOT}/Packages/opencv2"
if [ -d "${OPENCV_SRC}" ]; then
    echo "📦 Copying opencv2 package..."
    mkdir -p "${RELEASE_REPO}/Packages"
    cp -R "${OPENCV_SRC}" "${RELEASE_REPO}/Packages/"
    echo "✅ opencv2 copied"
else
    echo "⚠️  Warning: opencv2 not found at: ${OPENCV_SRC} (skipping)"
fi
echo ""

# ── Copy CoreML models ────────────────────────────────────────────────────────

echo "🧠 Copying CoreML models..."
MODELS_SOURCE="${PROJECT_ROOT}/AizipFishCountApp/FishCountModels"
MODELS_DEST="${RELEASE_REPO}/AizipFishCountApp/FishCountModels"

if [ ! -d "${MODELS_SOURCE}" ]; then
    echo "⚠️  Warning: Models not found at: ${MODELS_SOURCE} (skipping)"
else
    rm -rf "${MODELS_DEST}"
    mkdir -p "$(dirname "${MODELS_DEST}")"
    cp -R "${MODELS_SOURCE}" "${MODELS_DEST}"
    MODEL_COUNT=$(find "${MODELS_DEST}" -name "*.mlpackage" -type d | wc -l | tr -d ' ')
    echo "✅ CoreML models copied (${MODEL_COUNT} .mlpackage)"
fi
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Source release package ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Release repository contents:"
echo "  📄 Sources/AizipFishCount/         (backend source — tracking, counting, detection)"
echo "  📄 Sources/Visualization/          (UI source — customizable)"
echo "  📦 Package.swift                   (SPM package definition for both targets)"
echo "  📱 AizipFishCountApp/AizipFishCountApp/  (app source: ViewController, AppDelegate, storyboards, assets)"
echo "  📱 AizipFishCountApp/              (example Xcode app — SPM-based)"
echo "  📦 Packages/opencv2/               (OpenCV xcframework + bridge)"
echo "  🧠 AizipFishCountApp/FishCountModels/  (CoreML models)"
echo ""
echo "Integration:"
echo "  Open: AizipFishCountApp/AizipFishCountApp.xcodeproj"
echo "  Xcode will resolve the local SPM package from Package.swift at the repo root."
echo ""
echo "Notes:"
echo "  - AizipFishCount.xcframework NOT included (source replaces binary)"
echo "  - opencv2 IS included (no BUILD_LIBRARY_FOR_DISTRIBUTION restriction)"
echo ""
echo "Next steps:"
echo "  1. cd ${RELEASE_REPO}"
echo "  2. git add ."
echo "  3. git commit -m \"Source release vX.X.X\""
echo "  4. git push"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
