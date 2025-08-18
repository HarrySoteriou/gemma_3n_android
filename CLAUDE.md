# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

### Building the Project
```bash
./gradlew build
```

### Running on Device/Emulator
```bash
./gradlew installDebug
```

### Cleaning Build
```bash
./gradlew clean
```

### Gradle Properties
The project uses optimized Gradle settings in `gradle.properties`:
- Aggressive memory allocation: `-Xmx4g`
- Parallel builds and daemon enabled
- Uses `org.gradle.jvmargs` with ParallelGC and aggressive optimization

## Project Architecture

This is an Android application that implements real-time object detection using Google's MediaPipe with Gemma language model and NPU acceleration.

### Core Components

1. **MainActivity.kt** - Main activity that manages:
   - Camera permissions and lifecycle
   - Camera preview setup using CameraX
   - Model initialization coordination
   - UI state management (loading, error, success)
   - Resolution strategy optimized for 512x384 for faster inference

2. **ObjectDetection.kt** - Core AI inference engine:
   - MediaPipe LLM inference using Gemma 3N model (`gemma-3n-E2B-it-int4.task`)
   - Session management with reusable inference sessions
   - Memory management and NPU optimization
   - Frame sampling (0.2 FPS) for performance
   - Coordinate normalization from absolute to relative coordinates
   - Advanced parsing for detection output with fallback patterns

3. **OverlayView.kt** - Custom view for rendering detection results:
   - Draws bounding boxes and labels on camera preview
   - Handles coordinate transformation and rotation
   - Supports both bounded detections and text-only descriptions

### Key Technologies

- **MediaPipe + LiteRT Next**: For NPU-accelerated AI inference
- **CameraX**: Modern camera API with optimized resolution (512x384)
- **Kotlin Coroutines**: Async operations and lifecycle management
- **Data Binding**: View binding for type-safe UI access

### Model and Assets

- Model: `gemma-3n-E2B-it-int4.task` (located in `app/src/main/assets/`)
- Model is copied from assets to internal storage on first run for Android 13+ compatibility
- Supports vision-language multimodal inference

### Performance Optimizations

1. **Memory Management**:
   - Pre-allocates memory buffer to prevent model swapping
   - Session warming to keep model hot in RAM
   - Aggressive GC settings and memory monitoring

2. **Inference Optimization**:
   - Reusable inference sessions (max 2 inferences per session)
   - 30-second timeout for inference operations
   - Frame sampling at 0.2 FPS to prevent processing overload

3. **NPU Acceleration**:
   - Qualcomm QNN runtime libraries for NPU support
   - Graceful fallback to GPU/CPU if NPU unavailable

### Build Configuration

- **SDK Versions**: Compile SDK 35, Target SDK 35, Min SDK 31
- **ABI Filter**: `arm64-v8a` only for NPU support
- **Kotlin**: JVM Target 17 with coroutines support
- **No Compress**: `.litertlm`, `.task`, `.tflite` files preserved

### Dependencies Structure

- **MediaPipe**: `tasks-genai:0.10.25` and `tasks-vision:0.20230731`
- **LiteRT Next**: Core and Qualcomm NPU delegate libraries
- **CameraX**: Full BOM with core, camera2, lifecycle, and view components

## Development Notes

- The app requires Camera permission and handles Android 13+ storage permission changes
- Model loading happens on background thread with proper error handling
- Inference results are parsed using regex patterns with fallback parsing for robustness
- UI shows loading states during model initialization with retry functionality
- All coordinate systems are normalized (0-1 range) for consistent rendering

## Troubleshooting

- If model fails to load, check that `gemma-3n-E2B-it-int4.task` exists in assets folder
- NPU availability depends on device hardware (Snapdragon with NPU support)
- Memory pressure warnings indicate need to close other apps for better performance
- Coordinate parsing failures may require updating regex patterns in fallback parser