package ai.myapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.annotation.VisibleForTesting
import androidx.camera.core.ImageProxy
import androidx.lifecycle.LifecycleOwner

// MediaPipe frameworks imports
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
// MediaPipe core tasks
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
// MediaPipe genai tasks
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInference.LlmInferenceOptions
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession.LlmInferenceSessionOptions
import com.google.mediapipe.tasks.genai.llminference.GraphOptions
// kotlin imports
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
// java imports
import java.io.File
import java.util.concurrent.atomic.AtomicLong

class ObjectDetection(
    private val lifecycleOwner: LifecycleOwner,
    val context: Context,
    var detectorListener: DetectorListener? = null
) {

    private var imageRotation = 0
    private var modelPath = "gemma-3n-E2B-it-int4.task"
    private var llmInference: LlmInference? = null
    private val lastProcessedTime = AtomicLong(0)
    private val processingInterval = 1000L // 1 second - changed from 12 seconds for 1 FPS sampling
    private var isInitialized = false
    private var currentDelegate: Delegate = Delegate.CPU
    private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")
    private var isInferenceRunning = false
    private val inferenceStartTime = AtomicLong(0)
    private val maxInferenceTime = 30000L // 30 seconds max inference time (increased from 20)

    // Detection result data class
    data class Detection(
        val boundingBox: RectF? = null,
        val label: String,
        val confidence: Float,
        val classification: String
    )

    // Result bundle for returning detection results
    data class ResultBundle(
        val detections: List<Detection>,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
        val inputImageRotation: Int = 0
    )

    // Note: Initialization happens asynchronously via initializeAsync()
    // Call initializeAsync() from MainActivity after creating ObjectDetection

    /**
     * Initialize the ObjectDetection model asynchronously
     * Call this from a coroutine scope (e.g., lifecycleScope)
     */
    suspend fun initializeAsync() {
        setupLLMInference()
    }

    fun isReady(): Boolean {
        // Check if inference has been running too long and force reset
        if (isInferenceRunning) {
            val currentTime = System.currentTimeMillis()
            val inferenceStarted = inferenceStartTime.get()
            if (inferenceStarted > 0 && (currentTime - inferenceStarted) > maxInferenceTime) {
                Log.w(TAG, "🚨 Inference stuck for ${currentTime - inferenceStarted}ms - forcing reset")
                isInferenceRunning = false
                inferenceStartTime.set(0)
            }
        }
        
        val ready = isInitialized && llmInference != null && !isInferenceRunning
        // Only log when state changes or every few seconds to avoid spam
        if (!ready) {
            val currentTime = System.currentTimeMillis()
            val lastLogTime = lastProcessedTime.get()
            if (currentTime - lastLogTime > 4000L) { // Log state every 4 seconds max
                Log.v(TAG, "🔍 Model not ready - initialized: $isInitialized, llmExists: ${llmInference != null}, inferenceRunning: $isInferenceRunning")
            }
        }
        return ready
    }

    /**
     * Check if the model is fully initialized and ready for video streaming
     * This should be called before starting camera/video capture
     */
    fun isReadyForStreaming(): Boolean {
        val ready = isInitialized && llmInference != null
        Log.d(TAG, "🎥 Model streaming readiness check: initialized=$isInitialized, llmExists=${llmInference != null}, ready=$ready")
        return ready
    }

    suspend fun cleanup() {
        withContext(singleThreadDispatcher) {
            Log.d(TAG, "🧹 Cleaning up ObjectDetection...")
            
            // Force reset any stuck inference
            if (isInferenceRunning) {
                Log.w(TAG, "🚨 Forcing reset of stuck inference during cleanup")
                isInferenceRunning = false
                inferenceStartTime.set(0)
            }
            
            llmInference?.close()
            llmInference = null
            isInitialized = false
        }
    }

    private suspend fun copyModelFromAssets(): String? = withContext(Dispatchers.IO) {
        try {
            val assetManager = context.assets
            val outFile = File(context.filesDir, modelPath)
            if (outFile.exists()) {
                Log.d(TAG, "📁 Model already copied to ${outFile.absolutePath}")
                return@withContext outFile.absolutePath
            }

            Log.d(TAG, "📥 Copying model from assets...")
            assetManager.open(modelPath).use { input ->
                outFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            Log.d(TAG, "✅ Model copied to ${outFile.absolutePath}")
            return@withContext outFile.absolutePath

        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to copy model from assets", e)
            null
        }
    }

    // Initialize the LLM with multimodal capabilities for object detection
    private suspend fun setupLLMInference() {
        try {
            // First copy model from assets to local storage
            val modelFilePath = copyModelFromAssets()
            if (modelFilePath == null) {
                detectorListener?.onError("Failed to copy model from assets")
                return
            }

            withContext(Dispatchers.Main) {
                try {
                    // Create LLM Inference options with multimodal support
                    // Following MediaPipe documentation pattern
                    val options = LlmInferenceOptions.builder()
                        .setModelPath(modelFilePath) // Use copied file path
                        .setMaxTokens(512) // Increased to accommodate image tokens (258) + output tokens
                        .setMaxTopK(20)
                        .setMaxNumImages(1) // Allow one image per session
                        .build()

                    // Create LLM Inference instance
                    llmInference = LlmInference.createFromOptions(context, options)
                    isInitialized = true
                    
                    Log.d(TAG, "✅ LLM Inference initialized successfully")

                } catch (e: IllegalStateException) {
                    detectorListener?.onError("LLM failed to initialize. See error logs for details")
                    Log.e(TAG, "LLM failed to load model with error: " + e.message)
                } catch (e: RuntimeException) {
                    detectorListener?.onError("LLM failed to initialize. See error logs for details")
                    Log.e(TAG, "LLM failed to load model with error: " + e.message)
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "❌ Error during model setup", e)
            detectorListener?.onError("Model setup failed: ${e.message}")
        }
    }

    // Check if the detector is closed
    fun isClosed(): Boolean {
        return llmInference == null
    }

    // Detect objects in a single image using multimodal LLM
    suspend fun detectImage(image: Bitmap): ResultBundle? = withContext(singleThreadDispatcher) {
        if (llmInference == null) {
            Log.e(TAG, "❌ LLM not initialized")
            return@withContext null
        }

        // Prevent concurrent inference sessions
        if (isInferenceRunning) {
            Log.w(TAG, "⏸️ Skipping inference - another session is already running")
            return@withContext null
        }

        isInferenceRunning = true
        inferenceStartTime.set(System.currentTimeMillis())
        Log.d(TAG, "🔍 Starting inference for ${image.width}x${image.height} image")
        val startTime = SystemClock.uptimeMillis()

        try {
            // Add timeout to prevent hanging (30 seconds - increased to match max inference time)
            val result = withTimeout(30000L) {
                // Resize image if needed for better performance
                Log.v(TAG, "🖼️ Checking if image resize is needed...")
                val originalWidth = image.width
                val originalHeight = image.height
                val resizedImage = resizeBitmapIfNeeded(image)
                
                // Show processing info
                if (resizedImage != image) {
                    Log.d(TAG, "🔍 Processing resized image: ${originalWidth}x${originalHeight} → ${resizedImage.width}x${resizedImage.height}")
                } else {
                    Log.d(TAG, "🔍 Processing original image: ${originalWidth}x${originalHeight}")
                }
                
                // Convert bitmap to MPImage
                Log.v(TAG, "🔄 Converting ${resizedImage.width}x${resizedImage.height} bitmap to MPImage...")
                val mpImage = BitmapImageBuilder(resizedImage).build()

                // Create session with vision modality enabled
                Log.v(TAG, "🔧 Creating LLM session with vision modality...")
                val sessionOptions = LlmInferenceSessionOptions.builder()
                    .setTemperature(0.1f) // Lower temperature for faster, more focused responses
                    .setGraphOptions(
                        GraphOptions.builder()
                            .setEnableVisionModality(true)
                            .build()
                    )
                    .build()

                var resultBundle: ResultBundle? = null
                
                llmInference?.use { llm ->
                    Log.v(TAG, "💭 Creating inference session...")
                    try {
                        LlmInferenceSession.createFromOptions(llm, sessionOptions).use { session ->
                            Log.v(TAG, "✅ Session created successfully")
                            
                            // Use a more specific and concise prompt for faster response
                            Log.v(TAG, "📝 Adding prompt to session...")
                            session.addQueryChunk("Detect objects with bounding boxes (format: ObjectName [x1,y1,x2,y2]): 1.")
                            
                            Log.v(TAG, "🖼️ Adding image to session...")
                            session.addImage(mpImage)
                            
                            // Generate response
                            Log.d(TAG, "🧠 Generating LLM response...")
                            val result = session.generateResponse()
                            
                            val inferenceTime = SystemClock.uptimeMillis() - startTime
                            Log.d(TAG, "⏱️ Inference completed in ${inferenceTime}ms")
                            
                            // Warn if inference is taking too long for real-time
                            if (inferenceTime > 15000) {
                                Log.w(TAG, "🐌 Slow inference detected (${inferenceTime}ms)")
                            }
                            
                            // Parse the result into Detection objects
                            Log.v(TAG, "📄 Parsing LLM response...")
                            val detections = parseDetectionResult(result)
                            
                            Log.d(TAG, "🎯 Detection result (${detections.size} objects found, ${result.length} chars): ${result.take(100)}${if (result.length > 100) "..." else ""}")
                            
                            resultBundle = ResultBundle(
                                detections = detections,
                                inferenceTime = inferenceTime,
                                inputImageHeight = resizedImage.height,
                                inputImageWidth = resizedImage.width
                            )
                            
                            Log.d(TAG, "✅ Created result bundle with ${detections.size} detections")
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "❌ Error during session creation or inference", e)
                        throw e
                    }
                }
                
                // Return the result after use blocks are closed
                resultBundle?.let {
                    Log.d(TAG, "✅ Returning result bundle with ${it.detections.size} detections")
                    return@withTimeout it
                }
                
                // If we get here, something went wrong
                Log.w(TAG, "⚠️ Result bundle is null after inference")
                return@withTimeout null
            }
            
            // If we successfully got a result, return it
            result?.let {
                Log.d(TAG, "🎉 Detection successful - returning ${it.detections.size} detections")
                return@withContext it
            }
            
        } catch (e: kotlinx.coroutines.TimeoutCancellationException) {
            Log.e(TAG, "⏰ Inference timed out after 30 seconds")
            detectorListener?.onError("Inference timed out")
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error during detection", e)
            detectorListener?.onError("Detection failed: ${e.message}")
        } finally {
            isInferenceRunning = false
            inferenceStartTime.set(0)
            Log.d(TAG, "🔓 Released inference lock")
        }
        
        Log.w(TAG, "⚠️ Detection failed - returning null")
        return@withContext null
    }

    // Parse the LLM response into Detection objects
    private fun parseDetectionResult(response: String): List<Detection> {
        Log.v(TAG, "🔍 Parsing response: ${response.take(200)}${if (response.length > 200) "..." else ""}")
        val detections = mutableListOf<Detection>()
        
        // Parse different formats: numbered items, bullet points, and markdown
        val lines = response.split("\n")
        
        for (line in lines) {
            val trimmedLine = line.trim()
            var objectDescription = ""
            var boundingBox: RectF? = null
            
            // Handle different formats
            when {
                // Numbered items: "1. Object" or "1) Object" - with or without bounding boxes
                trimmedLine.matches(Regex("^\\d+[.)]\\s*.+")) -> {
                    val content = trimmedLine.replaceFirst(Regex("^\\d+[.)]\\s*"), "").trim()
                    val (description, bbox) = extractObjectAndBoundingBox(content)
                    objectDescription = description
                    boundingBox = bbox
                }
                // Bullet points: "* Object" or "- Object" - with or without bounding boxes
                trimmedLine.matches(Regex("^[*-]\\s*.+")) -> {
                    val content = trimmedLine.replaceFirst(Regex("^[*-]\\s*"), "").trim()
                    val (description, bbox) = extractObjectAndBoundingBox(content)
                    objectDescription = description
                    boundingBox = bbox
                }
                // Markdown bold items: "* **Object:**" or "- **Object:**"
                trimmedLine.matches(Regex("^[*-]\\s*\\*\\*.+\\*\\*:?.*")) -> {
                    // Extract the bold text and description
                    val boldMatch = Regex("\\*\\*(.+?)\\*\\*:?(.*)").find(trimmedLine)
                    if (boldMatch != null) {
                        val objectName = boldMatch.groupValues[1].trim()
                        val description = boldMatch.groupValues[2].trim()
                        val fullContent = if (description.isNotEmpty()) {
                            "$objectName: $description"
                        } else {
                            objectName
                        }
                        val (cleanDescription, bbox) = extractObjectAndBoundingBox(fullContent)
                        objectDescription = cleanDescription
                        boundingBox = bbox
                    }
                }
                // Simple list items that start with common object words
                trimmedLine.matches(Regex("^(a|an|the|some|several)?\\s*(person|water|fire|door|car|keyboard|mouse|monitor|screen|phone|bottle|cup|book|pen|laptop|computer|desk|chair|headphones|microphone|cable|wire|speaker|camera).*", RegexOption.IGNORE_CASE)) -> {
                    val (description, bbox) = extractObjectAndBoundingBox(trimmedLine)
                    objectDescription = description
                    boundingBox = bbox
                }
            }
            
            // Add detection if we found a valid object description
            if (objectDescription.isNotEmpty()) {
                // Clean up the description - remove extra formatting
                val cleanDescription = objectDescription
                    .replace(Regex("\\*\\*(.+?)\\*\\*"), "$1") // Remove markdown bold
                    .replace(Regex("^(a|an|the)\\s+", RegexOption.IGNORE_CASE), "") // Remove articles
                    .trim()
                
                if (cleanDescription.isNotEmpty()) {
                    val detection = Detection(
                        boundingBox = boundingBox,
                        label = cleanDescription,
                        confidence = if (boundingBox != null) 0.9f else 0.8f, // Higher confidence with bounding box
                        classification = "LLM_DETECTION"
                    )
                    detections.add(detection)
                }
            }
        }
        
        // If no structured items found, try to extract object names from the text
        if (detections.isEmpty() && response.isNotBlank()) {
            // Look for common object patterns in the entire text
            val objectPatterns = listOf(
                "person", "water", "fire", "door", "car"
            )
            
            val foundObjects = mutableSetOf<String>()
            for (pattern in objectPatterns) {
                if (response.contains(pattern, ignoreCase = true)) {
                    foundObjects.add(pattern.lowercase().replaceFirstChar { it.uppercase() })
                }
            }
            
            // If we found some objects, add them
            if (foundObjects.isNotEmpty()) {
                foundObjects.forEach { objectName ->
                    detections.add(
                        Detection(
                            boundingBox = null,
                            label = objectName,
                            confidence = 0.7f, // Lower confidence for pattern matching
                            classification = "LLM_DETECTION"
                        )
                    )
                }
            } else {
                // Fallback: treat whole response as one detection
                detections.add(
                    Detection(
                        boundingBox = null,
                        label = response.trim(),
                        confidence = 0.8f,
                        classification = "LLM_DETECTION"
                    )
                )
            }
        }
        
        Log.v(TAG, "📋 Parsed ${detections.size} detections: ${detections.map { "${it.label}${if (it.boundingBox != null) " [bbox]" else ""}" }}")
        return detections
    }

    // Helper function to extract object name and bounding box from a line
    private fun extractObjectAndBoundingBox(content: String): Pair<String, RectF?> {
        // Look for bounding box pattern [x1,y1,x2,y2] or (x1,y1,x2,y2)
        val bboxPattern = Regex("\\[([0-9.]+),([0-9.]+),([0-9.]+),([0-9.]+)\\]|\\(([0-9.]+),([0-9.]+),([0-9.]+),([0-9.]+)\\)")
        val bboxMatch = bboxPattern.find(content)
        
        return if (bboxMatch != null) {
            // Extract coordinates (handle both bracket formats)
            val coords = if (bboxMatch.groupValues[1].isNotEmpty()) {
                // Bracket format [x1,y1,x2,y2]
                listOf(
                    bboxMatch.groupValues[1].toFloatOrNull() ?: 0f,
                    bboxMatch.groupValues[2].toFloatOrNull() ?: 0f,
                    bboxMatch.groupValues[3].toFloatOrNull() ?: 0f,
                    bboxMatch.groupValues[4].toFloatOrNull() ?: 0f
                )
            } else {
                // Parenthesis format (x1,y1,x2,y2)
                listOf(
                    bboxMatch.groupValues[5].toFloatOrNull() ?: 0f,
                    bboxMatch.groupValues[6].toFloatOrNull() ?: 0f,
                    bboxMatch.groupValues[7].toFloatOrNull() ?: 0f,
                    bboxMatch.groupValues[8].toFloatOrNull() ?: 0f
                )
            }
            
            // Remove bounding box from object description
            val objectName = content.replace(bboxPattern, "").trim()
            val boundingBox = RectF(coords[0], coords[1], coords[2], coords[3])
            
            Log.v(TAG, "🎯 Extracted object: '$objectName' with bbox: [${coords[0]}, ${coords[1]}, ${coords[2]}, ${coords[3]}]")
            
            Pair(objectName, boundingBox)
        } else {
            // No bounding box found, return just the object name
            Pair(content, null)
        }
    }

    // Detect objects in video frames
    fun detectVideoFile(videoUri: Uri, inferenceIntervalMs: Long): ResultBundle? {
        // For video, we would need to extract frames and process them
        // This is a simplified version that processes the first frame
        val retriever = MediaMetadataRetriever()
        return try {
            retriever.setDataSource(context, videoUri)
            val firstFrame = retriever.getFrameAtTime(0)
            
            if (firstFrame != null) {
                // Convert the video frame to ARGB_8888 which is required by MediaPipe
                val argb8888Frame = if (firstFrame.config == Bitmap.Config.ARGB_8888) {
                    firstFrame
                } else {
                    firstFrame.copy(Bitmap.Config.ARGB_8888, false)
                }
                
                // Resize frame for better performance
                val resizedFrame = resizeBitmapIfNeeded(argb8888Frame)
                
                // Process the resized frame
                kotlinx.coroutines.runBlocking {
                    detectImage(resizedFrame)
                }
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing video", e)
            null
        } finally {
            retriever.release()
        }
    }

    // Detect objects in livestream frame
    suspend fun detectLivestreamFrame(imageProxy: ImageProxy) {
        val currentTime = System.currentTimeMillis()
        
        // Check if model is ready before any processing
        if (!isReady()) {
            if (isInferenceRunning) {
                // Frame skipping during inference - don't spam logs
                if ((currentTime - lastProcessedTime.get()) > 5000L) { // Log once every 5 seconds max
                    Log.d(TAG, "🎬 Skipping frames during inference (1 FPS sampling active)")
                }
            } else {
                Log.v(TAG, "⏸️ Skipping frame - model not ready")
            }
            imageProxy.close()
            return
        }
        
        // Frame sampling: process 1 frame per second (1 FPS)
        if (currentTime - lastProcessedTime.get() < processingInterval) {
            // Frame skipping for 1 FPS sampling
            Log.v(TAG, "⏭️ Skipping frame for 1 FPS sampling (${currentTime - lastProcessedTime.get()}ms since last)")
            imageProxy.close()
            return
        }
        
        // Prevent concurrent inference sessions - additional safety check
        if (isInferenceRunning) {
            Log.w(TAG, "⚠️ Inference running despite isReady() check - skipping frame")
            imageProxy.close()
            return
        }
        
        lastProcessedTime.set(currentTime)
        
        try {
            Log.d(TAG, "📸 Processing camera frame (1 FPS sampling) with multimodal detection...")
            
            // Check for rotation changes
            if (imageProxy.imageInfo.rotationDegrees != imageRotation) {
                imageRotation = imageProxy.imageInfo.rotationDegrees
                // Reinitialize if needed
                setupLLMInference()
                imageProxy.close()
                return
            }
            
            // Convert ImageProxy to Bitmap with optimized YUV conversion
            Log.v(TAG, "🔄 Converting ImageProxy (${imageProxy.width}x${imageProxy.height}, format: ${imageProxy.format}) to Bitmap...")
            val bitmapBuffer = imageProxyToBitmap(imageProxy)
            imageProxy.close()
            
            if (bitmapBuffer == null) {
                Log.e(TAG, "❌ Failed to convert ImageProxy to Bitmap")
                return
            }
            
            Log.v(TAG, "✅ Successfully converted to ${bitmapBuffer.width}x${bitmapBuffer.height} bitmap")
            
            // Process the frame
            Log.d(TAG, "🎬 Starting object detection (frame skipping active during inference)...")
            val result = detectImage(bitmapBuffer)
            result?.let { 
                Log.d(TAG, "📤 Detection complete - ${it.detections.size} objects found in ${it.inferenceTime}ms")
                detectorListener?.onResults(it)
            } ?: run {
                Log.w(TAG, "⚠️ Detection returned no results")
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing livestream frame", e)
            detectorListener?.onError("Livestream detection failed: ${e.message}")
            imageProxy.close()
        }
    }

    // Convert ImageProxy to Bitmap safely
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            when (imageProxy.format) {
                android.graphics.ImageFormat.YUV_420_888 -> {
                    // Most common format from camera - convert YUV to RGB
                    yuv420ToRgbBitmap(imageProxy)
                }
                android.graphics.PixelFormat.RGBA_8888 -> {
                    // Direct RGB format - can copy directly
                    rgbaImageProxyToBitmap(imageProxy)
                }
                else -> {
                    Log.w(TAG, "Unsupported image format: ${imageProxy.format}, attempting YUV conversion")
                    yuv420ToRgbBitmap(imageProxy)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error converting ImageProxy to Bitmap: ${e.message}", e)
            null
        }
    }
    
    private fun yuv420ToRgbBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val yBuffer = imageProxy.planes[0].buffer // Y
            val uBuffer = imageProxy.planes[1].buffer // U 
            val vBuffer = imageProxy.planes[2].buffer // V

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            // Get the pixel strides and row strides
            val yPixelStride = imageProxy.planes[0].pixelStride
            val yRowStride = imageProxy.planes[0].rowStride
            val uvPixelStride = imageProxy.planes[1].pixelStride
            val uvRowStride = imageProxy.planes[1].rowStride

            val width = imageProxy.width
            val height = imageProxy.height

            // Create output bitmap
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val pixels = IntArray(width * height)

            // Direct YUV to RGB conversion without JPEG compression
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val yIndex = y * yRowStride + x * yPixelStride
                    val uvIndex = (y / 2) * uvRowStride + (x / 2) * uvPixelStride
                    
                    // Get Y, U, V values
                    val yValue = (yBuffer.get(yIndex).toInt() and 0xFF) - 16
                    val uValue = (uBuffer.get(uvIndex).toInt() and 0xFF) - 128
                    val vValue = (vBuffer.get(uvIndex).toInt() and 0xFF) - 128
                    
                    // YUV to RGB conversion using standard formula
                    var r = ((298 * yValue + 409 * vValue + 128) shr 8).coerceIn(0, 255)
                    var g = ((298 * yValue - 100 * uValue - 208 * vValue + 128) shr 8).coerceIn(0, 255)
                    var b = ((298 * yValue + 516 * uValue + 128) shr 8).coerceIn(0, 255)
                    
                    // Set pixel (ARGB format)
                    pixels[y * width + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
            
            bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
            bitmap
            
        } catch (e: Exception) {
            Log.e(TAG, "Error converting YUV to RGB: ${e.message}", e)
            null
        }
    }
    
    private fun rgbaImageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val buffer = imageProxy.planes[0].buffer
            val pixelStride = imageProxy.planes[0].pixelStride
            val rowStride = imageProxy.planes[0].rowStride
            val rowPadding = rowStride - pixelStride * imageProxy.width
            
            val bitmap = Bitmap.createBitmap(
                imageProxy.width + rowPadding / pixelStride,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            )
            
            bitmap.copyPixelsFromBuffer(buffer)
            
            if (rowPadding == 0) {
                bitmap
            } else {
                val croppedBitmap = Bitmap.createBitmap(bitmap, 0, 0, imageProxy.width, imageProxy.height)
                bitmap.recycle()
                croppedBitmap
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error converting RGBA ImageProxy to Bitmap: ${e.message}", e)
            null
        }
    }

    // Helper function to resize bitmap while maintaining aspect ratio
    private fun resizeBitmapIfNeeded(bitmap: Bitmap, maxSize: Int = MAX_IMAGE_SIZE): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        
        // If already small enough, return original
        if (width <= maxSize && height <= maxSize) {
            Log.v(TAG, "✅ Image size ${width}x${height} is within limits, no resize needed")
            return bitmap
        }
        
        // Calculate scale factor to fit within maxSize while maintaining aspect ratio
        val scaleFactor = minOf(maxSize.toFloat() / width, maxSize.toFloat() / height)
        val newWidth = (width * scaleFactor).toInt()
        val newHeight = (height * scaleFactor).toInt()
        
        Log.d(TAG, "📏 Resizing image from ${width}x${height} to ${newWidth}x${newHeight} (scale: ${String.format("%.2f", scaleFactor)})")
        
        return try {
            val resized = Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
            
            // If we created a new bitmap and it's different from the original, recycle the original
            if (resized != bitmap) {
                bitmap.recycle()
            }
            
            Log.v(TAG, "✅ Successfully resized to ${resized.width}x${resized.height}")
            resized
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error resizing bitmap: ${e.message}", e)
            bitmap // Return original if resize fails
        }
    }

    // Emergency reset for stuck inference
    fun forceResetInference() {
        Log.w(TAG, "🚨 Force resetting inference state")
        isInferenceRunning = false
        inferenceStartTime.set(0)
    }
    
    // Get current inference status for debugging
    fun getInferenceStatus(): String {
        val runningTime = if (isInferenceRunning && inferenceStartTime.get() > 0) {
            System.currentTimeMillis() - inferenceStartTime.get()
        } else 0L
        
        return "Inference running: $isInferenceRunning, Duration: ${runningTime}ms, Initialized: $isInitialized"
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val OTHER_ERROR = 0
        const val GPU_ERROR = 1
        const val TAG = "ObjectDetection"
        const val MAX_IMAGE_SIZE = 256 // Reduced from 512 for faster processing
    }

    // Interface for detection callbacks
    interface DetectorListener {
        fun onError(error: String, errorCode: Int = OTHER_ERROR)
        fun onResults(resultBundle: ResultBundle)
    }
}
