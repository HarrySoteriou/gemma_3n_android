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
    private val processingInterval = 5000L // 5 seconds - increased from 2 for heavy LLM
    private var isInitialized = false
    private var currentDelegate: Delegate = Delegate.CPU
    private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")
    private var isInferenceRunning = false
    private val inferenceStartTime = AtomicLong(0)
    private val maxInferenceTime = 20000L // 20 seconds max inference time

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
        if (!ready) {
            Log.v(TAG, "🔍 Model not ready - initialized: $isInitialized, llmInference: ${llmInference != null}, inferenceRunning: $isInferenceRunning")
        }
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
                        .setMaxTokens(512)
                        .setMaxTopK(40)
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
            // Add timeout to prevent hanging (10 seconds - reduced further)
            withTimeout(10000L) {
                // Resize image if needed for better performance
                Log.v(TAG, "🖼️ Checking if image resize is needed...")
                val resizedImage = resizeBitmapIfNeeded(image)
                
                // Show processing info
                if (resizedImage != image) {
                    Log.d(TAG, "🔍 Processing resized image: ${image.width}x${image.height} → ${resizedImage.width}x${resizedImage.height}")
                } else {
                    Log.d(TAG, "🔍 Processing original image: ${image.width}x${image.height}")
                }
                
                // Convert bitmap to MPImage
                Log.v(TAG, "🔄 Converting ${resizedImage.width}x${resizedImage.height} bitmap to MPImage...")
                val mpImage = BitmapImageBuilder(resizedImage).build()

                // Create session with vision modality enabled
                Log.v(TAG, "🔧 Creating LLM session with vision modality...")
                val sessionOptions = LlmInferenceSessionOptions.builder()
                    .setTemperature(0.3f)
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
                            
                            // Use a more concise prompt for faster response
                            Log.v(TAG, "📝 Adding prompt to session...")
                            session.addQueryChunk("List objects in this image:")
                            
                            Log.v(TAG, "🖼️ Adding image to session...")
                            session.addImage(mpImage)
                            
                            // Generate response
                            Log.d(TAG, "🧠 Generating LLM response...")
                            val result = session.generateResponse()
                            
                            val inferenceTime = SystemClock.uptimeMillis() - startTime
                            Log.d(TAG, "⏱️ Inference completed in ${inferenceTime}ms")
                            
                            // Parse the result into Detection objects
                            Log.v(TAG, "📄 Parsing LLM response...")
                            val detections = parseDetectionResult(result)
                            
                            Log.d(TAG, "🎯 Detection result (${detections.size} objects): $result")
                            
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
            }
        } catch (e: kotlinx.coroutines.TimeoutCancellationException) {
            Log.e(TAG, "⏰ Inference timed out after 10 seconds")
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
        val detections = mutableListOf<Detection>()
        
        // Simple parsing - split by lines and look for numbered items
        val lines = response.split("\n")
        var itemNumber = 1
        
        for (line in lines) {
            val trimmedLine = line.trim()
            if (trimmedLine.startsWith("$itemNumber.") || trimmedLine.matches(Regex("^\\d+\\."))) {
                // Extract object description
                val objectDescription = trimmedLine.substringAfter(".").trim()
                
                // Create a detection with the description
                val detection = Detection(
                    boundingBox = null, // LLM doesn't provide exact coordinates
                    label = objectDescription,
                    confidence = 0.8f, // Default confidence for LLM detection
                    classification = "LLM_DETECTION"
                )
                detections.add(detection)
                itemNumber++
            }
        }
        
        // If no numbered items found, treat whole response as one detection
        if (detections.isEmpty() && response.isNotBlank()) {
            detections.add(
                Detection(
                    boundingBox = null,
                    label = response.trim(),
                    confidence = 0.8f,
                    classification = "LLM_DETECTION"
                )
            )
        }
        
        return detections
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
                // Don't spam logs when inference is running
                if ((currentTime - lastProcessedTime.get()) > 1000L) { // Log once per second max
                    Log.v(TAG, "⏸️ Skipping frames - inference in progress")
                }
            } else {
                Log.v(TAG, "⏸️ Skipping frame - model not ready")
            }
            imageProxy.close()
            return
        }
        
        // Throttle processing to avoid overwhelming the system
        if (currentTime - lastProcessedTime.get() < processingInterval) {
            imageProxy.close()
            return
        }
        
        // Prevent concurrent inference sessions - check here too
        if (isInferenceRunning) {
            Log.v(TAG, "⏸️ Skipping frame - inference already running")
            imageProxy.close()
            return
        }
        
        lastProcessedTime.set(currentTime)
        
        try {
            Log.d(TAG, "📸 Processing camera frame with multimodal detection...")
            
            // Check for rotation changes
            if (imageProxy.imageInfo.rotationDegrees != imageRotation) {
                imageRotation = imageProxy.imageInfo.rotationDegrees
                // Reinitialize if needed
                setupLLMInference()
                imageProxy.close()
                return
            }
            
            // Convert ImageProxy to Bitmap properly
            Log.v(TAG, "🔄 Converting ImageProxy (${imageProxy.width}x${imageProxy.height}, format: ${imageProxy.format}) to Bitmap...")
            val bitmapBuffer = imageProxyToBitmap(imageProxy)
            imageProxy.close()
            
            if (bitmapBuffer == null) {
                Log.e(TAG, "❌ Failed to convert ImageProxy to Bitmap")
                return
            }
            
            Log.v(TAG, "✅ Successfully converted to ${bitmapBuffer.width}x${bitmapBuffer.height} bitmap")
            
            // Process the frame
            Log.d(TAG, "🎬 Starting object detection...")
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

            val nv21 = ByteArray(ySize + uSize + vSize)

            // Copy Y
            yBuffer.get(nv21, 0, ySize)
            
            // Copy UV
            val uvPixelStride = imageProxy.planes[1].pixelStride
            if (uvPixelStride == 1) {
                uBuffer.get(nv21, ySize, uSize)
                vBuffer.get(nv21, ySize + uSize, vSize)
            } else {
                // Handle interleaved UV
                val uvBuffer = ByteArray(uSize + vSize)
                uBuffer.get(uvBuffer, 0, uSize)
                vBuffer.get(uvBuffer, uSize, vSize)
                
                var uvIndex = 0
                for (i in 0 until uSize + vSize step uvPixelStride) {
                    nv21[ySize + uvIndex] = uvBuffer[i]
                    uvIndex++
                }
            }

            // Convert to RGB bitmap
            val yuvImage = android.graphics.YuvImage(
                nv21,
                android.graphics.ImageFormat.NV21,
                imageProxy.width,
                imageProxy.height,
                null
            )

            val out = java.io.ByteArrayOutputStream()
            yuvImage.compressToJpeg(
                android.graphics.Rect(0, 0, imageProxy.width, imageProxy.height),
                100,
                out
            )
            val imageBytes = out.toByteArray()
            android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            
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
        const val MAX_IMAGE_SIZE = 512 // Maximum width/height for processing
    }

    // Interface for detection callbacks
    interface DetectorListener {
        fun onError(error: String, errorCode: Int = OTHER_ERROR)
        fun onResults(resultBundle: ResultBundle)
    }
}
