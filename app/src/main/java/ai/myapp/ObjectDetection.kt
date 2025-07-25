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
    private val processingInterval = 400L // ~2.5 FPS
    private var isInitialized = false
    private var currentDelegate: Delegate = Delegate.CPU
    private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")

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
        val ready = isInitialized && llmInference != null
        if (!ready) {
            Log.v(TAG, "🔍 Model not ready - initialized: $isInitialized")
        }
        return ready
    }

    suspend fun cleanup() {
        withContext(singleThreadDispatcher) {
            Log.d(TAG, "🧹 Cleaning up ObjectDetection...")
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
            Log.e(TAG, "LLM not initialized")
            return@withContext null
        }

        val startTime = SystemClock.uptimeMillis()

        try {
            // Convert bitmap to MPImage
            val mpImage = BitmapImageBuilder(image).build()

                         // Create session with vision modality enabled
             val sessionOptions = LlmInferenceSessionOptions.builder()
                 .setTopK(10)
                 .setTemperature(0.4f)
                 .setGraphOptions(
                     GraphOptions.builder()
                         .setEnableVisionModality(true)
                         .build()
                 )
                 .build()

            llmInference?.use { llm ->
                LlmInferenceSession.createFromOptions(llm, sessionOptions).use { session ->
                    // Add the object detection prompt first, then the image
                    session.addQueryChunk("Detect and describe all objects in this image. List each object with its type, location (if possible), and any notable characteristics. Format the response as a numbered list.")
                    session.addImage(mpImage)
                    
                    // Generate response
                    val result = session.generateResponse()
                    
                    val inferenceTime = SystemClock.uptimeMillis() - startTime
                    
                    // Parse the result into Detection objects
                    val detections = parseDetectionResult(result)
                    
                    Log.d(TAG, "🎯 Detection result: $result")
                    
                    return@withContext ResultBundle(
                        detections = detections,
                        inferenceTime = inferenceTime,
                        inputImageHeight = image.height,
                        inputImageWidth = image.width
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error during detection", e)
            detectorListener?.onError("Detection failed: ${e.message}")
        }
        
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
                
                // Process the first frame
                kotlinx.coroutines.runBlocking {
                    detectImage(argb8888Frame)
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
        
        // Throttle processing to avoid overwhelming the system
        if (currentTime - lastProcessedTime.get() < processingInterval) {
            imageProxy.close()
            return
        }
        
        lastProcessedTime.set(currentTime)
        
        try {
                         // Copy RGB bits from frame to bitmap buffer
             val bitmapBuffer = Bitmap.createBitmap(
                 imageProxy.width, 
                 imageProxy.height, 
                 Bitmap.Config.ARGB_8888
             )
             imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
            
            // Check for rotation changes
            if (imageProxy.imageInfo.rotationDegrees != imageRotation) {
                imageRotation = imageProxy.imageInfo.rotationDegrees
                // Reinitialize if needed
                setupLLMInference()
                return
            }
            
                         // Convert the input Bitmap object to an MPImage object to run inference
             // Process the frame
             val result = detectImage(bitmapBuffer)
             result?.let { 
                 detectorListener?.onResults(it)
             }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing livestream frame", e)
            detectorListener?.onError("Livestream detection failed: ${e.message}")
        }
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val OTHER_ERROR = 0
        const val GPU_ERROR = 1
        const val TAG = "ObjectDetection"
    }

    // Interface for detection callbacks
    interface DetectorListener {
        fun onError(error: String, errorCode: Int = OTHER_ERROR)
        fun onResults(resultBundle: ResultBundle)
    }
}
