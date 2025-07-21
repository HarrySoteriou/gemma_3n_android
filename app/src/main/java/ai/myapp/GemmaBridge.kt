package ai.myapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicLong

class GemmaBridge(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) {

    companion object {
        private const val TAG = "GemmaBridge"
    }

    private var llmInferenceTask: LLMInferenceTask? = null
    private val lastProcessedTime = AtomicLong(0)
    private val processingInterval = 400L // ~2.5 FPS
    private var isInitialized = false

    init {
        // Show loading UI when initialization starts
        if (context is MainActivity) {
            context.runOnUiThread { context.showLoading() }
        }
    }

    /**
     * Initialize the LLM asynchronously using lifecycle-aware scope
     */
    fun initializeAsync() {
        Log.d(TAG, "🔄 Starting async initialization...")
        
        lifecycleOwner.lifecycleScope.launch {
            try {
                Log.d(TAG, "🔄 Creating LLMInferenceTask...")
                llmInferenceTask = LLMInferenceTask(context)
                Log.d(TAG, "✅ LLMInferenceTask created successfully")
                
                // Check if model file is available before attempting to initialize
                val modelAvailable = llmInferenceTask?.isModelFileAvailable() ?: false
                Log.d(TAG, "📁 Model file available: $modelAvailable")
                
                if (!modelAvailable) {
                    Log.e(TAG, "❌ Cannot proceed with initialization: Model file not found")
                    Log.e(TAG, "📋 Please ensure 'gemma-3n-E2B-it-int4.task' is placed in one of the expected locations")
                    isInitialized = false
                    // Handle missing model file using callback
                    if (context is MainActivity) {
                        context.runOnUiThread { 
                            context.onModelInitializationFailed("Model file not found. Please ensure 'gemma-3n-E2B-it-int4.task' is available in the app's data directory.")
                        }
                    }
                    return@launch
                }
                
                Log.d(TAG, "🔄 Initializing LLM model...")
                llmInferenceTask?.initializeModel()
                Log.d(TAG, "✅ initializeModel() completed without throwing exception")
                
                // Double-check that initialization actually succeeded
                val taskReady = llmInferenceTask?.isReady() ?: false
                Log.d(TAG, "🔍 Task readiness check: $taskReady")
                
                if (taskReady) {
                    isInitialized = true
                    Log.d(TAG, "✅ GemmaBridge initialization completed successfully!")
                    // Notify MainActivity that model is ready
                    if (context is MainActivity) {
                        context.runOnUiThread { context.onModelInitialized() }
                    }
                } else {
                    isInitialized = false
                    Log.e(TAG, "❌ GemmaBridge initialization failed: LLM not ready after initialization")
                    
                    // Additional diagnostics
                    llmInferenceTask?.let { task ->
                        Log.e(TAG, "🔍 Diagnostic info:")
                        Log.e(TAG, "  - Model file available: ${task.isModelFileAvailable()}")
                        Log.e(TAG, "  - Task ready: ${task.isReady()}")
                    }
                    // Handle failure using callback
                    if (context is MainActivity) {
                        context.runOnUiThread { 
                            context.onModelInitializationFailed("LLM not ready after initialization. Check model file and permissions.")
                        }
                    }
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "❌ Initialization failed with exception!", e)
                Log.e(TAG, "❌ Exception type: ${e.javaClass.simpleName}")
                Log.e(TAG, "❌ Exception message: ${e.message}")
                Log.e(TAG, "❌ Stack trace:")
                e.printStackTrace()
                
                isInitialized = false
                llmInferenceTask?.cleanup()
                llmInferenceTask = null
                // Handle exception failure using callback
                if (context is MainActivity) {
                    context.runOnUiThread { 
                        context.onModelInitializationFailed("Error loading model: ${e.message}\nPlease ensure model file is available and try again.")
                    }
                }
            }
        }
    }

    fun processFrame(image: ImageProxy) {
        val currentTime = System.currentTimeMillis()
        
        // Skip processing if we're too busy or processed recently
        if (!shouldProcessFrame(currentTime)) {
            Log.v(TAG, "⏭️ Skipping frame processing (too frequent or LLM not ready)")
            image.close()
            Log.v(TAG, "🗑️ Frame discarded (skipped - throttled)")
            return
        }

        Log.d(TAG, "🖼️ Processing new camera frame")
        // Update last processed time
        lastProcessedTime.set(currentTime)

        // Capture rotation
        val rotation = image.imageInfo.rotationDegrees

        // Process the frame asynchronously using lifecycle scope
        lifecycleOwner.lifecycleScope.launch {
            try {
                val bitmap = withContext(Dispatchers.IO) {
                    imageProxyToBitmap(image)
                }
                
                if (bitmap != null && isReady()) {
                    val response = withContext(Dispatchers.IO) {
                        llmInferenceTask?.analyzeScene(
                            bitmap, 
                            "Analyze this camera feed for people, objects, and safety concerns. For each detected object, output in this exact format:\nDETECTED: name\nBOX: [left,top,right,bottom] (normalized 0-1 from left-top)\nCONFIDENCE: high/medium/low\nRISK: low/medium/high/critical\nSeparate multiple objects with --- Be concise and only output the formatted text."
                        )
                    }
                    
                    // Parse the LLM response and create detections
                    val detections = parseResponseToDetections(response, bitmap.width.toFloat(), bitmap.height.toFloat())
                    
                    // Update UI on main thread (we're already on Main due to lifecycleScope)
                    Log.d(TAG, "📱 Updating UI with ${detections.size} detections")
                    (context as? MainActivity)?.findViewById<OverlayView>(ai.myapp.R.id.overlay)
                        ?.setResults(detections, bitmap.height, bitmap.width, rotation)
                } else {
                    Log.w(TAG, "⚠️ Skipping inference: bitmap=${bitmap != null}, ready=${isReady()}")
                    (context as? MainActivity)?.findViewById<OverlayView>(ai.myapp.R.id.overlay)
                        ?.setResults(emptyList(), bitmap?.height ?: 480, bitmap?.width ?: 640, rotation)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error processing frame", e)
                (context as? MainActivity)?.findViewById<OverlayView>(ai.myapp.R.id.overlay)
                    ?.setResults(emptyList(), image.height, image.width, rotation)
            } finally {
                image.close()
                Log.v(TAG, "🗑️ Frame discarded after processing")
            }
        }
    }

    private fun shouldProcessFrame(currentTime: Long): Boolean {
        val timeSinceLastProcess = currentTime - lastProcessedTime.get()
        val isTimeOk = timeSinceLastProcess >= processingInterval
        val isLlmReady = isReady()
        
        Log.v(TAG, "⏰ Time check: ${timeSinceLastProcess}ms >= ${processingInterval}ms = $isTimeOk, LLM ready: $isLlmReady")
        
        return isTimeOk && isLlmReady
    }

    private suspend fun imageProxyToBitmap(image: ImageProxy): Bitmap? = withContext(Dispatchers.IO) {
        try {
            // Get the YUV_420_888 image format
            val yBuffer = image.planes[0].buffer // Y
            val uBuffer = image.planes[1].buffer // U
            val vBuffer = image.planes[2].buffer // V

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)

            // U and V are swapped
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = android.graphics.YuvImage(
                nv21,
                android.graphics.ImageFormat.NV21,
                image.width,
                image.height,
                null
            )

            val out = java.io.ByteArrayOutputStream()
            yuvImage.compressToJpeg(
                android.graphics.Rect(0, 0, image.width, image.height),
                85, // Quality
                out
            )

            val imageBytes = out.toByteArray()
            val bitmap = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            
            // Scale down the bitmap for faster processing
            val scaledBitmap = if (bitmap.width > 640 || bitmap.height > 640) {
                val scale = 640.0f / maxOf(bitmap.width, bitmap.height)
                val matrix = Matrix()
                matrix.setScale(scale, scale)
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, false)
            } else {
                bitmap
            }
            
            if (scaledBitmap != bitmap) {
                bitmap.recycle()
            }
            
            scaledBitmap
        } catch (e: Exception) {
            Log.e(TAG, "Error converting ImageProxy to Bitmap", e)
            null
        }
    }

    private fun parseResponseToDetections(response: String?, imgWidth: Float, imgHeight: Float): List<Detection> {
        if (response.isNullOrEmpty()) {
            Log.w(TAG, "Empty response received")
            return emptyList()
        }

        return try {
            val detections = mutableListOf<Detection>()
            
            // Split by separator for multiple objects
            val objects = response.split("---")
            
            objects.forEach { objText ->
                val lines = objText.trim().split("\n")
                var detected = ""
                var boxStr = ""
                var confidenceStr = ""
                var risk = "low"
                
                lines.forEach { line ->
                    when {
                        line.startsWith("DETECTED:") -> detected = line.substringAfter("DETECTED:").trim()
                        line.startsWith("BOX:") -> boxStr = line.substringAfter("BOX:").trim()
                        line.startsWith("CONFIDENCE:") -> confidenceStr = line.substringAfter("CONFIDENCE:").trim()
                        line.startsWith("RISK:") -> risk = line.substringAfter("RISK:").trim()
                    }
                }
                
                if (detected.isNotEmpty() && boxStr.isNotEmpty()) {
                    // Parse box [left,top,right,bottom]
                    val coords = boxStr.trim(' ', '[', ']').split(",").mapNotNull { it.trim().toFloatOrNull() }
                    if (coords.size == 4) {
                        val left = coords[0] * imgWidth
                        val top = coords[1] * imgHeight
                        val right = coords[2] * imgWidth
                        val bottom = coords[3] * imgHeight
                        
                        val confidenceValue = when (confidenceStr.lowercase()) {
                            "high" -> 0.9f
                            "medium" -> 0.7f
                            "low" -> 0.5f
                            else -> 0.6f
                        }
                        
                        detections.add(
                            Detection(
                                boundingBox = RectF(left, top, right, bottom),
                                label = detected,
                                confidence = confidenceValue,
                                classification = risk.lowercase()
                            )
                        )
                    }
                }
            }
            
            if (detections.isEmpty()) {
                Log.w(TAG, "No valid objects parsed from response")
            }
            detections
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing LLM response: $response", e)
            emptyList()
        }
    }

    fun isReady(): Boolean {
        return isInitialized && llmInferenceTask?.isReady() == true
    }

    fun cleanup() {
        Log.d(TAG, "🧹 Cleaning up GemmaBridge...")
        llmInferenceTask?.cleanup()
        llmInferenceTask = null
        isInitialized = false
    }

    data class Detection(
        val boundingBox: RectF,
        val label: String,
        val confidence: Float,
        val classification: String
    )
}