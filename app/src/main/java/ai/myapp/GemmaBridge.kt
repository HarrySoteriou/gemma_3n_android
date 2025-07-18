package ai.myapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageProxy
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicLong

class GemmaBridge(private val context: Context) {

    companion object {
        private const val TAG = "GemmaBridge"
    }

    private val llmInferenceTask: LLMInferenceTask
    private val processingScope: CoroutineScope
    private val lastProcessedTime = AtomicLong(0)
    private val processingInterval = 2000L // Process every 2 seconds

    init {
        Log.e(TAG, "🚀 GemmaBridge constructor started")
        
        Log.e(TAG, "🔄 Creating LLMInferenceTask...")
        llmInferenceTask = LLMInferenceTask(context)
        Log.e(TAG, "✅ LLMInferenceTask created")
        
        Log.e(TAG, "🔄 Creating CoroutineScope...")
        Log.e(TAG, "🔄 Creating SupervisorJob...")
        val supervisorJob = SupervisorJob()
        Log.e(TAG, "✅ SupervisorJob created")
        
        Log.e(TAG, "🔄 Getting Dispatchers.IO...")
        val ioDispatcher = Dispatchers.IO
        Log.e(TAG, "✅ Dispatchers.IO obtained")
        
        Log.e(TAG, "🔄 Combining job and dispatcher...")
        val combinedContext = supervisorJob + ioDispatcher
        Log.e(TAG, "✅ Context combined")
        
        Log.e(TAG, "🔄 Creating CoroutineScope with combined context...")
        processingScope = CoroutineScope(combinedContext)
        Log.e(TAG, "✅ CoroutineScope created successfully")
        
        Log.e(TAG, "🔄 Testing coroutine launch...")
        // TEMPORARY: Skip LLM initialization to test if coroutines work
        try {
            processingScope.launch {
                Log.e(TAG, "🎯 INSIDE COROUTINE - This proves coroutines work!")
                // Temporarily skip: llmInferenceTask.initializeModel()
                Log.e(TAG, "🏁 Test coroutine completed successfully")
            }
            Log.e(TAG, "✅ Coroutine launched successfully!")
        } catch (e: Exception) {
            Log.e(TAG, "❌ Coroutine launch failed!", e)
        }
        Log.e(TAG, "✅ Constructor finishing...")
    }

    fun processFrame(image: ImageProxy) {
        val currentTime = System.currentTimeMillis()
        
        // Skip processing if we're too busy or processed recently
        if (!shouldProcessFrame(currentTime)) {
            Log.v(TAG, "⏭️ Skipping frame processing (too frequent or LLM not ready)")
            image.close()
            return
        }

        Log.d(TAG, "🖼️ Processing new camera frame")
        // Update last processed time
        lastProcessedTime.set(currentTime)

        // Process the frame asynchronously
        processingScope.launch {
            try {
                val bitmap = imageProxyToBitmap(image)
                if (bitmap != null && llmInferenceTask.isReady()) {
                    val response = llmInferenceTask.analyzeScene(bitmap, 
                        "Analyze this camera feed for people, objects, and safety concerns. Be concise.")
                    
                    // Parse the LLM response and create detections
                    val detections = parseResponseToDetections(response)
                    
                    // Update UI on main thread
                    (context as? MainActivity)?.runOnUiThread {
                        Log.d(TAG, "📱 Updating UI with ${detections.size} detections")
                        context.findViewById<OverlayView>(ai.myapp.R.id.overlay)?.setDetections(detections)
                    }
                } else {
                    // Fallback to simulated detections if LLM not ready
                    Log.d(TAG, "🤖 LLM not ready, using simulated detections")
                    val detections = getSimulatedDetections()
                    (context as? MainActivity)?.runOnUiThread {
                        Log.d(TAG, "📱 Updating UI with ${detections.size} simulated detections")
                        context.findViewById<OverlayView>(ai.myapp.R.id.overlay)?.setDetections(detections)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error processing frame", e)
            } finally {
                image.close()
            }
        }
    }

    private fun shouldProcessFrame(currentTime: Long): Boolean {
        val timeSinceLastProcess = currentTime - lastProcessedTime.get()
        val isTimeOk = timeSinceLastProcess >= processingInterval
        val isLlmReady = llmInferenceTask.isReady()
        
        Log.v(TAG, "⏰ Time check: ${timeSinceLastProcess}ms >= ${processingInterval}ms = $isTimeOk, LLM ready: $isLlmReady")
        
        return isTimeOk && isLlmReady
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        return try {
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

    private fun parseResponseToDetections(response: String?): List<Detection> {
        if (response.isNullOrEmpty()) {
            Log.w(TAG, "Empty response received")
            return getSimulatedDetections()
        }

        return try {
            val detections = mutableListOf<Detection>()
            
            // Parse the structured response from LLM
            val lines = response.split("\n")
            var detected = ""
            var risk = "low"
            var action = ""
            var confidence = "medium"
            
            lines.forEach { line ->
                when {
                    line.startsWith("DETECTED:") -> detected = line.substringAfter("DETECTED:").trim()
                    line.startsWith("RISK:") -> risk = line.substringAfter("RISK:").trim()
                    line.startsWith("ACTION:") -> action = line.substringAfter("ACTION:").trim()
                    line.startsWith("CONFIDENCE:") -> confidence = line.substringAfter("CONFIDENCE:").trim()
                }
            }
            
            Log.d(TAG, "Parsed - Detected: $detected, Risk: $risk, Action: $action, Confidence: $confidence")
            
            // Create detection based on LLM analysis
            if (detected.isNotEmpty()) {
                val confidenceValue = when (confidence.lowercase()) {
                    "high" -> 0.9f
                    "medium" -> 0.7f
                    "low" -> 0.5f
                    else -> 0.6f
                }
                
                detections.add(
                    Detection(
                        boundingBox = RectF(50f, 100f, 550f, 400f), // Center area of screen
                        label = detected,
                        confidence = confidenceValue,
                        classification = risk.lowercase()
                    )
                )
                
                // Add action as separate detection if important
                if (action.isNotEmpty() && action.lowercase() != "none") {
                    detections.add(
                        Detection(
                            boundingBox = RectF(50f, 450f, 550f, 550f), // Bottom area
                            label = "Action: $action",
                            confidence = confidenceValue,
                            classification = when {
                                action.contains("urgent", ignoreCase = true) -> "critical"
                                action.contains("caution", ignoreCase = true) -> "high"
                                else -> "medium"
                            }
                        )
                    )
                }
            } else {
                Log.w(TAG, "No objects detected in response")
                return getSimulatedDetections()
            }
            
            detections
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing LLM response: $response", e)
            getSimulatedDetections()
        }
    }

    private fun getSimulatedDetections(): List<Detection> {
        return listOf(
            Detection(
                boundingBox = RectF(100f, 200f, 500f, 400f),
                label = "Initializing Analysis...",
                confidence = 0.75f,
                classification = "medium"
            )
        )
    }

    fun cleanup() {
        llmInferenceTask.cleanup()
    }

    data class Detection(
        val boundingBox: RectF,
        val label: String,
        val confidence: Float,
        val classification: String
    )
}