package ai.myapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.lifecycleScope

import com.google.ai.edge.litert.LlmInference
import com.google.ai.edge.litert.LlmInferenceOptions
import com.google.ai.edge.litert.TextGenerationResult
import com.google.ai.edge.next.NpuDelegatePlugin

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicLong

// Let's use the core LiteRT class directly instead of a custom wrapper
// to make the delegate logic clearer.
class GemmaBridge(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) {
    companion object {
        private const val TAG = "GemmaBridge"
        private const val MODEL_PATH = "gemma-3n-E2B-it-int4.task" // Define model path here
    }

    // This will hold our inference engine instance
    private var llmInference: LlmInference? = null
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
     * Initialize the LLM asynchronously, trying NPU -> GPU -> CPU.
     */
    fun initializeAsync() {
        Log.d(TAG, "🔄 Starting async initialization...")

        lifecycleOwner.lifecycleScope.launch(Dispatchers.Default) {
            try {
                // Attempt to initialize with NPU delegate
                var delegate = LlmInference.Delegate.NPU
                Log.d(TAG, "🚀 Attempting to initialize with NPU delegate...")
                llmInference = createLlmInference(delegate)

                // If NPU fails, fall back to GPU
                if (llmInference == null) {
                    Log.w(TAG, "⚠️ NPU initialization failed. Falling back to GPU.")
                    delegate = LlmInference.Delegate.GPU
                    Log.d(TAG, "🚀 Attempting to initialize with GPU delegate...")
                    llmInference = createLlmInference(delegate)
                }

                // If GPU also fails, fall back to CPU
                if (llmInference == null) {
                    Log.w(TAG, "⚠️ GPU initialization failed. Falling back to CPU.")
                    delegate = LlmInference.Delegate.CPU
                    Log.d(TAG, "🔧 Attempting to initialize with CPU delegate...")
                    llmInference = createLlmInference(delegate)
                }

                // Final check
                if (llmInference != null) {
                    isInitialized = true
                    Log.d(TAG, "✅ GemmaBridge initialization successful with [${delegate.name}] delegate!")
                    withContext(Dispatchers.Main) {
                        (context as? MainActivity)?.onModelInitialized()
                    }
                } else {
                    isInitialized = false
                    Log.e(TAG, "❌ GemmaBridge initialization failed on all delegates.")
                    withContext(Dispatchers.Main) {
                        (context as? MainActivity)?.onModelInitializationFailed("Failed to load model on NPU, GPU, or CPU.")
                    }
                }
            } catch (e: Exception) {
                isInitialized = false
                Log.e(TAG, "❌ Initialization failed with exception!", e)
                withContext(Dispatchers.Main) {
                    (context as? MainActivity)?.onModelInitializationFailed("Error during model initialization: ${e.message}")
                }
            }
        }
    }

    /**
     * Helper function to create an LlmInference instance with specific options.
     */
    private suspend fun createLlmInference(delegate: LlmInference.Delegate): LlmInference? {
        return try {
            val optionsBuilder = LlmInferenceOptions.builder()
                .setModelPath(MODEL_PATH)
                .setDelegate(delegate)

            // VERY IMPORTANT: For the NPU delegate, you must register the plugin.
            if (delegate == LlmInference.Delegate.NPU) {
                optionsBuilder.addPlugin(NpuDelegatePlugin())
            }

            withContext(Dispatchers.IO) {
                LlmInference.create(context, optionsBuilder.build())
            }
        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to create LlmInference with ${delegate.name} delegate", e)
            null
        }
    }


    suspend fun analyzeScene(bitmap: Bitmap, prompt: String): String? {
        if (!isReady()) {
            Log.w(TAG, "LLM not ready, cannot analyze scene.")
            return null
        }
        return try {
            // The new API takes a list of Bitmaps
            val result: TextGenerationResult = llmInference!!.generateResponse(listOf(bitmap), prompt)
            result.text()
        } catch (e: Exception) {
            Log.e(TAG, "Error during inference", e)
            null
        }
    }

    fun processFrame(image: ImageProxy) {
        val currentTime = System.currentTimeMillis()

        if (currentTime - lastProcessedTime.get() < processingInterval || !isReady()) {
            image.close()
            return
        }
        lastProcessedTime.set(currentTime)

        val rotation = image.imageInfo.rotationDegrees

        lifecycleOwner.lifecycleScope.launch {
            try {
                // The image conversion remains a potential bottleneck, but let's focus on the model first.
                val bitmap = imageProxyToBitmap(image)

                if (bitmap != null) {
                    val prompt = "Analyze this camera feed for people, objects, and safety concerns. For each detected object, output in this exact format:\nDETECTED: name\nBOX: [left,top,right,bottom] (normalized 0-1 from left-top)\nCONFIDENCE: high/medium/low\nRISK: low/medium/high/critical\nSeparate multiple objects with --- Be concise and only output the formatted text."

                    // Call our new analyzeScene method
                    val response = analyzeScene(bitmap, prompt)

                    val detections = parseResponseToDetections(response, bitmap.width.toFloat(), bitmap.height.toFloat())

                    // Update UI on main thread
                    withContext(Dispatchers.Main) {
                        (context as? MainActivity)?.findViewById<OverlayView>(R.id.overlay)
                            ?.setResults(detections, bitmap.height, bitmap.width, rotation)
                    }
                    // It's good practice to recycle the bitmap if you created a scaled copy
                    if (!bitmap.isRecycled) {
                        bitmap.recycle()
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error processing frame", e)
            } finally {
                image.close()
            }
        }
    }

    // *** Your other methods (imageProxyToBitmap, parseResponseToDetections, etc.) can remain mostly the same ***
    // They are well-written for their purpose.
    // I've included them here for completeness.

    private suspend fun imageProxyToBitmap(image: ImageProxy): Bitmap? = withContext(Dispatchers.IO) {
        try {
            val yBuffer = image.planes[0].buffer // Y
            val uBuffer = image.planes[1].buffer // U
            val vBuffer = image.planes[2].buffer // V
            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()
            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)
            val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, image.width, image.height, null)
            val out = java.io.ByteArrayOutputStream()
            yuvImage.compressToJpeg(android.graphics.Rect(0, 0, image.width, image.height), 85, out)
            val imageBytes = out.toByteArray()
            val bitmap = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            val scale = 640.0f / maxOf(bitmap.width, bitmap.height)
            if (scale < 1.0) {
                val matrix = Matrix()
                matrix.setScale(scale, scale)
                val scaledBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                bitmap.recycle() // Recycle the original large bitmap
                scaledBitmap
            } else {
                bitmap
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error converting ImageProxy to Bitmap", e)
            null
        }
    }

    private fun parseResponseToDetections(response: String?, imgWidth: Float, imgHeight: Float): List<Detection> {
        // This function is fine as-is
        if (response.isNullOrEmpty()) {
            return emptyList()
        }
        return try {
            val detections = mutableListOf<Detection>()
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
            detections
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing LLM response: $response", e)
            emptyList()
        }
    }

    fun isReady(): Boolean {
        return isInitialized && llmInference != null
    }

    fun cleanup() {
        Log.d(TAG, "🧹 Cleaning up GemmaBridge...")
        llmInference?.close()
        llmInference = null
        isInitialized = false
    }

    data class Detection(
        val boundingBox: RectF,
        val label: String,
        val confidence: Float,
        val classification: String
    )
}