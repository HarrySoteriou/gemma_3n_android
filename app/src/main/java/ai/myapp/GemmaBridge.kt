package ai.myapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.lifecycleScope

// Updated imports for LiteRT Next
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.ModelProvider

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

    // Updated for LiteRT Next
    private var compiledModel: CompiledModel? = null
    private var inputBuffers: List<com.google.ai.edge.litert.TensorBuffer>? = null
    private var outputBuffers: List<com.google.ai.edge.litert.TensorBuffer>? = null
    private val lastProcessedTime = AtomicLong(0)
    private val processingInterval = 400L // ~2.5 FPS
    private var isInitialized = false
    private var currentAccelerator: Accelerator? = null
    private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")


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
        Log.d(TAG, "🔄 Starting LiteRT Next initialization...")

        lifecycleOwner.lifecycleScope.launch(Dispatchers.Default) {
            try {
                // Create environment with NPU support
                val env = Environment.create(BuiltinNpuAcceleratorProvider(context))

                // Try accelerators in order: GPU -> NPU -> CPU
                val accelerators = listOf(Accelerator.GPU, Accelerator.NPU, Accelerator.CPU)
                
                for (accelerator in accelerators) {
                    try {
                        Log.d(TAG, "🚀 Attempting initialization with ${accelerator.name} accelerator...")
                        
                        withContext(singleThreadDispatcher) {
                            // Create model with specific accelerator
                            val model = CompiledModel.create(
                                context.assets,
                                MODEL_PATH,
                                CompiledModel.Options(accelerator),
                                env
                            )
                            
                            // Create input/output buffers
                            val inputs = model.createInputBuffers()
                            val outputs = model.createOutputBuffers()
                            
                            // If we get here, initialization succeeded
                            compiledModel = model
                            inputBuffers = inputs
                            outputBuffers = outputs
                            currentAccelerator = accelerator
                            isInitialized = true
                            
                            Log.d(TAG, "✅ LiteRT Next initialization successful with ${accelerator.name}!")
                        }
                        
                        // Success - notify UI and break the loop
                        withContext(Dispatchers.Main) {
                            (context as? MainActivity)?.onModelInitialized()
                        }
                        return@launch
                        
                    } catch (e: Exception) {
                        Log.w(TAG, "⚠️ ${accelerator.name} initialization failed: ${e.message}")
                        // Continue to next accelerator
                    }
                }
                
                // If we get here, all accelerators failed
                isInitialized = false
                Log.e(TAG, "❌ All accelerators failed")
                withContext(Dispatchers.Main) {
                    (context as? MainActivity)?.onModelInitializationFailed("Failed to initialize with any accelerator")
                }
                
            } catch (e: Exception) {
                isInitialized = false
                Log.e(TAG, "❌ Initialization failed with exception!", e)
                withContext(Dispatchers.Main) {
                    (context as? MainActivity)?.onModelInitializationFailed("Error: ${e.message}")
                }
            }
        }
    }

    suspend fun analyzeScene(bitmap: Bitmap, prompt: String): String? {
        if (!isReady()) {
            Log.w(TAG, "LLM not ready, cannot analyze scene.")
            return null
        }
        return try {
            withContext(singleThreadDispatcher) {
                Log.d(TAG, "🔄 Analyzing scene with ${currentAccelerator?.name ?: "unknown"} accelerator...")
                
                val model = compiledModel!!
                val inputs = inputBuffers!!
                val outputs = outputBuffers!!
                
                // For multimodal LLM, you'll need to write both text and image data
                // This depends on your specific model's input format
                // You may need to adapt this based on your model's expected input format
                
                // Example - adapt based on your model:
                // inputs[0].writeString(prompt)  // Text input
                // inputs[1].writeBitmap(bitmap)  // Image input
                
                // For now, let's assume a single input that combines text and image
                // You'll need to adapt this to your specific model format
                inputs[0].writeString(prompt) // This may need modification
                
                // Run inference
                model.run(inputs, outputs)
                
                // Read output
                val result = outputs[0].readString() // This may need modification
                
                Log.d(TAG, "✅ Scene analysis completed with ${currentAccelerator?.name}")
                result
            }
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error during inference with ${currentAccelerator?.name}", e)
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
                Log.v(TAG, "🖼️ Processing frame with ${currentAccelerator?.name ?: "unknown"}")
                
                val bitmap = imageProxyToBitmap(image)
                if (bitmap != null) {
                    val prompt = "Analyze this camera feed for people, objects, and safety concerns. For each detected object, output in this exact format:\nDETECTED: name\nBOX: [left,top,right,bottom] (normalized 0-1 from left-top)\nCONFIDENCE: high/medium/low\nRISK: low/medium/high/critical\nSeparate multiple objects with --- Be concise and only output the formatted text."

                    val response = analyzeScene(bitmap, prompt)
                    val detections = parseResponseToDetections(response, bitmap.width.toFloat(), bitmap.height.toFloat())

                    Log.v(TAG, "🎯 Found ${detections.size} detections using ${currentAccelerator?.name}")

                    withContext(Dispatchers.Main) {
                        (context as? MainActivity)?.findViewById<OverlayView>(R.id.overlay)
                            ?.setResults(detections, bitmap.height, bitmap.width, rotation)
                    }
                    
                    if (!bitmap.isRecycled) {
                        bitmap.recycle()
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "❌ Error processing frame", e)
            } finally {
                image.close()
            }
        }
    }

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
        val ready = isInitialized && compiledModel != null && inputBuffers != null && outputBuffers != null
        if (!ready) {
            Log.v(TAG, "🔍 Model not ready - initialized: $isInitialized, accelerator: ${currentAccelerator?.name ?: "none"}")
        }
        return ready
    }

    suspend fun cleanup() {
        withContext(singleThreadDispatcher) {
            Log.d(TAG, "🧹 Cleaning up GemmaBridge...")
            inputBuffers?.forEach { it.close() }
            outputBuffers?.forEach { it.close() }
            compiledModel?.close()
            
            inputBuffers = null
            outputBuffers = null
            compiledModel = null
            isInitialized = false
            currentAccelerator = null
        }
    }

    data class Detection(
        val boundingBox: RectF,
        val label: String,
        val confidence: Float,
        val classification: String
    )
}