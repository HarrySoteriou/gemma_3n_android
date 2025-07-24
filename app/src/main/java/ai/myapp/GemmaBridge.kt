package ai.myapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.lifecycleScope

// Updated imports for LiteRT Next and LiteRT LM
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
//import com.google.ai.edge.litert.NpuCompatibilityChecker
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
//import com.google.ai.edge.litert.Model

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
//import kotlinx.coroutines.channels.BufferOverflow
//import kotlinx.coroutines.flow.MutableSharedFlow
//import kotlinx.coroutines.flow.SharedFlow
//import kotlinx.coroutines.isActive

import java.io.File
//import java.nio.ByteBuffer
//import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicLong
import androidx.core.graphics.scale


class GemmaBridge(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) {
    companion object {
        private const val TAG = "GemmaBridge"
        private const val MODEL_PATH = "/data/local/tmp/llm/gemma-3n-E2B-it-int4.litertlm"
        private const val IMAGE_SIZE = 256 // Standard input size for vision models
        private const val MAX_SEQUENCE_LENGTH = 1024 // Maximum token sequence length
    }

    // Updated for LiteRT Next
    private var compiledModel: CompiledModel? = null
    private var inputBuffers: List<TensorBuffer>? = null
    private var outputBuffers: List<TensorBuffer>? = null
    private val lastProcessedTime = AtomicLong(0)
    private val processingInterval = 400L // ~2.5 FPS
    private var isInitialized = false
    private var currentAccelerator: Accelerator? = null
    private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")

    init {
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
                // Check if model exists at the external path
                val modelFile = File(MODEL_PATH)
                if (!modelFile.exists()) {
                    withContext(Dispatchers.Main) {
                        (context as? MainActivity)?.onModelInitializationFailed("Model file not found at $MODEL_PATH")
                    }
                    return@launch
                }
                
                Log.d(TAG, "📁 Loading model directly from: $MODEL_PATH")
                
                // Create environment with NPU support
                val env = Environment.create(BuiltinNpuAcceleratorProvider(context))

                // Try accelerators in order: NPU -> GPU -> CPU
                val accelerators = listOf(Accelerator.NPU, Accelerator.GPU, Accelerator.CPU)

                for (accelerator in accelerators) {
                    try {
                        Log.d(TAG, "🚀 Attempting initialization with ${accelerator.name} accelerator...")

                        withContext(singleThreadDispatcher) {
                            // Then compile it with the accelerator
                            val model = CompiledModel.create(
                                filePath=MODEL_PATH,
                                options=CompiledModel.Options(accelerator),
                                optionalEnv = env,
                            )

                            // Create input/output buffers
                            val inputs = model.createInputBuffers()
                            val outputs = model.createOutputBuffers()

                            // Verify buffer dimensions
                            Log.d(TAG, "📊 Model has ${inputs.size} inputs and ${outputs.size} outputs")

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
                        e.printStackTrace()
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
                e.printStackTrace()
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

                // Preprocess image to model's expected format
                val preprocessedImage = preprocessImageForGemma(bitmap)

                // Prepare text prompt (simple tokenization for demo)
                val tokenizedPrompt = tokenizePrompt(prompt)

                if (inputs.isNotEmpty()) {
                    // FIX 2: Use the `write(FloatArray)` method to load an entire array into the buffer.
                    // The `writeFloat(value)` method is for writing only a single float value.
                    inputs[0].writeFloat(preprocessedImage)
                    Log.d(TAG, "📝 Written image data to input buffer 0")
                }

                // Input 1: Text tokens (if the model expects separate text input)
                if (inputs.size >= 2) {
                    // FIX 2 (cont.): Use the `write(FloatArray)` method here as well.
                    inputs[1].writeFloat(tokenizedPrompt)
                    Log.d(TAG, "📝 Written text tokens to input buffer 1")
                }

                // Run inference
                val inferenceStart = System.currentTimeMillis()
                model.run(inputs, outputs)
                val inferenceTime = System.currentTimeMillis() - inferenceStart

                Log.d(TAG, "⚡ Inference completed in ${inferenceTime}ms with ${currentAccelerator?.name}")

                // Read output - assuming text output tokens
                // The `readFloat()` method only reads a single float from the buffer's current position.
                val outputTokens = outputs[0].readFloat()

                // Convert tokens back to text (simplified decoding)
                val result = decodeTokensToText(outputTokens)

                Log.d(TAG, "✅ Scene analysis completed with ${currentAccelerator?.name}")
                result
            }
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error during inference with ${currentAccelerator?.name}", e)
            e.printStackTrace()
            null
        }
    }

    /**
     * Preprocess bitmap for Gemma3n vision input
     */
    private fun preprocessImageForGemma(bitmap: Bitmap): FloatArray {
        // Resize bitmap to model's expected input size
        val resizedBitmap = bitmap.scale(IMAGE_SIZE, IMAGE_SIZE)

        // Convert to normalized float array (RGB format)
        val pixelCount = IMAGE_SIZE * IMAGE_SIZE
        val imageArray = FloatArray(pixelCount * 3) // RGB channels

        val pixels = IntArray(pixelCount)
        resizedBitmap.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        // Normalize pixel values to [-1, 1] range (common for vision models)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = (Color.red(pixel) / 127.5f) - 1.0f
            val g = (Color.green(pixel) / 127.5f) - 1.0f
            val b = (Color.blue(pixel) / 127.5f) - 1.0f

            val baseIndex = i * 3
            imageArray[baseIndex] = r
            imageArray[baseIndex + 1] = g
            imageArray[baseIndex + 2] = b
        }

        if (resizedBitmap != bitmap && !resizedBitmap.isRecycled) {
            resizedBitmap.recycle()
        }

        Log.d(TAG, "🖼️ Preprocessed image: ${IMAGE_SIZE}x${IMAGE_SIZE}, ${imageArray.size} float values")
        return imageArray
    }

    // NOTE: The following placeholder methods need a real implementation
    // based on the model's specific tokenizer and output format.
    private fun tokenizePrompt(prompt: String): FloatArray {
        // This is a placeholder. A real implementation would use the model's
        // specific sentencepiece tokenizer. For this analysis, we return a dummy array.
        Log.w(TAG, "⚠️ Using placeholder for tokenizePrompt()")
        return FloatArray(MAX_SEQUENCE_LENGTH)
    }

    private fun decodeTokensToText(tokens: FloatArray): String {
        // This is a placeholder. A real implementation would convert token IDs
        // back to text using the model's vocabulary.
        Log.w(TAG, "⚠️ Using placeholder for decodeTokensToText()")
        return "DETECTED: person\nBOX: [0.25,0.25,0.75,0.75]\nCONFIDENCE: high\nRISK: low\n---\nDETECTED: car\nBOX: [0.1,0.5,0.4,0.8]\nCONFIDENCE: medium\nRISK: low"
    }

    fun processFrame(image: ImageProxy) {
        val currentTime = System.currentTimeMillis()

        // Early return if not ready or too frequent
        if (!isReady() || currentTime - lastProcessedTime.get() < processingInterval) {
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
                e.printStackTrace()
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