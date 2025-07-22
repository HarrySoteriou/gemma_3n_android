package ai.myapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.lifecycleScope

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.core.graphics.scale

/**
 * A modern inference task handler using the LiteRT Next API.
 * This class replaces the previous MediaPipe-based implementation.
 *
 * Key changes:
 * - Uses LiteRT Next for inference (NPU -> GPU -> CPU).
 * - Loads the model directly from the app's 'assets' folder, which is more reliable
 *   and requires no storage permissions.
 * - Removes all manual native library loading and file-searching logic.
 */
class LLMInferenceTask(
    private val context: Context,
    // Providing a LifecycleOwner allows for safe coroutine management.
    private val lifecycleOwner: LifecycleOwner,
    // Optional callbacks for UI updates.
    private val onInitialized: (() -> Unit)? = null,
    private val onInitializationFailed: ((String) -> Unit)? = null
) {
    companion object {
        private const val TAG = "LLMInferenceTask"
        // The model file MUST be in the 'src/main/assets' folder.
        private const val MODEL_PATH = "gemma-3n-E2B-it-int4.task"
        private const val IMAGE_SIZE = 256 // Standard input size for vision models
    }

    private var compiledModel: CompiledModel? = null
    private var inputBuffers: List<TensorBuffer>? = null
    private var outputBuffers: List<TensorBuffer>? = null
    private var isInitialized = false
    private var currentAccelerator: Accelerator? = null
    // A dedicated single-threaded dispatcher ensures that model operations
    // (initialization, inference) run sequentially and off the main thread.
    private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "LLM-Dispatcher")

    /**
     * Initializes the inference engine asynchronously.
     * This method tries to use the best available hardware accelerator in order: NPU -> GPU -> CPU.
     */
    fun initialize() {
        Log.d(TAG, "🔄 Starting LiteRT Next initialization...")
        // Show loading state on the main thread if possible
        Handler(Looper.getMainLooper()).post {
            // e.g., (context as? MainActivity)?.showLoading()
        }

        lifecycleOwner.lifecycleScope.launch(Dispatchers.Default) {
            try {
                // Create an environment with NPU support if available.
                val env = Environment.create(BuiltinNpuAcceleratorProvider(context))
                val accelerators = listOf(Accelerator.NPU, Accelerator.GPU, Accelerator.CPU)

                for (accelerator in accelerators) {
                    try {
                        Log.d(TAG, "🚀 Attempting initialization with ${accelerator.name}...")
                        // All model operations are dispatched to our single-threaded context.
                        withContext(singleThreadDispatcher) {
                            // Load the model from the app's assets.
                            val compiledModelInstance = CompiledModel.create(
                                context.assets,
                                MODEL_PATH,
                                CompiledModel.Options(accelerator),
                                env
                            )

                            // If we reach here, initialization with this accelerator was successful.
                            this@LLMInferenceTask.compiledModel = compiledModelInstance
                            inputBuffers = compiledModelInstance.createInputBuffers()
                            outputBuffers = compiledModelInstance.createOutputBuffers()

                            currentAccelerator = accelerator
                            isInitialized = true
                        }
                        Log.d(TAG, "✅ LiteRT Next initialization successful with ${accelerator.name}!")
                        // Notify UI of success on the main thread and exit the loop.
                        withContext(Dispatchers.Main) { onInitialized?.invoke() }
                        return@launch

                    } catch (e: Exception) {
                        Log.w(TAG, "⚠️ ${accelerator.name} initialization failed: ${e.message}")
                        // Clean up any partial state before trying the next accelerator
                        cleanupInternal()
                    }
                }

                // If the loop completes without returning, all accelerators have failed.
                Log.e(TAG, "❌ All accelerators failed to initialize.")
                withContext(Dispatchers.Main) { onInitializationFailed?.invoke("Failed to initialize with any accelerator.") }

            } catch (e: Exception) {
                isInitialized = false
                Log.e(TAG, "❌ A critical error occurred during initialization!", e)
                withContext(Dispatchers.Main) { onInitializationFailed?.invoke("Fatal Error: ${e.message}") }
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
                Log.d(TAG, "🔄 Analyzing scene with ${currentAccelerator?.name ?: "unknown"}...")

                val model = compiledModel!!
                val inputs = inputBuffers!!
                val outputs = outputBuffers!!

                // Preprocess image and tokenize prompt (these need real implementations)
                val preprocessedImage = preprocessImage(bitmap)
                val tokenizedPrompt = tokenizePrompt(prompt)

                // Load data into the input buffers
                inputs.getOrNull(0)?.writeFloat(preprocessedImage)
                inputs.getOrNull(1)?.writeFloat(tokenizedPrompt)

                // Run inference
                val inferenceStart = System.currentTimeMillis()
                model.run(inputs, outputs)
                val inferenceTime = System.currentTimeMillis() - inferenceStart
                Log.d(TAG, "⚡ Inference completed in ${inferenceTime}ms with ${currentAccelerator?.name}")

                // Read and decode the output
                val outputTokens = outputs[0].readFloat()
                decodeTokensToText(outputTokens)
            }
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error during inference with ${currentAccelerator?.name}", e)
            null
        }
    }

    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        val resizedBitmap = bitmap.scale(IMAGE_SIZE, IMAGE_SIZE)
        val pixelCount = IMAGE_SIZE * IMAGE_SIZE
        val imageArray = FloatArray(pixelCount * 3)
        val pixels = IntArray(pixelCount)
        resizedBitmap.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            // Normalize pixel values to [-1, 1] range
            imageArray[i * 3] = (Color.red(pixel) / 127.5f) - 1.0f
            imageArray[i * 3 + 1] = (Color.green(pixel) / 127.5f) - 1.0f
            imageArray[i * 3 + 2] = (Color.blue(pixel) / 127.5f) - 1.0f
        }
        if (resizedBitmap != bitmap) resizedBitmap.recycle()
        return imageArray
    }

    // NOTE: These are placeholders and require a real implementation based on your model.
    private fun tokenizePrompt(prompt: String): FloatArray {
        Log.w(TAG, "⚠️ Using placeholder for tokenizePrompt()")
        return FloatArray(1) // Return a dummy array
    }

    private fun decodeTokensToText(tokens: FloatArray): String {
        Log.w(TAG, "⚠️ Using placeholder for decodeTokensToText()")
        // Return a sample response for testing purposes.
        return "DETECTED: person\nBOX: [0.25,0.25,0.75,0.75]\nCONFIDENCE: high\nRISK: low"
    }

    fun isReady(): Boolean = isInitialized && compiledModel != null

    suspend fun cleanup() {
        withContext(singleThreadDispatcher) {
            cleanupInternal()
        }
    }

    private fun cleanupInternal() {
        Log.d(TAG, "🧹 Cleaning up inference engine resources...")
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