package ai.myapp

import android.content.Context
import android.graphics.Bitmap
import android.os.Environment
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.genai.llminference.GraphOptions
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class LLMInferenceTask(private val context: Context) {

    private var llmInference: LlmInference? = null          // ❶ correct type
    private val isInitialized = AtomicBoolean(false)
    private val isProcessing = AtomicBoolean(false)

    companion object {
        private const val TAG = "LLMInferenceTask"
        private const val GEMMA_MODEL = "gemma-3n-E2B-it-int4.task"
        
        // Paths where the model might be located on the device
        private fun getExternalModelPaths(): List<String> {
            return listOf(
                // Modern approach - Downloads folder
                "${Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)}/$GEMMA_MODEL",
                // Alternative - root of external storage
                "${Environment.getExternalStorageDirectory()}/$GEMMA_MODEL",
                // Alternative - Downloads subfolder
                "${Environment.getExternalStorageDirectory()}/Download/$GEMMA_MODEL"
            )
        }
    }

    private suspend fun ensureModelAvailable(): String? = withContext(Dispatchers.IO) {
        val internalModelFile = File(context.filesDir, GEMMA_MODEL)
        
        // If model already exists in internal storage, use it
        if (internalModelFile.exists()) {
            Log.d(TAG, "Model found in internal storage: ${internalModelFile.absolutePath}")
            return@withContext internalModelFile.absolutePath
        }
        
        Log.d(TAG, "Model not in internal storage, looking for external copy...")
        
        // Look for model in external storage locations
        val externalPaths = getExternalModelPaths()
        Log.d(TAG, "Generated external paths:")
        externalPaths.forEach { Log.d(TAG, "  - $it") }
        
        val externalLocations = externalPaths.map { File(it) } + listOf(
            File("/sdcard/Download/", GEMMA_MODEL),
            File("/sdcard/", GEMMA_MODEL),
            File("/sdcard/Android/data/${context.packageName}/files/", GEMMA_MODEL),
            // App-specific external directory (no permissions needed)
            File(context.getExternalFilesDir(null), GEMMA_MODEL),
            File(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), GEMMA_MODEL)
        )
        
        Log.d(TAG, "Checking all possible locations:")
        var sourceFile: File? = null
        for (location in externalLocations) {
            Log.d(TAG, "  Checking: ${location.absolutePath} - exists: ${location.exists()}")
            if (location.exists()) {
                Log.i(TAG, "✓ FOUND MODEL AT: ${location.absolutePath}")
                sourceFile = location
                break
            }
        }
        
        if (sourceFile == null) {
            Log.e(TAG, "Model not found in any expected location.")
            Log.e(TAG, "All checked locations:")
            externalLocations.forEach { Log.e(TAG, "  - ${it.absolutePath} (exists: ${it.exists()})") }
            Log.e(TAG, "Please ensure the model file '$GEMMA_MODEL' is in one of these locations:")
            Log.e(TAG, "  RECOMMENDED (no permissions needed):")
            Log.e(TAG, "    ${context.getExternalFilesDir(null)?.absolutePath}/$GEMMA_MODEL")
            Log.e(TAG, "  Alternative (needs storage permission):")
            Log.e(TAG, "    /sdcard/Download/$GEMMA_MODEL")
            return@withContext null
        }
        
        // Copy model from external to internal storage
        try {
            val sourceSize = sourceFile.length() / 1024 / 1024 // MB
            Log.i(TAG, "📋 Copying model from ${sourceFile.absolutePath}")
            Log.i(TAG, "📋 Size: ${sourceSize}MB - This may take a few minutes...")
            Log.i(TAG, "📋 Destination: ${internalModelFile.absolutePath}")
            
            sourceFile.inputStream().use { input ->
                FileOutputStream(internalModelFile).use { output ->
                    input.copyTo(output)
                }
            }
            
            val finalSize = internalModelFile.length() / 1024 / 1024 // MB
            Log.i(TAG, "✅ MODEL COPIED SUCCESSFULLY!")
            Log.i(TAG, "✅ Location: ${internalModelFile.absolutePath}")
            Log.i(TAG, "✅ Size: ${finalSize}MB")
            return@withContext internalModelFile.absolutePath
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to copy model to internal storage", e)
            return@withContext null
        }
    }

    private fun getModelPath(): String {
        // Check if model exists in internal storage
        val internalFile = File(context.filesDir, GEMMA_MODEL)
        return if (internalFile.exists()) {
            Log.d(TAG, "Model found in internal storage")
            internalFile.absolutePath
        } else {
            Log.d(TAG, "Model not found in internal storage, will need to copy from external storage")
            // Return internal path - ensureModelAvailable() will handle the copying
            internalFile.absolutePath
        }
    }

    /** Call once (e.g. in a ViewModel's init) */
    suspend fun initializeModel() {
        if (isInitialized.get()) {
            Log.e(TAG, "✅ Model already initialized, skipping...")
            return
        }
        
        Log.e(TAG, "🔄 Starting LLM initialization process...")
        Log.e(TAG, "🔄 Looking for model: $GEMMA_MODEL")
        
        try {
            // Ensure model is available in internal storage
            val modelPath = ensureModelAvailable()
            if (modelPath == null) {
                Log.e(TAG, "❌ Cannot initialize LLM: Model file not available")
                Log.e(TAG, "📋 Instructions:")
                Log.e(TAG, "1. Connect your device via USB")
                Log.e(TAG, "2. Copy the model file to one of these locations:")
                Log.e(TAG, "   RECOMMENDED: ${context.getExternalFilesDir(null)?.absolutePath}/$GEMMA_MODEL")
                Log.e(TAG, "   Alternative: /sdcard/Download/$GEMMA_MODEL")
                Log.e(TAG, "3. Grant storage permissions if prompted")
                Log.e(TAG, "4. Restart the app")
                return
            }
            
            Log.i(TAG, "🔄 Initializing LLM with model: $modelPath")
            
            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelPath)
                .setMaxTokens(512)
                .setMaxNumImages(4)
                .build()

            llmInference = LlmInference.createFromOptions(context, options)
            isInitialized.set(true)
            Log.e(TAG, "🚀 LLM SUCCESSFULLY INITIALIZED AND READY FOR INFERENCE!")
            Log.e(TAG, "🚀 Model path: $modelPath")
            Log.e(TAG, "🚀 Max tokens: 512")
            Log.e(TAG, "🚀 Vision support: enabled")
        } catch (t: Throwable) {
            Log.e(TAG, "❌ LLM initialization failed!", t)
            Log.e(TAG, "❌ Error type: ${t.javaClass.simpleName}")
            Log.e(TAG, "❌ Error details: ${t.message}")
            isInitialized.set(false)
        }
    }

    /**
     * Synchronous generation inside a coroutine (Dispatcher.IO).
     * If you want streaming output, switch to generateResponseAsync.
     */
    suspend fun analyzeScene(
        bitmap: Bitmap,
        prompt: String =
            "Analyze this scene, list visible objects and any safety concerns."
    ): String? = withContext(Dispatchers.IO) {

        initializeModel()
        val engine = llmInference ?: return@withContext null

        // only one request at a time
        if (!isProcessing.compareAndSet(false, true)) return@withContext null
        try {
            // --- create a *new* session for each call ---
            val sessionOptions = LlmInferenceSession.LlmInferenceSessionOptions
                .builder()
                .setTopK(40)
                .setTemperature(0.7f)
                .setGraphOptions(                         // ❷ turn on vision
                    GraphOptions.builder()
                        .setEnableVisionModality(true)
                        .build()
                )
                .build()

            LlmInferenceSession.createFromOptions(engine, sessionOptions).use { session ->
                session.addQueryChunk(prompt)             // ❸ text first
                session.addImage(BitmapImageBuilder(bitmap).build()) // ❹ image second
                return@withContext session.generateResponse()        // blocking version
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Scene analysis failed", t)
            "DETECTED: error\nRISK: medium\nACTION: check logs\nCONFIDENCE: low"
        } finally {
            isProcessing.set(false)
        }
    }

    fun isReady(): Boolean {
        val initialized = isInitialized.get()
        val processing = isProcessing.get()
        val ready = initialized && !processing
        Log.v(TAG, "🔍 isReady() check: initialized=$initialized, processing=$processing, ready=$ready")
        return ready
    }

    fun cleanup() {
        runCatching { llmInference?.close() }
        isInitialized.set(false)
    }
}