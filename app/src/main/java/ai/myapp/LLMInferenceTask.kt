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
    
    init {
        Log.d(TAG, "🔄 LLMInferenceTask constructor started")
        Log.d(TAG, "✅ LLMInferenceTask constructor completed")
    }

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

    /** Initialize using official MediaPipe pattern */
    suspend fun initializeModel() = withContext(Dispatchers.IO) {
        if (isInitialized.get()) {
            Log.d(TAG, "✅ Model already initialized, skipping...")
            return@withContext
        }
        
        Log.d(TAG, "🔄 Starting LLM initialization process...")
        Log.d(TAG, "🔄 Looking for model: $GEMMA_MODEL")
        
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
                isInitialized.set(false)
                return@withContext
            }
            
            Log.i(TAG, "🔄 Initializing LLM with model: $modelPath")
            
            // Follow official MediaPipe pattern
            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelPath)
                .setMaxTokens(512)
                .setMaxNumImages(1) // Set to 1 for vision support
                .build()

            // Create the LLM inference engine
            val newLlmInference = LlmInference.createFromOptions(context, options)
            
            // Only set the instance and flag if creation was successful
            llmInference = newLlmInference
            isInitialized.set(true)
            
            Log.d(TAG, "🚀 LLM SUCCESSFULLY INITIALIZED AND READY FOR INFERENCE!")
            Log.d(TAG, "🚀 Model path: $modelPath")
            Log.d(TAG, "🚀 Max tokens: 512")
            Log.d(TAG, "🚀 Vision support: enabled")
            
        } catch (t: Throwable) {
            Log.e(TAG, "❌ LLM initialization failed!", t)
            Log.e(TAG, "❌ Error type: ${t.javaClass.simpleName}")
            Log.e(TAG, "❌ Error details: ${t.message}")
            
            // Clean up any partially initialized state
            llmInference?.let { 
                runCatching { it.close() }
                llmInference = null
            }
            isInitialized.set(false)
            
            throw t // Re-throw to allow caller to handle
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

        // ❶ Check if model is properly initialized before proceeding
        if (!isInitialized.get()) {
            Log.w(TAG, "⚠️ Cannot analyze scene: Model not initialized yet")
            return@withContext null
        }

        val engine = llmInference ?: run {
            Log.e(TAG, "❌ Cannot analyze scene: LLM engine is null despite initialization flag")
            return@withContext null
        }

        // ❷ only one request at a time
        if (!isProcessing.compareAndSet(false, true)) {
            Log.w(TAG, "⚠️ Cannot analyze scene: Already processing another request")
            return@withContext null
        }
        
        try {
            Log.d(TAG, "🔄 Starting scene analysis...")
            
            // ❃ create a *new* session for each call
            val sessionOptions = LlmInferenceSession.LlmInferenceSessionOptions
                .builder()
                .setTopK(40)
                .setTemperature(0.7f)
                .setGraphOptions(                         // ❹ turn on vision
                    GraphOptions.builder()
                        .setEnableVisionModality(true)
                        .build()
                )
                .build()

            LlmInferenceSession.createFromOptions(engine, sessionOptions).use { session ->
                session.addQueryChunk(prompt)             // ❺ text first
                session.addImage(BitmapImageBuilder(bitmap).build()) // ❻ image second
                val result = session.generateResponse()  // blocking version
                Log.d(TAG, "✅ Scene analysis completed successfully")
                return@withContext result
            }
        } catch (t: Throwable) {
            Log.e(TAG, "❌ Scene analysis failed", t)
            return@withContext "DETECTED: error\nRISK: medium\nACTION: check logs\nCONFIDENCE: low"
        } finally {
            isProcessing.set(false)
        }
    }

    fun isReady(): Boolean {
        val initialized = isInitialized.get()
        val processing = isProcessing.get()
        val engineAvailable = llmInference != null
        val ready = initialized && !processing && engineAvailable
        
        Log.v(TAG, "🔍 isReady() check: initialized=$initialized, processing=$processing, engineAvailable=$engineAvailable, ready=$ready")
        
        // Additional validation - if flags are inconsistent, fix them
        if (initialized && !engineAvailable) {
            Log.w(TAG, "⚠️ Inconsistent state detected: initialized=true but engine=null, fixing...")
            isInitialized.set(false)
            return false
        }
        
        return ready
    }

    /**
     * Check if the model file is available without attempting to load it
     */
    fun isModelFileAvailable(): Boolean {
        val internalFile = File(context.filesDir, GEMMA_MODEL)
        if (internalFile.exists()) {
            Log.d(TAG, "✅ Model file found in internal storage")
            return true
        }
        
        // Check external locations
        val externalPaths = getExternalModelPaths()
        val externalLocations = externalPaths.map { File(it) } + listOf(
            File("/sdcard/Download/", GEMMA_MODEL),
            File("/sdcard/", GEMMA_MODEL),
            File("/sdcard/Android/data/${context.packageName}/files/", GEMMA_MODEL),
            File(context.getExternalFilesDir(null), GEMMA_MODEL),
            File(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), GEMMA_MODEL)
        )
        
        for (location in externalLocations) {
            if (location.exists()) {
                Log.d(TAG, "✅ Model file found at: ${location.absolutePath}")
                return true
            }
        }
        
        Log.w(TAG, "❌ Model file not found in any expected location")
        return false
    }

    fun cleanup() {
        runCatching { llmInference?.close() }
        isInitialized.set(false)
    }
}