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
        try {
            Log.d(TAG, "🔍 Context received: ${context.javaClass.simpleName}")
            Log.d(TAG, "🔍 Looking for model file: $GEMMA_MODEL")
            
            // Check if we can access external storage
            val externalStorageState = Environment.getExternalStorageState()
            Log.d(TAG, "🔍 External storage state: $externalStorageState")
            
            Log.d(TAG, "✅ LLMInferenceTask constructor completed")
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error in LLMInferenceTask constructor", e)
            throw e
        }
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
        
        // PRIORITIZE app-specific directories that don't require permissions
        val priorityLocations = listOf(
            // App-specific external directory (no permissions needed)
            File(context.getExternalFilesDir(null), GEMMA_MODEL),
            File(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), GEMMA_MODEL),
            // App-specific directory in Android/data (no permissions needed)
            File("/sdcard/Android/data/${context.packageName}/files/", GEMMA_MODEL)
        )
        
        // Secondary locations that might require permissions
        val externalPaths = getExternalModelPaths()
        val secondaryLocations = externalPaths.map { File(it) } + listOf(
            File("/sdcard/Download/", GEMMA_MODEL),
            File("/sdcard/", GEMMA_MODEL)
        )
        
        // Combine all locations with priority order
        val allLocations = priorityLocations + secondaryLocations
        
        Log.d(TAG, "Checking all possible locations (priority order):")
        var sourceFile: File? = null
        for ((index, location) in allLocations.withIndex()) {
            val isPriority = index < priorityLocations.size
            val exists = location.exists() && location.canRead()
            Log.d(TAG, "  ${if (isPriority) "PRIORITY" else "FALLBACK"}: ${location.absolutePath} - exists: $exists")
            if (exists) {
                Log.i(TAG, "✓ FOUND MODEL AT: ${location.absolutePath}")
                sourceFile = location
                break
            }
        }
        
        if (sourceFile == null) {
            Log.e(TAG, "❌ Model not found in any location!")
            Log.e(TAG, "📋 SOLUTION: Place your model file in one of these locations:")
            Log.e(TAG, "")
            Log.e(TAG, "🎯 RECOMMENDED (no permissions needed):")
            context.getExternalFilesDir(null)?.let { dir ->
                Log.e(TAG, "   ${dir.absolutePath}/$GEMMA_MODEL")
                // Try to create the directory if it doesn't exist
                if (!dir.exists()) {
                    Log.d(TAG, "Creating app-specific directory: ${dir.absolutePath}")
                    dir.mkdirs()
                }
            }
            Log.e(TAG, "   OR: /sdcard/Android/data/${context.packageName}/files/$GEMMA_MODEL")
            Log.e(TAG, "")
            Log.e(TAG, "📱 MANUAL STEPS:")
            Log.e(TAG, "   1. Connect your device via USB")
            Log.e(TAG, "   2. Enable 'File Transfer' mode on your device")
            Log.e(TAG, "   3. Copy $GEMMA_MODEL to one of the recommended paths above")
            Log.e(TAG, "   4. Restart the app")
            Log.e(TAG, "")
            Log.e(TAG, "⚠️  Alternative (requires all files access permission):")
            Log.e(TAG, "   /sdcard/Download/$GEMMA_MODEL")
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
                    input.copyTo(output, bufferSize = 8192)
                }
            }
            
            val finalSize = internalModelFile.length() / 1024 / 1024 // MB
            Log.i(TAG, "✅ MODEL COPIED SUCCESSFULLY!")
            Log.i(TAG, "✅ Location: ${internalModelFile.absolutePath}")
            Log.i(TAG, "✅ Size: ${finalSize}MB")
            return@withContext internalModelFile.absolutePath
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to copy model to internal storage", e)
            Log.e(TAG, "💡 This might be a permission issue. Try:")
            Log.e(TAG, "   1. Grant 'All files access' permission in Settings > Apps > YourApp > Permissions")
            Log.e(TAG, "   2. Or place the model in: ${context.getExternalFilesDir(null)?.absolutePath}/$GEMMA_MODEL")
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
                Log.e(TAG, "")
                Log.e(TAG, "🎯 QUICK FIX: Place your model file here:")
                context.getExternalFilesDir(null)?.let { dir ->
                    if (!dir.exists()) dir.mkdirs() // Ensure directory exists
                    Log.e(TAG, "   ${dir.absolutePath}/$GEMMA_MODEL")
                }
                Log.e(TAG, "")
                Log.e(TAG, "📱 STEPS:")
                Log.e(TAG, "   1. Connect device via USB, enable File Transfer")
                Log.e(TAG, "   2. Navigate to Android/data/ai.myapp/files/ on your device")
                Log.e(TAG, "   3. Copy $GEMMA_MODEL to that folder")
                Log.e(TAG, "   4. Restart the app")
                Log.e(TAG, "")
                Log.e(TAG, "💡 TIP: The 'Android/data/ai.myapp/files' folder doesn't require special permissions!")
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
            Log.e(TAG, "❌ Full stack trace:")
            t.printStackTrace()
            
            // Additional diagnostic info
            Log.e(TAG, "🔍 Diagnostic information:")
            Log.e(TAG, "  - Model name: $GEMMA_MODEL")
            Log.e(TAG, "  - Context: ${context.javaClass.simpleName}")
            Log.e(TAG, "  - Files dir: ${context.filesDir?.absolutePath}")
            Log.e(TAG, "  - External files dir: ${context.getExternalFilesDir(null)?.absolutePath}")
            
            // Clean up any partially initialized state
            llmInference?.let { 
                runCatching { 
                    Log.d(TAG, "🧹 Cleaning up partially initialized LLM engine...")
                    it.close() 
                }
                llmInference = null
            }
            isInitialized.set(false)
            
            // Don't re-throw here, let the caller handle the failure gracefully
            // throw t // Re-throw to allow caller to handle
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
        
        // Check with same priority order as ensureModelAvailable
        val priorityLocations = listOf(
            // App-specific external directory (no permissions needed)
            File(context.getExternalFilesDir(null), GEMMA_MODEL),
            File(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), GEMMA_MODEL),
            // App-specific directory in Android/data (no permissions needed)
            File("/sdcard/Android/data/${context.packageName}/files/", GEMMA_MODEL)
        )
        
        // Check priority locations first
        for (location in priorityLocations) {
            if (location.exists() && location.canRead()) {
                Log.d(TAG, "✅ Model file found at priority location: ${location.absolutePath}")
                return true
            }
        }
        
        // Check secondary locations
        val externalPaths = getExternalModelPaths()
        val secondaryLocations = externalPaths.map { File(it) } + listOf(
            File("/sdcard/Download/", GEMMA_MODEL),
            File("/sdcard/", GEMMA_MODEL)
        )
        
        for (location in secondaryLocations) {
            if (location.exists() && location.canRead()) {
                Log.d(TAG, "✅ Model file found at secondary location: ${location.absolutePath}")
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