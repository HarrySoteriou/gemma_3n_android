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
        val externalLocations = externalPaths.map { File(it) } + listOf(
            File("/sdcard/Download/", GEMMA_MODEL),
            File("/sdcard/", GEMMA_MODEL),
            File("/sdcard/Android/data/${context.packageName}/files/", GEMMA_MODEL)
        )
        
        var sourceFile: File? = null
        for (location in externalLocations) {
            if (location.exists()) {
                Log.d(TAG, "Found model at: ${location.absolutePath}")
                sourceFile = location
                break
            }
        }
        
        if (sourceFile == null) {
            Log.e(TAG, "Model not found in any expected location.")
            Log.e(TAG, "Checked locations:")
            externalLocations.forEach { Log.e(TAG, "  - ${it.absolutePath}") }
            Log.e(TAG, "Please push the model to device using:")
            Log.e(TAG, "adb push \"<path-to-your-model>\" /sdcard/Download/$GEMMA_MODEL")
            return@withContext null
        }
        
        // Copy model from external to internal storage
        try {
            Log.d(TAG, "Copying model from ${sourceFile.absolutePath} to internal storage...")
            Log.d(TAG, "This may take a few minutes for a 3GB file...")
            
            sourceFile.inputStream().use { input ->
                FileOutputStream(internalModelFile).use { output ->
                    input.copyTo(output)
                }
            }
            
            Log.d(TAG, "Model copied successfully to: ${internalModelFile.absolutePath}")
            return@withContext internalModelFile.absolutePath
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to copy model to internal storage", e)
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
        if (isInitialized.get()) return
        
        try {
            // Ensure model is available in internal storage
            val modelPath = ensureModelAvailable()
            if (modelPath == null) {
                Log.e(TAG, "Cannot initialize LLM: Model file not available")
                Log.e(TAG, "Instructions:")
                Log.e(TAG, "1. Connect your device via USB")
                Log.e(TAG, "2. Run: adb push \"C:\\Users\\mario\\OneDrive\\Desktop\\Machine Learning Skillset\\android-app\\data\\local\\tmp\\llm\\${GEMMA_MODEL}\" /sdcard/Download/")
                Log.e(TAG, "3. Restart the app")
                return
            }
            
            Log.d(TAG, "Initializing LLM with model: $modelPath")
            
            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelPath)
                .setMaxTokens(512)
                .setMaxNumImages(4)
                .build()

            llmInference = LlmInference.createFromOptions(context, options)
            isInitialized.set(true)
            Log.d(TAG, "LLM initialised")
        } catch (t: Throwable) {
            Log.e(TAG, "LLM initialization failed.", t)
            Log.e(TAG, "Error details: ${t.message}")
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

    fun isReady(): Boolean = isInitialized.get() && !isProcessing.get()

    fun cleanup() {
        runCatching { llmInference?.close() }
        isInitialized.set(false)
    }
}