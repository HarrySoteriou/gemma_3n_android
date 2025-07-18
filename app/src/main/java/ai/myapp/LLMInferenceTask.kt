package ai.myapp

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.genai.llminference.GraphOptions
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

class LLMInferenceTask(private val context: Context) {

    private var llmInference: LlmInference? = null          // ❶ correct type
    private val isInitialized = AtomicBoolean(false)
    private val isProcessing = AtomicBoolean(false)

    companion object {
        private const val TAG = "LLMInferenceTask"
        private const val GEMMA_MODEL =
            "/data/local/tmp/llm/gemma‑3n‑E2B‑it‑int4.task" // example path
    }

    /** Call once (e.g. in a ViewModel’s init) */
    fun initializeModel() {
        if (isInitialized.get()) return
        try {
            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(GEMMA_MODEL)
                .setMaxTokens(512)
                .setMaxNumImages(1)                       // allow exactly one image
                .build()

            llmInference = LlmInference.createFromOptions(context, options)
            isInitialized.set(true)
            Log.d(TAG, "LLM initialised")
        } catch (t: Throwable) {
            Log.e(TAG, "LLM init failed", t)
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