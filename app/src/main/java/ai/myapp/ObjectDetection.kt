package ai.myapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MediaImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInference.LlmInferenceOptions
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession.LlmInferenceSessionOptions
import com.google.mediapipe.tasks.genai.llminference.GraphOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.io.File
import java.util.concurrent.atomic.AtomicLong

/**
 * ObjectDetection — streamlined for fastest frame ingestion **with single‑time model load**.
 */
class ObjectDetection(
    private val context: Context,
    private val listener: DetectorListener? = null
) {

    /* --------------------------- Model / session state --------------------------- */

    private var llm: LlmInference? = null
    private val isInitialised = AtomicLong(0)         // 0 = not yet, 1 = ready
    private val lastFrameTs = AtomicLong(0)

    private val inferenceLock = Any()
    private var inferenceRunning = false
    private val inferenceStartedAt = AtomicLong(0)

    /* ------------------------------- Parameters ---------------------------------- */

    private val modelAssetName = "gemma-3n-E2B-it-int4.task"
    private val frameIntervalMs = 1_000L        // 1 fps sampler
    private val inferenceTimeoutMs = 30_000L
    private val maxEdge = 256                   // max image edge sent to model

    /* --------------------------- Public lifecycle API ---------------------------- */

    suspend fun initialise() { loadModelIfNeeded() }

    fun isReady(): Boolean = isInitialised.get() == 1L && !inferenceRunning

    suspend fun cleanup() = withContext(Dispatchers.IO) {
        llm?.close(); llm = null; isInitialised.set(0)
    }

    /* ---------------------------- CameraX entry point ---------------------------- */

    @OptIn(ExperimentalGetImage::class)
    suspend fun detectLivestreamFrame(proxy: ImageProxy) {
        val now = System.currentTimeMillis()
        
        if (!isReady()) {
            Log.v(TAG, "⏭️ [5/5] FRAME DISCARDED: Model not ready - discarding frame to load next one")
            proxy.close()
            return
        }
        
        if (now - lastFrameTs.get() < frameIntervalMs) {
            Log.v(TAG, "⏭️ [5/5] FRAME DISCARDED: Too frequent (${now - lastFrameTs.get()}ms < ${frameIntervalMs}ms) - discarding frame to load next one")
            proxy.close()
            return
        }
        
        lastFrameTs.set(now)
        Log.v(TAG, "📸 [3/5] FRAME CAPTURE STARTED: Processing camera frame ${proxy.image?.width}x${proxy.image?.height}")

        val mediaImg = proxy.image ?: run { 
            Log.w(TAG, "⏭️ [5/5] FRAME DISCARDED: Null image - discarding frame to load next one")
            proxy.close()
            return
        }

        val mp = MediaImageBuilder(mediaImg).build()
        proxy.close()
        Log.v(TAG, "⏭️ [5/5] FRAME DISCARDED: Camera frame closed, ready to load next one")

        val rotation = proxy.imageInfo.rotationDegrees
        Log.d(TAG, "🔄 Starting detection inference for frame (rotation: ${rotation}°)...")
        detectMpImage(mp, rotation).let { 
            Log.v(TAG, "📤 Sending detection results to listener...")
            listener?.onResults(it)
        }
        mp.close()
    }

    /* ----------------------------- .mp4 entry point ------------------------------ */

    fun detectVideoFile(uri: Uri): ResultBundle? {
        val retriever = MediaMetadataRetriever()
        return try {
            retriever.setDataSource(context, uri)
            val bmp = retriever.getFrameAtTime(0) ?: return null
            kotlinx.coroutines.runBlocking { detectImage(bmp) }
        } catch (e: Exception) {
            Log.e(TAG, "Video detection error", e); null
        } finally { retriever.release() }
    }

    /* ---------------------------- Bitmap entry point ----------------------------- */

    suspend fun detectImage(bmp: Bitmap): ResultBundle? {
        val resized = resizeIfNeeded(src=bmp)
        val mp = BitmapImageBuilder(resized).build()
        val result: ResultBundle? = detectMpImage(img=mp, imageRotation = 0)
        mp.close()
        return result
    }

    /* ------------------------------ Core inference ------------------------------- */

    private suspend fun detectMpImage(img: MPImage, imageRotation: Int): ResultBundle? = withContext(Dispatchers.IO) {
        if (llm == null) {
            Log.w(TAG, "⚠️ Cannot run detection: LLM is null")
            return@withContext null
        }
        
        synchronized(inferenceLock) {
            if (inferenceRunning) {
                Log.v(TAG, "⏭️ [5/5] FRAME DISCARDED: Previous inference still running - discarding frame to load next one")
                return@withContext null
            }
            inferenceRunning = true
            inferenceStartedAt.set(System.currentTimeMillis())
        }

        val start = SystemClock.uptimeMillis()
        Log.d(TAG, "🧠 Starting object detection inference on ${img.width}x${img.height} image...")
        
        try {
            withTimeout(inferenceTimeoutMs) {
                val opts = LlmInferenceSessionOptions.builder()
                    .setTemperature(0.1f)
                    .setGraphOptions(GraphOptions.builder().setEnableVisionModality(true).build())
                    .build()
                var bundle: ResultBundle? = null
                
                llm!!.use { engine ->
                    LlmInferenceSession.createFromOptions(engine, opts).use { session ->
                        Log.v(TAG, "📝 Adding query chunk and image to inference session...")
                        session.addQueryChunk("Detect objects: 1.")
                        session.addImage(img)
                        
                        Log.v(TAG, "⚡ Generating detection response...")
                        val raw = session.generateResponse()
                        val detections = parseDetections(raw)
                        
                        val inferenceTime = SystemClock.uptimeMillis() - start
                        Log.i(TAG, "🎯 [4/5] DETECTION RESULTS: Found ${detections.size} objects in ${inferenceTime}ms")
                        
                        detections.forEachIndexed { index, detection ->
                            Log.d(TAG, "🎯 [4/5] Detection #$index: '${detection.label}' (confidence: ${String.format("%.2f", detection.confidence)}, box: ${detection.boundingBox})")
                        }
                        
                        bundle = ResultBundle(
                            detections,
                            inferenceTime,
                            img.height,
                            img.width,
                            imageRotation
                        )
                        
                        Log.v(TAG, "📦 Created ResultBundle with ${detections.size} detections for ${img.width}x${img.height} image")
                    }
                }
                return@withTimeout bundle
            }
        } catch (e: Exception) {
            Log.e(TAG, "❌ Inference error after ${SystemClock.uptimeMillis() - start}ms", e)
            listener?.onError("Inference error: ${e.message}")
            null
        } finally {
            inferenceRunning = false
            inferenceStartedAt.set(0)
            Log.v(TAG, "🏁 Inference session completed, ready for next frame")
        }
    }

    /* ------------------------------ Model loading -------------------------------- */

    private suspend fun loadModelIfNeeded() = withContext(Dispatchers.IO) {
        if (isInitialised.get() == 1L) {
            Log.d(TAG, "✅ [2/5] MODEL ALREADY INITIALIZED: Skipping reload")
            return@withContext  // Already loaded
        }
        
        try {
            Log.i(TAG, "🚀 Starting model initialization process...")
            val path = copyAssetToFile(modelAssetName)
            
            Log.i(TAG, "⚙️ [2/5] INITIALIZING MODEL: Creating LlmInference with path: $path")
            val initStartTime = System.currentTimeMillis()
            
            val opts = LlmInferenceOptions.builder()
                .setModelPath(path)
                .setMaxTokens(512)
                .setMaxTopK(20)
                .setMaxNumImages(1)
                .build()
                
            Log.d(TAG, "⚙️ Creating LlmInference with options: maxTokens=512, maxTopK=20, maxImages=1")
            llm = LlmInference.createFromOptions(context, opts)
            isInitialised.set(1)
            
            val initTime = System.currentTimeMillis() - initStartTime
            Log.i(TAG, "✅ [2/5] MODEL INITIALIZED SUCCESSFULLY: LLM ready in ${initTime}ms from $path")
            Log.d(TAG, "🎯 Model is now ready for object detection inference")
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ [2/5] MODEL INITIALIZATION FAILED: ${e.message}", e)
            listener?.onError("Model init failed: ${e.message}")
        }
    }

    private fun copyAssetToFile(assetName: String): String {
        val out = File(context.filesDir, assetName)
        Log.i(TAG, "📁 [1/5] CHECKING MODEL LOCATION: ${out.absolutePath}")
        
        if (!out.exists()) {
            Log.w(TAG, "📥 [1/5] MODEL FILE NOT FOUND: $assetName not in internal storage")
            Log.i(TAG, "📥 [1/5] EXPECTED LOCATION: ${out.absolutePath}")
            Log.i(TAG, "📥 [1/5] INSTRUCTIONS: Please copy the model file to the device using:")
            Log.i(TAG, "📥     - Android Studio Device File Explorer, or")  
            Log.i(TAG, "📥     - ADB: adb push gemma-3n-E2B-it-int4.task /data/data/ai.myapp/files/")
            
            throw RuntimeException("Model file not found at: ${out.absolutePath}. Please copy the model file to this location.")
        } else {
            Log.i(TAG, "📥 [1/5] MODEL FOUND IN INTERNAL MEMORY: Found model file (${out.length()} bytes)")
            Log.i(TAG, "📥 [1/5] MODEL VERIFICATION: File size is ${String.format("%.2f", out.length() / (1024.0 * 1024.0 * 1024.0))} GB")
        }
        
        return out.absolutePath
    }

    /* ------------------------------- Utilities ------------------------------------ */

    private fun resizeIfNeeded(src: Bitmap): Bitmap {
        val w = src.width; val h = src.height
        if (w <= maxEdge && h <= maxEdge) return src
        val scale = minOf(maxEdge.toFloat() / w, maxEdge.toFloat() / h)
        val nw = (w * scale).toInt(); val nh = (h * scale).toInt()
        return Bitmap.createScaledBitmap(src, nw, nh, true).also { if (it != src) src.recycle() }
    }

    private fun parseDetections(text: String): List<Detection> {
        val list = mutableListOf<Detection>()
        text.split("\n").forEach { line ->
            val (label, box) = extractLabelAndBox(line.trim())
            if (label.isNotEmpty()) list += Detection(box, label, 0.8f, "LLM_DETECTION")
        }
        return list
    }

    private fun extractLabelAndBox(s: String): Pair<String, RectF?> {
        val r = Regex("\\[([0-9.]+),([0-9.]+),([0-9.]+),([0-9.]+)]")
        val m = r.find(s) ?: return Pair(s, null)
        val (x1, y1, x2, y2) = m.groupValues.drop(1).map { it.toFloat() }
        return Pair(s.replace(r, "").trim(), RectF(x1, y1, x2, y2))
    }

    /* --------------------------------- Data -------------------------------------- */

    data class Detection(
        val boundingBox: RectF? = null,
        val label: String,
        val confidence: Float,
        val classification: String
    )

    // AFTER
    data class ResultBundle(
        val detections: List<Detection>,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
        val inputImageRotation: Int // Add this property
    )

    interface DetectorListener {
        fun onError(msg: String)
        fun onResults(result: ResultBundle?)
    }

    companion object { const val TAG = "ObjectDetection" }
}
