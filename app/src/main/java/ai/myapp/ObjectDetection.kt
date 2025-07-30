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
        if (!isReady() || now - lastFrameTs.get() < frameIntervalMs) { proxy.close(); return }
        lastFrameTs.set(now)

        val mediaImg = proxy.image ?: run { proxy.close(); return }

        val mp = MediaImageBuilder(mediaImg).build()
        proxy.close()

        val rotation = proxy.imageInfo.rotationDegrees
        detectMpImage(mp, rotation).let { listener?.onResults(it) }
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
        if (llm == null) return@withContext null
        synchronized(inferenceLock) {
            if (inferenceRunning) return@withContext null
            inferenceRunning = true; inferenceStartedAt.set(System.currentTimeMillis())
        }

        val start = SystemClock.uptimeMillis()
        try {
            withTimeout(inferenceTimeoutMs) {
                val opts = LlmInferenceSessionOptions.builder()
                    .setTemperature(0.1f)
                    .setGraphOptions(GraphOptions.builder().setEnableVisionModality(true).build())
                    .build()
                var bundle: ResultBundle? = null
                llm!!.use { engine ->
                    LlmInferenceSession.createFromOptions(engine, opts).use { session ->
                        session.addQueryChunk("Detect objects: 1.")
                        session.addImage(img)
                        val raw = session.generateResponse()
                        val detections = parseDetections(raw)
                        // AFTER
                        bundle = ResultBundle(
                            detections,
                            SystemClock.uptimeMillis() - start,
                            img.height,
                            img.width,
                            // Note: You will need to pass the rotation to this function.
                            // See the next change. For now, let's assume a variable `imageRotation`.
                            imageRotation
                        )
                    }
                }
                return@withTimeout bundle
            }
        } catch (e: Exception) {
            Log.e(TAG, "Inference error", e)
            listener?.onError("Inference error: ${e.message}")
            null
        } finally {
            inferenceRunning = false; inferenceStartedAt.set(0)
        }
    }

    /* ------------------------------ Model loading -------------------------------- */

    private suspend fun loadModelIfNeeded() = withContext(Dispatchers.IO) {
        if (isInitialised.get() == 1L) return@withContext  // Already loaded
        try {
            val path = copyAssetToFile(modelAssetName)
            val opts = LlmInferenceOptions.builder()
                .setModelPath(path)
                .setMaxTokens(512)
                .setMaxTopK(20)
                .setMaxNumImages(1)
                .build()
            llm = LlmInference.createFromOptions(context, opts)
            isInitialised.set(1)
            Log.d(TAG, "LLM initialised from $path")
        } catch (e: Exception) {
            Log.e(TAG, "Model init failed", e)
            listener?.onError("Model init failed: ${e.message}")
        }
    }

    private fun copyAssetToFile(assetName: String): String {
        val out = File(context.filesDir, assetName)
        if (!out.exists()) {
            context.assets.open(assetName).use { input ->
                out.outputStream().use { output -> input.copyTo(output) }
            }
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
