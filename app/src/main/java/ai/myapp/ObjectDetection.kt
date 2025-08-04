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
import androidx.core.graphics.scale
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
import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicLong
import java.util.regex.Pattern

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
    private val inferenceTimeoutMs = 60_000L    // Increased timeout for generative model
    private val maxEdge = 512                   // max image edge sent to model

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
            Log.v(TAG, "⏭️ [5/5] FRAME DISCARDED: Model not ready")
            proxy.close()
            return
        }

        if (now - lastFrameTs.get() < frameIntervalMs) {
            Log.v(TAG, "⏭️ [5/5] FRAME DISCARDED: Too frequent")
            proxy.close()
            return
        }

        lastFrameTs.set(now)
        Log.v(TAG, "📸 [3/5] FRAME CAPTURE STARTED: Processing camera frame ${proxy.image?.width}x${proxy.image?.height}")

        val mediaImg = proxy.image ?: run {
            Log.w(TAG, "⏭️ [5/5] FRAME DISCARDED: Null image")
            proxy.close()
            return
        }

        val mp = MediaImageBuilder(mediaImg).build()

        try {
            val rotation = proxy.imageInfo.rotationDegrees
            Log.d(TAG, "🔄 Starting detection inference for frame (rotation: ${rotation}°)...")

            val results = detectMpImage(mp, rotation)

            Log.v(TAG, "📤 Sending detection results to listener...")
            listener?.onResults(results)

        } finally {
            mp.close()
            proxy.close()
            Log.v(TAG, "✅ [5/5] FRAME PROCESSED: Frame and MPImage closed, ready for next one.")
        }
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
                Log.v(TAG, "⏭️ [5/5] FRAME DISCARDED: Previous inference still running")
                return@withContext null
            }
            inferenceRunning = true
            inferenceStartedAt.set(System.currentTimeMillis())
        }

        val start = SystemClock.uptimeMillis()
        Log.d(TAG, "🧠 Starting object detection inference on ${img.width}x${img.height} image...")

        val engine = llm ?: return@withContext null

        try {
            withTimeout(inferenceTimeoutMs) {
                val opts = LlmInferenceSessionOptions.builder()
                    .setTemperature(0.2f)
                    .setGraphOptions(GraphOptions.builder().setEnableVisionModality(true).build())
                    .build()
                var bundle: ResultBundle? = null

                LlmInferenceSession.createFromOptions(engine, opts).use { session ->
                    Log.v(TAG, "📝 Adding query chunk and image to inference session...")
                    // This prompt guides the model to produce the structured output we can parse.
                    session.addQueryChunk("Detect all people and objects. For each, provide a classification on a new line starting with '**Class:**', a bounding box on a new line as '**Bounding Box:** (x1, y1, x2, y2)', and a confidence score on a new line as '**Confidence:** score'.")
                    session.addImage(img)

                    Log.v(TAG, "⚡ Generating detection response...")
                    val rawText = session.generateResponse()
                    val lines = rawText.split('\n')

                    // *** NEW PARSING STEP ***
                    val detections = parseDetectionsFromText(lines)

                    val inferenceTime = SystemClock.uptimeMillis() - start
                    Log.i(TAG, "🎯 [4/5] DETECTION RESULTS: Parsed ${detections.size} objects in ${inferenceTime}ms")

                    detections.forEachIndexed { index, detection ->
                        Log.d(TAG, "🎯 [4/5] Parsed Detection #$index: '${detection.label}' (confidence: ${String.format("%.2f", detection.confidence)}, box: ${detection.boundingBox})")
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
            return@withContext
        }

        try {
            Log.i(TAG, "🚀 Starting model initialization process...")
            val path = copyAssetToFile(modelAssetName)

            Log.i(TAG, "⚙️ [2/5] INITIALIZING MODEL: Creating LlmInference with path: $path")
            val initStartTime = System.currentTimeMillis()

            val opts = LlmInferenceOptions.builder()
                .setModelPath(path)
                .setMaxTokens(1024) // Increased tokens for more detailed output
                .setMaxTopK(10)
                .setMaxNumImages(1)
                .build()

            Log.d(TAG, "⚙️ Creating LlmInference with options: maxTokens=1024, maxTopK=10, maxImages=1")
            llm = LlmInference.createFromOptions(context, opts)
            isInitialised.set(1)

            val initTime = System.currentTimeMillis() - initStartTime
            Log.i(TAG, "✅ [2/5] MODEL INITIALIZED SUCCESSFULLY: LLM ready in ${initTime}ms from $path")

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

            val sdcardFile = File("/sdcard/$assetName")
            Log.i(TAG, "📥 [1/5] CHECKING SDCARD: Looking for model at ${sdcardFile.absolutePath}")

            if (sdcardFile.exists()) {
                Log.i(TAG, "📥 [1/5] MODEL FOUND ON SDCARD: Found model file (${sdcardFile.length()} bytes)")
                Log.i(TAG, "📥 [1/5] COPYING MODEL: Copying from ${sdcardFile.absolutePath} to ${out.absolutePath}")

                try {
                    copyFile(sdcardFile, out)
                    Log.i(TAG, "✅ [1/5] MODEL COPIED SUCCESSFULLY: Model copied to internal storage (${out.length()} bytes)")
                } catch (e: Exception) {
                    Log.e(TAG, "❌ [1/5] MODEL COPY FAILED: Failed to copy model file", e)
                    throw RuntimeException("Failed to copy model file from SDCard: ${e.message}")
                }
            } else {
                Log.w(TAG, "📥 [1/5] MODEL FILE NOT FOUND: $assetName not found in SDCard either")
                throw RuntimeException("Model file not found. Please copy $assetName to /sdcard/ or internal storage.")
            }
        } else {
            Log.i(TAG, "📥 [1/5] MODEL FOUND IN INTERNAL MEMORY: Found model file (${out.length()} bytes)")
        }

        return out.absolutePath
    }

    private fun copyFile(source: File, destination: File) {
        FileInputStream(source).use { inputStream ->
            FileOutputStream(destination).use { outputStream ->
                val buffer = ByteArray(8192)
                var length: Int
                while (inputStream.read(buffer).also { length = it } > 0) {
                    outputStream.write(buffer, 0, length)
                }
                outputStream.flush()
            }
        }
    }

    /* ------------------------------- Utilities ------------------------------------ */

    private fun resizeIfNeeded(src: Bitmap): Bitmap {
        val w = src.width; val h = src.height
        if (w <= maxEdge && h <= maxEdge) return src
        val scale = minOf(maxEdge.toFloat() / w, maxEdge.toFloat() / h)
        val nw = (w * scale).toInt(); val nh = (h * scale).toInt()
        return src.scale(nw, nh).also { if (it != src) src.recycle() }
    }

    /**
     * Helper data class for parsing multi-line detection entries from the LLM.
     */
    private data class ParsedDetectionData(
        var label: String = "",
        var boundingBox: RectF? = null,
        var confidence: Float = 0f
    )

    /**
     * Parses a list of text lines from the LLM into a structured List of Detection objects.
     */
    private fun parseDetectionsFromText(lines: List<String>): List<Detection> {
        val finalDetections = mutableListOf<Detection>()

        // Regex to find the structured data
        val classRegex = Pattern.compile("""\*\s+\*\*Class:\*\*\s+(.*)""")
        val boxRegex = Pattern.compile("""\*\s+\*\*Bounding Box:\*\*\s+\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)""")
        val confidenceRegex = Pattern.compile("""\*\s+\*\*Confidence:\*\*\s+([\d.]+)""")

        var currentParsedData: ParsedDetectionData? = null

        for (line in lines) {
            val trimmedLine = line.trim()

            val classMatcher = classRegex.matcher(trimmedLine)
            if (classMatcher.find()) {
                // A new object is starting. If we were parsing a previous one, save it.
                currentParsedData?.let {
                    if (it.label.isNotEmpty() && it.boundingBox != null) {
                        finalDetections.add(Detection(it.boundingBox, it.label, it.confidence, "llm_detection"))
                    }
                }
                currentParsedData = ParsedDetectionData(label = classMatcher.group(1).trim())
                continue
            }

            if (currentParsedData == null) continue

            val boxMatcher = boxRegex.matcher(trimmedLine)
            if (boxMatcher.find()) {
                try {
                    val left = boxMatcher.group(1).toFloat()
                    val top = boxMatcher.group(2).toFloat()
                    val right = boxMatcher.group(3).toFloat()
                    val bottom = boxMatcher.group(4).toFloat()
                    currentParsedData.boundingBox = RectF(left, top, right, bottom)
                } catch (e: Exception) {
                    Log.e(TAG, "Could not parse bounding box in line: $trimmedLine", e)
                }
                continue
            }

            val confidenceMatcher = confidenceRegex.matcher(trimmedLine)
            if (confidenceMatcher.find()) {
                try {
                    currentParsedData.confidence = confidenceMatcher.group(1).toFloat()
                    // This is the last piece of info. The object is complete.
                    if (currentParsedData.label.isNotEmpty() && currentParsedData.boundingBox != null) {
                        finalDetections.add(Detection(currentParsedData.boundingBox, currentParsedData.label, currentParsedData.confidence, "llm_detection"))
                    }
                    currentParsedData = null // Reset for the next object
                } catch (e: Exception) {
                    Log.e(TAG, "Could not parse confidence in line: $trimmedLine", e)
                }
            }
        }

        // Add the last detection if the loop ended before its confidence line was found.
        currentParsedData?.let {
            if (it.label.isNotEmpty() && it.boundingBox != null) {
                finalDetections.add(Detection(it.boundingBox, it.label, it.confidence, "llm_detection"))
            }
        }

        return finalDetections
    }


    /* --------------------------------- Data -------------------------------------- */

    data class Detection(
        val boundingBox: RectF? = null,
        val label: String,
        val confidence: Float,
        val classification: String
    )

    data class ResultBundle(
        val detections: List<Detection>,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
        val inputImageRotation: Int
    )

    interface DetectorListener {
        fun onError(msg: String)
        fun onResults(result: ResultBundle?)
    }

    companion object { const val TAG = "ObjectDetection" }
}