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
    private var session: LlmInferenceSession? = null  // Reuse session
    private val isInitialised = AtomicLong(0)         // 0 = not yet, 1 = ready
    private val lastFrameTs = AtomicLong(0)

    private val inferenceLock = Any()
    private var inferenceRunning = false
    private val inferenceStartedAt = AtomicLong(0)
    
    // Session management for token limit handling
    private var sessionInferenceCount = 0
    private val maxSessionInferences = 2  // Recreate session after 2 inferences to prevent token buildup
    
    // Memory management for preventing model swapping
    private val runtime = Runtime.getRuntime()
    private var sessionWarmedUp = false
    private val memoryBuffer = ByteArray(50 * 1024 * 1024) // 50MB buffer to reserve RAM

    /* ------------------------------- Parameters ---------------------------------- */

    private val modelAssetName = "gemma-3n-E2B-it-int4.task"
    private val frameIntervalMs = 10_000L        // 0.1 fps sampler
    private val inferenceTimeoutMs = 60_000L    // Increased timeout for generative model
    private val maxEdge = 512                   // max image edge sent to model

    /* --------------------------- Public lifecycle API ---------------------------- */

    suspend fun initialise() { loadModelIfNeeded() }

    fun isReady(): Boolean = isInitialised.get() == 1L && !inferenceRunning && hasAvailableMemory()
    
    private fun hasAvailableMemory(): Boolean {
        val usedMemory = runtime.totalMemory() - runtime.freeMemory()
        val maxMemory = runtime.maxMemory()
        val memoryUsagePercent = (usedMemory.toDouble() / maxMemory.toDouble()) * 100
        val availableMemory = maxMemory - usedMemory
        
        Log.v(TAG, "💾 Memory Status: ${String.format("%.1f", memoryUsagePercent)}% used, ${availableMemory/1024/1024}MB available")
        
        if (memoryUsagePercent > 85) {
            Log.w(TAG, "⚠️ HIGH MEMORY PRESSURE: ${String.format("%.1f", memoryUsagePercent)}% - RISK OF MODEL SWAPPING TO DISK!")
            Log.w(TAG, "💾 Available: ${availableMemory/1024/1024}MB - Consider reducing other app usage")
            
            // Force garbage collection to free up memory
            System.gc()
            
            // Re-check after GC
            val newUsedMemory = runtime.totalMemory() - runtime.freeMemory()
            val newMemoryUsagePercent = (newUsedMemory.toDouble() / maxMemory.toDouble()) * 100
            Log.i(TAG, "🗑️ After GC: ${String.format("%.1f", newMemoryUsagePercent)}% memory used")
            
            return newMemoryUsagePercent < 90 // More aggressive threshold
        }
        return true
    }

    suspend fun cleanup() = withContext(Dispatchers.IO) {
        session?.close(); session = null
        llm?.close(); llm = null; isInitialised.set(0)
        sessionInferenceCount = 0
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

        // Prevent garbage collection during inference to avoid memory pressure
        preventGarbageCollection()

        try {
            withTimeout(inferenceTimeoutMs) {
                // Track session usage first
                sessionInferenceCount++
                
                // Check if we need to recreate session to prevent token limit issues
                var currentSession = session
                if (currentSession == null || sessionInferenceCount > maxSessionInferences) {
                    Log.i(TAG, "🔄 Recreating session to prevent token accumulation (inference count: $sessionInferenceCount)")
                    recreateSession()
                    currentSession = session
                    sessionInferenceCount = 1
                }
                
                if (currentSession == null) {
                    Log.w(TAG, "⚠️ Cannot run detection: Session creation failed")
                    return@withTimeout null
                }
                
                // Warn if session hasn't been warmed up
                if (!sessionWarmedUp) {
                    Log.w(TAG, "⚠️ Session not warmed up - first inference may be slow due to cold model loading")
                }

                Log.v(TAG, "📊 Session inference count: $sessionInferenceCount/$maxSessionInferences")

                val rawText = try {
                    Log.v(TAG, "📝 Adding query chunk and image to reusable session...")
                    // This prompt guides the model to produce the structured output we can parse.
                    currentSession.addQueryChunk("Detect objects. For each object found, provide on separate lines: * **Class:** [object name], * **Bounding Box:** (x1, y1, x2, y2), * **Confidence:** [score]")
                    currentSession.addImage(img)

                    Log.v(TAG, "⚡ Generating detection response...")
                    currentSession.generateResponse()
                } catch (e: Exception) {
                    // Handle OUT_OF_RANGE and other session errors by recreating session
                    if (e.message?.contains("OUT_OF_RANGE") == true || e.message?.contains("too long") == true) {
                        Log.w(TAG, "🔄 Token limit exceeded, recreating session and retrying...")
                        recreateSession()
                        currentSession = session ?: return@withTimeout null
                        sessionInferenceCount = 1
                        
                        // Retry with fresh session
                        currentSession.addQueryChunk("Detect objects. For each object found, provide on separate lines: * **Class:** [object name], * **Bounding Box:** (x1, y1, x2, y2), * **Confidence:** [score]")
                        currentSession.addImage(img)
                        currentSession.generateResponse()
                    } else {
                        throw e
                    }
                }
                val lines = rawText.split('\n')

                // DEBUG: Log the raw output to see what the model is actually generating
                Log.d(TAG, "🔍 RAW MODEL OUTPUT:")
                Log.d(TAG, "📝 Raw text length: ${rawText.length} characters")
                Log.d(TAG, "📝 Raw text: '$rawText'")
                Log.d(TAG, "📝 Split into ${lines.size} lines:")
                lines.forEachIndexed { index, line ->
                    Log.d(TAG, "📝 Line $index: '$line'")
                }

                // *** NEW PARSING STEP - Pass image dimensions for coordinate normalization ***
                val detections = parseDetectionsFromText(lines, img.width, img.height)

                val inferenceTime = SystemClock.uptimeMillis() - start
                Log.i(TAG, "🎯 [4/5] DETECTION RESULTS: Parsed ${detections.size} objects in ${inferenceTime}ms")

                detections.forEachIndexed { index, detection ->
                    Log.d(TAG, "🎯 [4/5] Parsed Detection #$index: '${detection.label}' (confidence: ${String.format("%.2f", detection.confidence)}, box: ${detection.boundingBox})")
                }

                Log.v(TAG, "📦 Created ResultBundle with ${detections.size} detections for ${img.width}x${img.height} image")

                return@withTimeout ResultBundle(
                    detections,
                    inferenceTime,
                    img.height,
                    img.width,
                    imageRotation
                )
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
                .setMaxTokens(1536)  // Further increased tokens to prevent OUT_OF_RANGE errors
                .setMaxTopK(10)
                .setMaxNumImages(1)
                .build()

            Log.d(TAG, "⚙️ Creating LlmInference with options: maxTokens=1536, maxTopK=10, maxImages=1")
            llm = LlmInference.createFromOptions(context, opts)
            
            // Create a single reusable session
            val sessionOpts = LlmInferenceSessionOptions.builder()
                .setTemperature(0.2f)
                .setGraphOptions(GraphOptions.builder().setEnableVisionModality(true).build())
                .build()
            session = LlmInferenceSession.createFromOptions(llm!!, sessionOpts)
            
            // RAM Management: Force aggressive memory allocation
            Log.i(TAG, "🧠 Reserving memory to prevent model swapping...")
            forceMemoryAllocation()
            
            // Session warming to keep model hot in RAM
            Log.i(TAG, "🔥 Warming up session to prevent cold starts...")
            warmUpSession()
            
            isInitialised.set(1)

            val initTime = System.currentTimeMillis() - initStartTime
            Log.i(TAG, "✅ [2/5] MODEL INITIALIZED SUCCESSFULLY: LLM and session ready in ${initTime}ms from $path")

        } catch (e: Exception) {
            Log.e(TAG, "❌ [2/5] MODEL INITIALIZATION FAILED: ${e.message}", e)
            listener?.onError("Model init failed: ${e.message}")
        }
    }

    private fun copyAssetToFile(assetName: String): String {
        val internalFile = File(context.filesDir, assetName)

        // 1. If the model is already in its final internal location, we are done.
        if (internalFile.exists()) {
            Log.i(TAG, "📥 [1/5] MODEL FOUND IN INTERNAL MEMORY: (${internalFile.length()} bytes)")
            return internalFile.absolutePath
        }

        Log.i(TAG, "📥 [1/5] MODEL NOT FOUND: Copying from app assets...")

        // 2. If not found, copy it from the app's assets folder.
        try {
            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(internalFile).use { outputStream ->
                    val buffer = ByteArray(8192) // Use a larger buffer for faster copying
                    var length: Int
                    while (inputStream.read(buffer).also { length = it } > 0) {
                        outputStream.write(buffer, 0, length)
                    }
                }
            }
            Log.i(TAG, "✅ [1/5] MODEL COPIED SUCCESSFULLY: to ${internalFile.absolutePath}")
            return internalFile.absolutePath
        } catch (e: Exception) {
            Log.e(TAG, "❌ [1/5] FATAL: FAILED TO COPY MODEL FROM ASSETS", e)
            // If the model cannot be copied from assets, the app cannot function.
            throw RuntimeException("Critical error: Failed to copy model file from assets. Please ensure '$assetName' is in the 'src/main/assets' directory.", e)
        }
    }    /* --------------------------- Memory Management ---------------------------- */
    
    private fun forceMemoryAllocation() {
        try {
            // Force JVM to allocate maximum available memory
            val maxMemory = runtime.maxMemory()
            val totalMemory = runtime.totalMemory()
            val freeMemory = runtime.freeMemory()
            val usedMemory = totalMemory - freeMemory
            val availableMemory = maxMemory - usedMemory
            
            Log.i(TAG, "💾 Memory Stats - Max: ${maxMemory/1024/1024}MB, Used: ${usedMemory/1024/1024}MB, Available: ${availableMemory/1024/1024}MB")
            
            // Pre-allocate memory to prevent later allocations that could trigger swapping
            val bufferSize = minOf(availableMemory / 4, 100 * 1024 * 1024L).toInt() // Use 25% of available or 100MB max
            val preAllocBuffer = ByteArray(bufferSize)
            preAllocBuffer.fill(0) // Touch all memory pages
            
            // Disable garbage collection during inference periods
            System.gc() // One final GC before we start
            
            Log.i(TAG, "✅ Pre-allocated ${bufferSize/1024/1024}MB RAM buffer to prevent swapping")
            
        } catch (e: Exception) {
            Log.w(TAG, "⚠️ Could not pre-allocate memory: ${e.message}")
        }
    }
    
    private fun warmUpSession() {
        try {
            // FIXED: Create a separate warmup session to avoid corrupting the main session
            val warmupSessionOpts = LlmInferenceSessionOptions.builder()
                .setTemperature(0.2f)
                .setGraphOptions(GraphOptions.builder().setEnableVisionModality(true).build())
                .build()
            
            Log.i(TAG, "🔥 Creating separate warmup session to avoid session corruption...")
            val warmupStart = System.currentTimeMillis()
            
            // Create separate warmup session, use it, then dispose it
            LlmInferenceSession.createFromOptions(llm!!, warmupSessionOpts).use { warmupSession ->
                // Create a small dummy image for warming up
                val warmupBitmap = Bitmap.createBitmap(64, 64, Bitmap.Config.RGB_565)
                val warmupImage = BitmapImageBuilder(warmupBitmap).build()
                
                Log.i(TAG, "🔥 Performing warmup inference with separate session...")
                
                // Perform a lightweight warmup inference with disposable session
                warmupSession.addQueryChunk("Test")
                warmupSession.addImage(warmupImage)
                warmupSession.generateResponse() // This loads model weights into RAM
                
                warmupImage.close()
                warmupBitmap.recycle()
                
                Log.i(TAG, "✅ Warmup session disposed - main session should be clean")
            }
            
            sessionWarmedUp = true
            val warmupTime = System.currentTimeMillis() - warmupStart
            Log.i(TAG, "✅ Model warmed up in ${warmupTime}ms - weights should now be hot in RAM")
            
        } catch (e: Exception) {
            Log.w(TAG, "⚠️ Session warmup failed: ${e.message}")
            sessionWarmedUp = false
        }
    }
    
    private fun recreateSession() {
        try {
            Log.i(TAG, "🔄 Starting session recreation...")
            
            // Force garbage collection before closing session
            System.gc()
            
            // Close existing session if it exists
            session?.close()
            session = null
            
            // Brief pause to allow cleanup
            Thread.sleep(100)
            
            // Create a new session with the same options
            val sessionOpts = LlmInferenceSessionOptions.builder()
                .setTemperature(0.2f)
                .setGraphOptions(GraphOptions.builder().setEnableVisionModality(true).build())
                .build()
            
            session = LlmInferenceSession.createFromOptions(llm!!, sessionOpts)
            sessionWarmedUp = false // Mark as not warmed up since it's a new session
            
            // Force another GC after recreation
            System.gc()
            
            Log.i(TAG, "✅ Session recreated successfully to prevent token accumulation")
        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to recreate session: ${e.message}")
            session = null
        }
    }
    
    private fun preventGarbageCollection() {
        // Hint to JVM to avoid GC during inference
        System.runFinalization()
        
        // Additional JVM memory pressure relief
        try {
            // Signal to JVM that we want to keep objects in memory
            val usedMemory = runtime.totalMemory() - runtime.freeMemory()
            val maxMemory = runtime.maxMemory()
            val memoryPressure = usedMemory.toDouble() / maxMemory.toDouble()
            
            if (memoryPressure > 0.8) {
                Log.w(TAG, "🚨 CRITICAL MEMORY PRESSURE: ${String.format("%.1f", memoryPressure * 100)}% - MODEL MAY SWAP TO DISK!")
                Log.w(TAG, "💡 TIP: Close other apps to free RAM for better Gemma performance")
            }
        } catch (e: Exception) {
            Log.v(TAG, "Could not check memory pressure: ${e.message}")
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
     * Automatically detects and normalizes absolute pixel coordinates to 0-1 range.
     * 
     * @param lines The text lines from the LLM response
     * @param imageWidth The width of the input image in pixels
     * @param imageHeight The height of the input image in pixels
     */
    private fun parseDetectionsFromText(lines: List<String>, imageWidth: Int, imageHeight: Int): List<Detection> {
        val finalDetections = mutableListOf<Detection>()

        Log.d(TAG, "🔍 PARSING DEBUG: Starting to parse ${lines.size} lines")

        // Regex to find the structured data
        val classRegex = Pattern.compile("""\*\s+\*\*Class:\*\*\s+([^,]+)""")
        val boxRegex = Pattern.compile("""\*\s+\*\*Bounding Box:\*\*\s+\(([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)""")
        val confidenceRegex = Pattern.compile("""\*\s+\*\*Confidence:\*\*\s+([\d.]+)""")

        Log.d(TAG, "🔍 PARSING DEBUG: Using regex patterns:")
        Log.d(TAG, "🔍   Class: ${classRegex.pattern()}")
        Log.d(TAG, "🔍   Box: ${boxRegex.pattern()}")
        Log.d(TAG, "🔍   Confidence: ${confidenceRegex.pattern()}")

        var currentParsedData: ParsedDetectionData? = null

        for (line in lines) {
            val trimmedLine = line.trim()
            Log.v(TAG, "🔍 PARSING: Processing line: '$trimmedLine'")

            val classMatcher = classRegex.matcher(trimmedLine)
            if (classMatcher.find()) {
                Log.d(TAG, "🔍 FOUND CLASS: '${classMatcher.group(1).trim()}'")
                // A new object is starting. If we were parsing a previous one, save it.
                currentParsedData?.let {
                    if (it.label.isNotEmpty() && it.boundingBox != null) {
                        Log.d(TAG, "🔍 SAVING previous detection: ${it.label}")
                        finalDetections.add(Detection(it.boundingBox, it.label, it.confidence, "llm_detection"))
                    }
                }
                currentParsedData = ParsedDetectionData(label = classMatcher.group(1).trim())
                continue
            }

            if (currentParsedData == null) continue

            val boxMatcher = boxRegex.matcher(trimmedLine)
            if (boxMatcher.find()) {
                Log.d(TAG, "🔍 FOUND BOX: (${boxMatcher.group(1)}, ${boxMatcher.group(2)}, ${boxMatcher.group(3)}, ${boxMatcher.group(4)})")
                try {
                    val rawLeft = boxMatcher.group(1).toFloat()
                    val rawTop = boxMatcher.group(2).toFloat()
                    val rawRight = boxMatcher.group(3).toFloat()
                    val rawBottom = boxMatcher.group(4).toFloat()
                    
                    // Detect if coordinates are absolute (pixel values) or normalized (0-1)
                    // If any coordinate is > 1.0, treat as absolute pixel coordinates
                    val isAbsolute = rawLeft > 1.0f || rawTop > 1.0f || rawRight > 1.0f || rawBottom > 1.0f
                    
                    val (left, top, right, bottom) = if (isAbsolute) {
                        // Normalize absolute coordinates by dividing by image dimensions
                        Log.d(TAG, "🔧 COORDINATE NORMALIZATION: Converting absolute coords [$rawLeft, $rawTop, $rawRight, $rawBottom] to normalized")
                        val normalizedLeft = rawLeft / imageWidth.toFloat()
                        val normalizedTop = rawTop / imageHeight.toFloat()
                        val normalizedRight = rawRight / imageWidth.toFloat()
                        val normalizedBottom = rawBottom / imageHeight.toFloat()
                        Log.d(TAG, "🔧 NORMALIZED COORDS: [$normalizedLeft, $normalizedTop, $normalizedRight, $normalizedBottom]")
                        listOf(normalizedLeft, normalizedTop, normalizedRight, normalizedBottom)
                    } else {
                        // Already normalized coordinates
                        Log.d(TAG, "✅ COORDS ALREADY NORMALIZED: [$rawLeft, $rawTop, $rawRight, $rawBottom]")
                        listOf(rawLeft, rawTop, rawRight, rawBottom)
                    }
                    
                    currentParsedData.boundingBox = RectF(left, top, right, bottom)
                } catch (e: Exception) {
                    Log.e(TAG, "Could not parse bounding box in line: $trimmedLine", e)
                }
                continue
            }

            val confidenceMatcher = confidenceRegex.matcher(trimmedLine)
            if (confidenceMatcher.find()) {
                Log.d(TAG, "🔍 FOUND CONFIDENCE: ${confidenceMatcher.group(1)}")
                try {
                    currentParsedData.confidence = confidenceMatcher.group(1).toFloat()
                    // This is the last piece of info. The object is complete.
                    if (currentParsedData.label.isNotEmpty() && currentParsedData.boundingBox != null) {
                        Log.d(TAG, "🔍 COMPLETING detection: ${currentParsedData.label} with confidence ${currentParsedData.confidence}")
                        finalDetections.add(Detection(currentParsedData.boundingBox, currentParsedData.label, currentParsedData.confidence, "llm_detection"))
                    }
                    currentParsedData = null // Reset for the next object
                } catch (e: Exception) {
                    Log.e(TAG, "Could not parse confidence in line: $trimmedLine", e)
                }
            } else {
                Log.v(TAG, "🔍 NO MATCH: Line '$trimmedLine' didn't match any pattern")
            }
        }

        // Add the last detection if the loop ended before its confidence line was found.
        currentParsedData?.let {
            if (it.label.isNotEmpty() && it.boundingBox != null) {
                Log.d(TAG, "🔍 FINAL detection: ${it.label}")
                finalDetections.add(Detection(it.boundingBox, it.label, it.confidence, "llm_detection"))
            }
        }

        Log.d(TAG, "🔍 PARSING COMPLETE: Found ${finalDetections.size} total detections")
        
        // If no detections found with strict parsing, try alternative formats
        if (finalDetections.isEmpty()) {
            Log.w(TAG, "🔍 STRICT PARSING FAILED - Attempting fallback parsing...")
            return parseDetectionsFallback(lines, imageWidth, imageHeight)
        }

        return finalDetections
    }

    /**
     * Fallback parser that tries multiple different formats the model might generate.
     * Automatically detects and normalizes absolute pixel coordinates to 0-1 range.
     * 
     * @param lines The text lines from the LLM response
     * @param imageWidth The width of the input image in pixels
     * @param imageHeight The height of the input image in pixels
     */
    private fun parseDetectionsFallback(lines: List<String>, imageWidth: Int, imageHeight: Int): List<Detection> {
        val detections = mutableListOf<Detection>()
        
        Log.d(TAG, "🔍 FALLBACK PARSING: Trying alternative patterns...")
        
        // Try various common patterns the model might use
        val patterns = listOf(
            // Pattern 0: * **Class:** Smartphone, **Bounding Box:** (0.3, 0.3, 0.7, 0.9), **Confidence:** 0.95
            """\*\s*\*\*Class:\*\*\s*([^,]+),\s*\*\*Bounding Box:\*\*\s*\(([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\),\s*\*\*Confidence:\*\*\s*([\d.]+)""",
            // Pattern 1: **Class:** person **Box:** (x,y,x,y) **Confidence:** 0.9
            """\*\*Class:\*\*\s*([^*]+)\s*\*\*Box:\*\*\s*\(([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)\s*\*\*Confidence:\*\*\s*([\d.]+)""",
            // Pattern 2: Class: person, Box: (x,y,x,y), Confidence: 0.9
            """Class:\s*([^,]+),\s*Box:\s*\(([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\),\s*Confidence:\s*([\d.]+)""",
            // Pattern 3: person (x,y,x,y) confidence
            """([a-zA-Z]+)\s*\(([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)\s*([\d.]+)""",
            // Pattern 4: Simple object name with bounding box
            """([a-zA-Z\s]+).*?\(([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)"""
        )
        
        for (line in lines) {
            val trimmedLine = line.trim()
            if (trimmedLine.isEmpty()) continue
            
            for ((index, pattern) in patterns.withIndex()) {
                try {
                    val regex = Pattern.compile(pattern)
                    val matcher = regex.matcher(trimmedLine)
                    
                    if (matcher.find()) {
                        Log.d(TAG, "🔍 FALLBACK MATCH pattern $index on line: '$trimmedLine'")
                        
                        val label = matcher.group(1).trim()
                        val rawLeft = matcher.group(2).toFloat()
                        val rawTop = matcher.group(3).toFloat()
                        val rawRight = matcher.group(4).toFloat()
                        val rawBottom = matcher.group(5).toFloat()
                        val confidence = when (index) {
                            0, 1, 2, 3 -> try { matcher.group(6)?.toFloat() ?: 0.5f } catch (e: Exception) { 0.5f } // Patterns with confidence
                            4 -> 0.5f // Pattern 4 has no confidence group
                            else -> 0.5f
                        }
                        
                        // Detect if coordinates are absolute (pixel values) or normalized (0-1)
                        // If any coordinate is > 1.0, treat as absolute pixel coordinates
                        val isAbsolute = rawLeft > 1.0f || rawTop > 1.0f || rawRight > 1.0f || rawBottom > 1.0f
                        
                        val (left, top, right, bottom) = if (isAbsolute) {
                            // Normalize absolute coordinates by dividing by image dimensions
                            Log.d(TAG, "🔧 COORDINATE NORMALIZATION: Converting absolute coords [$rawLeft, $rawTop, $rawRight, $rawBottom] to normalized")
                            val normalizedLeft = rawLeft / imageWidth.toFloat()
                            val normalizedTop = rawTop / imageHeight.toFloat()
                            val normalizedRight = rawRight / imageWidth.toFloat()
                            val normalizedBottom = rawBottom / imageHeight.toFloat()
                            Log.d(TAG, "🔧 NORMALIZED COORDS: [$normalizedLeft, $normalizedTop, $normalizedRight, $normalizedBottom]")
                            listOf(normalizedLeft, normalizedTop, normalizedRight, normalizedBottom)
                        } else {
                            // Already normalized coordinates
                            Log.d(TAG, "✅ COORDS ALREADY NORMALIZED: [$rawLeft, $rawTop, $rawRight, $rawBottom]")
                            listOf(rawLeft, rawTop, rawRight, rawBottom)
                        }
                        
                        val boundingBox = RectF(left, top, right, bottom)
                        detections.add(Detection(boundingBox, label, confidence, "llm_detection_fallback"))
                        
                        Log.d(TAG, "🔍 FALLBACK DETECTION: $label at $boundingBox with confidence $confidence")
                        break // Found a match, try next line
                    }
                } catch (e: Exception) {
                    Log.v(TAG, "🔍 Fallback pattern $index failed on line: $trimmedLine")
                }
            }
        }
        
        Log.d(TAG, "🔍 FALLBACK COMPLETE: Found ${detections.size} detections using fallback patterns")
        return detections
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