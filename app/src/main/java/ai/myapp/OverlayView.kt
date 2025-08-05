package ai.myapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.util.Log
import android.view.View
import kotlin.math.max

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<ObjectDetection.Detection> = emptyList()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()
    private var scaleFactor: Float = 1f
    private var bounds = Rect()
    private var outputWidth = 0
    private var outputHeight = 0
    private var outputRotate = 0

    private val boxRect = RectF()
    private val matrix = Matrix()
    private val drawableRect = RectF()

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.strokeWidth = 8f
        boxPaint.style = Paint.Style.STROKE
    }
    init {
        initPaints()
    }
    fun clear() {
        detections = emptyList()
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    fun setResults(
        detections: List<ObjectDetection.Detection>,
        outputHeight: Int,
        outputWidth: Int,
        imageRotation: Int
    ) {
        Log.d("OverlayView", "📊 Setting ${detections.size} detections on overlay (${outputWidth}x${outputHeight}, rotation: ${imageRotation}°)")

        this.detections = detections
        this.outputWidth = outputWidth
        this.outputHeight = outputHeight
        this.outputRotate = imageRotation

        val rotatedWidthHeight = when (imageRotation) {
            0, 180 -> Pair(outputWidth, outputHeight)
            90, 270 -> Pair(outputHeight, outputWidth)
            else -> return
        }

        // Assume LIVE_STREAM mode for scaling (FILL_START)
        scaleFactor = max(
            width * 1f / rotatedWidthHeight.first,
            height * 1f / rotatedWidthHeight.second
        )

        Log.d("OverlayView", "📏 Overlay view size: ${width}x${height}, scale factor: $scaleFactor")

        // Log detections with bounding boxes
        detections.forEachIndexed { index, detection ->
            if (detection.boundingBox != null) {
                Log.d("OverlayView", "🎯 Detection $index: ${detection.label} bbox[${detection.boundingBox.left}, ${detection.boundingBox.top}, ${detection.boundingBox.right}, ${detection.boundingBox.bottom}]")
            } else {
                Log.d("OverlayView", "📝 Detection $index: ${detection.label} (text only)")
            }
        }

        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        detections.forEachIndexed { index, detection ->
            // Set color based on classification
            boxPaint.color = when (detection.classification.lowercase()) {
                "critical" -> Color.RED
                "high" -> Color.rgb(255, 165, 0) // Orange
                "medium" -> Color.YELLOW
                "low" -> Color.GREEN
                "llm_detection" -> Color.CYAN
                else -> Color.WHITE
            }

            // Handle detections with bounding boxes
            detection.boundingBox?.let { boundingBox ->
                // Convert normalized coordinates (0.0-1.0) to pixel coordinates with rounding
                val pixelLeft = kotlin.math.round(boundingBox.left * outputWidth)
                val pixelTop = kotlin.math.round(boundingBox.top * outputHeight)
                val pixelRight = kotlin.math.round(boundingBox.right * outputWidth)
                val pixelBottom = kotlin.math.round(boundingBox.bottom * outputHeight)
                
                boxRect.set(pixelLeft, pixelTop, pixelRight, pixelBottom)
                
                Log.v("OverlayView", "🎨 Drawing bbox for ${detection.label}: original[${boundingBox.left}, ${boundingBox.top}, ${boundingBox.right}, ${boundingBox.bottom}]")
                Log.v("OverlayView", "🎨 Converted to pixels: [${pixelLeft}, ${pixelTop}, ${pixelRight}, ${pixelBottom}]")

                // Apply rotation matrix
                matrix.reset()
                matrix.postTranslate(-outputWidth / 2f, -outputHeight / 2f)
                matrix.postRotate(outputRotate.toFloat())
                if (outputRotate == 90 || outputRotate == 270) {
                    matrix.postTranslate(outputHeight / 2f, outputWidth / 2f)
                } else {
                    matrix.postTranslate(outputWidth / 2f, outputHeight / 2f)
                }
                matrix.mapRect(boxRect)

                // Scale to view with rounding to avoid floating point precision issues
                val top = kotlin.math.round(boxRect.top * scaleFactor)
                val bottom = kotlin.math.round(boxRect.bottom * scaleFactor)
                val left = kotlin.math.round(boxRect.left * scaleFactor)
                val right = kotlin.math.round(boxRect.right * scaleFactor)

                Log.v("OverlayView", "🖼️ Final screen coords for ${detection.label}: [${left}, ${top}, ${right}, ${bottom}] (scaleFactor: $scaleFactor)")

                // Draw bounding box
                drawableRect.set(left, top, right, bottom)
                canvas.drawRect(drawableRect, boxPaint)

                // Create text for bounding box detection
                val drawableText = "${detection.label} (${String.format("%.2f", detection.confidence)})"

                // Draw text background
                textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
                val textWidth = bounds.width()
                val textHeight = bounds.height()
                drawableRect.set(
                    left,
                    top,
                    left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                    top + textHeight + BOUNDING_RECT_TEXT_PADDING
                )
                canvas.drawRect(drawableRect, textBackgroundPaint)

                // Draw text with rounded positioning
                canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
                
                Log.d("OverlayView", "✅ Drew bounding box for ${detection.label} at screen coords [${left.toInt()}, ${top.toInt()}, ${right.toInt()}, ${bottom.toInt()}]")
            } ?: run {
                // Handle detections without bounding boxes (LLM text-only descriptions)
                // Display them as a list on the side
                val yPosition = 100f + (index * 80f) // Stack vertically
                val xPosition = 20f
                
                // Create text for description-only detection
                val drawableText = "${detection.label} (${String.format("%.2f", detection.confidence)})"
                
                // Draw text background
                textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
                val textWidth = bounds.width()
                val textHeight = bounds.height()
                // REUSE drawableRect here as well
                drawableRect.set(
                    xPosition,
                    yPosition - textHeight,
                    xPosition + textWidth + BOUNDING_RECT_TEXT_PADDING,
                    yPosition + BOUNDING_RECT_TEXT_PADDING
                )
                canvas.drawRect(drawableRect, textBackgroundPaint)
                // Draw text
                canvas.drawText(drawableText, xPosition, yPosition, textPaint)
                
                // Draw small indicator circle
                boxPaint.style = Paint.Style.FILL
                canvas.drawCircle(xPosition - 15f, yPosition - textHeight/2, 8f, boxPaint)
                boxPaint.style = Paint.Style.STROKE // Reset to stroke
            }
        }
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}