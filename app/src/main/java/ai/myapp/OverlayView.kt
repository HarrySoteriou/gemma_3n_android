package ai.myapp

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import kotlin.math.max
import kotlin.math.min

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<GemmaBridge.Detection> = emptyList()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()
    private var scaleFactor: Float = 1f
    private var bounds = Rect()
    private var outputWidth = 0
    private var outputHeight = 0
    private var outputRotate = 0

    init {
        initPaints()
    }

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

    fun clear() {
        detections = emptyList()
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    fun setResults(
        detections: List<GemmaBridge.Detection>,
        outputHeight: Int,
        outputWidth: Int,
        imageRotation: Int
    ) {
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

        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        detections.forEach { detection ->
            val boxRect = RectF(detection.boundingBox)

            // Apply rotation matrix
            val matrix = Matrix()
            matrix.postTranslate(-outputWidth / 2f, -outputHeight / 2f)
            matrix.postRotate(outputRotate.toFloat())
            if (outputRotate == 90 || outputRotate == 270) {
                matrix.postTranslate(outputHeight / 2f, outputWidth / 2f)
            } else {
                matrix.postTranslate(outputWidth / 2f, outputHeight / 2f)
            }
            matrix.mapRect(boxRect)

            // Scale to view
            val top = boxRect.top * scaleFactor
            val bottom = boxRect.bottom * scaleFactor
            val left = boxRect.left * scaleFactor
            val right = boxRect.right * scaleFactor

            boxPaint.color = when (detection.classification) {
                "critical" -> Color.RED
                "high-risk" -> Color.YELLOW
                "medium-risk" -> Color.BLUE
                else -> Color.GREEN
            }

            // Draw bounding box
            val drawableRect = RectF(left, top, right, bottom)
            canvas.drawRect(drawableRect, boxPaint)

            // Create text
            val drawableText =
                "${detection.label} (${String.format("%.2f", detection.confidence)}) - ${detection.classification}"

            // Draw text background
            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            // Draw text
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
        }
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}