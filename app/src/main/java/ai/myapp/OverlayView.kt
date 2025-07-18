package ai.myapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<GemmaBridge.Detection> = emptyList()
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
    }
    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }

    fun setDetections(detections: List<GemmaBridge.Detection>) {
        this.detections = detections
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        detections.forEach { detection ->
            val box = detection.boundingBox
            val label = "${detection.label} (${String.format("%.2f", detection.confidence)}) - ${detection.classification}"

            boxPaint.color = when (detection.classification) {
                "critical" -> Color.RED
                "high-risk" -> Color.YELLOW
                "medium-risk" -> Color.BLUE
                else -> Color.GREEN
            }

            canvas.drawRect(box, boxPaint)
            canvas.drawText(label, box.left, box.top - 10, textPaint)
        }
    }
}