package com.dunghn2201.cameradetectionobject

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import com.dunghn2201.cameradetectionobject.ml.SsdMobilenetV11Metadata1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

/**
 * Helper class that encapsulates object detection logic using the bundled
 * TensorFlow Lite model. The [detect] function runs inference on a bitmap and
 * returns a new bitmap with bounding boxes and labels drawn over it.
 */
class Detector(context: Context) {

    private val colors = listOf(
        Color.BLUE,
        Color.GREEN,
        Color.RED,
        Color.CYAN,
        Color.GRAY,
        Color.BLACK,
        Color.DKGRAY,
        Color.MAGENTA,
        Color.YELLOW,
        Color.RED
    )

    private val labels: List<String> = FileUtil.loadLabels(context, "labels.txt")
    private val paint = Paint()
    private val imageProcessor: ImageProcessor =
        ImageProcessor.Builder()
            .add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR))
            .build()
    private val model: SsdMobilenetV11Metadata1 =
        SsdMobilenetV11Metadata1.newInstance(context)

    /**
     * Runs object detection on [bitmap] and draws results above the image.
     * @param threshold Minimum confidence score required to display a detection.
     */
    fun detect(bitmap: Bitmap, threshold: Float): Bitmap {
        var image = TensorImage.fromBitmap(bitmap)
        image = imageProcessor.process(image)

        val outputs = model.process(image)
        val locations = outputs.locationsAsTensorBuffer.floatArray
        val classes = outputs.classesAsTensorBuffer.floatArray
        val scores = outputs.scoresAsTensorBuffer.floatArray

        val mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutable)

        val h = mutable.height
        val w = mutable.width

        paint.textSize = h / 15f
        paint.strokeWidth = h / 85f

        scores.forEachIndexed { index, score ->
            val offset = index * 4
            if (score > threshold) {
                paint.color = colors[index % colors.size]
                paint.style = Paint.Style.STROKE
                canvas.drawRect(
                    RectF(
                        locations[offset + 1] * w,
                        locations[offset] * h,
                        locations[offset + 3] * w,
                        locations[offset + 2] * h
                    ), paint
                )
                paint.style = Paint.Style.FILL
                canvas.drawText(
                    labels[classes[index].toInt()] + " " + "%.2f".format(score),
                    locations[offset + 1] * w,
                    locations[offset] * h,
                    paint
                )
            }
        }

        return mutable
    }

    /** Releases the underlying ML model. */
    fun close() {
        model.close()
    }
}

