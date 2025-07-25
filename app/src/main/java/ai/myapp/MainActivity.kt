package ai.myapp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import ai.myapp.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity(), ObjectDetection.DetectorListener {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var objectDetection: ObjectDetection

    private lateinit var loadingOverlay: View
    private lateinit var loadingProgress: ProgressBar
    private lateinit var loadingMessage: TextView
    private lateinit var errorMessage: TextView
    private lateinit var retryButton: Button
    
    // Track initialization state
    private var isModelInitialized = false
    private var arePermissionsGranted = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Initialize loading UI
        loadingOverlay = viewBinding.loadingOverlay
        loadingProgress = viewBinding.loadingProgress
        loadingMessage = viewBinding.loadingMessage
        errorMessage = viewBinding.errorMessage
        retryButton = viewBinding.retryButton

        retryButton.setOnClickListener {
            showLoading()
            loadingMessage.text = "Retrying model initialization..."
            isModelInitialized = false
            initializeModel()
        }

        // Initialize camera executor
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Show initial loading
        showLoading()
        loadingMessage.text = "Checking permissions..."

        // Check permissions first, then initialize model
        if (!allPermissionsGranted()) {
            Log.d(TAG, "🔐 Requesting permissions...")
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        } else {
            arePermissionsGranted = true
            initializeModel()
        }
    }

    private fun initializeModel() {
        if (!arePermissionsGranted) {
            Log.w(TAG, "⚠️ Cannot initialize model without permissions")
            return
        }

        loadingMessage.text = "Loading AI model..."
        Log.d(TAG, "🔄 Creating ObjectDetection...")
        
        try {
            // Create ObjectDetection with this as the DetectorListener
            objectDetection = ObjectDetection(this, this, this)
            
            // Initialize the model asynchronously
            loadingMessage.text = "Initializing multimodal model..."
            lifecycleScope.launch {
                try {
                    objectDetection.initializeAsync()
                    
                    // Check if initialization was successful
                    if (objectDetection.isReady()) {
                        Log.d(TAG, "✅ ObjectDetection initialized successfully")
                        onModelInitialized()
                    } else {
                        onModelInitializationFailed("Model failed to initialize")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "❌ Model initialization failed", e)
                    onModelInitializationFailed("Model initialization failed: ${e.message}")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to create ObjectDetection", e)
            onModelInitializationFailed("Failed to create ObjectDetection: ${e.message}")
        }
    }

    /**
     * Called when object detection model initialization is complete
     */
    fun onModelInitialized() {
        Log.d(TAG, "✅ Model initialization complete - ready to start camera")
        isModelInitialized = true
        loadingMessage.text = "Starting camera..."
        
        // Now safe to start camera since model is loaded
        startCamera()
    }

    /**
     * Called when object detection model initialization fails
     */
    fun onModelInitializationFailed(errorMsg: String) {
        Log.e(TAG, "❌ Model initialization failed: $errorMsg")
        isModelInitialized = false
        showError("Model initialization failed: $errorMsg")
    }

    // Implementation of ObjectDetection.DetectorListener interface
    override fun onError(error: String, errorCode: Int) {
        Log.e(TAG, "🚨 ObjectDetection error (code: $errorCode): $error")
        runOnUiThread {
            // You could show error messages to user here if needed
            Log.e(TAG, "Detection error: $error")
        }
    }

    override fun onResults(resultBundle: ObjectDetection.ResultBundle) {
        Log.v(TAG, "🎯 Received ${resultBundle.detections.size} detections")
        
        runOnUiThread {
            // Update overlay with detection results
            viewBinding.overlay.setResults(
                resultBundle.detections,
                resultBundle.inputImageHeight,
                resultBundle.inputImageWidth,
                resultBundle.inputImageRotation
            )
            
            // Log the detections for debugging
            resultBundle.detections.forEach { detection ->
                Log.v(TAG, "📋 Detection: ${detection.label} (confidence: ${detection.confidence})")
            }
        }
    }

    fun showLoading() {
        loadingOverlay.visibility = View.VISIBLE
        loadingProgress.visibility = View.VISIBLE
        loadingMessage.visibility = View.VISIBLE
        errorMessage.visibility = View.GONE
        retryButton.visibility = View.GONE
    }

    fun showError(message: String) {
        loadingProgress.visibility = View.GONE
        loadingMessage.visibility = View.GONE
        errorMessage.text = message
        errorMessage.visibility = View.VISIBLE
        retryButton.visibility = View.VISIBLE
    }

    fun hideLoading() {
        loadingOverlay.visibility = View.GONE
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                Log.i(TAG, "✅ All permissions granted!")
                arePermissionsGranted = true
                // Now that we have permissions, initialize the model
                initializeModel()
            } else {
                // Log which permissions are missing
                val deniedPermissions = permissions.filterIndexed { index, permission ->
                    grantResults[index] != PackageManager.PERMISSION_GRANTED
                }
                Log.e(TAG, "❌ Permissions denied: $deniedPermissions")
                Log.e(TAG, "❌ The app needs these permissions to function properly:")
                Log.e(TAG, "   - CAMERA: For camera access")
                Log.e(TAG, "   - READ_EXTERNAL_STORAGE: To access the AI model file")
                showError("Permissions denied. Please grant camera and storage permissions to continue.")
                arePermissionsGranted = false
            }
        }
    }

    fun startCamera() {
        if (!isModelInitialized) {
            Log.w(TAG, "⚠️ Cannot start camera - model not initialized yet")
            return
        }

        if (!allPermissionsGranted()) {
            Log.w(TAG, "⚠️ Cannot start camera - permissions not granted")
            return
        }

        Log.d(TAG, "📹 Starting camera with initialized model...")
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        // Double-check model is ready before processing
                        if (objectDetection.isReady()) {
                            Log.v(TAG, "📸 Processing camera frame with multimodal detection...")
                            lifecycleScope.launch {
                                objectDetection.detectLivestreamFrame(image)
                            }
                        } else {
                            Log.w(TAG, "⏭️ Skipping frame - model not ready (isInitialized: ${::objectDetection.isInitialized && objectDetection.isReady()})")
                            image.close()
                        }
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer)
                Log.d(TAG, "✅ Camera started successfully with multimodal object detection")
                hideLoading()
            } catch(exc: Exception) {
                Log.e(TAG, "❌ Use case binding failed", exc)
                showError("Failed to start camera: ${exc.message}")
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::objectDetection.isInitialized) {
            // Call suspend cleanup function in a coroutine
            lifecycleScope.launch {
                objectDetection.cleanup()
            }
        }
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "ObjectDetection"
        private const val REQUEST_CODE_PERMISSIONS = 10
        
        private val REQUIRED_PERMISSIONS = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // Android 13+ (API 33+)
            arrayOf(
                Manifest.permission.CAMERA,
                Manifest.permission.READ_MEDIA_IMAGES,
                Manifest.permission.READ_MEDIA_VIDEO
            )
        } else {
            // Android 12 and below
            arrayOf(
                Manifest.permission.CAMERA,
                Manifest.permission.READ_EXTERNAL_STORAGE
            )
        }
    }
}