package ai.myapp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import ai.myapp.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

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

        // Initialize camera executor
        cameraExecutor = Executors.newSingleThreadExecutor()

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
            objectDetection = ObjectDetection(context=this, listener=this)
            
            // Initialize the model asynchronously
            loadingMessage.text = "Initializing multimodal model..."
            lifecycleScope.launch {
                try {
                    objectDetection.initialise()
                    
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
        Log.d(TAG, "✅ Model initialization complete - verifying readiness...")
        
        // Double-check that model is truly ready for streaming
        if (::objectDetection.isInitialized && objectDetection.isReady()) {
            isModelInitialized = true
            loadingMessage.text = "Model ready! Starting camera..."
            Log.d(TAG, "✅ Model verified ready - starting camera")
            
            // Small delay to ensure UI updates
            lifecycleScope.launch {
                kotlinx.coroutines.delay(500)
                startCamera()
            }
        } else {
            Log.w(TAG, "⚠️ Model initialization reported complete but model not ready")
            onModelInitializationFailed("Model initialization incomplete")
        }
    }

    /**
     * Called when object detection model initialization fails
     */
    fun onModelInitializationFailed(errorMsg: String) {
        Log.e(TAG, "❌ Model initialization failed: $errorMsg")
        isModelInitialized = false
        showError("Model initialization failed: $errorMsg")
    }

    // AFTER
    override fun onError(msg: String) {
        Log.e(TAG, "🚨 ObjectDetection error: $msg")
        runOnUiThread {
            // You could show error messages to the user here if needed
            showError("An error occurred: $msg")
        }
    }

    // FIX: Handle the nullable ResultBundle correctly
    override fun onResults(result: ObjectDetection.ResultBundle?) {
        // This is the main fix for the null-safety errors
        result?.let {
            Log.v(TAG, "🎯 Received ${it.detections.size} detections")
            runOnUiThread {
                viewBinding.overlay.setResults(
                    it.detections,
                    it.inputImageHeight,
                    it.inputImageWidth,
                    // NOTE: You still need to add `inputImageRotation` to your ResultBundle
                    // For now, we'll pass 0 as a placeholder.
                    0 // it.inputImageRotation
                )
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
            showError("Model not ready for camera")
            return
        }

        if (!allPermissionsGranted()) {
            Log.w(TAG, "⚠️ Cannot start camera - permissions not granted")
            showError("Camera permissions not granted")
            return
        }

        // Final check that ObjectDetection is ready for streaming
        if (!::objectDetection.isInitialized || !objectDetection.isReady()) {
            Log.w(TAG, "⚠️ Cannot start camera - ObjectDetection not ready for streaming")
            showError("Object detection model not ready")
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
                        // Triple-check model is ready before processing any frame
                        if (isModelInitialized && 
                            ::objectDetection.isInitialized && 
                            objectDetection.isReady()) {
                            Log.v(TAG, "📸 Processing camera frame with multimodal detection...")
                            lifecycleScope.launch {
                                objectDetection.detectLivestreamFrame(image)
                            }
                        } else {
                            Log.w(TAG, "⏭️ Skipping frame - model not ready (initialized: $isModelInitialized, objectDetectionReady: ${if (::objectDetection.isInitialized) objectDetection.isReady() else false})")
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