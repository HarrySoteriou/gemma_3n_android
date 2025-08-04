package ai.myapp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.camera.core.resolutionselector.ResolutionSelector
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
import android.util.Size
import androidx.camera.core.resolutionselector.ResolutionStrategy
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
        Log.e("APP_DEBUG", "🚀 APP STARTING - MainActivity onCreate() called!")
        
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)
        
        Log.e("APP_DEBUG", "🎯 APP LAYOUT SET - UI should be visible now!")

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

        Log.e("APP_DEBUG", "🎯 PERMISSIONS CHECK STARTED")
        
        // Check permissions first, then initialize model
        if (!allPermissionsGranted()) {
            Log.e("APP_DEBUG", "🔐 Requesting permissions...")
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        } else {
            arePermissionsGranted = true
            Log.e("APP_DEBUG", "✅ PERMISSIONS ALREADY GRANTED")
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
        Log.i(TAG, "✅ [2/5] MODEL INITIALIZATION CALLBACK: Received model initialization complete signal")
        Log.d(TAG, "🔍 Verifying model readiness for camera streaming...")
        
        // Double-check that model is truly ready for streaming
        if (::objectDetection.isInitialized && objectDetection.isReady()) {
            isModelInitialized = true
            loadingMessage.text = "Model ready! Starting camera..."
            Log.i(TAG, "✅ [2/5] MODEL VERIFIED READY: All checks passed - proceeding to camera startup")
            
            // Small delay to ensure UI updates
            lifecycleScope.launch {
                kotlinx.coroutines.delay(500)
                Log.d(TAG, "📹 Initiating camera startup sequence...")
                startCamera()
            }
        } else {
            Log.w(TAG, "⚠️ [2/5] MODEL INITIALIZATION INCONSISTENT: Callback received but model not ready")
            Log.w(TAG, "🔍 Debug info: objectDetection.isInitialized=${::objectDetection.isInitialized}, objectDetection.isReady()=${if (::objectDetection.isInitialized) objectDetection.isReady() else "N/A"}")
            onModelInitializationFailed("Model initialization incomplete")
        }
    }

    /**
     * Called when object detection model initialization fails
     */
    fun onModelInitializationFailed(errorMsg: String) {
        Log.e(TAG, "❌ [2/5] MODEL INITIALIZATION FAILED: $errorMsg")
        isModelInitialized = false
        
        if (errorMsg.contains("Model file not found")) {
            showError("Model file missing. Please copy gemma-3n-E2B-it-int4.task to device internal storage using Android Studio Device File Explorer.")
        } else {
            showError("Model initialization failed: $errorMsg")
        }
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
            Log.i(TAG, "🎯 [4/5] DETECTION RESULTS RECEIVED: ${it.detections.size} objects detected in ${it.inferenceTime}ms")
            Log.d(TAG, "📊 [4/5] Result details: Image ${it.inputImageWidth}x${it.inputImageHeight}, rotation ${it.inputImageRotation}°")
            
            it.detections.forEachIndexed { index, detection ->
                Log.v(TAG, "🎯 [4/5] Object #${index + 1}: '${detection.label}' at ${detection.boundingBox}")
            }
            
            runOnUiThread {
                Log.v(TAG, "🖼️ Updating UI overlay with detection results...")
                viewBinding.overlay.setResults(
                    it.detections,
                    it.inputImageHeight,
                    it.inputImageWidth,
                    it.inputImageRotation
                )
                Log.v(TAG, "✅ UI overlay updated successfully")
            }
        } ?: run {
            Log.w(TAG, "⚠️ [4/5] DETECTION RESULTS: Received null result bundle")
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

        if (!::objectDetection.isInitialized || !objectDetection.isReady()) {
            Log.w(TAG, "⚠️ Cannot start camera - ObjectDetection not ready for streaming")
            showError("Object detection model not ready")
            return
        }

        Log.d(TAG, "📹 Starting camera with initialized model...")
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(
                        Size(512, 384),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                    )
                )
                .build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        if (isModelInitialized &&
                            ::objectDetection.isInitialized &&
                            objectDetection.isReady()
                        ) {
                            Log.v(
                                TAG,
                                "📸 [3/5] FRAME CAPTURE: Processing camera frame ${image.width}x${image.height} with multimodal detection..."
                            )
                            lifecycleScope.launch {
                                objectDetection.detectLivestreamFrame(image)
                            }
                        } else {
                            Log.v(
                                TAG,
                                "⏭️ [5/5] FRAME DISCARDED: Model not ready - skipping frame to load next one"
                            )
                            Log.v(
                                TAG,
                                "🔍 Debug: isModelInitialized=$isModelInitialized, objectDetectionReady=${if (::objectDetection.isInitialized) objectDetection.isReady() else false}"
                            )
                            image.close()
                        }
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            // Create a ViewPort to crop the output to a 1:1 aspect ratio
            val viewPort = androidx.camera.core.ViewPort.Builder(
                android.util.Rational(1, 1),
                preview.targetRotation
            ).build()

            val useCaseGroup = androidx.camera.core.UseCaseGroup.Builder()
                .addUseCase(preview)
                .addUseCase(imageAnalyzer)
                .setViewPort(viewPort)
                .build()

            try {
                cameraProvider.unbindAll()
                // Bind the UseCaseGroup to the lifecycle
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, useCaseGroup
                )
                Log.i(
                    TAG,
                    "✅ [3/5] FRAME CAPTURE STARTED: Camera successfully bound with multimodal object detection"
                )
                Log.d(TAG, "📹 Live camera feed is now active and ready to capture frames")
                hideLoading()
            } catch (exc: Exception) {
                Log.e(TAG, "❌ [3/5] FRAME CAPTURE FAILED: Use case binding failed", exc)
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