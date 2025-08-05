package ai.myapp

import ai.myapp.databinding.ActivityMainBinding
import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), ObjectDetection.DetectorListener {
    // The single, most important object that gives access to all your views.
    private lateinit var viewBinding: ActivityMainBinding

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var objectDetection: ObjectDetection

    // Track initialization state
    private var isModelInitialized = false
    private var arePermissionsGranted = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.e("APP_DEBUG", "🚀 APP STARTING - MainActivity onCreate() called!")

        // Step 1: Inflate the layout and set the content view.
        // From this point on, use 'viewBinding' to access all views.
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        Log.e("APP_DEBUG", "🎯 APP LAYOUT SET - UI should be visible now!")
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Step 2: Set up listeners using the viewBinding object.
        viewBinding.retryButton.setOnClickListener {
            showLoading()
            viewBinding.loadingMessage.text = "Retrying model initialization..."
            isModelInitialized = false
            initializeModel()
        }

        // Show initial loading screen
        showLoading()
        viewBinding.loadingMessage.text = "Checking permissions..."

        Log.e("APP_DEBUG", "🎯 PERMISSIONS CHECK STARTED")
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
            // *** FIX: Use the correct TAG from the companion object ***
            Log.w(MainActivity.TAG, "⚠️ Cannot initialize model without permissions")
            return
        }

        viewBinding.loadingMessage.text = "Loading AI model..."
        // *** FIX: Use the correct TAG from the companion object ***
        Log.d(MainActivity.TAG, "🔄 Creating ObjectDetection...")

        objectDetection = ObjectDetection(context = this, listener = this)

        viewBinding.loadingMessage.text = "Initializing multimodal model..."
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // This is a suspend function and needs to be called from a coroutine
                objectDetection.initialise()

                // Switch back to the main thread to update the UI.
                withContext(Dispatchers.Main) {
                    onModelInitialized()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    onModelInitializationFailed("Fatal Error: ${e.message}")
                }
            }
        }
    }
    private fun onModelInitialized() {
        Log.i(TAG, "✅ Model initialization callback received.")
        if (::objectDetection.isInitialized && objectDetection.isReady()) {
            isModelInitialized = true
            viewBinding.loadingMessage.text = "Model ready! Starting camera..."
            // Small delay to ensure UI updates before starting the camera
            lifecycleScope.launch {
                delay(500)
                startCamera()
            }
        } else {
            onModelInitializationFailed("Model initialization incomplete.")
        }
    }

    private fun onModelInitializationFailed(errorMsg: String) {
        Log.e(TAG, "❌ MODEL INITIALIZATION FAILED: $errorMsg")
        isModelInitialized = false
        showError(errorMsg)
    }

    override fun onError(msg: String) {
        Log.e(TAG, "🚨 ObjectDetection error: $msg")
        runOnUiThread {
            Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
            // Optionally, show the error in the main error view
            showError("An error occurred: $msg")
        }
    }

    override fun onResults(result: ObjectDetection.ResultBundle?) {
        result?.let {
            Log.i(TAG, "🎯 DETECTION RESULTS: ${it.detections.size} objects in ${it.inferenceTime}ms")
            runOnUiThread {
                viewBinding.overlay.setResults(
                    it.detections,
                    it.inputImageHeight,
                    it.inputImageWidth,
                    it.inputImageRotation
                )
            }
        } ?: run {
            Log.w(TAG, "⚠️ DETECTION RESULTS: Received null result bundle")
        }
    }

    private fun showLoading() {
        viewBinding.loadingOverlay.visibility = View.VISIBLE
        viewBinding.loadingProgress.visibility = View.VISIBLE
        viewBinding.loadingMessage.visibility = View.VISIBLE
        viewBinding.errorMessage.visibility = View.GONE
        viewBinding.retryButton.visibility = View.GONE
    }

    private fun showError(message: String) {
        viewBinding.loadingOverlay.visibility = View.VISIBLE
        viewBinding.loadingProgress.visibility = View.GONE
        viewBinding.loadingMessage.visibility = View.GONE
        viewBinding.errorMessage.text = message
        viewBinding.errorMessage.visibility = View.VISIBLE
        viewBinding.retryButton.visibility = View.VISIBLE
    }

    private fun hideLoading() {
        viewBinding.loadingOverlay.visibility = View.GONE
        viewBinding.mainContentGroup.visibility = View.VISIBLE
        Toast.makeText(this, "Model Ready! Camera Starting...", Toast.LENGTH_SHORT).show()
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
                initializeModel()
            } else {
                val deniedPermissions = permissions.filterIndexed { index, _ ->
                    grantResults[index] != PackageManager.PERMISSION_GRANTED
                }
                Log.e(TAG, "❌ Permissions denied: $deniedPermissions")
                showError("Permissions denied. The app requires Camera and Storage access to function.")
                arePermissionsGranted = false
            }
        }
    }

    private fun startCamera() {
        if (!isModelInitialized || !arePermissionsGranted) {
            showError("Cannot start camera. Model or permissions are not ready.")
            return
        }

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
            }

            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(Size(512, 384), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                ).build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        if (isModelInitialized && ::objectDetection.isInitialized && objectDetection.isReady()) {
                            lifecycleScope.launch {
                                objectDetection.detectLivestreamFrame(image)
                            }
                        } else {
                            image.close()
                        }
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
                Log.i(TAG, "✅ Camera use cases bound successfully.")
                hideLoading()
            } catch (exc: Exception) {
                Log.e(TAG, "❌ Use case binding failed", exc)
                showError("Failed to start camera: ${exc.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        if (::objectDetection.isInitialized) {
            lifecycleScope.launch(Dispatchers.IO) {
                objectDetection.cleanup()
            }
        }
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_CODE_PERMISSIONS = 10

        private val REQUIRED_PERMISSIONS = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            arrayOf(Manifest.permission.CAMERA) // Only camera needed for Android 13+ if model is in assets
        } else {
            arrayOf(Manifest.permission.CAMERA, Manifest.permission.READ_EXTERNAL_STORAGE)
        }
    }
}