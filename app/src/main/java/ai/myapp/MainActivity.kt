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

class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var gemmaBridge: GemmaBridge
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
            gemmaBridge.initializeAsync()
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
        Log.d(TAG, "🔄 Creating GemmaBridge...")
        gemmaBridge = GemmaBridge(this, this) // context, lifecycleOwner
        Log.d(TAG, "✅ GemmaBridge created successfully")
        
        Log.d(TAG, "🔄 Starting async initialization...")
        gemmaBridge.initializeAsync()
    }

    /**
     * Called by GemmaBridge when model initialization is complete
     */
    fun onModelInitialized() {
        Log.d(TAG, "✅ Model initialization complete - ready to start camera")
        isModelInitialized = true
        loadingMessage.text = "Starting camera..."
        
        // Now safe to start camera since model is loaded
        startCamera()
    }

    /**
     * Called by GemmaBridge when model initialization fails
     */
    fun onModelInitializationFailed(errorMsg: String) {
        Log.e(TAG, "❌ Model initialization failed: $errorMsg")
        isModelInitialized = false
        showError("Model initialization failed: $errorMsg")
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
                        if (gemmaBridge.isReady()) {
                            Log.v(TAG, "📸 Processing camera frame...")
                            gemmaBridge.processFrame(image)
                        } else {
                            Log.v(TAG, "⏭️ Skipping frame - model not ready")
                            image.close()
                        }
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer)
                Log.d(TAG, "✅ Camera started successfully with model ready")
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
        if (::gemmaBridge.isInitialized) {
            // Call suspend cleanup function in a coroutine
            lifecycleScope.launch {
                gemmaBridge.cleanup()
            }
        }
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "Gemma3N"
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