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

class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var gemmaBridge: GemmaBridge

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        Log.d(TAG, "🔄 Creating GemmaBridge...")
        gemmaBridge = GemmaBridge(this, this) // context, lifecycleOwner
        Log.d(TAG, "✅ GemmaBridge created successfully")
        
        Log.d(TAG, "🔄 Starting async initialization...")
        gemmaBridge.initializeAsync()

        if (allPermissionsGranted()) {
            Log.i(TAG, "✅ All permissions already granted")
            startCamera()
        } else {
            Log.i(TAG, "🔄 Requesting permissions...")
            Log.i(TAG, "   - Camera permission (for live camera feed)")
            Log.i(TAG, "   - Storage permission (to access AI model)")
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
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
                startCamera()
            } else {
                // Log which permissions are missing
                val deniedPermissions = permissions.filterIndexed { index, permission ->
                    grantResults[index] != PackageManager.PERMISSION_GRANTED
                }
                Log.e(TAG, "❌ Permissions denied: $deniedPermissions")
                Log.e(TAG, "❌ The app needs these permissions to function properly:")
                Log.e(TAG, "   - CAMERA: For camera access")
                Log.e(TAG, "   - READ_EXTERNAL_STORAGE: To access the AI model file")
                finish()
            }
        }
    }

    private fun startCamera() {
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
                    it.setAnalyzer(cameraExecutor, { image -> 
                        Log.v(TAG, "📸 Processing camera frame...")
                        gemmaBridge.processFrame(image)
                    })
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer)
                Log.d(TAG, "Camera started successfully")
            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        gemmaBridge.cleanup()
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