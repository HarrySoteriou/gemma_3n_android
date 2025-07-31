// plugins block is the first thing in the file.
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("kotlin-kapt") // You need this for dataBinding

}

android {
    namespace = "ai.myapp"
    compileSdk = 36  // Use stable Android 14 API

    defaultConfig {
        applicationId = "ai.myapp"
        minSdk = 31
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // NPU only supports arm64-v8a
        ndk { abiFilters.add("arm64-v8a") }
        // Needed for Qualcomm NPU runtimes
        packaging { jniLibs { useLegacyPackaging = true } }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    // Modern way to set the Kotlin JVM target
    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }

    buildFeatures {
        viewBinding = true
        dataBinding = true
    }

    // Modern way to handle asset compression
    // NOTE: This is only needed if you bundle the model in your assets,
    // which you are not currently doing. It's good practice to have it.
    androidResources {
        noCompress.addAll(listOf(".litertlm", ".task", ".tflite"))
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
    //buildToolsVersion = rootProject.extra["buildToolsVersion"] as String
}


// NPU runtime libraries
//dynamicFeatures.add(":litert_npu_runtime_libraries:mediatek_runtime")
//dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v69")
//dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v73")
//dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v75")
//dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v79")



dependencies {
    implementation("androidx.core:core-ktx:1.16.0")
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.1")
    // Coroutines for async processing
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
    // Lifecycle KTX for lifecycleScope
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.9.2")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")

    // CameraX dependencies
    val cameraxVersion = "1.4.2"
    implementation("androidx.camera:camera-core:${cameraxVersion}")
    implementation("androidx.camera:camera-camera2:${cameraxVersion}")
    implementation("androidx.camera:camera-lifecycle:${cameraxVersion}")
    implementation("androidx.camera:camera-view:${cameraxVersion}")

    // --- MediaPipe Dependencies ---
    implementation("com.google.mediapipe:tasks-genai:0.10.25")
    //noinspection Aligned16KB
    implementation("com.google.mediapipe:tasks-vision:0.10.26.1")
    
    // --- LiteRT Next Dependencies (for advanced NPU support) ---
    // The core LiteRT Next API (includes accelerator providers)
    implementation("com.google.ai.edge.litert:litert:1.4.0")

    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")

}