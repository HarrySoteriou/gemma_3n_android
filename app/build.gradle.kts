// plugins block is the first thing in the file.
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    // Note: kotlin-kapt is not needed for ViewBinding, only for libraries that use annotation processing like Dagger/Hilt or Room.
    // You can likely remove it if you are not using such libraries.
    // id("kotlin-kapt")
}

android {
    namespace = "ai.myapp"
    // It's best practice to align compileSdk and targetSdk.
    // As of mid-2024, API 34 (Android 14) is the latest stable target.
    // API 35/36 are for Android 15 Beta/Preview and can be unstable.
    compileSdk = 34

    defaultConfig {
        applicationId = "ai.myapp"
        minSdk = 31
        targetSdk = 34 // Align with compileSdk for stability
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters.add("arm64-v8a")
        }
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

    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }

    buildFeatures {
        viewBinding = true
    }

    androidResources {
        noCompress.addAll(listOf(".litertlm", ".task", ".tflite"))
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

// NPU runtime libraries
//dynamicFeatures.add(":litert_npu_runtime_libraries:mediatek_runtime")
//dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v69")
//dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v73")
//dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v75")
//dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v79")

dependencies {
    // Use the latest stable versions
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0") // Note: 1.7.1 had issues, 1.7.0 is safer
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4") // Note: 2.2.x is in alpha

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.1")

    // Lifecycle KTX
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.3")

    // CameraX dependencies - USE THE BILL OF MATERIALS (BoM)
    // The BoM ensures that all CameraX modules are version-compatible.
    val cameraxVersion = "1.3.4" // Use the latest stable version
    implementation(platform("androidx.camera:camera-bom:$cameraxVersion"))
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

    // Testing - Consolidate duplicate dependencies
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
}