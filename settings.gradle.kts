pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "Gemma3N"
include(":app")

// NPU runtime libraries
include(":litert_npu_runtime_libraries:runtime_strings")

include(":litert_npu_runtime_libraries:mediatek_runtime")
include(":litert_npu_runtime_libraries:qualcomm_runtime_v69")
include(":litert_npu_runtime_libraries:qualcomm_runtime_v73")
include(":litert_npu_runtime_libraries:qualcomm_runtime_v75")
include(":litert_npu_runtime_libraries:qualcomm_runtime_v79")