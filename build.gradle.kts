plugins {
    id("com.android.application") version "8.12.0" apply false
    id("org.jetbrains.kotlin.android") version "2.2.0" apply false
}
val buildToolsVersion by extra("36.0.0")  // Use stable build tools

tasks.register<Delete>("clean") {
    delete(layout.buildDirectory)
}