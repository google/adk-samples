plugins {
    kotlin("jvm") version "2.1.20"
    id("com.google.devtools.ksp") version "2.1.20-2.0.1"
    application
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("com.google.adk:google-adk-kotlin-core:0.1.0")
    implementation("com.google.adk:google-adk-kotlin-webserver:0.1.0")
    ksp("com.google.adk:google-adk-kotlin-processor:0.1.0")
}

kotlin {
    jvmToolchain(17)
}

application {
    mainClass.set(
        project.findProperty("mainClass") as? String
            ?: "com.google.adk.samples.agents.funfacts.MainKt"
    )
}
