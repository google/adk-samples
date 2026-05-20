package com.example.adkdemoapp

import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.adkdemoapp.agents.FunFactsAgent
import com.google.adk.kt.agents.RunConfig
import com.google.adk.kt.agents.StreamingMode
import com.google.adk.kt.runners.InMemoryRunner
import com.google.adk.kt.types.Content
import com.google.adk.kt.types.Role
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ChatViewModel : ViewModel() {
    private val TAG = "ChatViewModel"
    var inputText by mutableStateOf("")
        private set

    val messages = mutableStateListOf<Pair<String, String>>()

    private val agent = FunFactsAgent.rootAgent
    private val runner = InMemoryRunner(agent)

    fun onInputTextChange(newValue: String) {
        inputText = newValue
    }

    fun sendMessage() {
        if (inputText.isNotBlank()) {
            val userMsg = inputText
            Log.d(TAG, "sendMessage: $userMsg")
            messages.add("User" to userMsg)
            inputText = ""
            viewModelScope.launch {
                var agentMessageIndex = -1
                var accumulatedText = ""
                try {
                    Log.d(TAG, "Starting runAsync")
                    withContext(Dispatchers.IO) {
                        runner.runAsync(
                            userId = "android-user",
                            sessionId = "android-session",
                            newMessage = Content.fromText(Role.USER, userMsg),
                            runConfig = RunConfig(streamingMode = StreamingMode.SSE)
                        ).onEach { event ->
                            Log.d(TAG, "Received event: $event")
                            val content = event.content
                            if (content != null) {
                                val chunkText = content.parts.mapNotNull { it.text }.joinToString("")
                                if (event.partial) {
                                    accumulatedText += chunkText
                                } else {
                                    // Final aggregated content
                                    accumulatedText = chunkText
                                }

                                if (accumulatedText.isNotBlank()) {
                                    withContext(Dispatchers.Main) {
                                        if (agentMessageIndex == -1) {
                                            messages.add("Agent" to accumulatedText)
                                            agentMessageIndex = messages.size - 1
                                        } else {
                                            messages[agentMessageIndex] = "Agent" to accumulatedText
                                        }
                                    }
                                }
                            }
                        }.collect()
                    }
                    Log.d(TAG, "Finished collecting events")
                } catch (e: Exception) {
                    Log.e(TAG, "Error in runAsync", e)
                    withContext(Dispatchers.Main) {
                        messages.add("Error" to (e.message ?: "Unknown error"))
                    }
                }
            }
        }
    }
}
