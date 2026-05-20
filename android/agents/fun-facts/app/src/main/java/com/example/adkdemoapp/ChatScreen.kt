/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.adkdemoapp

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp

@Composable
fun ChatScreen(
    viewModel: ChatViewModel,
    modifier: Modifier = Modifier
) {
    ChatScreenContent(
        messages = viewModel.messages,
        inputText = viewModel.inputText,
        onInputTextChange = viewModel::onInputTextChange,
        onSendMessage = viewModel::sendMessage,
        modifier = modifier
    )
}

@Composable
fun ChatScreenContent(
    messages: List<Pair<String, String>>,
    inputText: String,
    onInputTextChange: (String) -> Unit,
    onSendMessage: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp)
            .imePadding()
    ) {
        LazyColumn(modifier = Modifier.weight(1f).fillMaxWidth()) {
            items(messages) { (sender, text) ->
                Text(
                    text = "$sender: $text",
                    modifier = Modifier.padding(vertical = 4.dp)
                )
            }
        }

        Row(verticalAlignment = Alignment.CenterVertically) {
            TextField(
                value = inputText,
                onValueChange = onInputTextChange,
                modifier = Modifier.weight(1f),
                placeholder = { Text("Ask for a fun fact...") }
            )
            Spacer(modifier = Modifier.width(8.dp))
            Button(onClick = onSendMessage) {
                Text("Send")
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun ChatScreenPreview() {
    MaterialTheme {
        ChatScreenContent(
            messages = listOf(
                "User" to "Hello!",
                "Agent" to "Hi there! I'm a fun facts agent. Ask me anything!"
            ),
            inputText = "Tell me a fun fact",
            onInputTextChange = {},
            onSendMessage = {}
        )
    }
}
