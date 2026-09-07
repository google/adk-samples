/**
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

/**
 * API service for Background Changer functionality
 * Handles communication with the backend background change endpoints
 */

const API_BASE_URL = '/api/other/background-changer';

export interface ChangeBackgroundRequest {
  personImage: File | Blob;
  backgroundDescription?: string;
  backgroundImage?: File | Blob;
  numVariations?: number;
}

export interface BackgroundChangeEvaluation {
  similarity_percentage: number;
  distance: number;
  model: string;
  face_detected: boolean;
  error?: string;
}

export interface BackgroundChangeResult {
  index: number;
  status: 'ready' | 'failed';
  image_base64?: string;
  evaluation?: BackgroundChangeEvaluation;
  error?: string;
}

/**
 * Generates background change images using Server-Sent Events for streaming results
 */
export async function changeBackground(
  request: ChangeBackgroundRequest,
  onProgress: (result: BackgroundChangeResult) => void,
  onComplete: () => void,
  onError: (error: string) => void
): Promise<void> {
  const formData = new FormData();
  formData.append('person_image', request.personImage);

  if (request.backgroundDescription) {
    formData.append('background_description', request.backgroundDescription);
  }

  if (request.backgroundImage) {
    formData.append('background_image', request.backgroundImage);
  }

  formData.append('num_variations', String(request.numVariations ?? 4));

  try {
    const response = await fetch(`${API_BASE_URL}/change-background`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `Request failed with status ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));

          if (data.error) {
            onError(data.error);
            return;
          }

          if (data.status === 'complete') {
            onComplete();
            return;
          }

          if (data.index !== undefined) {
            onProgress(data as BackgroundChangeResult);
          }
        }
      }
    }
  } catch (error) {
    onError(error instanceof Error ? error.message : 'Unknown error');
  }
}

/**
 * Converts a data URL (base64 with prefix) to raw base64 string
 */
export function dataUrlToBase64(dataUrl: string): string {
  if (!dataUrl.startsWith('data:')) {
    return dataUrl;
  }
  const base64Index = dataUrl.indexOf(',');
  if (base64Index === -1) {
    return dataUrl;
  }
  return dataUrl.substring(base64Index + 1);
}

/**
 * Converts a data URL to a Blob
 */
export function dataUrlToBlob(dataUrl: string): Blob {
  const base64 = dataUrlToBase64(dataUrl);
  const byteString = atob(base64);
  const mimeString = dataUrl.split(',')[0].split(':')[1].split(';')[0];
  const ab = new ArrayBuffer(byteString.length);
  const ia = new Uint8Array(ab);
  for (let i = 0; i < byteString.length; i++) {
    ia[i] = byteString.charCodeAt(i);
  }
  return new Blob([ab], { type: mimeString });
}

/**
 * Converts base64 string to data URL
 */
export function base64ToDataUrl(base64: string, mimeType: string = 'image/png'): string {
  return `data:${mimeType};base64,${base64}`;
}
