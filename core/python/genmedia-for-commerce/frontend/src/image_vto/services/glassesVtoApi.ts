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
 * API service for Glasses Image VTO (Virtual Try-On) functionality
 * Handles communication with the backend glasses VTO generation endpoints using SSE streaming
 */

import { dataUrlToBlob, base64ToImageUrl } from './imageVtoApi'
import type { VTOResult, VTOStreamEvent } from './imageVtoApi'

const API_BASE_URL = '/api/glasses';

export interface GenerateGlassesVTORequest {
  glassesImage: string; // data URL
  glassesImage2?: string; // optional second glasses image data URL
  modelImage: string; // data URL (front face)
  numVariations?: number;
}

/**
 * Generates glasses VTO images with SSE streaming
 * Returns results progressively as they complete
 */
export async function generateGlassesVTO(
  request: GenerateGlassesVTORequest,
  onResult: (result: VTOResult) => void,
  onComplete: (total: number) => void,
  onError: (error: string) => void,
  onReferenceFace?: (base64: string) => void
): Promise<void> {
  const formData = new FormData();

  formData.append('model_image', dataUrlToBlob(request.modelImage), 'model.png');
  formData.append('product_image', dataUrlToBlob(request.glassesImage), 'glasses.png');

  if (request.glassesImage2) {
    formData.append('product_image2', dataUrlToBlob(request.glassesImage2), 'glasses2.png');
  }

  formData.append('num_variations', String(request.numVariations ?? 3));

  try {
    const response = await fetch(`${API_BASE_URL}/generate-vto`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `Request failed with status ${response.status}`);
    }

    // Parse SSE stream
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

      // Process complete SSE events
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.substring(6).trim();
          if (!jsonStr) continue;

          try {
            const event: VTOStreamEvent = JSON.parse(jsonStr);

            if (event.error) {
              onError(event.error);
              continue;
            }

            if (event.reference_face_base64 && event.index === undefined && event.status === undefined) {
              onReferenceFace?.(event.reference_face_base64);
              continue;
            }

            if (event.status === 'complete') {
              onComplete(event.total ?? 0);
              continue;
            }

            if (event.index !== undefined) {
              const result: VTOResult = {
                index: event.index,
                status: event.status === 'ready' ? 'ready' : 'failed',
                error: event.error,
              };

              if (event.image_base64) {
                result.imageUrl = base64ToImageUrl(event.image_base64);
                result.imageBase64 = event.image_base64;
              }

              if (event.evaluation) {
                result.evaluation = event.evaluation;
              }

              if (event.final_score !== undefined) {
                result.final_score = event.final_score;
              }
              if (event.face_score !== undefined) {
                result.face_score = event.face_score;
              }
              if (event.glasses_evaluation?.glasses_score !== undefined) {
                result.glasses_score = event.glasses_evaluation.glasses_score;
              }

              onResult(result);
            }
          } catch (parseError) {
            console.error('Failed to parse SSE event:', parseError, jsonStr);
          }
        }
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error occurred';
    onError(message);
  }
}
