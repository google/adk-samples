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
 * API service for Image VTO (Virtual Try-On) functionality
 * Handles communication with the backend VTO generation endpoints using SSE streaming
 */

const API_BASE_URL = '/api/clothes';

export interface GenerateVTORequest {
  faceImage?: string | null; // data URL (optional - for face correction step)
  fullBodyImage: string; // data URL
  garments: string[]; // array of data URLs
  garmentUris?: string[]; // GCS URIs passed directly to Gemini
  scenario: string;
  numVariations?: number;
}

export interface VTOEvaluation {
  similarity_percentage?: number;
  confidence?: string;
  face_detected: boolean;
}

export interface VTOResult {
  index: number;
  status: 'ready' | 'failed' | 'pending' | 'discarded';
  imageUrl?: string;
  imageBase64?: string;
  evaluation?: VTOEvaluation;
  final_score?: number;
  face_score?: number;
  garments_score?: number;
  glasses_score?: number;
  error?: string;
}

export interface VTOStreamEvent {
  index?: number;
  status?: 'ready' | 'failed' | 'complete' | 'discarded';
  image_base64?: string;
  reference_face_base64?: string;
  evaluation?: VTOEvaluation;
  final_score?: number;
  face_score?: number;
  garments_evaluation?: { garments_score?: number };
  glasses_evaluation?: { glasses_score?: number };
  error?: string;
  total?: number;
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
 * Converts a data URL to a Blob for API upload
 */
export function dataUrlToBlob(dataUrl: string): Blob {
  const base64 = dataUrlToBase64(dataUrl);
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  // Extract mime type from data URL
  const mimeMatch = dataUrl.match(/^data:([^;]+);/);
  const mimeType = mimeMatch ? mimeMatch[1] : 'image/png';

  return new Blob([bytes], { type: mimeType });
}

/**
 * Converts a base64 image string to a blob URL for display
 */
export function base64ToImageUrl(base64: string): string {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: 'image/png' });
  return URL.createObjectURL(blob);
}

/**
 * Downloads an image from base64 data
 */
export function downloadImage(base64: string, filename: string = 'vto-result.png'): void {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: 'image/png' });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  setTimeout(() => URL.revokeObjectURL(url), 100);
}

/**
 * Generates VTO images with SSE streaming
 * Returns results progressively as they complete
 */
export async function generateVTO(
  request: GenerateVTORequest,
  onResult: (result: VTOResult) => void,
  onComplete: (total: number) => void,
  onError: (error: string) => void,
  onReferenceFace?: (base64: string) => void
): Promise<void> {
  const formData = new FormData();

  // Convert data URLs to blobs and append to form
  // Face image is optional - only append if provided
  if (request.faceImage) {
    formData.append('face_image', dataUrlToBlob(request.faceImage), 'face.png');
  }
  formData.append('full_body_image', dataUrlToBlob(request.fullBodyImage), 'body.png');

  // Append each garment (uploaded files)
  request.garments.forEach((garment, index) => {
    formData.append('garments', dataUrlToBlob(garment), `garment_${index}.png`);
  });

  // Append GCS URIs for catalog garments (passed directly to Gemini)
  if (request.garmentUris && request.garmentUris.length > 0) {
    formData.append('garment_uris', JSON.stringify(request.garmentUris));
  }

  formData.append('scenario', request.scenario);
  formData.append('num_variations', String(request.numVariations ?? 4));

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
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

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
                status: event.status === 'ready' ? 'ready' : event.status === 'discarded' ? 'discarded' : 'failed',
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
              if (event.garments_evaluation?.garments_score !== undefined) {
                result.garments_score = event.garments_evaluation.garments_score;
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
