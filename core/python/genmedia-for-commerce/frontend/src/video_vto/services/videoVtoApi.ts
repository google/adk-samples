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
 * API service for Video VTO (Animate Frame) functionality
 * Handles communication with the backend video generation endpoints
 */

const API_BASE_URL = '/api/glasses';

export interface GenerateVideoRequest {
  prompt: string;
  modelImage?: File | Blob;
  isAnimationMode?: boolean;
  numberOfVideos?: number;
  productImage?: File | Blob;
}

export interface GenerateVideoResponse {
  videos: string[];
  filenames: string[];
  collage_data: string;
}

export interface EnhancePromptRequest {
  text: string;
  modelImage?: File | Blob;
}

export interface EnhancePromptResponse {
  enhanced_prompt: string;
}

export interface ApiError {
  detail: string;
}

/**
 * Generates animated videos from a model image and prompt
 */
export async function generateAnimatedVideo(
  request: GenerateVideoRequest
): Promise<GenerateVideoResponse> {
  const formData = new FormData();
  formData.append('prompt', request.prompt);
  if (request.modelImage) {
    formData.append('model_image', request.modelImage);
  }
  formData.append('is_animation_mode', String(request.isAnimationMode ?? true));
  formData.append('number_of_videos', String(request.numberOfVideos ?? 4));
  if (request.productImage) {
    formData.append('product_image', request.productImage);
  }

  const response = await fetch(`${API_BASE_URL}/generate-video`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}

/**
 * Enhances a user's animation prompt using AI
 */
export async function generateAnimationPrompt(
  request: EnhancePromptRequest
): Promise<EnhancePromptResponse> {
  const formData = new FormData();
  formData.append('text', request.text);
  if (request.modelImage) {
    formData.append('model_image', request.modelImage);
  }

  const response = await fetch(`${API_BASE_URL}/generate-animation-prompt`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}

/**
 * Converts a data URL (base64 with prefix) to raw base64 string
 * Handles both data URLs and raw base64 strings
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
 * Converts a base64 video string to a blob URL for video playback
 */
export function base64ToVideoUrl(base64: string): string {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: 'video/mp4' });
  return URL.createObjectURL(blob);
}

/**
 * Creates a downloadable link for a base64 video
 */
export function downloadVideo(base64: string, filename: string = 'animation.mp4'): void {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: 'video/mp4' });
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
 * Downloads a video from a blob URL
 */
export function downloadVideoFromUrl(url: string, filename: string = 'animation.mp4'): void {
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

/**
 * Clothes Video VTO API (unified SSE pipeline)
 */

const CLOTHES_VIDEO_API_BASE_URL = '/api/clothes/video';

export interface GenerateVideoVTORequest {
  fullBodyImage: Blob;
  garments: Blob[];
  garmentUris?: string[];
  faceImage?: Blob;
  scenario?: string;
  numVariations?: number;
  numberOfVideos?: number;
  prompt?: string;
}

export interface GenerateAnimateModelRequest {
  modelImage: Blob;
  numberOfVideos?: number;
  prompt?: string;
}

export interface VideoVTOEvent {
  status: 'generating_image' | 'image_ready' | 'generating_videos' | 'videos' | 'error';
  image_base64?: string;
  final_score?: number;
  face_score?: number;
  videos?: string[];
  scores?: number[];
  filenames?: string[];
  detail?: string;
}

/**
 * Runs the unified video VTO pipeline via SSE.
 * Calls onEvent for each stage; the caller decides what to render.
 */
export async function generateVideoVTO(
  request: GenerateVideoVTORequest,
  onEvent: (event: VideoVTOEvent) => void,
  onError: (error: string) => void,
): Promise<void> {
  const formData = new FormData();
  formData.append('full_body_image', request.fullBodyImage);
  request.garments.forEach((garment) => {
    formData.append('garments', garment);
  });
  if (request.garmentUris && request.garmentUris.length > 0) {
    formData.append('garment_uris', JSON.stringify(request.garmentUris));
  }
  if (request.faceImage) {
    formData.append('face_image', request.faceImage);
  }
  formData.append('scenario', request.scenario ?? 'a plain white studio background');
  formData.append('num_variations', String(request.numVariations ?? 3));
  formData.append('number_of_videos', String(request.numberOfVideos ?? 4));
  if (request.prompt) {
    formData.append('prompt', request.prompt);
  }

  try {
    const response = await fetch(`${CLOTHES_VIDEO_API_BASE_URL}/generate-video-vto`, {
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
          const jsonStr = line.substring(6).trim();
          if (!jsonStr) continue;
          try {
            const event: VideoVTOEvent = JSON.parse(jsonStr);
            if (event.status === 'error') {
              onError(event.detail || 'Unknown pipeline error');
            } else {
              onEvent(event);
            }
          } catch (parseError) {
            console.error('Failed to parse SSE event:', parseError, jsonStr);
          }
        }
      }
    }
  } catch (error) {
    onError(error instanceof Error ? error.message : 'Unknown error occurred');
  }
}

/**
 * Animate a model image into videos (video-only, no image VTO).
 * Calls /generate-animate-model via SSE.
 */
export async function generateAnimateModel(
  request: GenerateAnimateModelRequest,
  onEvent: (event: VideoVTOEvent) => void,
  onError: (error: string) => void,
): Promise<void> {
  const formData = new FormData();
  formData.append('model_image', request.modelImage);
  formData.append('number_of_videos', String(request.numberOfVideos ?? 4));
  if (request.prompt) {
    formData.append('prompt', request.prompt);
  }

  try {
    const response = await fetch(`${CLOTHES_VIDEO_API_BASE_URL}/generate-animate-model`, {
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
          const jsonStr = line.substring(6).trim();
          if (!jsonStr) continue;
          try {
            const event: VideoVTOEvent = JSON.parse(jsonStr);
            if (event.status === 'error') {
              onError(event.detail || 'Unknown pipeline error');
            } else {
              onEvent(event);
            }
          } catch (parseError) {
            console.error('Failed to parse SSE event:', parseError, jsonStr);
          }
        }
      }
    }
  } catch (error) {
    onError(error instanceof Error ? error.message : 'Unknown error occurred');
  }
}

/**
 * Converts a base64 string (raw, no prefix) to a Blob
 */
export function base64ToBlob(base64: string, mimeType: string = 'image/png'): Blob {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return new Blob([bytes], { type: mimeType });
}
