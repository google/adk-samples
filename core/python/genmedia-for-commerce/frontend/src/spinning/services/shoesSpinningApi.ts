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
 * API service for Shoes Spinning (360° spin) functionality
 * Handles communication with the backend R2V pipeline endpoints
 */

const API_BASE_URL = '/api/shoes/spinning';

export interface PipelineRequest {
  images_base64: string[];
  max_retries?: number;
  upscale_images?: boolean;
}

export interface PipelineResponse {
  video_base64: string;
  frames_base64: string[];
  num_frames: number;
  retry_count: number;
}

export interface GalleryProduct {
  folder_name: string;
  images: Array<{
    url: string;
    name: string;
  }>;
}

export interface GalleryResponse {
  products: GalleryProduct[];
}

export interface ApiError {
  detail: string;
}

/**
 * Runs the complete R2V spinning pipeline
 * Takes base64-encoded images and returns a generated video
 */
export async function runSpinningPipeline(request: PipelineRequest): Promise<PipelineResponse> {
  const response = await fetch(`${API_BASE_URL}/run-pipeline-r2v`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      images_base64: request.images_base64,
      max_retries: request.max_retries ?? 4,
      upscale_images: request.upscale_images ?? true,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}

/**
 * Fetches available gallery images from the backend
 */
export async function getGalleryImages(): Promise<GalleryResponse> {
  const response = await fetch(`${API_BASE_URL}/get_gallery_images`);

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
  // If it's already raw base64 (no data: prefix), return as-is
  if (!dataUrl.startsWith('data:')) {
    return dataUrl;
  }
  // Extract base64 content after the comma
  const base64Index = dataUrl.indexOf(',');
  if (base64Index === -1) {
    return dataUrl;
  }
  return dataUrl.substring(base64Index + 1);
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
export function downloadVideo(base64: string, filename: string = 'spinning-360.mp4'): void {
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

  // Clean up the blob URL after download
  setTimeout(() => URL.revokeObjectURL(url), 100);
}
