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
 * API service for Interpolation (360° spin) functionality
 * Handles communication with the backend interpolation pipeline endpoints
 */

const API_BASE_URL = '/api/spinning/interpolation/other';

export interface PreprocessResponse {
  images: Array<{
    index: number;
    data: string; // base64
  }>;
}

export interface GeneratePromptResponse {
  prompt: string;
}

export interface VideoSegment {
  index: number;
  data: string; // base64
  retries: number;
  is_valid: boolean;
  validation_reason: string;
}

export interface GenerateAllResponse {
  videos: VideoSegment[];
  num_videos: number;
  num_generated: number;
  num_valid: number;
  num_failed: number;
  indices: number[];
  total_retries: number;
}

export interface ApiError {
  detail: string;
}

/**
 * Preprocess images for interpolation (background removal, upscaling, canvas creation)
 */
export async function preprocessImages(imageFiles: File[]): Promise<PreprocessResponse> {
  const formData = new FormData();
  imageFiles.forEach((file) => {
    formData.append('images', file);
  });

  const response = await fetch(`${API_BASE_URL}/preprocess`, {
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
 * Generate a video prompt based on two images
 */
export async function generatePrompt(img1: File, img2: File): Promise<GeneratePromptResponse> {
  const formData = new FormData();
  formData.append('img1', img1);
  formData.append('img2', img2);

  const response = await fetch(`${API_BASE_URL}/generate-prompt`, {
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
 * Generate all video segments for interpolation
 */
export async function generateAllVideos(
  imageFiles: File[],
  prompt: string,
  backgroundColor: string = '#FFFFFF',
  indices?: number[]
): Promise<GenerateAllResponse> {
  const formData = new FormData();

  imageFiles.forEach((file) => {
    formData.append('images', file);
  });

  formData.append('prompt', prompt);
  formData.append('backgroundColor', backgroundColor);

  if (indices && indices.length > 0) {
    formData.append('indices', JSON.stringify(indices));
  } else {
    formData.append('indices', '');
  }

  const response = await fetch(`${API_BASE_URL}/generate-all`, {
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
 * Merge video segments into final 360° video
 */
export async function mergeVideos(
  videoBlobs: Blob[],
  speeds: number[]
): Promise<Blob> {
  const formData = new FormData();

  videoBlobs.forEach((blob, index) => {
    formData.append('videos', blob, `video_${index}.mp4`);
  });

  formData.append('speeds', JSON.stringify(speeds));

  const response = await fetch(`${API_BASE_URL}/merge`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || `Request failed with status ${response.status}`);
  }

  return response.blob();
}

/**
 * Converts a base64 string to a Blob
 */
export function base64ToBlob(base64: string, mimeType: string = 'video/mp4'): Blob {
  try {
    // Clean the base64 string (remove any whitespace/newlines)
    const cleanedBase64 = base64.replace(/\s/g, '');

    const binaryString = atob(cleanedBase64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return new Blob([bytes], { type: mimeType });
  } catch (error) {
    console.error('Failed to convert base64 to blob:', error);
    throw new Error(`Failed to decode base64 data: ${error}`);
  }
}

/**
 * Converts a base64 video string to a blob URL for video playback
 */
export function base64ToVideoUrl(base64: string): string {
  const blob = base64ToBlob(base64);
  return URL.createObjectURL(blob);
}

/**
 * Converts a data URL to a File object
 */
export function dataUrlToFile(dataUrl: string, filename: string): File {
  try {
    if (!dataUrl) {
      throw new Error('dataUrl is empty or undefined');
    }

    const arr = dataUrl.split(',');
    if (arr.length < 2) {
      throw new Error('Invalid data URL format - missing comma separator');
    }

    const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/png';
    const base64Data = arr[1];

    if (!base64Data) {
      throw new Error('Base64 data is empty after splitting');
    }

    // Clean the base64 string (remove any whitespace/newlines)
    const cleanedBase64 = base64Data.replace(/\s/g, '');

    const bstr = atob(cleanedBase64);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
  } catch (error) {
    console.error(`Failed to convert dataUrl to file for ${filename}:`, error);
    throw new Error(`Failed to convert data URL to file: ${error}`);
  }
}
