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
 * API service for Other Product R2V Spinning (360° spin) functionality
 * Hits /api/spinning/r2v/other/pipeline — the "other" R2V backend,
 * NOT the shoes endpoint.
 */

const API_BASE_URL = '/api/spinning/r2v/other';

/**
 * Converts a data URL or raw base64 string to a File object for multipart upload
 */
function dataUrlToFile(dataUrl: string, filename: string): File {
  let base64: string;
  let mimeType = 'image/png';

  if (dataUrl.startsWith('data:')) {
    const mimeMatch = dataUrl.match(/^data:([^;]+);/);
    if (mimeMatch) mimeType = mimeMatch[1];
    const base64Index = dataUrl.indexOf(',');
    base64 = base64Index === -1 ? dataUrl : dataUrl.substring(base64Index + 1);
  } else {
    base64 = dataUrl;
  }

  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  return new File([bytes], filename, { type: mimeType });
}

export interface OtherPipelineResponse {
  video_base64: string;
}

/**
 * Runs the Other R2V spinning pipeline.
 * Sends images as multipart form data to /api/spinning/r2v/other/pipeline.
 * The backend returns raw video/mp4 bytes; we convert to base64 for the caller.
 */
export async function runOtherSpinningPipeline(
  imagesBase64: string[],
): Promise<OtherPipelineResponse> {
  const formData = new FormData();

  imagesBase64.forEach((img, index) => {
    const filename = 'image_' + index + '.png';
    formData.append('images', dataUrlToFile(img, filename));
  });

  const response = await fetch(`${API_BASE_URL}/pipeline`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || `Request failed with status ${response.status}`);
  }

  // Backend returns raw video/mp4 bytes — convert to base64
  const videoBlob = await response.blob();
  const base64 = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      // Strip the data:video/mp4;base64, prefix
      const commaIndex = result.indexOf(',');
      resolve(commaIndex === -1 ? result : result.substring(commaIndex + 1));
    };
    reader.onerror = reject;
    reader.readAsDataURL(videoBlob);
  });

  return { video_base64: base64 };
}
