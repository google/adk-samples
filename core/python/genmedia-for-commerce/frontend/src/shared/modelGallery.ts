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

export interface ModelPreset {
  id: string
  label: string
  gender: 'woman' | 'man'
  thumbnail: string
}

export const MODEL_PRESETS: ModelPreset[] = [
  { id: 'african_woman', label: 'African', gender: 'woman', thumbnail: '/assets/models/african_woman.png' },
  { id: 'asian_woman', label: 'Asian', gender: 'woman', thumbnail: '/assets/models/asian_woman.png' },
  { id: 'european_woman', label: 'European', gender: 'woman', thumbnail: '/assets/models/european_woman.png' },
  { id: 'european_woman2', label: 'European 2', gender: 'woman', thumbnail: '/assets/models/european_woman2.png' },
  { id: 'african_man', label: 'African', gender: 'man', thumbnail: '/assets/models/african_man.png' },
  { id: 'asian_man', label: 'Asian', gender: 'man', thumbnail: '/assets/models/asian_man.png' },
  { id: 'european_man', label: 'European', gender: 'man', thumbnail: '/assets/models/european_man.png' },
  { id: 'european_man2', label: 'European 2', gender: 'man', thumbnail: '/assets/models/european_man2.png' },
]
