#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copy MediaPipe files to public directory for bundling

mkdir -p public/mediapipe/face_mesh
cp -r node_modules/@mediapipe/face_mesh/* public/mediapipe/face_mesh/
# Remove empty SIMD data file that causes loading errors
rm -f public/mediapipe/face_mesh/face_mesh_solution_simd_wasm_bin.data
echo "✅ MediaPipe files copied to public/mediapipe/face_mesh"
