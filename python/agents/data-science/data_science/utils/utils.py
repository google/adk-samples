# Copyright 2025 Google LLC
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

import json
import os

from vertexai.preview.extensions import Extension


def list_all_extensions():
  extensions = Extension.list(location='us-central1')
  for extension in extensions:
    print('Name:', extension.gca_resource.name)
    print('Display Name:', extension.gca_resource.display_name)
    print('Description:', extension.gca_resource.description)


def get_env_var(var_name):
  """Retrieves the value of an environment variable.

  Args:
    var_name: The name of the environment variable.

  Returns:
    The value of the environment variable, or None if it is not set.

  Raises:
    ValueError: If the environment variable is not set.
  """
  try:
    value = os.environ[var_name]
    return value
  except KeyError:
    raise ValueError(f'Missing environment variable: {var_name}')


def get_image_bytes(filepath):
  """Reads an image file and returns its bytes.

  Args:
    filepath: The path to the image file.

  Returns:
    The bytes of the image file, or None if the file does not exist or cannot be
    read.
  """
  try:
    with open(filepath, 'rb') as f:  # "rb" mode for reading in binary
      image_bytes = f.read()
    return image_bytes
  except FileNotFoundError:
    print(f'Error: File not found at {filepath}')
    return None
  except Exception as e:
    print(f'Error reading file: {e}')
    return None


def extract_json_from_model_output(model_output):
  """Extracts JSON object from a string that potentially contains markdown

  code fences.

  Args:
    model_output: A string potentially containing a JSON object wrapped in
      markdown code fences (```json ... ```).

  Returns:
    A Python dictionary representing the extracted JSON object,
    or None if JSON extraction fails.
  """
  try:
    cleaned_output = (
        model_output.replace('```json', '').replace('```', '').strip()
    )
    json_object = json.loads(cleaned_output)
    return json_object
  except json.JSONDecodeError as e:
    msg = f'Error decoding JSON: {e}'
    print(msg)
    return {'error': msg}


if __name__ == '__main__':
  list_all_extensions()
