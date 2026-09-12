"""Common util functions for notebook."""

import base64
from collections.abc import Sequence
import datetime
import io
import json
import os
import subprocess
import time
from typing import Any

from google import auth
from google.cloud import storage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
import tensorflow as tf
import yaml


GCS_URI_PREFIX = "gs://"
CHECKPOINT_BUCKET = "gs://model_garden_checkpoints"


def convert_numpy_array_to_byte_string_via_tf_tensor(
    np_array: np.ndarray,
) -> str:
  """Serializes a numpy array to tensor bytes.

  Args:
    np_array: A numpy array.

  Returns:
    A tensor bytes.
  """
  tensor_array = tf.convert_to_tensor(np_array)
  tensor_byte_string = tf.io.serialize_tensor(tensor_array)
  return tensor_byte_string.numpy()


def get_jpeg_bytes(local_image_path: str, new_width: int = -1) -> bytes:
  """Returns jpeg bytes given an image path and resizes if required.

  Args:
    local_image_path: A string of local image path.
    new_width: An integer of new image width.

  Returns:
    A jpeg bytes.
  """
  image = Image.open(local_image_path)
  if new_width <= 0:
    new_image = image
  else:
    width, height = image.size
    print("original input image size: ", width, " , ", height)
    new_height = int(height * new_width / width)
    print("new input image size: ", new_width, " , ", new_height)
    new_image = image.resize((new_width, new_height))
  buffered = io.BytesIO()
  new_image.save(buffered, format="JPEG")
  return buffered.getvalue()


def gcs_fuse_path(path: str) -> str:
  """Try to convert path to gcsfuse path if it starts with gs:// else do not modify it.

  Args:
    path: A string of path.

  Returns:
    A gcsfuse path.
  """
  path = path.strip()
  if path.startswith("gs://"):
    return "/gcs/" + path[5:]
  return path


def get_job_name_with_datetime(prefix: str) -> str:
  """Gets a job name by adding current time to prefix.

  Args:
    prefix: A string of job name prefix.

  Returns:
    A job name.
  """
  now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  job_name = f"{prefix}-{now}".replace("_", "-")
  return job_name


def create_job_name(prefix: str) -> str:
  """Creates a job name.

  Args:
    prefix: A string of job name prefix.

  Returns:
    A job name.
  """
  user = os.environ.get("USER")
  now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  job_name = f"{prefix}-{user}-{now}".replace("_", "-")
  return job_name


def save_subset_annotation(
    input_annotation_path: str, output_annotation_path: str
):
  """Saves a subset of COCO annotation json file with CCA 4.0 license.

  Args:
    input_annotation_path: A string of input annotation path.
    output_annotation_path: A string of output annotation path.
  """

  with open(input_annotation_path) as f:
    coco_json = json.load(f)

  img_ids = set()
  images = []
  annotations = []

  for img in coco_json["images"]:
    if img["license"] in [4, 5]:  # CCA 4.0 license.
      img_ids.add(img["id"])
      images.append(img)

  for ann in coco_json["annotations"]:
    if ann["image_id"] in img_ids:
      annotations.append(ann)

  new_json = {
      "info": coco_json["info"],
      "licenses": coco_json["licenses"],
      "images": images,
      "annotations": annotations,
      "categories": coco_json["categories"],
  }

  with open(output_annotation_path, "w") as f:
    json.dump(new_json, f)


def image_to_base64(image: Any, image_format: str = "JPEG") -> str:
  """Converts an image to base64.

  Args:
    image: A PIL.Image instance.
    image_format: A string of image format.

  Returns:
    A base64 string.
  """
  buffer = io.BytesIO()
  image.save(buffer, format=image_format)
  image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
  return image_str


def base64_to_image(image_str: str) -> Any:
  """Convert base64 encoded string to an image.

  Args:
    image_str: A string of base64 encoded image.

  Returns:
    A PIL.Image instance.
  """
  image = Image.open(io.BytesIO(base64.b64decode(image_str)))
  return image


def image_grid(imgs: Sequence[Any], rows: int = 2, cols: int = 2) -> Any:
  """Creates an image grid.

  Args:
    imgs: A list of PIL.Image instances.
    rows: An integer of number of rows.
    cols: An integer of number of columns.

  Returns:
    A PIL.Image instance.
  """
  w, h = imgs[0].size
  grid = Image.new(
      mode="RGB", size=(cols * w + 10 * cols, rows * h), color=(255, 255, 255)
  )
  for i, img in enumerate(imgs):
    grid.paste(img, box=(i % cols * w + 10 * i, i // cols * h))
  return grid


def display_image(image: Any):
  """Displays an image.

  Args:
    image: A PIL.Image instance.
  """
  _ = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)


def download_gcs_file_to_local(gcs_uri: str, local_path: str):
  """Download a gcs file to a local path.

  Args:
    gcs_uri: A string of file path on GCS.
    local_path: A string of local file path.
  """
  if not gcs_uri.startswith(GCS_URI_PREFIX):
    raise ValueError(
        f"{gcs_uri} is not a GCS path starting with {GCS_URI_PREFIX}."
    )
  client = storage.Client()
  os.makedirs(os.path.dirname(local_path), exist_ok=True)
  with open(local_path, "wb") as f:
    client.download_blob_to_file(gcs_uri, f)


def download_image(url: str) -> str:
  """Downloads an image from the given URL.

  Args:
    url: The URL of the image to download.

  Returns:
    base64 encoded image.
  """
  response = requests.get(url)
  return Image.open(io.BytesIO(response.content))  # pytype: disable=bad-return-type  # pillow-102-upgrade


def resize_image(image: Any, new_width: int = 1000) -> Any:
  """Resizes an image to a certain width.

  Args:
    image: The image which has to be resized.
    new_width: New width of the image.

  Returns:
    New resized image.
  """
  width, height = image.size
  new_height = int(height * new_width / width)
  new_img = image.resize((new_width, new_height))
  return new_img


def load_img(path: str) -> Any:
  """Reads image from path and return PIL.Image instance.

  Args:
    path: A string of image path.

  Returns:
    A PIL.Image instance.
  """
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return Image.fromarray(np.uint8(img)).convert("RGB")


def decode_image(
    image_str_tensor: tf.string, new_height: int, new_width: int
) -> tf.float32:
  """Converts and resizes image bytes to image tensor.

  Args:
    image_str_tensor: A string of image bytes.
    new_height: An integer of new image height.
    new_width: An integer of new image width.

  Returns:
    An image tensor.
  """
  image = tf.io.decode_image(image_str_tensor, 3, expand_animations=False)
  image = tf.image.resize(image, (new_height, new_width))
  return image


def get_label_map(label_map_yaml_filepath: str) -> dict[int, str]:
  """Returns class id to label mapping given a filepath to the label map.

  Args:
    label_map_yaml_filepath: A string of label map yaml file path.

  Returns:
    A dictionary of class id to label mapping.
  """
  with tf.io.gfile.GFile(label_map_yaml_filepath, "rb") as input_file:
    label_map = yaml.safe_load(input_file.read())["label_map"]
  return label_map


def get_prediction_instances(test_filepath: str, new_width: int = -1) -> Any:
  """Generate instance from image path to pass to Vertex AI Endpoint for prediction.

  Args:
    test_filepath: A string of test image path.
    new_width: An integer of new image width.

  Returns:
    A list of instances.
  """
  if new_width <= 0:
    with tf.io.gfile.GFile(test_filepath, "rb") as input_file:
      encoded_string = base64.b64encode(input_file.read()).decode("utf-8")
  else:
    img = load_img(test_filepath)
    width, height = img.size
    print("original input image size: ", width, " , ", height)
    new_height = int(height * new_width / width)
    new_img = img.resize((new_width, new_height))
    print("resized input image size: ", new_width, " , ", new_height)
    buffered = io.BytesIO()
    new_img.save(buffered, format="JPEG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

  instances = [{
      "encoded_image": {"b64": encoded_string},
  }]
  return instances


def vqa_predict(
    endpoint: Any,
    question_prompts: Sequence[str],
    image: Any,
    language_code: str = "en",
    new_width: int = 1000,
    use_dedicated_endpoint: bool = False,
) -> Sequence[str]:
  """Predicts the answer to a question about an image using an Endpoint."""
  # Resize and convert image to base64 string.
  resized_image = resize_image(image, new_width)
  resized_image_base64 = image_to_base64(resized_image)

  instances = []
  if question_prompts:
    # Format question prompt
    question_prompt_format = "answer {} {}\n"
    for question_prompt in question_prompts:
      if question_prompt:
        instances.append({
            "prompt": question_prompt_format.format(
                language_code, question_prompt
            ),
            "image": resized_image_base64,
        })
  else:
    instances.append({
        "image": resized_image_base64,
    })

  response = endpoint.predict(
      instances=instances, use_dedicated_endpoint=use_dedicated_endpoint
  )
  return [pred.get("response") for pred in response.predictions]


def caption_predict(
    endpoint: Any,
    language_code: str,
    image: Any,
    caption_prompt: bool = False,
    new_width: int = 1000,
    use_dedicated_endpoint: bool = False,
) -> str:
  """Predicts a caption for a given image using an Endpoint."""
  # Resize and convert image to base64 string.
  resized_image = resize_image(image, new_width)
  resized_image_base64 = image_to_base64(resized_image)

  instance = {"image": resized_image_base64}

  if caption_prompt:
    # Format caption prompt
    caption_prompt_format = "caption {}\n"
    instance["prompt"] = caption_prompt_format.format(language_code)

  instances = [instance]
  response = endpoint.predict(
      instances=instances, use_dedicated_endpoint=use_dedicated_endpoint
  )
  return response.predictions[0].get("response")


def ocr_predict(
    endpoint: Any,
    ocr_prompt: str,
    image: Any,
    new_width: int = 1000,
    use_dedicated_endpoint: bool = False,
) -> str:
  """Extracts text from a given image using an Endpoint."""
  # Resize and convert image to base64 string.
  resized_image = resize_image(image, new_width)
  resized_image_base64 = image_to_base64(resized_image)

  instance = {"image": resized_image_base64}
  if ocr_prompt:
    instance["prompt"] = ocr_prompt
  instances = [instance]

  response = endpoint.predict(
      instances=instances, use_dedicated_endpoint=use_dedicated_endpoint
  )
  return response.predictions[0].get("response")


def detect_predict(
    endpoint: Any,
    detect_prompt: str,
    image: Any,
    new_width: int = 1000,
    use_dedicated_endpoint: bool = False,
) -> str:
  """Predicts the answer to a question about an image using an Endpoint."""
  # Resize and convert image to base64 string.
  resized_image = resize_image(image, new_width)
  resized_image_base64 = image_to_base64(resized_image)

  instance = {"image": resized_image_base64}
  if detect_prompt:
    instance["prompt"] = detect_prompt
  instances = [instance]

  response = endpoint.predict(
      instances=instances, use_dedicated_endpoint=use_dedicated_endpoint
  )
  return response.predictions[0].get("response")


def copy_model_artifacts(
    model_id: str,
    model_source: str,
    model_destination: str,
) -> None:
  """Copies model artifacts from model_source to model_destination.

  model_source and model_destination should be GCS path.

  Args:
    model_id: The model id.
    model_source: The source of the model artifact.
    model_destination: The destination of the model artifact.
  """
  if not model_source.startswith(GCS_URI_PREFIX):
    raise ValueError(
        f"{model_source} is not a GCS path starting with {GCS_URI_PREFIX}."
    )
  if not model_destination.startswith(GCS_URI_PREFIX):
    raise ValueError(
        f"{model_destination} is not a GCS path starting with {GCS_URI_PREFIX}."
    )
  model_source = f"{model_source}/{model_id}"
  model_destination = f"{model_destination}/{model_id}"
  print("Copying model artifact from ", model_source, " to ", model_destination)
  subprocess.check_output([
      "gcloud",
      "storage",
      "cp",
      "-r",
      model_source,
      model_destination,
  ])


def get_quota(project_id: str, region: str, resource_id: str) -> int:
  """Returns the quota for a resource in a region.

  Args:
    project_id: The project id.
    region: The region.
    resource_id: The resource id.

  Returns:
    The quota for the resource in the region. Returns -1 if can not figure out
    the quota.

  Raises:
    RuntimeError: If the command to get quota fails.
  """
  service_endpoint = "aiplatform.googleapis.com"

  command = (
      "gcloud alpha services quota list"
      f" --service={service_endpoint} --consumer=projects/{project_id}"
      f" --filter='{service_endpoint}/{resource_id}' --format=json"
  )
  process = subprocess.run(
      command, shell=True, capture_output=True, text=True, check=True
  )
  if process.returncode == 0:
    quota_data = json.loads(process.stdout)
  else:
    raise RuntimeError(f"Error fetching quota data: {process.stderr}")

  if not quota_data or "consumerQuotaLimits" not in quota_data[0]:
    return -1
  if (
      not quota_data[0]["consumerQuotaLimits"]
      or "quotaBuckets" not in quota_data[0]["consumerQuotaLimits"][0]
  ):
    return -1
  all_regions_data = quota_data[0]["consumerQuotaLimits"][0]["quotaBuckets"]

  # If the quota data does not have dimensions, it is global quota. However,
  # global quota may be overridden by regional quota. So we need to check the
  # global quota first.
  global_quota = -1
  if (
      all_regions_data
      and "dimensions" not in all_regions_data[0]
      and "effectiveLimit" in all_regions_data[0]
  ):
    global_quota = int(all_regions_data[0]["effectiveLimit"])
  for region_data in all_regions_data:
    if (
        region_data.get("dimensions")
        and region_data["dimensions"]["region"] == region
    ):
      if "effectiveLimit" in region_data:
        return int(region_data["effectiveLimit"])
      else:
        return 0
  return global_quota


def get_resource_id(
    accelerator_type: str,
    is_for_training: bool,
    is_spot: bool = False,
    is_restricted_image: bool = False,
    is_dynamic_workload_scheduler: bool = False,
) -> str:
  """Returns the resource id for a given accelerator type and the use case.

  Args:
    accelerator_type: The accelerator type.
    is_for_training: Whether the resource is used for training. Set false for
      serving use case.
    is_spot: Whether the resource is used with Spot.
    is_restricted_image: Whether the image is hosted in `vertex-ai-restricted`.
    is_dynamic_workload_scheduler: Whether the resource is used with Dynamic
      Workload Scheduler.

  Returns:
    The resource id.
  """
  accelerator_suffix_map = {
      "NVIDIA_TESLA_V100": "nvidia_v100_gpus",
      "NVIDIA_TESLA_P100": "nvidia_p100_gpus",
      "NVIDIA_L4": "nvidia_l4_gpus",
      "NVIDIA_TESLA_A100": "nvidia_a100_gpus",
      "NVIDIA_A100_80GB": "nvidia_a100_80gb_gpus",
      "NVIDIA_H100_80GB": "nvidia_h100_gpus",
      "NVIDIA_H100_MEGA_80GB": "nvidia_h100_mega_gpus",
      "NVIDIA_H200_141GB": "nvidia_h200_gpus",
      "NVIDIA_TESLA_T4": "nvidia_t4_gpus",
      "TPU_V6e": "tpu_v6e",
      "TPU_V5e": "tpu_v5e",
      "TPU_V3": "tpu_v3",
  }
  default_training_accelerator_map = {
      key: f"custom_model_training_{accelerator_suffix_map[key]}"
      for key in accelerator_suffix_map
  }
  dws_training_accelerator_map = {
      key: f"custom_model_training_preemptible_{accelerator_suffix_map[key]}"
      for key in accelerator_suffix_map
  }
  restricted_image_training_accelerator_map = {
      "NVIDIA_A100_80GB": "restricted_image_training_nvidia_a100_80gb_gpus",
  }
  spot_serving_accelerator_map = {
      key: f"custom_model_serving_preemptible_{accelerator_suffix_map[key]}"
      for key in accelerator_suffix_map
  }
  serving_accelerator_map = {
      key: f"custom_model_serving_{accelerator_suffix_map[key]}"
      for key in accelerator_suffix_map
  }

  if is_for_training:
    if is_restricted_image and is_dynamic_workload_scheduler:
      raise ValueError(
          "Dynamic Workload Scheduler does not work for restricted image"
          " training."
      )
    training_accelerator_map = (
        restricted_image_training_accelerator_map
        if is_restricted_image
        else default_training_accelerator_map
    )
    if accelerator_type in training_accelerator_map:
      if is_dynamic_workload_scheduler:
        return dws_training_accelerator_map[accelerator_type]
      else:
        return training_accelerator_map[accelerator_type]
    else:
      raise ValueError(
          f"Could not find accelerator type: {accelerator_type} for training."
      )
  else:
    if is_dynamic_workload_scheduler:
      raise ValueError("Dynamic Workload Scheduler does not work for serving.")
    accelerator_map = (
        spot_serving_accelerator_map if is_spot else serving_accelerator_map
    )
    if accelerator_type in accelerator_map:
      return accelerator_map[accelerator_type]
    else:
      raise ValueError(
          f"Could not find accelerator type: {accelerator_type} for serving."
      )


def check_quota(
    project_id: str,
    region: str,
    accelerator_type: str,
    accelerator_count: int,
    is_for_training: bool,
    is_spot: bool = False,
    is_restricted_image: bool = False,
    is_dynamic_workload_scheduler: bool = False,
) -> None:
  """Checks if the project and the region has the required quota.

  Args:
    project_id: The project id.
    region: The region.
    accelerator_type: The accelerator type.
    accelerator_count: The number of accelerators to check quota for.
    is_for_training: Whether the resource is used for training. Set false for
      serving use case.
    is_spot: Whether the resource is used with Spot.
    is_restricted_image: Whether the image is hosted in `vertex-ai-restricted`.
    is_dynamic_workload_scheduler: Whether the resource is used with Dynamic
      Workload Scheduler.
  """
  resource_id = get_resource_id(
      accelerator_type,
      is_for_training=is_for_training,
      is_spot=is_spot,
      is_restricted_image=is_restricted_image,
      is_dynamic_workload_scheduler=is_dynamic_workload_scheduler,
  )
  quota = get_quota(project_id, region, resource_id)
  quota_request_instruction = (
      "Either use "
      "a different region or request additional quota. Follow "
      "instructions here "
      "https://cloud.google.com/docs/quotas/view-manage#requesting_higher_quota"
      " to check quota in a region or request additional quota for "
      "your project."
  )
  if quota == -1:
    raise ValueError(
        f"Quota not found for: {resource_id} in {region}."
        f" {quota_request_instruction}"
    )
  if quota < accelerator_count:
    raise ValueError(
        f"Quota not enough for {resource_id} in {region}: {quota} <"
        f" {accelerator_count}. {quota_request_instruction}"
    )


def get_deploy_source() -> str:
  """Gets deploy_source string based on running environment."""
  vertex_product = os.environ.get("VERTEX_PRODUCT", "")
  match vertex_product:
    case "COLAB_ENTERPRISE":
      return "notebook_colab_enterprise"
    case "WORKBENCH_INSTANCE":
      return "notebook_workbench"
    case _:
      # Legacy workbench, legacy colab, or other custom environments.
      return "notebook_environment_unspecified"


def _is_operation_done(op_name: str, region: str) -> bool:
  """Checks if the operation is done.

  Args:
    op_name: The name of the operation to poll.
    region: The region of the operation.

  Returns:
    True if the operation is done, False otherwise.

  Raises:
    ValueError: If the operation failed.
  """
  creds, _ = auth.default()
  auth_req = auth.transport.requests.Request()
  creds.refresh(auth_req)
  headers = {
      "Authorization": f"Bearer {creds.token}",
  }
  url = f"https://{region}-aiplatform.googleapis.com/ui/{op_name}"
  response = requests.get(url, headers=headers)
  operation_data = response.json()
  if "error" in operation_data:
    raise ValueError(f"Operation failed: {operation_data['error']}")
  return operation_data.get("done", False)


def poll_and_wait(
    op_name: str, region: str, total_wait: int, interval: int = 60
) -> None:
  """Polls the operation and waits for it to complete.

  Args:
    op_name: The name of the operation to poll.
    region: The region of the operation.
    total_wait: The total wait time in seconds.
    interval: The interval between each poll in seconds.

  Raises:
    TimeoutError: If the operation times out.
  """
  start_time = time.time()
  while True:
    if _is_operation_done(op_name, region):
      break
    time_elapsed = time.time() - start_time
    if time_elapsed > total_wait:
      raise TimeoutError(
          f"Operation timed out after {int(time_elapsed)} seconds."
      )
    print(
        "\rStill waiting for operation... Elapsed time in seconds:"
        f" {int(time_elapsed):<6}",
        end="",
        flush=True,
    )
    time.sleep(interval)

