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

import pytest
import requests
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner


@pytest.fixture(autouse=True, scope="session")
def normalize_requests_url():
    original_request = requests.Session.request

    def new_request(self, method, url, *args, **kwargs):
        # Normalize double slashes like http://127.0.0.1:8000//apps/ to http://127.0.0.1:8000/apps/
        if isinstance(url, str) and "//apps/" in url:
            url = url.replace("//apps/", "/apps/")
        return original_request(self, method, url, *args, **kwargs)

    requests.Session.request = new_request
    yield
    requests.Session.request = original_request


@pytest.fixture(autouse=True, scope="session")
def default_artifact_service_for_runner():
    original_init = Runner.__init__

    def new_init(self, *args, **kwargs):
        if kwargs.get("artifact_service") is None:
            kwargs["artifact_service"] = InMemoryArtifactService()
        original_init(self, *args, **kwargs)

    Runner.__init__ = new_init
    yield
    Runner.__init__ = original_init
