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

"""A2UI v0.8 schema and video examples for the product strategy agent.

Based on the working a2a_a2ui_sample reference implementation.
See https://a2ui.org/specification/v0.8-a2ui/ for details.
"""

# ---------------------------------------------------------------------------
# A2UI v0.8 JSON Schema  (comprehensive, matching working sample)
# surfaceId is REQUIRED on surfaceUpdate, dataModelUpdate, beginRendering
# ---------------------------------------------------------------------------

A2UI_SCHEMA = r"""
{
  "title": "A2UI Message Schema",
  "description": "Describes a JSON payload for an A2UI (Agent to UI) message, which is used to dynamically construct and update user interfaces. A message MUST contain exactly ONE of the action properties: 'beginRendering', 'surfaceUpdate', 'dataModelUpdate', or 'deleteSurface'.",
  "type": "object",
  "properties": {
    "beginRendering": {
      "type": "object",
      "description": "Signals the client to begin rendering a surface with a root component and specific styles.",
      "properties": {
        "surfaceId": {
          "type": "string",
          "description": "The unique identifier for the UI surface to be rendered."
        },
        "root": {
          "type": "string",
          "description": "The ID of the root component to render."
        },
        "styles": {
          "type": "object",
          "description": "Styling information for the UI.",
          "properties": {
            "font": { "type": "string" },
            "primaryColor": { "type": "string", "pattern": "^#[0-9a-fA-F]{6}$" }
          }
        }
      },
      "required": ["root", "surfaceId"]
    },
    "surfaceUpdate": {
      "type": "object",
      "description": "Updates a surface with a new set of components.",
      "properties": {
        "surfaceId": {
          "type": "string",
          "description": "The unique identifier for the UI surface to be updated."
        },
        "components": {
          "type": "array",
          "description": "A list containing all UI components for the surface.",
          "minItems": 1,
          "items": {
            "type": "object",
            "description": "Represents a single component in a UI widget tree.",
            "properties": {
              "id": { "type": "string" },
              "weight": { "type": "number" },
              "component": {
                "type": "object",
                "description": "Wrapper containing exactly one component type key.",
                "properties": {
                  "Text": {
                    "type": "object",
                    "properties": {
                      "text": {
                        "type": "object",
                        "properties": {
                          "literalString": { "type": "string" },
                          "path": { "type": "string" }
                        }
                      },
                      "usageHint": {
                        "type": "string",
                        "enum":["h1","h2","h3","h4","h5","caption","body"]
                      }
                    },
                    "required": ["text"]
                  },
                  "Image": {
                    "type": "object",
                    "properties": {
                      "url": {
                        "type": "object",
                        "properties": {
                          "literalString": { "type": "string" },
                          "path": { "type": "string" }
                        }
                      },
                      "fit": { "type": "string", "enum":["contain","cover","fill","none","scale-down"] },
                      "usageHint": { "type": "string", "enum":["icon","avatar","smallFeature","mediumFeature","largeFeature","header"] }
                    },
                    "required": ["url"]
                  },
                  "Icon": {
                    "type": "object",
                    "properties": {
                      "name": {
                        "type": "object",
                        "properties": {
                          "literalString": { "type": "string" },
                          "path": { "type": "string" }
                        }
                      }
                    },
                    "required": ["name"]
                  },
                  "Video": {
                    "type": "object",
                    "properties": {
                      "url": {
                        "type": "object",
                        "properties": {
                          "literalString": { "type": "string" },
                          "path": { "type": "string" }
                        }
                      }
                    },
                    "required": ["url"]
                  },
                  "Row": {
                    "type": "object",
                    "properties": {
                      "children": {
                        "type": "object",
                        "properties": {
                          "explicitList": { "type": "array", "items": { "type": "string" } },
                          "template": {
                            "type": "object",
                            "properties": {
                              "componentId": { "type": "string" },
                              "dataBinding": { "type": "string" }
                            },
                            "required": ["componentId", "dataBinding"]
                          }
                        }
                      },
                      "distribution": { "type": "string", "enum":["center","end","spaceAround","spaceBetween","spaceEvenly","start"] },
                      "alignment": { "type": "string", "enum": ["start","center","end","stretch"] }
                    },
                    "required":["children"]
                  },
                  "Column": {
                    "type": "object",
                    "properties": {
                      "children": {
                        "type": "object",
                        "properties": {
                          "explicitList": { "type": "array", "items": { "type": "string" } },
                          "template": {
                            "type": "object",
                            "properties": {
                              "componentId": { "type": "string" },
                              "dataBinding": { "type": "string" }
                            },
                            "required": ["componentId", "dataBinding"]
                          }
                        }
                      },
                      "distribution": { "type": "string", "enum":["start","center","end","spaceBetween","spaceAround","spaceEvenly"] },
                      "alignment": { "type": "string", "enum":["center","end","start","stretch"] }
                    },
                    "required": ["children"]
                  },
                  "List": {
                    "type": "object",
                    "properties": {
                      "children": {
                        "type": "object",
                        "properties": {
                          "explicitList": { "type": "array", "items": { "type": "string" } },
                          "template": {
                            "type": "object",
                            "properties": {
                              "componentId": { "type": "string" },
                              "dataBinding": { "type": "string" }
                            },
                            "required":["componentId", "dataBinding"]
                          }
                        }
                      },
                      "direction": { "type": "string", "enum": ["vertical","horizontal"] },
                      "alignment": { "type": "string", "enum": ["start","center","end","stretch"] }
                    },
                    "required":["children"]
                  },
                  "Card": {
                    "type": "object",
                    "properties": {
                      "child": { "type": "string" }
                    },
                    "required": ["child"]
                  },
                  "Divider": {
                    "type": "object",
                    "properties": {
                      "axis": { "type": "string", "enum": ["horizontal","vertical"] }
                    }
                  },
                  "Button": {
                    "type": "object",
                    "properties": {
                      "child": { "type": "string" },
                      "primary": { "type": "boolean" },
                      "action": {
                        "type": "object",
                        "properties": {
                          "name": { "type": "string" },
                          "context": {
                            "type": "array",
                            "items": {
                              "type": "object",
                              "properties": {
                                "key": { "type": "string" },
                                "value": {
                                  "type": "object",
                                  "properties": {
                                    "path": { "type": "string" },
                                    "literalString": { "type": "string" },
                                    "literalNumber": { "type": "number" },
                                    "literalBoolean": { "type": "boolean" }
                                  }
                                }
                              },
                              "required": ["key", "value"]
                            }
                          }
                        },
                        "required":["name"]
                      }
                    },
                    "required": ["child", "action"]
                  }
                }
              }
            },
            "required": ["id", "component"]
          }
        }
      },
      "required": ["surfaceId", "components"]
    },
    "dataModelUpdate": {
      "type": "object",
      "description": "Updates the data model for a surface.",
      "properties": {
        "surfaceId": {
          "type": "string"
        },
        "path": {
          "type": "string"
        },
        "contents": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "key": { "type": "string" },
              "valueString": { "type": "string" },
              "valueNumber": { "type": "number" },
              "valueBoolean": { "type": "boolean" },
              "valueMap": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "key": { "type": "string" },
                    "valueString": { "type": "string" },
                    "valueNumber": { "type": "number" },
                    "valueBoolean": { "type": "boolean" }
                  },
                  "required": ["key"]
                }
              }
            },
            "required": ["key"]
          }
        }
      },
      "required":["contents", "surfaceId"]
    },
    "deleteSurface": {
      "type": "object",
      "properties": {
        "surfaceId": { "type": "string" }
      },
      "required": ["surfaceId"]
    }
  }
}
"""

# ---------------------------------------------------------------------------
# Few-shot example: A2UI Video output for campaign videos
# Includes surfaceId and beginRendering as required by the spec
# FS Corp Fabric App specific examples
# ---------------------------------------------------------------------------

VIDEO_A2UI_EXAMPLE = """
[
  { "beginRendering": { "surfaceId": "campaign-videos", "root": "root" } },
  { "surfaceUpdate": {
    "surfaceId": "campaign-videos",
    "components":[
      { "id": "root", "component": { "Column": { "children": { "explicitList":["video_1_label", "video_1", "video_2_label", "video_2", "video_3_label", "video_3"] } } } },
      { "id": "video_1_label", "component": { "Text": { "usageHint": "caption", "text": { "literalString": "Young Family Peace of Mind" } } } },
      { "id": "video_1", "component": { "Video": { "url": { "literalString": "https://storage.googleapis.com/bucket/videos/video1.mp4" } } } },
      { "id": "video_2_label", "component": { "Text": { "usageHint": "caption", "text": { "literalString": "Frictionless Mobile App Experience" } } } },
      { "id": "video_2", "component": { "Video": { "url": { "literalString": "https://storage.googleapis.com/bucket/videos/video2.mp4" } } } },
      { "id": "video_3_label", "component": { "Text": { "usageHint": "caption", "text": { "literalString": "Long-Term Protection" } } } },
      { "id": "video_3", "component": { "Video": { "url": { "literalString": "https://storage.googleapis.com/bucket/videos/video3.mp4" } } } }
    ]
  } },
  { "dataModelUpdate": { "surfaceId": "campaign-videos", "path": "/", "contents": [] } }
]
"""
