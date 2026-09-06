# Gemma 3 on Vertex AI & GKE with ADK 🚀

This repository provides a comprehensive guide and all the necessary code to deploy a fine-tuned **Gemma 3** model on Google Cloud. You have two deployment options:
1.  **Google Kubernetes Engine (GKE)** using the high-performance **vLLM** inference server.
2.  **Vertex AI Model Garden & Endpoints** for a fully managed, serverless experience.

Once deployed, you can integrate your Gemma 3 model with intelligent agents built using the **Agent Development Kit (ADK)**.


### 📂 Folder Structure

```.
├── agents
│   ├── gke
│   │   └── gke-agent
│   │       ├── agent.py
│   │       └── __init__.py
│   ├── README.md
│   └── vertexai
│       └── vertexai-agent
│           ├── agent.py
│           └── __init__.py
├── GEMINI.md
├── notebooks
│   ├── common_util.py
│   ├── deploy_gemma3_vllm_on_gke.ipynb
│   ├── deploy_gemma3_vllm_on_vertex.ipynb
├── pyproject.toml
├── README.md
└── uv.lock
```


### 🔧 Prerequisites

Before you begin, ensure you have the following tools installed and configured on your system (Please note Cloud Shell have these pre-installed):

* **Google Cloud SDK:** [Install gcloud CLI](https://cloud.google.com/sdk/docs/install)
    * *Make sure to authenticate by running `gcloud auth login` and `gcloud auth application-default login`.*
* **kubectl:** [Install kubectl](https://kubernetes.io/docs/tasks/tools/)
* **uv:** [Install uv](https://docs.astral.sh/uv/getting-started/installation/) (an extremely fast Python package installer)


### ⚙️ Setup

 **Clone the repository:**
```bash
    git clone <your-repo-url> 

    cd <your-repo-name>
```
--

### 🚀 Deployment Options

Follow one of the notebooks below to deploy your Gemma 3 model.

### Option 1: Deploy to Google Kubernetes Engine (GKE)

This approach uses **vLLM** to serve the Gemma 3 model on a GKE cluster, giving you full control over the serving environment and hardware.

➡️ **Follow the notebook:** [`notebooks/deploy_gemma3_vllm_on_gke.ipynb`](./notebooks/deploy_gemma3_vllm_on_gke.ipynb)

### Option 2: Deploy to Vertex AI

This approach uses the **Vertex AI Model Garden** to deploy Gemma 3 to a **Vertex AI Endpoint**. This is a fully managed, auto-scaling solution that simplifies deployment and maintenance.

➡️ **Follow the notebook:** [`notebooks/deploy_gemma3_vllm_on_vertex.ipynb`](./notebooks/deploy_gemma3_vllm_on_vertex.ipynb)

---

### 🤖 Run the ADK Agents

After successfully deploying your model using either GKE or Vertex AI, you can run the corresponding ADK agent to interact with it.

Navigate to the `agents` directory and follow the instructions in the README file there to start your agent.

```bash
cd agents
```
Inside the agents folder, you will find specific instructions for running the gke-agent or the vertexai-agent, depending on which deployment path you chose.
