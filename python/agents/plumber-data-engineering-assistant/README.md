# Plumber - Data Engineering Buddy

Plumber is your data engineering buddy/agent which has capability of creating and deploying data pipelines on Dataflow, Dataproc and dBT on GKE. It also has capability of commiting code on github and veiwing logs and metrices via cloud monitoring. This is achieved by using multiple sub agents as defined in folder structure below. Idea is to accelerate day to day development of repeatetive data pipelines via agent which can do it just via conversations. Currently this agent leverages publicly available Google provided Dataproc and Dataflow Templates which is open sourced. 

Ref-: 
- https://github.com/GoogleCloudPlatform/DataflowTemplates
- https://github.com/GoogleCloudPlatform/dataproc-templates

*ADK Versiong=1.16.0* 

Note: This sample is to be used by users as is. Users are requested to use this as an sample make changes as fit for purpose for them, including issues specific to their environment.  

## Folder Structure
```
plumber/
├── plumber_agent/
│   ├── sub_agents/
│   │   ├── dataflow_agent/
│   │   ├── dataproc_agent/
│   │   ├── dataproc_template_agent/
│   │   ├── dbt_agent/
│   │   ├── github_agent/
│   │   └── monitoring_agent/
│   ├── agent.py
│   ├── contants.py
│   ├── prompts.py
│   └-  __init__.py
├── .env.example
├── 
├── 
├── README.md
└── requirements.txt
```

<h2 id="configuration">Configuration</h2>

Required environment variables in `.env`:
- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud project ID.
- `GCP_PROJECT`: Project ID where Vertex AI API and data pipelines will be deployed (used in dataproc sub-agent)
- `GOOGLE_GENAI_USE_VERTEXAI`: Set to `TRUE` to use Vertex AI for generative models.
- `GOOGLE_CLOUD_LOCATION`: The default Google Cloud location for your resources (e.g., `us-central1`).

---

#### Sub-Agents Overview

The ADK-based agent is composed of several specialized sub-agents. For more details on each agent and its tools, please refer to the `README.md` file in the respective agent's directory.

- **[Dataflow Agent](plumber_agent/sub_agents/dataflow_agent/README.md)**
- **[Dataproc Agent](plumber_agent/sub_agents/dataproc_agent/README.md)**
- **[Dataproc Template Agent](plumber_agent/sub_agents/dataproc_template_agent/README.md)**
- **[dbt Agent](plumber_agent/sub_agents/dbt_agent/README.md)**
- **[GitHub Agent](plumber_agent/sub_agents/github_agent/README.md)**
- **[Monitoring Agent](plumber_agent/sub_agents/monitoring_agent/README.md)**

## Launching the Agent

You can launch the Plumber agent in two ways: as a command-line web service or through the integrated UI platform.

<h3 id="with-adk-web-command">With ADK Web Command</h3>

The core of Plumber is powered by the Agent Development Kit (ADK), which enables the creation of intelligent agents that can perform complex tasks via command-line or programmatic interaction.

#### How to Launch
1. **Set up your environment:** Ensure you have a `.env` file with the necessary variables as described in the [Configuration](#configuration) section.
2. **Authenticate with GCP and enable VertexAI:**
   ```bash
   gcloud auth login --update-adc
   gcloud config set project PROJECT_ID
   gcloud services enable aiplatform.googleapis.com
   ```
3. **Navigate to the agent directory:**
   ```bash
   cd plumber/
   ```
4. **Install dependencies and run the ADK web server:**
   ```bash
   uv sync --dev
   uv run adk web
   ```


### Google Agents CLI (Recommended)

Use the [Google Agents CLI](https://github.com/google/agents-cli) to create a production-ready version of this agent with additional deployment options.

**Install the CLI** (one-time):

```bash
uvx google-agents-cli setup
```

**Create the project from this sample** (replace `my-plumber` with your project name):

```bash
agents-cli create my-plumber -a adk@plumber-data-engineering-assistant
```

The Google Agents CLI will prompt you to select deployment options and provides additional production-ready features including automated CI/CD deployment scripts.

## Security

- Keep `.env` file private
- Use minimal permissions for service accounts and API keys
- Rotate API keys and credentials regularly

## Support

- [Google ADK Docs](https://developers.google.com/adk)
- [Google Cloud Documentation](https://cloud.google.com/docs)

## Contributors

- Anish Sarangi - anish97IND (Agent Owner)
- Mayank Kalsan - mayank1811
- Aditya- Ignotus-21
- Bala Priya - balapriya01
- Aman Usmani - Amanusmani242
- Pradeep Kumar K - pradeepk556
- Bhabani Mahapatra 


## Copyrights
This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).
