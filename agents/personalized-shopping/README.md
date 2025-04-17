# Personalized Shopping

## Overview of the Agent

This agent sample efficiently provides tailored product recommendations within the ecosystem of a specific brand, merchant, or online marketplace. It enhances the shopping experience within its own context by using targeted data and offering relevant suggestions.

### Agent Details

The personalized-shopping agent has the following capabilities:

* Navigates specific websites to gather product information and understand available options.

* Identifies suitable products using text and image-based search applied to a specific catalog.

* Compares product features or within a defined brand or marketplace scope.

* Recommends products based on user behavior and profile data.

The agent’s default configuration allows you to simulate interactions with a focused shopper. It demonstrates how an agent navigates a specific retail environment.

| <div align="center">Feature</div> | <div align="center">Description</div> |
| --- | --- |
| <div align="center">**Interaction Type**</div> | <div align="center">Conversational</div> |
| <div align="center">**Complexity**</div>  | <div align="center">Easy</div> |
| <div align="center">**Agent Type**</div>  | <div align="center">Single Agent</div> |
| <div align="center">**Components**</div>  | <div align="center">Web Environment: Access to a pre-indexed product website</div> |
|  | <div align="center">SearchTool (to retrieve relevant product information)</div> |
|  | <div align="center">ClickTool (to navigate the website)</div> |
|  | <div align="center">Conversational Memory</div> |
| <div align="center">**Vertical**</div>  | <div align="center">E-Commerce</div> |


### Architecture
![Personalized Shopping Agent Architecture](ps_architecture.png)

### Key Features

The key features of the personalized-shopping agent include:
*  **Environment:** The agent can interact in a e-commerce web environment with 1.18M of products.
*  **Memory:** The agent maintains a conversational memory with all the previous-turns information in its context window.
*  **Tools:**

    _Search_: The agent has access to a search-retrieval engine, where it can perform key-word search for the related products.

    _Click_: The agent has access to the product website and it can navigate the website by clicking buttons.
*  **Evaluation:**
    The agent uses `tool_trajectory_avg_score` and `response_match_score` to measure user satisfaction.

    The evaluation code is located under `eval/test_eval.py`.

    The unittest for tools is located under `tests/test_tools.py`.


## Setup and Installation

IMPORTANT HIGHLIGHTS:
* You should be using [poetry](https://python-poetry.org/docs/) for dependency management and packaging in Python
* Your repo should contain a .env example file highlighting what environmental variables are used and how to active .env

1.  **Prerequisites:**

* To begin with, please firstly _clone this repo_, then install the required packages with the poetry command below.

    ```bash
    cd ~
    python3 -m venv myenv
    source ~/myenv/bin/activate
    pip install poetry
    ```

* Then go to the project folder, and run this:

    ```bash
    cd adk-samples/agents/personalized-shopping
    poetry install
    ```

2.  **Installation:**

* To run the agent, first download the JSON files containing the product information, which are necessary to initialize the web environment the agent will interact with.

    ```bash
    cd personalized_shopping/shared_libraries
    mkdir data
    cd data

    # Download items_shuffle_1000 (4.5MB)
    gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib;

    # Download items_ins_v2_1000 (147KB)
    gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu;

    # Download items_shuffle (5.1GB)
    gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB;

    # Download items_ins_v2 (178MB)
    gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi;

    # Download items_human_ins (4.9MB)
    gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O
    ```

* Then you need to index the product data so that they can be used by the search engine:

    ```bash
    # Convert items.json => required doc format
    cd ../search_engine
    mkdir -p resources_100 resources_1k resources_10k resources_50k
    python convert_product_file_format.py

    # Index the products
    mkdir -p indexes
    bash run_indexing.sh
    cd ../../
    ```
3.  **Configuration:**

* Update the `.env.example` file with your cloud project name and region, then rename it to `.env`.

* Authenticate your GCloud account.

    ```bash
    gcloud auth application-default login
    ```

## Running the Agent

- **Option 1**: You may talk to the agent using the CLI:

    ```bash
    adk run personalized_shopping
    ```

- **Option 2**: You may run the agent on a web interface. This will start a web server on your machine. You may go to the URL printed on the screen and interact with the agent in a chatbot interface.

    ```bash
    cd personalized-shopping
    adk web
    ```
    Please select the `personalized_shopping` option from the dropdown list located at the top left of the screen. Now you can start talking to the agent!


> **Note**: The first run may take some time as the system loads approximately 50,000 product entries into the web environment for the search engine. :)

### Example Interaction

The user can look for product recommendations with both text-based search and image-based search. Here're two quick examples of how a user might interact with the agent.

____________

#### Example 1: text based search

* Session file: [text_search_floral_dress.session.json](tests/example_interactions/text_search_floral_dress.session.md)

#### Example 2: image based search

* Session file: [image_search_denim_skirt.session.json](tests/example_interactions/text_search_floral_dress.session.md) (The input image file for this session is [example_product.png](tests/example_interactions/example_product.png))

## Running Eval

The evaluation assesses the agent's performance on tasks defined in the evalset. Each example in the evalset includes a query, expected tool use, and a reference answer. The judgment criteria are specified in `test_config.json`.

The evaluation of the agent can be run from the `personalized-shopping` directory using
the `pytest` module:

```bash
python3 -m pytest eval
```

You can add more eval prompts by adding your dataset into the `eval/eval_data` folder.

To run unittest for tools, you can run the following command from the `personalized-shopping` directory:

```bash
python3 -m pytest tests
```

## Deployment

* The personalized shopping agent sample can be deployed to Vertex AI Agent Engine. In order to inherit all dependencies of your agent you can build the wheel file of the agent and run the deployment.

1.  **Build Personalized Shopping Agent WHL File**

    ```bash
    cd agents/personalized-shopping
    poetry build --format=wheel --output=deployment
    ```

1.  **Deploy the Agent to Agents Engine**

    ```bash
    cd agents/personalized-shopping/deployment
    python3 deploy.py
    ```

    > **Note**: This process could take more than 10 minutes to finish, please be patient.

When the deployment finishes, it will print a line like this:

```
Created remote agent: projects/<PROJECT_NUMBER>/locations/<PROJECT_LOCATION>/reasoningEngines/<AGENT_ENGINE_ID>
```

You may interact with the deployed agent programmatically in Python:

```python
import dotenv
dotenv.load_dotenv()  # May skip if you have exported environment variables.
from vertexai import agent_engines

agent_engine_id = "AGENT_ENGINE_ID" #Remember to update the ID here. 
user_input = "Hello, can you help me find a summer dress? I want something flowy and floral."

agent_engine = agent_engines.get(agent_engine_id)
session = agent_engine.create_session(user_id="new_user")
for event in agent_engine.stream_query(
    user_id=session["user_id"], session_id=session["id"], message=user_input
):
    for part in event["content"]["parts"]:
        print(part["text"])
```

To delete the deployed agent, you may run the following command:

```bash
python3 deployment/deploy.py --delete --resource_id=${AGENT_ENGINE_ID}
```

## Customization

This agent sample uses the webshop environment from [princeton-nlp/WebShop](https://github.com/princeton-nlp/WebShop), which includes 1.18 million real-world products and 12,087 crowd-sourced text instructions.

By default, the agent loads only 50,000 products into the environment to prevent out-of-memory (OOM) issues. You can adjust this by modifying the `num_product_items` parameter in [init_env.py](personalized_shopping/shared_libraries/init_env.py).

For customization, you can add your own product data and place the annotations in `items_human_ins.json`, `items_ins_v2.json`, and `items_shuffle.json`, then launch the agent sample easily.

## Troubleshooting

* **Q1:** I'm having issues with `gdown` during agent setup. What should I do?
* **A1:** You can manually download the files from the individual Google Drive links and place them in the `personalized-shopping/personalized_shopping/shared_libraries/data` folder.

## Acknowledgement
We are grateful to the developers of [princeton-nlp/WebShop](https://github.com/princeton-nlp/WebShop) for their simulated environment. This agent incorporates modified code from their project.
