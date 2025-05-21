# Cymbal Home & Garden Customer Service Agent

This project implements an AI-powered customer service agent for Cymbal Home & Garden, a big-box retailer specializing in home improvement, gardening, and related supplies. The agent is designed to provide excellent customer service, assist customers with product selection, manage orders, schedule services, and offer personalized recommendations.

## Overview

The Cymbal Home & Garden Customer Service Agent is designed to provide a seamless and personalized shopping experience for customers. It leverages Gemini to understand customer needs, offer tailored product recommendations, manage orders, and schedule services. The agent is designed to be friendly, empathetic, and highly efficient, ensuring that customers receive the best possible service.

## Agent Details

The key features of the Customer Service Agent include:

| Feature            | Description             |
| ------------------ | ----------------------- |
| _Interaction Type_ | Conversational          |
| _Complexity_       | Intermediate            |
| _Agent Type_       | Single Agent            |
| _Components_       | Tools, Multimodal, Live |
| _Vertical_         | Retail                  |

### Agent Architecture

![Customer Service Agent Workflow](customer_service_workflow.png)

The agent is built using a multi-modal architecture, combining text and video inputs to provide a rich and interactive experience. It mocks interactions with various tools and services, including a product catalog, inventory management, order processing, and appointment scheduling systems. The agent also utilizes a session management system to maintain context across interactions and personalize the customer experience.

It is important to notice that this agent is not integrated to an actual backend and the behaviour is based on mocked tools. If you would like to implement this agent with actual backend integration you will need to edit [customer_service/tools.py](./customer_service/tools/tools.py)

Because the tools are mocked you might notice that some requested changes will not be applied. For instance newly added item to cart will not show if later a user asks the agent to list all items.

### Key Features

- **Personalized Customer Assistance:**
  - Greets returning customers by name and acknowledges their purchase history.
  - Maintains a friendly, empathetic, and helpful tone.
- **Product Identification and Recommendation:**
  - Assists customers in identifying plants, even from vague descriptions.
  - Requests and utilizes visual aids (video) to accurately identify plants.
  - Provides tailored product recommendations based on identified plants, customer needs, and location (e.g., Las Vegas, NV).
  - Offers alternatives to items in the customer's cart if better options exist.
- **Order Management:**
  - Accesses and displays the contents of a customer's shopping cart.
  - Modifies the cart by adding and removing items based on recommendations and customer approval.
  - Informs customers about relevant sales and promotions.
- **Upselling and Service Promotion:**
  - Suggests relevant services, such as professional planting services.
  - Handles inquiries about pricing and discounts, including competitor offers.
  - Requests manager approval for discounts when necessary.
- **Appointment Scheduling:**
  - Schedules appointments for planting services (or other services).
  - Checks available time slots and presents them to the customer.
  - Confirms appointment details and sends a confirmation/calendar invite.
- **Customer Support and Engagement:**
  - Sends via sms or email plant care instructions relevant to the customer's purchases and location.
  - Offers a discount QR code for future in-store purchases to loyal customers.
- **Tool-Based Interactions:**
  - The agent interacts with the user using a set of tools.
  - The agent can use multiple tools in a single interaction.
  - The agent can use the tools to get information and to modify the user's transaction state.
- **Evaluation:**
  - The agent can be evaluated using a set of test cases.
  - The evaluation is based on the agent's ability to use the tools and to respond to the user's requests.

#### Agent State - Default customer information

The agent's session state is preloaded with sample customer data, simulating a real conversation. Ideally, this state should be loaded from a CRM system at the start of the conversation, using the user's information. This assumes that either the agent authenticates the user or the user is already logged in. If this behavior is expected to be modified edit the [get_customer(current_customer_id: str) in customer.py](./customer_service/entities/customer.py)

#### Tools

The agent has access to the following tools:

- `send_call_companion_link(phone_number: str) -> str`: Sends a link for video connection.
- `approve_discount(type: str, value: float, reason: str) -> str`: Approves a discount (within pre-defined limits).
- `sync_ask_for_approval(type: str, value: float, reason: str) -> str`: Requests discount approval from a manager.
- `update_salesforce_crm(customer_id: str, details: str) -> dict`: Updates customer records in Salesforce.
- `access_cart_information(customer_id: str) -> dict`: Retrieves the customer's cart contents.
- `modify_cart(customer_id: str, items_to_add: list, items_to_remove: list) -> dict`: Updates the customer's cart.
- `get_product_recommendations(plant_type: str, customer_id: str) -> dict`: Suggests suitable products.
- `check_product_availability(product_id: str, store_id: str) -> dict`: Checks product stock.
- `schedule_planting_service(customer_id: str, date: str, time_range: str, details: str) -> dict`: Books a planting service appointment.
- `get_available_planting_times(date: str) -> list`: Retrieves available time slots.
- `send_care_instructions(customer_id: str, plant_type: str, delivery_method: str) -> dict`: Sends plant care information.
- `generate_qr_code(customer_id: str, discount_value: float, discount_type: str, expiration_days: int) -> dict`: Creates a discount QR code.

## Setup and Installations

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- Google ADK SDK (installed via Poetry)
- Google Cloud Project (for Vertex AI Gemini integration)

### Installation
1.  **Prerequisites:**

    For the Agent Engine deployment steps, you will need
    a Google Cloud Project. Once you have created your project,
    [install the Google Cloud SDK](https://cloud.google.com/sdk/docs/install).
    Then run the following command to authenticate with your project:
    ```bash
    gcloud auth login
    ```
    You also need to enable certain APIs. Run the following command to enable
    the required APIs:
    ```bash
    gcloud services enable aiplatform.googleapis.com
    ```

1.  Clone the repository:

    ```bash
    git clone https://github.com/google/adk-samples.git
    cd adk-samples/python/agents/customer-service
    ```

    For the rest of this tutorial **ensure you remain in the `agents/customer-service` directory**.

2.  Install dependencies using Poetry:

- if you have not installed poetry before then run `pip install poetry` first. then you can create your virtual environment and install all dependencies using:

  ```bash
  poetry install
  ```

  To activate the virtual environment run:

  ```bash
  poetry env activate
  ```

3.  Set up Google Cloud credentials:

    - Ensure you have a Google Cloud project.
    - Make sure you have the Vertex AI API enabled in your project.
    - Set the `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION` environment variables. You can set them in your `.env` file (modify and rename .env_sample file to .env) or directly in your shell. Alternatively you can edit [customer_service/config.py](./customer_service/config.py)

    ```bash
    export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_NAME_HERE
    export GOOGLE_GENAI_USE_VERTEXAI=1
    export GOOGLE_CLOUD_LOCATION=us-central1
    ```

## Running the Agent

You can run the agent using the ADK commant in your terminal.
from the root project directory:

1.  Run agent in CLI:

    ```bash
    adk run customer_service
    ```

2.  Run agent with ADK Web UI:
    ```bash
    adk web
    ```
    Select the customer_service from the dropdown

### Example Interaction

Here's a quick example of how a user might interact with the agent:

**User:**:
hi there

**Agent**:
Hey Alex, I see you've been a Cymbal Home & Garden customer for a couple of years now - we appreciate your business! I also see you have a few things in your cart already. How can I help you get your garden growing today?

**User**:
Yeah, so I'm a bit confused. I just bought some...uh...plants or flowers that I wanted to put in the backyard. They just say "sun loving annuals" on them. I totally spaced on picking up potting soil so I'm placing an order for pickup now. The problem is, I'm not sure if the potting soil and fertilizer I picked out are the right ones.

**Agent**:
I understand – it can be tricky when you're not quite sure what you've got! Let's figure this out together. Don't worry, we'll get your plants the right care. To best help you, would you be willing to share a quick video of the plants you've purchased? That will help me identify them and make sure you have the right supplies.

**User**:
Sure, how do I do that?

**Agent**:
I just texted you a link to the number in your profile. Just click on that, and it'll open a secure video connection.

```tool_code
print(send_call_companion_link(phone_number='+12065550123'))
```

```tool_outputs
{'status': 'success', 'message': 'Link sent to +12065550123'}
```

**User**:
Okay, I clicked the link...can you see this?

**Agent**:
Yes I can see you Alex! Can you go ahead and point the camera at your plants so I can inspect them?

**User**:
Yes of course!
...

## Evaluating the Agent

Evaluation tests assess the overall performance and capabilities of the agent in a holistic manner.

**Steps:**

1.  **Run Evaluation Tests:**

    ```bash
    pytest eval
    ```

    - This command executes all test files within the `eval` directory.

## Unit Tests

Unit tests focus on testing individual units or components of the code in isolation.

**Steps:**

1.  **Run Unit Tests:**

    ```bash
    pytest tests/unit
    ```

    - This command executes all test files within the `tests/unit` directory.

## Configuration

You can find further configuration parameters in [customer_service/config.py](./customer_service/config.py). This incudes parameters such as agent name, app name and llm model used by the agent.

## Deployment on Google Agent Engine

In order to inherit all dependencies of your agent you can build the wheel file of the agent and run the deployment.

1.  **Build Customer Service Agent WHL file**

    ```bash
    poetry build --format=wheel --output=deployment
    ```

1.  **Deploy the agent to agents engine**
    It is important to run deploy.py from within deployment folder so paths are correct

    ```bash
    cd deployment
    python deploy.py
    ```

### Testing deployment

This code snippet is an example of how to test the deployed agent.

```
import vertexai
from customer_service.config import Config
from vertexai.preview.reasoning_engines import AdkApp


configs = Config()

vertexai.init(
    project="<GOOGLE_CLOUD_LOCATION_PROJECT_ID>",
    location="<GOOGLE_CLOUD_LOCATION>"
)

# get the agent based on resource id
agent_engine = vertexai.agent_engines.get('DEPLOYMENT_RESOURCE_NAME') # looks like this projects/PROJECT_ID/locations/LOCATION/reasoningEngines/REASONING_ENGINE_ID

for event in remote_agent.stream_query(
    user_id=USER_ID,
    session_id=session["id"],
    message="Hello!",
):
    print(event)

```

## Disclaimer

This agent sample is provided for illustrative purposes only and is not intended for production use. It serves as a basic example of an agent and a foundational starting point for individuals or teams to develop their own agents.

This sample has not been rigorously tested, may contain bugs or limitations, and does not include features or optimizations typically required for a production environment (e.g., robust error handling, security measures, scalability, performance considerations, comprehensive logging, or advanced configuration options).

Users are solely responsible for any further development, testing, security hardening, and deployment of agents based on this sample. We recommend thorough review, testing, and the implementation of appropriate safeguards before using any derived agent in a live or critical system.