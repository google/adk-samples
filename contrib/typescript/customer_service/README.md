# Cymbal Home & Garden Customer Service Agent (TypeScript)

This project implements an AI-powered customer service agent for Cymbal Home & Garden, a big-box retailer specializing in home improvement, gardening, and related supplies. The agent is designed to provide excellent customer service, assist customers with product selection, manage orders, schedule services, and offer personalized recommendations.

## Overview

The Cymbal Home & Garden Customer Service Agent is designed to provide a seamless and personalized shopping experience for customers. It leverages Gemini to understand customer needs, offer tailored product recommendations, manage orders, and schedule services. The agent is designed to be friendly, empathetic, and highly efficient.

## Agent Details

The key features of the Customer Service Agent include:

| Feature            | Description       |
| ------------------ | ----------------- |
| _Interaction Type_ | Conversational    |
| _Complexity_       | Intermediate      |
| _Agent Type_       | Single Agent      |
| _Components_       | Tools, Multimodal |
| _Vertical_         | Retail            |

### Agent Architecture

The agent is built using the Google ADK (Agent Development Kit) for TypeScript. It combines text and video inputs to provide a rich and interactive experience. It mocks interactions with various tools and services, including a product catalog, inventory management, order processing, and appointment scheduling systems.

**Note:** This agent is not integrated into an actual backend; behaviors are based on mocked tools defined in `customer_service/tools/tools.ts`.

### Key Features

- **Personalized Customer Assistance:** Greets returning customers and acknowledges purchase history.
- **Product Identification:** Assists customers in identifying plants using visual aids (video/multimodal inputs).
- **Order Management:** Accesses cart contents and modifies orders (add/remove items).
- **Upselling:** Suggests relevant services (e.g., planting services) and handles pricing inquiries.
- **Appointment Scheduling:** Checks availability and books slots for services.
- **Customer Engagement:** Sends care instructions via SMS/Email and generates discount QR codes.

#### Tools

The agent has access to the following tools (defined in `customer_service/tools/tools.ts`):

- `sendCallCompanionLink`: Sends a link for video connection to the user's phone.
- `approveDiscount`: Approves a discount (within pre-defined limits).
- `syncAskForApproval`: Requests discount approval from a manager.
- `updateSalesforceCrm`: Updates customer records in Salesforce.
- `accessCartInformation`: Retrieves the customer's cart contents.
- `modifyCart`: Updates the customer's cart (add/remove items).
- `getProductRecommendations`: Suggests suitable products based on plant type.
- `checkProductAvailability`: Checks product stock at specific stores.
- `schedulePlantingService`: Books a planting service appointment.
- `getAvailablePlantingTimes`: Retrieves available time slots.
- `sendCareInstructions`: Sends plant care information via email or SMS.
- `generateQrCode`: Creates a discount QR code.

## Setup and Installation

### Prerequisites

- Node.js (v20 or higher)
- Google Cloud Project (for Vertex AI Gemini integration)

### Installation

1.  **Clone the repository** (if applicable) and navigate to the project directory:
    ```bash
    cd adk-samples/typescript/agents/customer_service
    ```

2.  **Install dependencies**:
    ```bash
    npm install
    ```

3.  **Set up Environment Variables**:

    The agent requires Google Cloud credentials. You can set these in your shell or a `.env` file.
    
    Refer to `customer_service/config.ts` for the configuration loading logic.

    ```bash
    export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
    export GOOGLE_CLOUD_LOCATION=us-central1
    export GOOGLE_GENAI_USE_VERTEXAI=1
    export GOOGLE_API_KEY=MY_GOOGLE_API_KEY
    ```

## Running the Agent

The entry point for the agent is `customer_service/agent.ts`. This file contains a rootAgent that is used to run the agent.

To run the agent using `adk run`:

```bash
npx adk run customer_service/agent.ts
```