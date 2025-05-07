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
# add docstring to this module
"""Tools module for the customer service agent."""

import logging
import uuid
from datetime import datetime, timedelta
from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)


def send_call_companion_link(phone_number: str) -> str:
    """
    Sends a link to the user's phone number to start a video session.

    Args:
        phone_number (str): The phone number to send the link to.

    Returns:
        dict: A dictionary with the status and message.

    Example:
        >>> send_call_companion_link(phone_number='+12065550123')
        {'status': 'success', 'message': 'Link sent to +12065550123'}
    """

    logger.info("Sending call companion link to %s", phone_number)

    return {"status": "success", "message": f"Link sent to {phone_number}"}


def approve_discount(discount_type: str, value: float, reason: str) -> str:
    """
    Approve the flat rate or percentage discount requested by the user.

    Args:
        discount_type (str): The type of discount, either "percentage" or "flat".
        value (float): The value of the discount.
        reason (str): The reason for the discount.

    Returns:
        str: A JSON string indicating the status of the approval.

    Example:
        >>> approve_discount(type='percentage', value=10.0, reason='Customer loyalty')
        '{"status": "ok"}'
    """
    if value > 10:
        logger.info("Denying %s discount of %s", discount_type, value)
        # Send back a reason for the error so that the model can recover.
        return {"status": "rejected",
                "message": "discount too large. Must be 10 or less."}
    logger.info(
        "Approving a %s discount of %s because %s", discount_type, value, reason
    )
    return {"status": "ok"}

def sync_ask_for_approval(discount_type: str, value: float, reason: str) -> str:
    """
    Asks the manager for approval for a discount.

    Args:
        discount_type (str): The type of discount, either "percentage" or "flat".
        value (float): The value of the discount.
        reason (str): The reason for the discount.

    Returns:
        str: A JSON string indicating the status of the approval.

    Example:
        >>> sync_ask_for_approval(type='percentage', value=15, reason='Customer loyalty')
        '{"status": "approved"}'
    """
    logger.info(
        "Asking for approval for a %s discount of %s because %s",
        discount_type,
        value,
        reason,
    )
    return {"status": "approved"}


def update_salesforce_crm(customer_id: str, details: dict) -> dict:
    """
    Updates the Salesforce CRM with customer details.

    Args:
        customer_id (str): The ID of the customer.
        details (str): A dictionary of details to update in Salesforce.

    Returns:
        dict: A dictionary with the status and message.

    Example:
        >>> update_salesforce_crm(customer_id='123', details={
            'appointment_date': '2024-07-25',
            'appointment_time': '9-12',
            'services': 'Planting',
            'discount': '15% off planting',
            'qr_code': '10% off next in-store purchase'})
        {'status': 'success', 'message': 'Salesforce record updated.'}
    """
    logger.info(
        "Updating Salesforce CRM for customer ID %s with details: %s",
        customer_id,
        details,
    )
    return {"status": "success", "message": "Salesforce record updated."}


def access_cart_information(customer_id: str) -> dict:
    """
    Args:
        customer_id (str): The ID of the customer.

    Returns:
        dict: A dictionary representing the cart contents.

    Example:
        >>> access_cart_information(customer_id='123')
        {'items': [{'product_id': 'soil-123', 'name': 'Standard Potting Soil', 'quantity': 1}, {'product_id': 'fert-456', 'name': 'General Purpose Fertilizer', 'quantity': 1}], 'subtotal': 25.98}
    """
    logger.info("Accessing cart information for customer ID: %s", customer_id)

    # MOCK API RESPONSE - Replace with actual API call
    mock_cart = {
        "items": [
            {
                "product_id": "soil-123",
                "name": "Standard Potting Soil",
                "quantity": 1,
            },
            {
                "product_id": "fert-456",
                "name": "General Purpose Fertilizer",
                "quantity": 1,
            },
        ],
        "subtotal": 25.98,
    }
    return mock_cart


def modify_cart(
    customer_id: str, items_to_add: list[dict], items_to_remove: list[dict]
) -> dict:
    """Modifies the user's shopping cart by adding and/or removing items.

    Args:
        customer_id (str): The ID of the customer.
        items_to_add (list): A list of dictionaries, each with 'product_id' and 'quantity'.
        items_to_remove (list): A list of product_ids to remove.

    Returns:
        dict: A dictionary indicating the status of the cart modification.
    Example:
        >>> modify_cart(customer_id='123', items_to_add=[{'product_id': 'soil-456', 'quantity': 1}, {'product_id': 'fert-789', 'quantity': 1}], items_to_remove=[{'product_id': 'fert-112', 'quantity': 1}])
        {'status': 'success', 'message': 'Cart updated successfully.', 'items_added': True, 'items_removed': True}
    """

    logger.info("Modifying cart for customer ID: %s", customer_id)
    logger.info("Adding items: %s", items_to_add)
    logger.info("Removing items: %s", items_to_remove)
    # MOCK API RESPONSE - Replace with actual API call
    return {
        "status": "success",
        "message": "Cart updated successfully.",
        "items_added": True,
        "items_removed": True,
    }


def get_product_recommendations(plant_type: str, customer_id: str) -> dict:
    """Provides product recommendations based on the type of plant.

    Args:
        plant_type: The type of plant (e.g., 'Petunias', 'Sun-loving annuals').
        customer_id: Optional customer ID for personalized recommendations.

    Returns:
        A dictionary of recommended products. Example:
        {'recommendations': [
            {'product_id': 'soil-456', 'name': 'Bloom Booster Potting Mix', 'description': '...'},
            {'product_id': 'fert-789', 'name': 'Flower Power Fertilizer', 'description': '...'}
        ]}
    """
    #
    logger.info(
        "Getting product recommendations for plant " "type: %s and customer %s",
        plant_type,
        customer_id,
    )
    # MOCK API RESPONSE - Replace with actual API call or recommendation engine
    if plant_type.lower() == "petunias":
        recommendations = {
            "recommendations": [
                {
                    "product_id": "soil-456",
                    "name": "Bloom Booster Potting Mix",
                    "description": "Provides extra nutrients that Petunias love.",
                },
                {
                    "product_id": "fert-789",
                    "name": "Flower Power Fertilizer",
                    "description": "Specifically formulated for flowering annuals.",
                },
            ]
        }
    else:
        recommendations = {
            "recommendations": [
                {
                    "product_id": "soil-123",
                    "name": "Standard Potting Soil",
                    "description": "A good all-purpose potting soil.",
                },
                {
                    "product_id": "fert-456",
                    "name": "General Purpose Fertilizer",
                    "description": "Suitable for a wide variety of plants.",
                },
            ]
        }
    return recommendations


def check_product_availability(product_id: str, store_id: str) -> dict:
    """Checks the availability of a product at a specified store (or for pickup).

    Args:
        product_id: The ID of the product to check.
        store_id: The ID of the store (or 'pickup' for pickup availability).

    Returns:
        A dictionary indicating availability.  Example:
        {'available': True, 'quantity': 10, 'store': 'Main Store'}

    Example:
        >>> check_product_availability(product_id='soil-456', store_id='pickup')
        {'available': True, 'quantity': 10, 'store': 'pickup'}
    """
    logger.info(
        "Checking availability of product ID: %s at store: %s",
        product_id,
        store_id,
    )
    # MOCK API RESPONSE - Replace with actual API call
    return {"available": True, "quantity": 10, "store": store_id}


def schedule_planting_service(
    customer_id: str, date: str, time_range: str, details: str
) -> dict:
    """Schedules a planting service appointment.

    Args:
        customer_id: The ID of the customer.
        date:  The desired date (YYYY-MM-DD).
        time_range: The desired time range (e.g., "9-12").
        details: Any additional details (e.g., "Planting Petunias").

    Returns:
        A dictionary indicating the status of the scheduling. Example:
        {'status': 'success', 'appointment_id': '12345', 'date': '2024-07-29', 'time': '9:00 AM - 12:00 PM'}

    Example:
        >>> schedule_planting_service(customer_id='123', date='2024-07-29', time_range='9-12', details='Planting Petunias')
        {'status': 'success', 'appointment_id': 'some_uuid', 'date': '2024-07-29', 'time': '9-12', 'confirmation_time': '2024-07-29 9:00'}
    """
    logger.info(
        "Scheduling planting service for customer ID: %s on %s (%s)",
        customer_id,
        date,
        time_range,
    )
    logger.info("Details: %s", details)
    # MOCK API RESPONSE - Replace with actual API call to your scheduling system
    # Calculate confirmation time based on date and time_range
    start_time_str = time_range.split("-")[0]  # Get the start time (e.g., "9")
    confirmation_time_str = (
        f"{date} {start_time_str}:00"  # e.g., "2024-07-29 9:00"
    )

    return {
        "status": "success",
        "appointment_id": str(uuid.uuid4()),
        "date": date,
        "time": time_range,
        "confirmation_time": confirmation_time_str,  # formatted time for calendar
    }


def get_available_planting_times(date: str) -> list:
    """Retrieves available planting service time slots for a given date.

    Args:
        date: The date to check (YYYY-MM-DD).

    Returns:
        A list of available time ranges.

    Example:
        >>> get_available_planting_times(date='2024-07-29')
        ['9-12', '13-16']
    """
    logger.info("Retrieving available planting times for %s", date)
    # MOCK API RESPONSE - Replace with actual API call
    # Generate some mock time slots, ensuring they're in the correct format:
    return ["9-12", "13-16"]


def send_care_instructions(
    customer_id: str, plant_type: str, delivery_method: str
) -> dict:
    """Sends an email or SMS with instructions on how to take care of a specific plant type.

    Args:
        customer_id:  The ID of the customer.
        plant_type: The type of plant.
        delivery_method: 'email' (default) or 'sms'.

    Returns:
        A dictionary indicating the status.

    Example:
        >>> send_care_instructions(customer_id='123', plant_type='Petunias', delivery_method='email')
        {'status': 'success', 'message': 'Care instructions for Petunias sent via email.'}
    """
    logger.info(
        "Sending care instructions for %s to customer: %s via %s",
        plant_type,
        customer_id,
        delivery_method,
    )
    # MOCK API RESPONSE - Replace with actual API call or email/SMS sending logic
    return {
        "status": "success",
        "message": f"Care instructions for {plant_type} sent via {delivery_method}.",
    }


def generate_qr_code(
    customer_id: str,
    discount_value: float,
    discount_type: str,
    expiration_days: int,
) -> dict:
    """Generates a QR code for a discount.

    Args:
        customer_id: The ID of the customer.
        discount_value: The value of the discount (e.g., 10 for 10%).
        discount_type: "percentage" (default) or "fixed".
        expiration_days: Number of days until the QR code expires.

    Returns:
        A dictionary containing the QR code data (or a link to it). Example:
        {'status': 'success', 'qr_code_data': '...', 'expiration_date': '2024-08-28'}

    Example:
        >>> generate_qr_code(customer_id='123', discount_value=10.0, discount_type='percentage', expiration_days=30)
        {'status': 'success', 'qr_code_data': 'MOCK_QR_CODE_DATA', 'expiration_date': '2024-08-24'}
    """
    
    # Guardrails to validate the amount of discount is acceptable for a auto-approved discount.
    # Defense-in-depth to prevent malicious prompts that could circumvent system instructions and
    # be able to get arbitrary discounts.
    if discount_type == "" or discount_type == "percentage":
        if discount_value > 10:
            return "cannot generate a QR code for this amount, must be 10% or less"
    if discount_type == "fixed" and discount_value > 20:
        return "cannot generate a QR code for this amount, must be 20 or less"
    
    logger.info(
        "Generating QR code for customer: %s with %s - %s discount.",
        customer_id,
        discount_value,
        discount_type,
    )
    # MOCK API RESPONSE - Replace with actual QR code generation library
    expiration_date = (
        datetime.now() + timedelta(days=expiration_days)
    ).strftime("%Y-%m-%d")
    return {
        "status": "success",
        "qr_code_data": "MOCK_QR_CODE_DATA",  # Replace with actual QR code
        "expiration_date": expiration_date,
    }
