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

import logging
from datetime import datetime, timedelta

from customer_service.tools.tools import (
    access_cart_information,
    approve_discount,
    check_product_availability,
    generate_qr_code,
    get_available_planting_times,
    get_product_recommendations,
    modify_cart,
    schedule_planting_service,
    send_call_companion_link,
    send_care_instructions,
    update_salesforce_crm,
)

# Configure logging for the test file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_send_call_companion_link():
    phone_number = "+1-555-123-4567"
    result = send_call_companion_link(phone_number)
    assert result == {
        "status": "success",
        "message": f"Link sent to {phone_number}",
    }


def test_approve_discount_ok():
    result = approve_discount(
        discount_type="percentage", value=10.0, reason="Test discount"
    )
    assert result == {"status": "ok"}


def test_approve_discount_rejected():
    result = approve_discount(
        discount_type="percentage", value=15.0, reason="Test large discount"
    )
    assert result == {
        "message": "discount too large. Must be 10 or less.",
        "status": "rejected",
    }


def test_update_salesforce_crm():
    customer_id = "123"
    details = "Updated customer details"
    result = update_salesforce_crm(customer_id, details)
    assert result == {
        "status": "success",
        "message": "Salesforce record updated.",
    }


def test_access_cart_information():
    customer_id = "123"
    result = access_cart_information(customer_id)
    assert result == {
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


def test_modify_cart_add_and_remove():
    customer_id = "123"
    items_to_add = [{"product_id": "tree-789", "quantity": 1}]
    items_to_remove = [{"product_id": "soil-123"}]
    result = modify_cart(customer_id, items_to_add, items_to_remove)
    assert result == {
        "status": "success",
        "message": "Cart updated successfully.",
        "items_added": True,
        "items_removed": True,
    }


def test_get_product_recommendations_petunias():
    plant_type = "petunias"
    customer_id = "123"
    result = get_product_recommendations(plant_type, customer_id)
    assert result == {
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


def test_get_product_recommendations_other():
    plant_type = "other"
    customer_id = "123"
    result = get_product_recommendations(plant_type, customer_id)
    assert result == {
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


def test_check_product_availability():
    product_id = "soil-123"
    store_id = "Main Store"
    result = check_product_availability(product_id, store_id)
    assert result == {"available": True, "quantity": 10, "store": store_id}


def test_schedule_planting_service():
    customer_id = "123"
    date = "2024-07-29"
    time_range = "9-12"
    details = "Planting Petunias"
    result = schedule_planting_service(customer_id, date, time_range, details)
    assert result["status"] == "success"
    assert result["date"] == date
    assert result["time"] == time_range
    assert "appointment_id" in result
    assert "confirmation_time" in result


def test_get_available_planting_times():
    date = "2024-07-29"
    result = get_available_planting_times(date)
    assert result == ["9-12", "13-16"]


def test_send_care_instructions():
    customer_id = "123"
    plant_type = "Petunias"
    delivery_method = "email"
    result = send_care_instructions(customer_id, plant_type, delivery_method)
    assert result == {
        "status": "success",
        "message": f"Care instructions for {plant_type} sent via {delivery_method}.",
    }


def test_generate_qr_code():
    customer_id = "123"
    discount_value = 10.0
    discount_type = "percentage"
    expiration_days = 30
    result = generate_qr_code(
        customer_id, discount_value, discount_type, expiration_days
    )
    assert result["status"] == "success"
    assert result["qr_code_data"] == "MOCK_QR_CODE_DATA"
    assert "expiration_date" in result
    expiration_date = datetime.now() + timedelta(days=expiration_days)
    assert result["expiration_date"] == expiration_date.strftime("%Y-%m-%d")
