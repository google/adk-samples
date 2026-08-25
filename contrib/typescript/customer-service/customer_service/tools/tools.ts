/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Tools module for the customer service agent.
 */

import { v4 as uuidv4 } from 'uuid';

export interface StatusResponse {
  status: string;
  message?: string;
  [key: string]: any;
}

/**
 * Sends a link to the user's phone number to start a video session.
 *
 * @param params.phoneNumber The phone number to send the link to.
 * @returns An object with the status and message.
 */
export function sendCallCompanionLink({
  phoneNumber,
}: {
  phoneNumber: string;
}): StatusResponse {
  console.info(`Sending call companion link to ${phoneNumber}`);
  return { status: 'success', message: `Link sent to ${phoneNumber}` };
}

/**
 * Approve the flat rate or percentage discount requested by the user.
 *
 * @param params.discountType The type of discount, either "percentage" or "flat".
 * @param params.value The value of the discount.
 * @param params.reason The reason for the discount.
 * @returns An object indicating the status of the approval.
 */
export function approveDiscount({
  discountType,
  value,
  reason,
}: {
  discountType: string;
  value: number;
  reason: string;
}): StatusResponse {
  if (value > 10) {
    console.info(`Denying ${discountType} discount of ${value}`);
    // Send back a reason for the error so that the model can recover.
    return {
      status: 'rejected',
      message: 'discount too large. Must be 10 or less.',
    };
  }
  console.info(
    `Approving a ${discountType} discount of ${value} because ${reason}`
  );
  return { status: 'ok' };
}

/**
 * Asks the manager for approval for a discount.
 *
 * @param params.discountType The type of discount, either "percentage" or "flat".
 * @param params.value The value of the discount.
 * @param params.reason The reason for the discount.
 * @returns An object indicating the status of the approval.
 */
export function syncAskForApproval({
  discountType,
  value,
  reason,
}: {
  discountType: string;
  value: number;
  reason: string;
}): StatusResponse {
  console.info(
    `Asking for approval for a ${discountType} discount of ${value} because ${reason}`
  );
  return { status: 'approved' };
}

/**
 * Updates the Salesforce CRM with customer details.
 *
 * @param params.customerId The ID of the customer.
 * @param params.details A dictionary of details to update in Salesforce.
 * @returns An object with the status and message.
 */
export function updateSalesforceCrm({
  customerId,
  details,
}: {
  customerId: string;
  details: Record<string, any>;
}): StatusResponse {
  console.info(
    `Updating Salesforce CRM for customer ID ${customerId} with details: ${JSON.stringify(
      details
    )}`
  );
  return { status: 'success', message: 'Salesforce record updated.' };
}

interface CartItem {
  product_id: string;
  name: string;
  quantity: number;
}

interface Cart {
  items: CartItem[];
  subtotal: number;
}

/**
 * Accesses cart information for a customer.
 *
 * @param params.customerId The ID of the customer.
 * @returns An object representing the cart contents.
 */
export function accessCartInformation({
  customerId,
}: {
  customerId: string;
}): Cart {
  console.info(`Accessing cart information for customer ID: ${customerId}`);

  // MOCK API RESPONSE - Replace with actual API call
  const mockCart: Cart = {
    items: [
      {
        product_id: 'soil-123',
        name: 'Standard Potting Soil',
        quantity: 1,
      },
      {
        product_id: 'fert-456',
        name: 'General Purpose Fertilizer',
        quantity: 1,
      },
    ],
    subtotal: 25.98,
  };
  return mockCart;
}

/**
 * Modifies the user's shopping cart by adding and/or removing items.
 *
 * @param params.customerId The ID of the customer.
 * @param params.itemsToAdd A list of objects, each with 'product_id' and 'quantity'.
 * @param params.itemsToRemove A list of items to remove (often just product_id).
 * @returns An object indicating the status of the cart modification.
 */
export function modifyCart({
  customerId,
  itemsToAdd,
  itemsToRemove,
}: {
  customerId: string;
  itemsToAdd: Record<string, any>[];
  itemsToRemove: Record<string, any>[];
}): StatusResponse {
  console.info(`Modifying cart for customer ID: ${customerId}`);
  console.info(`Adding items: ${JSON.stringify(itemsToAdd)}`);
  console.info(`Removing items: ${JSON.stringify(itemsToRemove)}`);

  // MOCK API RESPONSE - Replace with actual API call
  return {
    status: 'success',
    message: 'Cart updated successfully.',
    items_added: true,
    items_removed: true,
  };
}

interface ProductRecommendation {
  product_id: string;
  name: string;
  description: string;
}

/**
 * Provides product recommendations based on the type of plant.
 *
 * @param params.plantType The type of plant (e.g., 'Petunias', 'Sun-loving annuals').
 * @param params.customerId Optional customer ID for personalized recommendations.
 * @returns An object of recommended products.
 */
export function getProductRecommendations({
  plantType,
  customerId,
}: {
  plantType: string;
  customerId: string;
}): { recommendations: ProductRecommendation[] } {
  console.info(
    `Getting product recommendations for plant type: ${plantType} and customer ${customerId}`
  );

  // MOCK API RESPONSE - Replace with actual API call or recommendation engine
  let recommendations: { recommendations: ProductRecommendation[] };

  if (plantType.toLowerCase() === 'petunias') {
    recommendations = {
      recommendations: [
        {
          product_id: 'soil-456',
          name: 'Bloom Booster Potting Mix',
          description: 'Provides extra nutrients that Petunias love.',
        },
        {
          product_id: 'fert-789',
          name: 'Flower Power Fertilizer',
          description: 'Specifically formulated for flowering annuals.',
        },
      ],
    };
  } else {
    recommendations = {
      recommendations: [
        {
          product_id: 'soil-123',
          name: 'Standard Potting Soil',
          description: 'A good all-purpose potting soil.',
        },
        {
          product_id: 'fert-456',
          name: 'General Purpose Fertilizer',
          description: 'Suitable for a wide variety of plants.',
        },
      ],
    };
  }
  return recommendations;
}

interface ProductAvailability {
  available: boolean;
  quantity: number;
  store: string;
}

/**
 * Checks the availability of a product at a specified store (or for pickup).
 *
 * @param params.productId The ID of the product to check.
 * @param params.storeId The ID of the store (or 'pickup' for pickup availability).
 * @returns An object indicating availability.
 */
export function checkProductAvailability({
  productId,
  storeId,
}: {
  productId: string;
  storeId: string;
}): ProductAvailability {
  console.info(
    `Checking availability of product ID: ${productId} at store: ${storeId}`
  );
  // MOCK API RESPONSE - Replace with actual API call
  return { available: true, quantity: 10, store: storeId };
}

/**
 * Schedules a planting service appointment.
 *
 * @param params.customerId The ID of the customer.
 * @param params.date The desired date (YYYY-MM-DD).
 * @param params.timeRange The desired time range (e.g., "9-12").
 * @param params.details Any additional details (e.g., "Planting Petunias").
 * @returns An object indicating the status of the scheduling.
 */
export function schedulePlantingService({
  customerId,
  date,
  timeRange,
  details,
}: {
  customerId: string;
  date: string;
  timeRange: string;
  details: string;
}): StatusResponse {
  console.info(
    `Scheduling planting service for customer ID: ${customerId} on ${date} (${timeRange})`
  );
  console.info(`Details: ${details}`);

  // MOCK API RESPONSE - Replace with actual API call to your scheduling system
  // Calculate confirmation time based on date and time_range
  const startTimeStr = timeRange.split('-')[0]; // Get the start time (e.g., "9")
  const confirmationTimeStr = `${date} ${startTimeStr}:00`; // e.g., "2024-07-29 9:00"

  return {
    status: 'success',
    appointment_id: uuidv4(),
    date: date,
    time: timeRange,
    confirmation_time: confirmationTimeStr, // formatted time for calendar
  };
}

/**
 * Retrieves available planting service time slots for a given date.
 *
 * @param params.date The date to check (YYYY-MM-DD).
 * @returns A list of available time ranges.
 */
export function getAvailablePlantingTimes({
  date,
}: {
  date: string;
}): string[] {
  console.info(`Retrieving available planting times for ${date}`);
  // MOCK API RESPONSE - Replace with actual API call
  // Generate some mock time slots
  return ['9-12', '13-16'];
}

/**
 * Sends an email or SMS with instructions on how to take care of a specific plant type.
 *
 * @param params.customerId The ID of the customer.
 * @param params.plantType The type of plant.
 * @param params.deliveryMethod 'email' (default) or 'sms'.
 * @returns An object indicating the status.
 */
export function sendCareInstructions({
  customerId,
  plantType,
  deliveryMethod,
}: {
  customerId: string;
  plantType: string;
  deliveryMethod: string;
}): StatusResponse {
  console.info(
    `Sending care instructions for ${plantType} to customer: ${customerId} via ${deliveryMethod}`
  );
  // MOCK API RESPONSE - Replace with actual API call or email/SMS sending logic
  return {
    status: 'success',
    message: `Care instructions for ${plantType} sent via ${deliveryMethod}.`,
  };
}

/**
 * Generates a QR code for a discount.
 *
 * @param params.customerId The ID of the customer.
 * @param params.discountValue The value of the discount (e.g., 10 for 10%).
 * @param params.discountType "percentage" (default) or "fixed".
 * @param params.expirationDays Number of days until the QR code expires.
 * @returns An object containing the QR code data (or a link to it), or an error string if invalid.
 */
export function generateQrCode({
  customerId,
  discountValue,
  discountType,
  expirationDays,
}: {
  customerId: string;
  discountValue: number;
  discountType: string;
  expirationDays: number;
}): StatusResponse | string {
  // Guardrails to validate the amount of discount is acceptable for a auto-approved discount.
  if (discountType === '' || discountType === 'percentage') {
    if (discountValue > 10) {
      return 'cannot generate a QR code for this amount, must be 10% or less';
    }
  }
  if (discountType === 'fixed' && discountValue > 20) {
    return 'cannot generate a QR code for this amount, must be 20 or less';
  }

  console.info(
    `Generating QR code for customer: ${customerId} with ${discountValue} - ${discountType} discount.`
  );

  // MOCK API RESPONSE - Replace with actual QR code generation library
  const expirationDateObj = new Date();
  expirationDateObj.setDate(expirationDateObj.getDate() + expirationDays);
  // Format as YYYY-MM-DD
  const expirationDate = expirationDateObj.toISOString().split('T')[0];

  return {
    status: 'success',
    qr_code_data: 'MOCK_QR_CODE_DATA', // Replace with actual QR code
    expiration_date: expirationDate,
  };
}