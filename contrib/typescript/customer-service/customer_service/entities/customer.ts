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
 * Customer entity module.
 */

export interface Address {
  street: string;
  city: string;
  state: string;
  zip: string;
}

export interface Product {
  product_id: string;
  name: string;
  quantity: number;
}

export interface Purchase {
  date: string;
  items: Product[];
  total_amount: number;
}

export interface CommunicationPreferences {
  email: boolean;
  sms: boolean;
  push_notifications: boolean;
}

export interface GardenProfile {
  type: string;
  size: string;
  sun_exposure: string;
  soil_type: string;
  interests: string[];
}

export class Customer {
  account_number: string;
  customer_id: string;
  customer_first_name: string;
  customer_last_name: string;
  email: string;
  phone_number: string;
  customer_start_date: string;
  years_as_customer: number;
  billing_address: Address;
  purchase_history: Purchase[];
  loyalty_points: number;
  preferred_store: string;
  communication_preferences: CommunicationPreferences;
  garden_profile: GardenProfile;
  scheduled_appointments: Record<string, any>;

  constructor(data: Customer) {
    this.account_number = data.account_number;
    this.customer_id = data.customer_id;
    this.customer_first_name = data.customer_first_name;
    this.customer_last_name = data.customer_last_name;
    this.email = data.email;
    this.phone_number = data.phone_number;
    this.customer_start_date = data.customer_start_date;
    this.years_as_customer = data.years_as_customer;
    this.billing_address = data.billing_address;
    this.purchase_history = data.purchase_history;
    this.loyalty_points = data.loyalty_points;
    this.preferred_store = data.preferred_store;
    this.communication_preferences = data.communication_preferences;
    this.garden_profile = data.garden_profile;
    this.scheduled_appointments = data.scheduled_appointments || {};
  }

  /**
   * Converts the Customer object to a JSON string.
   *
   * @returns A JSON string representing the Customer object.
   */
  toJson(): string {
    return JSON.stringify(this, null, 4);
  }

  /**
   * Retrieves a customer based on their ID.
   *
   * @param currentCustomerId The ID of the customer to retrieve.
   * @returns The Customer object if found, null otherwise.
   */
  static getCustomer(currentCustomerId: string): Customer | null {
    // In a real application, this would involve a database lookup.
    // For this example, we'll just return a dummy customer.
    return new Customer({
      customer_id: currentCustomerId,
      account_number: '428765091',
      customer_first_name: 'Alex',
      customer_last_name: 'Johnson',
      email: 'alex.johnson@example.com',
      phone_number: '+1-702-555-1212',
      customer_start_date: '2022-06-10',
      years_as_customer: 2,
      billing_address: {
        street: '123 Main St',
        city: 'Anytown',
        state: 'CA',
        zip: '12345',
      },
      purchase_history: [
        {
          date: '2023-03-05',
          items: [
            {
              product_id: 'fert-111',
              name: 'All-Purpose Fertilizer',
              quantity: 1,
            },
            {
              product_id: 'trowel-222',
              name: 'Gardening Trowel',
              quantity: 1,
            },
          ],
          total_amount: 35.98,
        },
        {
          date: '2023-07-12',
          items: [
            {
              product_id: 'seeds-333',
              name: 'Tomato Seeds (Variety Pack)',
              quantity: 2,
            },
            {
              product_id: 'pots-444',
              name: 'Terracotta Pots (6-inch)',
              quantity: 4,
            },
          ],
          total_amount: 42.5,
        },
        {
          date: '2024-01-20',
          items: [
            {
              product_id: 'gloves-555',
              name: 'Gardening Gloves (Leather)',
              quantity: 1,
            },
            {
              product_id: 'pruner-666',
              name: 'Pruning Shears',
              quantity: 1,
            },
          ],
          total_amount: 55.25,
        },
      ],
      loyalty_points: 133,
      preferred_store: 'Anytown Garden Store',
      communication_preferences: {
        email: true,
        sms: false,
        push_notifications: true,
      },
      garden_profile: {
        type: 'backyard',
        size: 'medium',
        sun_exposure: 'full sun',
        soil_type: 'unknown',
        interests: ['flowers', 'vegetables'],
      },
      scheduled_appointments: {},
    } as Customer);
  }
}