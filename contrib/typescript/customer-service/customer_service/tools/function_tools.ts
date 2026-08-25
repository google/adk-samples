import { z } from 'zod';
import { FunctionTool } from '@google/adk';
// Importing functions from the sibling file 'tools.ts'
import {
  sendCallCompanionLink,
  approveDiscount,
  syncAskForApproval,
  updateSalesforceCrm,
  accessCartInformation,
  modifyCart,
  getProductRecommendations,
  checkProductAvailability,
  schedulePlantingService,
  getAvailablePlantingTimes,
  sendCareInstructions,
  generateQrCode
} from './tools';

// --- 1. Zod Input Schemas ---

const SendCallCompanionLinkInput = z.object({
  phoneNumber: z.string().describe("The phone number to send the link to."),
});

const ApproveDiscountInput = z.object({
  discountType: z.string().describe("The type of discount, either 'percentage' or 'flat'."),
  value: z.number().describe("The value of the discount."),
  reason: z.string().describe("The reason for the discount."),
});

const SyncAskForApprovalInput = z.object({
  discountType: z.string().describe("The type of discount, either 'percentage' or 'flat'."),
  value: z.number().describe("The value of the discount."),
  reason: z.string().describe("The reason for the discount."),
});

const UpdateSalesforceCrmInput = z.object({
  customerId: z.string().describe("The ID of the customer."),
  details: z.object({}).passthrough().describe("A dictionary of details to update in Salesforce."),
});

const AccessCartInformationInput = z.object({
  customerId: z.string().describe("The ID of the customer."),
});

const ModifyCartInput = z.object({
  customerId: z.string().describe("The ID of the customer."),
  itemsToAdd: z.array(z.object({ product_id: z.string(), quantity: z.number() }).passthrough()).describe("A list of objects, each with 'product_id' and 'quantity'."),
  itemsToRemove: z.array(z.object({ product_id: z.string() }).passthrough()).describe("A list of items to remove (often just product_id)."),
});

const GetProductRecommendationsInput = z.object({
  plantType: z.string().describe("The type of plant (e.g., 'Petunias', 'Sun-loving annuals')."),
  customerId: z.string().describe("Optional customer ID for personalized recommendations."),
});

const CheckProductAvailabilityInput = z.object({
  productId: z.string().describe("The ID of the product to check."),
  storeId: z.string().describe("The ID of the store (or 'pickup' for pickup availability)."),
});

const SchedulePlantingServiceInput = z.object({
  customerId: z.string().describe("The ID of the customer."),
  date: z.string().describe("The desired date (YYYY-MM-DD)."),
  timeRange: z.string().describe("The desired time range (e.g., '9-12')."),
  details: z.string().describe("Any additional details (e.g., 'Planting Petunias')."),
});

const GetAvailablePlantingTimesInput = z.object({
  date: z.string().describe("The date to check (YYYY-MM-DD)."),
});

const SendCareInstructionsInput = z.object({
  customerId: z.string().describe("The ID of the customer."),
  plantType: z.string().describe("The type of plant."),
  deliveryMethod: z.string().describe("'email' (default) or 'sms'."),
});

const GenerateQrCodeInput = z.object({
  customerId: z.string().describe("The ID of the customer."),
  discountValue: z.number().describe("The value of the discount (e.g., 10 for 10%)."),
  discountType: z.string().describe("'percentage' (default) or 'fixed'."),
  expirationDays: z.number().describe("Number of days until the QR code expires."),
});

// --- 2. Function Tool Definitions ---

export const sendCallCompanionLinkTool = new FunctionTool({
  name: 'sendCallCompanionLink',
  description: 'Sends a link to the user\'s phone number to start a video session.',
  parameters: SendCallCompanionLinkInput,
  execute: sendCallCompanionLink, // Now types match!
});

export const approveDiscountTool = new FunctionTool({
  name: 'approveDiscount',
  description: 'Approve the flat rate or percentage discount requested by the user.',
  parameters: ApproveDiscountInput,
  execute: approveDiscount,
});

export const syncAskForApprovalTool = new FunctionTool({
  name: 'syncAskForApproval',
  description: 'Asks the manager for approval for a discount.',
  parameters: SyncAskForApprovalInput,
  execute: syncAskForApproval,
});

export const updateSalesforceCrmTool = new FunctionTool({
  name: 'updateSalesforceCrm',
  description: 'Updates the Salesforce CRM with customer details.',
  parameters: UpdateSalesforceCrmInput,
  execute: updateSalesforceCrm,
});

export const accessCartInformationTool = new FunctionTool({
  name: 'accessCartInformation',
  description: 'Accesses cart information for a customer.',
  parameters: AccessCartInformationInput,
  execute: accessCartInformation,
});

export const modifyCartTool = new FunctionTool({
  name: 'modifyCart',
  description: 'Modifies the user\'s shopping cart by adding and/or removing items.',
  parameters: ModifyCartInput,
  execute: modifyCart,
});

export const getProductRecommendationsTool = new FunctionTool({
  name: 'getProductRecommendations',
  description: 'Provides product recommendations based on the type of plant.',
  parameters: GetProductRecommendationsInput,
  execute: getProductRecommendations,
});

export const checkProductAvailabilityTool = new FunctionTool({
  name: 'checkProductAvailability',
  description: 'Checks the availability of a product at a specified store (or for pickup).',
  parameters: CheckProductAvailabilityInput,
  execute: checkProductAvailability,
});

export const schedulePlantingServiceTool = new FunctionTool({
  name: 'schedulePlantingService',
  description: 'Schedules a planting service appointment.',
  parameters: SchedulePlantingServiceInput,
  execute: schedulePlantingService,
});

export const getAvailablePlantingTimesTool = new FunctionTool({
  name: 'getAvailablePlantingTimes',
  description: 'Retrieves available planting service time slots for a given date.',
  parameters: GetAvailablePlantingTimesInput,
  execute: getAvailablePlantingTimes,
});

export const sendCareInstructionsTool = new FunctionTool({
  name: 'sendCareInstructions',
  description: 'Sends an email or SMS with instructions on how to take care of a specific plant type.',
  parameters: SendCareInstructionsInput,
  execute: sendCareInstructions,
});

export const generateQrCodeTool = new FunctionTool({
  name: 'generateQrCode',
  description: 'Generates a QR code for a discount.',
  parameters: GenerateQrCodeInput,
  execute: generateQrCode,
});