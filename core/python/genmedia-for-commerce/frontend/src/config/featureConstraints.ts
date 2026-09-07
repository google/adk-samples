/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * GenMedia for Retail - Feature Configuration
 *
 * This file defines the capabilities and constraints for each product type.
 * Update this file to modify which features are available for each product.
 */

export const CAPABILITIES = {
  IMAGE_VTO: 'image-vto',
  VIDEO_VTO: 'video-vto',
  PRODUCT_360: 'product-360',
  BACKGROUND_CHANGE: 'background-change',
  ASSET_TOOLS: 'asset-tools'
} as const

export type Capability = typeof CAPABILITIES[keyof typeof CAPABILITIES]

export const PRODUCTS = {
  GLASSES: 'glasses',
  CLOTHES: 'clothes',
  SHOES: 'shoes',
  CARS: 'cars',
  SMARTPHONES: 'smartphones',
  OTHER: 'other'
} as const

export type Product = typeof PRODUCTS[keyof typeof PRODUCTS]

interface InputConfig {
  required: string[]
  optional: string[]
  uploadBoxes: string[]
}

interface ProductConfigEntry {
  capabilities: Capability[]
  inputs: Partial<Record<Capability, InputConfig>>
}

/**
 * Product Configuration
 *
 * Define which capabilities are available for each product type.
 * Also specify the required inputs for each product/capability combination.
 */
export const PRODUCT_CONFIG: Record<Product, ProductConfigEntry> = {
  [PRODUCTS.GLASSES]: {
    capabilities: [CAPABILITIES.IMAGE_VTO, CAPABILITIES.VIDEO_VTO, CAPABILITIES.ASSET_TOOLS],
    inputs: {
      [CAPABILITIES.IMAGE_VTO]: {
        required: ['product_image', 'face_photo'],
        optional: ['model_side_image'],
        uploadBoxes: ['face_photo']
      },
      [CAPABILITIES.VIDEO_VTO]: {
        required: ['product_image', 'face_photo'],
        optional: ['model_side_image', 'animation_description'],
        uploadBoxes: ['face_photo']
      },
      [CAPABILITIES.ASSET_TOOLS]: {
        required: ['product_image'],
        optional: ['background_description', 'placement_scene'],
        uploadBoxes: ['product_image']
      }
    }
  },

  [PRODUCTS.CLOTHES]: {
    capabilities: [CAPABILITIES.IMAGE_VTO, CAPABILITIES.VIDEO_VTO, CAPABILITIES.ASSET_TOOLS],
    inputs: {
      [CAPABILITIES.IMAGE_VTO]: {
        required: ['face_photo', 'full_body_photo', 'garments'],
        optional: ['scenario'],
        uploadBoxes: ['face_photo', 'full_body_photo']
      },
      [CAPABILITIES.VIDEO_VTO]: {
        required: ['full_body_photo', 'garments'],
        optional: ['animation_description'],
        uploadBoxes: ['full_body_photo']
      },
      [CAPABILITIES.ASSET_TOOLS]: {
        required: ['product_image'],
        optional: ['background_description', 'placement_scene'],
        uploadBoxes: ['product_image']
      }
    }
  },

  [PRODUCTS.SHOES]: {
    capabilities: [CAPABILITIES.PRODUCT_360, CAPABILITIES.ASSET_TOOLS],
    inputs: {
      [CAPABILITIES.PRODUCT_360]: {
        required: ['product_images'],
        optional: ['description', 'veo_model', 'reference_type'],
        uploadBoxes: ['product_images']
      },
      [CAPABILITIES.ASSET_TOOLS]: {
        required: ['product_image'],
        optional: ['background_description', 'placement_scene'],
        uploadBoxes: ['product_image']
      }
    }
  },

  [PRODUCTS.CARS]: {
    capabilities: [CAPABILITIES.PRODUCT_360, CAPABILITIES.ASSET_TOOLS],
    inputs: {
      [CAPABILITIES.PRODUCT_360]: {
        required: ['product_images'],
        optional: ['description', 'veo_model', 'reference_type'],
        uploadBoxes: ['product_images']
      },
      [CAPABILITIES.ASSET_TOOLS]: {
        required: ['product_image'],
        optional: ['background_description', 'placement_scene'],
        uploadBoxes: ['product_image']
      }
    }
  },

  [PRODUCTS.SMARTPHONES]: {
    capabilities: [CAPABILITIES.PRODUCT_360, CAPABILITIES.ASSET_TOOLS],
    inputs: {
      [CAPABILITIES.PRODUCT_360]: {
        required: ['product_images'],
        optional: ['description', 'veo_model', 'reference_type'],
        uploadBoxes: ['product_images']
      },
      [CAPABILITIES.ASSET_TOOLS]: {
        required: ['product_image'],
        optional: ['background_description', 'placement_scene'],
        uploadBoxes: ['product_image']
      }
    }
  },

  // Note: For Product 360, only 'shoes' and 'other' are exposed in the UI.
  // Cars and smartphones are treated as 'other' in the Product 360 flow.
  [PRODUCTS.OTHER]: {
    capabilities: [CAPABILITIES.PRODUCT_360, CAPABILITIES.ASSET_TOOLS],
    inputs: {
      [CAPABILITIES.PRODUCT_360]: {
        required: ['front_view', 'right_view', 'back_view', 'left_view'],
        optional: ['description', 'veo_model'],
        uploadBoxes: ['front_view', 'right_view', 'back_view', 'left_view']
      },
      [CAPABILITIES.ASSET_TOOLS]: {
        required: ['product_image'],
        optional: ['background_description', 'placement_scene'],
        uploadBoxes: ['product_image']
      }
    }
  }
}

interface CapabilityConfigEntry {
  products: Product[]
  description: string
}

/**
 * Capability Configuration
 *
 * Define which products support each capability (reverse mapping).
 */
export const CAPABILITY_CONFIG: Record<Capability, CapabilityConfigEntry> = {
  [CAPABILITIES.IMAGE_VTO]: {
    products: [PRODUCTS.CLOTHES, PRODUCTS.GLASSES],
    description: 'Generate static images with virtual try-on'
  },

  [CAPABILITIES.VIDEO_VTO]: {
    products: [PRODUCTS.CLOTHES, PRODUCTS.GLASSES],
    description: 'Generate animated videos with virtual try-on'
  },

  [CAPABILITIES.PRODUCT_360]: {
    products: [PRODUCTS.SHOES, PRODUCTS.OTHER],
    description: 'Generate 360° product spinning videos'
  },

  [CAPABILITIES.BACKGROUND_CHANGE]: {
    products: [PRODUCTS.GLASSES, PRODUCTS.CLOTHES, PRODUCTS.SHOES],
    description: 'Change product backgrounds with AI'
  },

  [CAPABILITIES.ASSET_TOOLS]: {
    products: [PRODUCTS.GLASSES, PRODUCTS.CLOTHES, PRODUCTS.SHOES, PRODUCTS.OTHER],
    description: 'AI-powered ingredients for creative product photography - background changing and product placement'
  }
}

/**
 * Helper Functions
 */

export const getAvailableCapabilities = (productType: Product): Capability[] => {
  return PRODUCT_CONFIG[productType]?.capabilities || []
}

export const getAvailableProducts = (capability: Capability): Product[] => {
  return CAPABILITY_CONFIG[capability]?.products || []
}

export const getRequiredInputs = (productType: Product, capability: Capability): InputConfig => {
  return PRODUCT_CONFIG[productType]?.inputs?.[capability] || { required: [], optional: [], uploadBoxes: [] }
}

export const getUploadBoxes = (productType: Product, capability: Capability): string[] => {
  return PRODUCT_CONFIG[productType]?.inputs?.[capability]?.uploadBoxes || []
}

export const isValidCombination = (productType: Product, capability: Capability): boolean => {
  const availableCapabilities = getAvailableCapabilities(productType)
  return availableCapabilities.includes(capability)
}

export const getDefaultCapability = (productType: Product): Capability => {
  const capabilities = getAvailableCapabilities(productType)
  return capabilities[0] || CAPABILITIES.IMAGE_VTO
}

export const getDefaultProduct = (capability: Capability): Product => {
  const products = getAvailableProducts(capability)
  return products[0] || PRODUCTS.GLASSES
}

/**
 * Industry/Use Case Configuration
 */

export const INDUSTRIES = {
  ECOMMERCE: 'ecommerce',
  FASHION_APPAREL: 'fashion-apparel',
  EYEWEAR: 'eyewear',
  FOOTWEAR: 'footwear',
  LUXURY_GOODS: 'luxury-goods',
  MARKETING_AGENCIES: 'marketing-agencies',
  RETAIL_BRANDS: 'retail-brands'
} as const

export type Industry = typeof INDUSTRIES[keyof typeof INDUSTRIES]

interface IndustryConfigEntry {
  name: string
  description: string
  icon: string
  capabilities: Capability[]
  products: Product[]
  useCases: string[]
}

export const INDUSTRY_CONFIG: Record<Industry, IndustryConfigEntry> = {
  [INDUSTRIES.ECOMMERCE]: {
    name: 'E-Commerce',
    description: 'Enhance product listings with 360° views and try-on features',
    icon: 'shopping-cart',
    capabilities: [CAPABILITIES.PRODUCT_360, CAPABILITIES.IMAGE_VTO, CAPABILITIES.VIDEO_VTO],
    products: [PRODUCTS.SHOES, PRODUCTS.CARS, PRODUCTS.SMARTPHONES, PRODUCTS.GLASSES],
    useCases: [
      'Interactive product galleries',
      'Virtual try-on experiences',
      'Product detail pages',
      'Conversion rate optimization'
    ]
  },

  [INDUSTRIES.FASHION_APPAREL]: {
    name: 'Fashion & Apparel',
    description: 'Show clothes and accessories on diverse models instantly',
    icon: 'shirt',
    capabilities: [CAPABILITIES.IMAGE_VTO, CAPABILITIES.VIDEO_VTO],
    products: [PRODUCTS.CLOTHES, PRODUCTS.GLASSES],
    useCases: [
      'Model diversity at scale',
      'Virtual fitting rooms',
      'Seasonal campaign content',
      'Social media content'
    ]
  },

  [INDUSTRIES.EYEWEAR]: {
    name: 'Eyewear',
    description: 'Create virtual try-on experiences and product animations for glasses',
    icon: 'glasses',
    capabilities: [CAPABILITIES.IMAGE_VTO, CAPABILITIES.VIDEO_VTO],
    products: [PRODUCTS.GLASSES],
    useCases: [
      'Virtual try-on for sunglasses',
      'Prescription glasses visualization',
      'Style comparison tools',
      'Product launch videos'
    ]
  },

  [INDUSTRIES.FOOTWEAR]: {
    name: 'Footwear',
    description: 'Generate 360° spinning views and product showcases for shoes',
    icon: 'shoe',
    capabilities: [CAPABILITIES.PRODUCT_360],
    products: [PRODUCTS.SHOES],
    useCases: [
      '360° product spins',
      'Product catalogs',
      'Detail highlighting',
      'Marketing campaigns'
    ]
  },

  [INDUSTRIES.LUXURY_GOODS]: {
    name: 'Luxury Goods',
    description: 'Create premium product experiences for cars and smartphones',
    icon: 'diamond',
    capabilities: [CAPABILITIES.PRODUCT_360, CAPABILITIES.VIDEO_VTO],
    products: [PRODUCTS.CARS, PRODUCTS.SMARTPHONES],
    useCases: [
      'Premium product showcases',
      'Detail and craftsmanship highlights',
      'Virtual product demonstrations',
      'Brand storytelling videos'
    ]
  },

  [INDUSTRIES.MARKETING_AGENCIES]: {
    name: 'Marketing Agencies',
    description: 'Create stunning product videos for campaigns at scale',
    icon: 'megaphone',
    capabilities: [CAPABILITIES.IMAGE_VTO, CAPABILITIES.VIDEO_VTO, CAPABILITIES.PRODUCT_360],
    products: [PRODUCTS.GLASSES, PRODUCTS.CLOTHES, PRODUCTS.SHOES, PRODUCTS.CARS, PRODUCTS.SMARTPHONES],
    useCases: [
      'Multi-client campaigns',
      'Rapid content generation',
      'A/B testing creative',
      'Social media assets'
    ]
  },

  [INDUSTRIES.RETAIL_BRANDS]: {
    name: 'Retail Brands',
    description: 'Generate consistent, professional content across channels',
    icon: 'store',
    capabilities: [CAPABILITIES.IMAGE_VTO, CAPABILITIES.VIDEO_VTO, CAPABILITIES.PRODUCT_360],
    products: [PRODUCTS.GLASSES, PRODUCTS.CLOTHES, PRODUCTS.SHOES, PRODUCTS.CARS, PRODUCTS.SMARTPHONES],
    useCases: [
      'Omnichannel content',
      'Brand consistency',
      'Seasonal collections',
      'Product launches'
    ]
  }
}

export interface FeaturedCapability {
  id: Capability
  title: string
  description: string
  icon: string
  benefits: string[]
  industries: Industry[]
}

export const FEATURED_CAPABILITIES: FeaturedCapability[] = [
  {
    id: CAPABILITIES.PRODUCT_360,
    title: '360° Product Views',
    description: 'Create spinning product videos from simple product images. Perfect for e-commerce and product showcases.',
    icon: 'rotate-360',
    benefits: ['Increase engagement', 'Reduce returns', 'Improve conversion'],
    industries: [INDUSTRIES.ECOMMERCE, INDUSTRIES.FOOTWEAR, INDUSTRIES.LUXURY_GOODS]
  },
  {
    id: CAPABILITIES.IMAGE_VTO,
    title: 'Virtual Try-On',
    description: 'Let customers see products on real models with AI-powered virtual try-on for glasses, clothes, and accessories.',
    icon: 'user-check',
    benefits: ['Boost confidence', 'Personalize experience', 'Drive sales'],
    industries: [INDUSTRIES.FASHION_APPAREL, INDUSTRIES.EYEWEAR, INDUSTRIES.ECOMMERCE]
  },
  {
    id: CAPABILITIES.VIDEO_VTO,
    title: 'Custom Video Scenes',
    description: 'Generate professional product videos in custom environments. Describe your scene and watch it come to life.',
    icon: 'video',
    benefits: ['Scale content production', 'Reduce costs', 'Creative flexibility'],
    industries: [INDUSTRIES.MARKETING_AGENCIES, INDUSTRIES.RETAIL_BRANDS, INDUSTRIES.FASHION_APPAREL]
  }
]

export const getAllIndustries = () => {
  return Object.values(INDUSTRIES).map(industryId => ({
    id: industryId,
    ...INDUSTRY_CONFIG[industryId]
  }))
}

export const getIndustriesByCapability = (capability: Capability) => {
  return Object.entries(INDUSTRY_CONFIG)
    .filter(([, config]) => config.capabilities.includes(capability))
    .map(([industryId, config]) => ({
      id: industryId,
      ...config
    }))
}

export const getCapabilitiesByIndustry = (industryId: Industry): Capability[] => {
  return INDUSTRY_CONFIG[industryId]?.capabilities || []
}

export const getProductsByIndustry = (industryId: Industry): Product[] => {
  return INDUSTRY_CONFIG[industryId]?.products || []
}

export const getFeaturedCapabilities = (): FeaturedCapability[] => {
  return FEATURED_CAPABILITIES
}

export const getIndustryById = (industryId: Industry) => {
  return {
    id: industryId,
    ...INDUSTRY_CONFIG[industryId]
  }
}
