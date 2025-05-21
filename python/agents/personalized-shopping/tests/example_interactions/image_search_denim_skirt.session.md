# Example Session: image_search_denim_skirt

**\[user\]:** (Upload [an image](example_product.png)) I really like this one, can you help me find something similar?

**\[personalized_shopping_agent\]:** Okay, I can help with that! Based on the image, it looks like you're interested in a denim skirt. I'll search for similar items.


**\[personalized_shopping_agent\]:**
```json
function call: "search"
args: {
  "keywords": "denim skirt"
}
```


**\[personalized_shopping_agent\]:**
```json
Response from: "search"
response: {
  "result": "Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B01CO4H6TC [SEP] Hard Tail Forever Pull On Pencil Skirt, Cotton, Knee Length Style: W-321 [SEP] $60.0 to $88.0 [SEP] B083CPGNCN [SEP] Women Denim Shorts Jeans New Summer Short Denim Female Pockets Wash pants [SEP] $100.0 [SEP] B0923YHZQP [SEP] Women Denim Overall Dress Adjustable Straps American Flag Jumpsuit Outfits Streetwear [SEP] $22.99 to $31.99 [SEP] B09Q592C74 [SEP] LLDYYDS Holiday Dresses for Women Formal Midi Party Casual Short Sleeve Plus Size That Hide Belly Fat Crewneck Vintage [SEP] $20.99 to $21.99 [SEP] B09SHWQYQM [SEP] Teprec Summer Shorts for Women 2022 High Waisted Butt Lifting Casual Denim Shorts Frayed Ripped Jeans Shorts [SEP] $0.19 to $8.99 [SEP] B076KZJ3T9 [SEP] Woolrich Bitter Creek Oversized Comforter Set Grey/Brown Queen [SEP] $116.76 to $101.53 [SEP] B09H7GG7W9 [SEP] Mens Waffle Cotton Henley T Shirts Long Sleeves Casual Slim Solid Quick Dry Big & Tall Yoga Athletic Button Down Tops [SEP] $11.99 [SEP] B085T3HMGR [SEP] Woman Within Women's Plus Size Petite Flared Denim Skirt [SEP] $24.41 to $29.95 [SEP] B09HS4D3NV [SEP] Meladyan Women Y2K V Neck Checkered Graphic Knitted Sweater Vest Plaid Preppy Sleeveless Argyle Tank Top Jumper [SEP] $12.99 to $22.99 [SEP] B09KVCFQN4 [SEP] Knitted Sweaters for Women V-Neck Color Block Pullover Ribbed Cuff Thermal Fall Clothes Fashion Tops [SEP] $100.0"
}
```


**\[personalized_shopping_agent\]:** Okay, I see a few options here. Based on the search results, "Woman Within Women's Plus Size Petite Flared Denim Skirt" (B085T3HMGR) seems like it could be similar to the skirt in the image. Would you like to explore that product further?


**\[user\]:** yeah B085T3HMGR looks good!!

**\[personalized_shopping_agent\]:** Okay, let's explore "Woman Within Women's Plus Size Petite Flared Denim Skirt" (B085T3HMGR). I will gather information from the "Description," "Features," and "Reviews" sections to give you a comprehensive overview of the product.


**\[personalized_shopping_agent\]:**
```json
function call: "click"
args: {
  "button_name": "B085T3HMGR"
}
```


**\[personalized_shopping_agent\]:**
```json
Response from: "click"
response: {
  "result": "Back to Search [SEP] < Prev [SEP] size [SEP] 12 plus [SEP] 12 plus petite [SEP] 14 plus [SEP] 14 plus petite [SEP] 16 plus [SEP] 16 plus petite [SEP] 18 plus [SEP] 18 plus petite [SEP] 20 plus [SEP] 20 plus petite [SEP] 22 plus [SEP] 22 plus petite [SEP] 24 plus [SEP] 24 plus petite [SEP] 26 plus [SEP] 26 plus petite [SEP] 28 plus [SEP] 28 plus petite [SEP] 30 plus [SEP] 30 plus petite [SEP] 32 plus [SEP] 32 plus petite [SEP] 34 plus [SEP] 34 plus petite [SEP] 36 plus [SEP] 36 plus petite [SEP] 38 plus [SEP] 38 plus petite [SEP] color [SEP] indigo [SEP] natural khaki [SEP] black [SEP] light wash [SEP] stonewash [SEP] white [SEP] Woman Within Women's Plus Size Petite Flared Denim Skirt [SEP] Price: $24.41 to $29.95 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
}
```


**\[personalized_shopping_agent\]:** Okay, I'm now on the product page for the "Woman Within Women's Plus Size Petite Flared Denim Skirt".

I will start by gathering information from the "Description" section.


**\[personalized_shopping_agent\]:**
```json
function call: "click"
args: {
  "button_name": "Description"
}
```


**\[personalized_shopping_agent\]:**
```json
Response from: "click"
response: {
  "result": "Back to Search [SEP] < Prev"
}
```


**\[personalized_shopping_agent\]:** Looks like there is no description available.

Now I will check the "Features".


**\[personalized_shopping_agent\]:**
```json
function call: "click"
args: {
  "button_name": "< Prev"
}
```


**\[personalized_shopping_agent\]:**
```json
Response from: "click"
response: {
  "result": "Back to Search [SEP] < Prev [SEP] size [SEP] 12 plus [SEP] 12 plus petite [SEP] 14 plus [SEP] 14 plus petite [SEP] 16 plus [SEP] 16 plus petite [SEP] 18 plus [SEP] 18 plus petite [SEP] 20 plus [SEP] 20 plus petite [SEP] 22 plus [SEP] 22 plus petite [SEP] 24 plus [SEP] 24 plus petite [SEP] 26 plus [SEP] 26 plus petite [SEP] 28 plus [SEP] 28 plus petite [SEP] 30 plus [SEP] 30 plus petite [SEP] 32 plus [SEP] 32 plus petite [SEP] 34 plus [SEP] 34 plus petite [SEP] 36 plus [SEP] 36 plus petite [SEP] 38 plus [SEP] 38 plus petite [SEP] color [SEP] indigo [SEP] natural khaki [SEP] black [SEP] light wash [SEP] stonewash [SEP] white [SEP] Woman Within Women's Plus Size Petite Flared Denim Skirt [SEP] Price: $24.41 to $29.95 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
}
```


**\[personalized_shopping_agent\]:**
```json
function call: "click"
args: {
  "button_name": "Features"
}
```


**\[personalized_shopping_agent\]:**
```json
Response from: "click"
response: {
  "result": "Back to Search [SEP] < Prev [SEP] PLUS SIZING: Please reference our size charts [SEP] Redefine your silhouette in this great,\u00a0plus size drawstring denim skirt. The fabric is soft with an elastic waist that ties with a drawstring. You can fine tune your waist fit with ease. [SEP] Full elastic drawstring waist. A-line silhouette [SEP] Regular length: 34\". Petite length: 32\". Side seam pockets. Woven cotton denim or twill; imported [SEP] ABOUT THE BRAND: At Woman Within we\u2019re the experts in plus size comfort, and we\u2019ve been doing this for over 100 years. No surprise then that we\u2019re one of the go-to brands in the plus-size clothing industry for incredible fit and of-the-moment style. Our goal is to make sure you always feel great and look good in our comfortable clothes in all the colors you love."
}
```


**\[personalized_shopping_agent\]:** Okay, I have the features now.

Finally, let's check the "Reviews".


**\[personalized_shopping_agent\]:**
```json
function call: "click"
args: {
  "button_name": "< Prev"
}
```


**\[personalized_shopping_agent\]:**
```json
Response from: "click"
response: {
  "result": "Back to Search [SEP] < Prev [SEP] size [SEP] 12 plus [SEP] 12 plus petite [SEP] 14 plus [SEP] 14 plus petite [SEP] 16 plus [SEP] 16 plus petite [SEP] 18 plus [SEP] 18 plus petite [SEP] 20 plus [SEP] 20 plus petite [SEP] 22 plus [SEP] 22 plus petite [SEP] 24 plus [SEP] 24 plus petite [SEP] 26 plus [SEP] 26 plus petite [SEP] 28 plus [SEP] 28 plus petite [SEP] 30 plus [SEP] 30 plus petite [SEP] 32 plus [SEP] 32 plus petite [SEP] 34 plus [SEP] 34 plus petite [SEP] 36 plus [SEP] 36 plus petite [SEP] 38 plus [SEP] 38 plus petite [SEP] color [SEP] indigo [SEP] natural khaki [SEP] black [SEP] light wash [SEP] stonewash [SEP] white [SEP] Woman Within Women's Plus Size Petite Flared Denim Skirt [SEP] Price: $24.41 to $29.95 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
}
```


**\[personalized_shopping_agent\]:**
```json
function call: "click"
args: {
  "button_name": "Reviews"
}
```


**\[personalized_shopping_agent\]:**
```json
Response from: "click"
response: {
  "result": "Back to Search [SEP] < Prev"
}
```


**\[personalized_shopping_agent\]:** There are no reviews available for this product.

Okay, so here's a summary of what I found:

*   **Description:** Not Available
*   **Features:** Plus size denim skirt with a full elastic drawstring waist and A-line silhouette. Available in regular (34") and petite (32") lengths. Side seam pockets. Made from woven cotton denim or twill.
*   **Reviews:** Not Available

Given the lack of a description and reviews, it's a bit difficult to say for sure if this skirt is exactly what you're looking for. However, the features mention an A-line silhouette, which is similar to the skirt in your image.

Would you like to proceed with purchasing this skirt, or would you prefer to go back to the search results and explore other options?


**\[user\]:** Yeah proceed with purchasing. Can you go back to the previous page and show me what options are available?

**\[personalized_shopping_agent\]:** Okay, I will go back to the product page to show you the available options.


**\[personalized_shopping_agent\]:**
```json
function call: "click"
args: {
  "button_name": "< Prev"
}
```


**\[personalized_shopping_agent\]:**
```json
Response from: "click"
response: {
  "result": "Back to Search [SEP] < Prev [SEP] size [SEP] 12 plus [SEP] 12 plus petite [SEP] 14 plus [SEP] 14 plus petite [SEP] 16 plus [SEP] 16 plus petite [SEP] 18 plus [SEP] 18 plus petite [SEP] 20 plus [SEP] 20 plus petite [SEP] 22 plus [SEP] 22 plus petite [SEP] 24 plus [SEP] 24 plus petite [SEP] 26 plus [SEP] 26 plus petite [SEP] 28 plus [SEP] 28 plus petite [SEP] 30 plus [SEP] 30 plus petite [SEP] 32 plus [SEP] 32 plus petite [SEP] 34 plus [SEP] 34 plus petite [SEP] 36 plus [SEP] 36 plus petite [SEP] 38 plus [SEP] 38 plus petite [SEP] color [SEP] indigo [SEP] natural khaki [SEP] black [SEP] light wash [SEP] stonewash [SEP] white [SEP] Woman Within Women's Plus Size Petite Flared Denim Skirt [SEP] Price: $24.41 to $29.95 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
}
```


**\[personalized_shopping_agent\]:** Okay, here are the available options:

**Sizes:**

*   12 plus to 38 plus
*   12 plus petite to 38 plus petite

**Colors:**

*   Indigo
*   Natural Khaki
*   Black
*   Light Wash
*   Stonewash
*   White

What size and color would you like to purchase?


**\[user\]:** select 12 plus, in natural khaki

**\[personalized_shopping_agent\]:** Okay, I have selected size 12 plus and color natural khaki.


**\[personalized_shopping_agent\]:**
```json
function call: "click"
args: {
  "button_name": "12 plus"
}
```


**\[personalized_shopping_agent\]:**
```json
Response from: "click"
response: {
  "result": "Back to Search [SEP] < Prev [SEP] size [SEP] 12 plus [SEP] 12 plus petite [SEP] 14 plus [SEP] 14 plus petite [SEP] 16 plus [SEP] 16 plus petite [SEP] 18 plus [SEP] 18 plus petite [SEP] 20 plus [SEP] 20 plus petite [SEP] 22 plus [SEP] 22 plus petite [SEP] 24 plus [SEP] 24 plus petite [SEP] 26 plus [SEP] 26 plus petite [SEP] 28 plus [SEP] 28 plus petite [SEP] 30 plus [SEP] 30 plus petite [SEP] 32 plus [SEP] 32 plus petite [SEP] 34 plus [SEP] 34 plus petite [SEP] 36 plus [SEP] 36 plus petite [SEP] 38 plus [SEP] 38 plus petite [SEP] color [SEP] indigo [SEP] natural khaki [SEP] black [SEP] light wash [SEP] stonewash [SEP] white [SEP] Woman Within Women's Plus Size Petite Flared Denim Skirt [SEP] Price: $24.41 to $29.95 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
}
```


**\[personalized_shopping_agent\]:**
```json
function call: "click"
args: {
  "button_name": "natural khaki"
}
```


**\[personalized_shopping_agent\]:**
```json
Response from: "click"
response: {
  "result": "Back to Search [SEP] < Prev [SEP] size [SEP] 12 plus [SEP] 12 plus petite [SEP] 14 plus [SEP] 14 plus petite [SEP] 16 plus [SEP] 16 plus petite [SEP] 18 plus [SEP] 18 plus petite [SEP] 20 plus [SEP] 20 plus petite [SEP] 22 plus [SEP] 22 plus petite [SEP] 24 plus [SEP] 24 plus petite [SEP] 26 plus [SEP] 26 plus petite [SEP] 28 plus [SEP] 28 plus petite [SEP] 30 plus [SEP] 30 plus petite [SEP] 32 plus [SEP] 32 plus petite [SEP] 34 plus [SEP] 34 plus petite [SEP] 36 plus [SEP] 36 plus petite [SEP] 38 plus [SEP] 38 plus petite [SEP] color [SEP] indigo [SEP] natural khaki [SEP] black [SEP] light wash [SEP] stonewash [SEP] white [SEP] Woman Within Women's Plus Size Petite Flared Denim Skirt [SEP] Price: $24.41 to $29.95 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
}
```


**\[personalized_shopping_agent\]:** You have selected a size 12 plus and the color natural khaki. Are you ready to finalize the purchase?


**\[user\]:** Yes, go ahead!

**\[personalized_shopping_agent\]:**
```json
function call: "click"
args: {
  "button_name": "Buy Now"
}
```


**\[personalized_shopping_agent\]:**
```json
Response from: "click"
response: {
  "result": "Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [SEP] asin [SEP] B085T3HMGR [SEP] options [SEP] {\"color\": \"natural khaki\", \"size\": \"12 plus\"} [SEP] attrs [SEP] None [SEP] category [SEP] None [SEP] query [SEP] None [SEP] product category [SEP] None [SEP] Target [SEP] asin [SEP] options [SEP] attrs [SEP] price upper [SEP] instuction text [SEP] category [SEP] product category [SEP] query [SEP] Goal [SEP] None [SEP] Reward [SEP] Your score (min 0.0, max 1.0) [SEP] 0.0 [SEP] Reward Details [SEP] None"
}
```


**\[personalized_shopping_agent\]:** Okay, the purchase is being processed. Thank you for shopping with us! The order details are as follows:

*   **Product:** Woman Within Women's Plus Size Petite Flared Denim Skirt
*   **Size:** 12 plus
*   **Color:** Natural Khaki

Please paste the code "None" in your MTurk interface.