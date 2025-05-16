from google import genai
from google.genai import types

from google.adk.agents import Agent
from google.adk.tools import ToolContext 
import traceback
from datetime import datetime 
from google.cloud import storage
from ... import config

image_uris=[]
client = genai.Client()
def generate_images(prompt: str,  tool_context: ToolContext):

    try:
        
        response = client.models.generate_images(
            model='imagen-3.0-generate-002',
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images= 1,
                aspect_ratio='9:16',
                safety_filter_level="block_low_and_above",
                person_generation="allow_adult"
               
            )     
        )
        generated_image_paths = [] 
        if response.generated_images is not None:
            for generated_image in response.generated_images:
                    # Get the image bytes
                image_bytes = generated_image.image.image_bytes  
                counter= str(tool_context.state.get('loop_iteration', 0))
                artifact_name = f"generated_image_"+ counter + ".png"
                # call save to gcs function
                if config.GCS_BUCKET_NAME:
                    save_to_gcs(tool_context, image_bytes, artifact_name, counter)
                
                # Save as ADK artifact (optional, if still needed by other ADK components)
                report_artifact = types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png"
                )
           
                
                tool_context.save_artifact(artifact_name, report_artifact)
                print(f"Image also saved as ADK artifact: {artifact_name}")

                return {
                    'status': 'success',
                    'message': f'Image generated .  ADK artifact: {artifact_name}.',
                    'artifact_name': artifact_name
                }
        else:
            # model_dump_json might not exist or be the best way to get error details
            error_details = str(response) # Or a more specific error field if available
            print(f'No images generated. Response: {error_details}')
            return {'status': 'error', 'message': f'No images generated. Response: {error_details}'}

    except Exception as e:

         print('exception occured'+ traceback.format_exc())
         traceback.print_exc()
         return {'status': 'error', 'message': 'No images generated.  {e}' }
    
def save_to_gcs(tool_context: ToolContext, image_bytes, filename: str, counter:str): 
        # --- Save to GCS ---
    storage_client = storage.Client() # Initialize GCS client
    bucket_name = config.GCS_BUCKET_NAME

    unique_id = tool_context.state.get('unique_id','')
    current_date_str = datetime.utcnow().strftime("%Y-%m-%d")
    unique_filename = filename
    gcs_blob_name = f"{current_date_str}/{unique_id}/{unique_filename}"

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_name)


    try:
        blob.upload_from_string(image_bytes, content_type="image/png")
        gcs_uri = f"gs://{bucket_name}/{gcs_blob_name}"
        print(f"Image successfully uploaded to GCS: {gcs_uri}")

        # Store GCS URI in session context
        # Store GCS URI in session context
        tool_context.state["generated_image_gcs_uri_"+ counter] = gcs_uri
        

    except Exception as e_gcs:
        print(f"Error uploading image to GCS: {e_gcs}\n{traceback.format_exc()}")
        # Decide if this is a fatal error for the tool
        return {'status': 'error', 'message': f'Image generated but failed to upload to GCS: {e_gcs}'}
                # --- End Save to GCS ---

image_generation_agent = Agent(
    name="image_generation_agent",
    model="gemini-2.0-flash",
    description=(
        "You are an expert in creating images with imagen 3"
    ),
    instruction=(
        "Invoke the 'generate_images' tool by passing the prompt in the 'imagen_prompts' variable as the prompt." 
    
    
    ),
    tools=[generate_images],
    output_key="output_image"
)
  