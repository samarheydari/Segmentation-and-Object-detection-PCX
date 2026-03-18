from flask import Flask, request, jsonify, render_template, g
import requests
import os
import logging
from rq.registry import FailedJobRegistry
from rq.job import Job
import traceback
import json
import uuid
from PIL import Image
import io
import numpy as np
import time

# Import your existing modules
from src.explanator import Explanator
from src.minio_client import MinIOClient, FHHI_MINIO_BUCKET, NAPLES_MINIO_BUCKET
from common_app_funcs import update_entity, get_bm_id, set_bm_id,get_uav_id,set_uav_id,get_flight_number,set_flight_number, set_alert_ref_id , get_alert_ref_id , update_job_status, get_job_status, get_redis_conn, get_job_queue
from tasks import process_image_task


# Set up Redis connection and queue
redis_conn = get_redis_conn() 
job_queue = get_job_queue(redis_conn) 


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

app = Flask(__name__, template_folder='./')

# Configure logging, set logging level to DEBUG
logging.basicConfig(level=logging.DEBUG)

# Configure logging to silence specific libraries
logging.getLogger('PIL').setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Get base path from environment variable
BASE_PATH = os.environ.get('BASE_PATH', '/tfa02')


def get_minio_client():
    if 'minio_client' not in g:
        g.minio_client = MinIOClient()
    return g.minio_client



def get_explanator():
    if 'explanator' not in g:
        g.explanator = Explanator(project_root=PROJECT_ROOT, logger=app.logger)
    return g.explanator


# Route for the index page
@app.route(f'{BASE_PATH}')
def index():
    return render_template('index.html', message='Welcome! On this endpoint you can find the TFA02 component of the TEMA project.')


# Route for the ping endpoint
@app.route(f'{BASE_PATH}/ping')
def ping(): 
    app.logger.debug('Ping endpoint called.')
    return jsonify({'status': 'ok'})

# Route to check task status
@app.route(f'{BASE_PATH}/task_status/<task_id>', methods=['GET'])
def task_status(task_id):
    status = get_job_status(redis_conn, task_id)
    if status:
        return jsonify(status)
    else:
        return jsonify({'error': 'Task not found'}), 404

# Route to get all running tasks
@app.route(f'{BASE_PATH}/tasks', methods=['GET'])
def list_tasks():
    # Get all task keys from Redis
    task_keys = redis_conn.keys('job_status:*')
    tasks = {}
    
    # Clean up old completed tasks (older than 1 hour)
    current_time = time.time()
    
    # Process each task
    for key in task_keys:
        # Extract task_id from the key
        task_id = key.decode('utf-8').split(':', 1)[1]
        
        # Get task status data
        status_json = redis_conn.get(key)
        if status_json:
            status_data = json.loads(status_json)
            
            # Check if it's an old completed/failed task
            if status_data.get('status') in ['completed', 'failed']:
                if current_time - status_data.get('created_at', 0) > 3600:  # 1 hour
                    # Delete old tasks
                    redis_conn.delete(key)
                    continue
            
            # Add to the tasks dictionary
            tasks[task_id] = status_data
    
    # Get the current queue length
    queue_length = job_queue.count
    
    return jsonify({
        'tasks': tasks,
        'queue_length': queue_length
    })


@app.route(f'{BASE_PATH}/clean_tasks', methods=['GET'])
def clean_tasks():
    # Delete all job status keys
    task_keys = redis_conn.keys('job_status:*')
    for key in task_keys:
        redis_conn.delete(key)
    
    # Clear any remaining RQ data
    failed_registry = FailedJobRegistry(queue=job_queue)
    failed_job_ids = failed_registry.get_job_ids()
    for job_id in failed_job_ids:
        Job.fetch(job_id, connection=redis_conn).delete()
    
    return jsonify({
        'message': f'Cleaned {len(task_keys)} task records',
    })

@app.route(f'{BASE_PATH}/requeue_tasks', methods=['GET'])
def requeue_tasks():
    # Get all task keys from Redis
    task_keys = redis_conn.keys('job_status:*')
    requeued_count = 0
    
    # Get current job IDs in the queue to avoid duplicates
    queue_job_ids = set(job_queue.job_ids)
    
    for key in task_keys:
        task_id = key.decode('utf-8').split(':', 1)[1]
        status_json = redis_conn.get(key)
        
        if status_json:
            status_data = json.loads(status_json)
            
            # Skip if task is already completed or in the queue
            if status_data.get('status') in ['completed'] or task_id in queue_job_ids:
                continue
                
            # For queued or processing tasks (which might be stuck)
            entity_type = status_data.get('entity_type')
            minio_filename = status_data.get('minio_filename')
            src_image_bucket = status_data.get('src_image_bucket')
            
            if entity_type and minio_filename and src_image_bucket:
                try:
                    # Create a minimal entity with the required fields
                    minimal_entity = {
                        "type": entity_type,
                        "parameters": {"value": {"FileName": minio_filename}},
                        "bucket": {"value": src_image_bucket}
                    }
                    
                    # Re-enqueue the task
                    job = job_queue.enqueue(
                        process_image_task,
                        entity_type, 
                        # minimal_entity, 
                        src_image_bucket, 
                        minio_filename,
                        task_id,
                        job_timeout='12h'
                    )
                    
                    # Update status back to queued
                    updated_status = {
                        'status': 'queued',
                        'progress': 0,
                        'entity_type': entity_type,
                        'minio_filename': minio_filename,
                        'src_image_bucket': src_image_bucket,
                        'created_at': status_data.get('created_at')
                    }
                    update_job_status(redis_conn, task_id, updated_status)
                    
                    requeued_count += 1
                    
                except Exception as e:
                    app.logger.error(f"Failed to requeue task {task_id}: {str(e)}")
    
    return jsonify({
        'message': f'Requeued {requeued_count} tasks',
    })


# Route for the GET request
@app.route(f'{BASE_PATH}/get_data', methods=['GET'])
def get_data():
    try:
        app.logger.debug('GET request received.')
        # Logic to retrieve data would go here
        return jsonify({'message': 'Welcome! This is a sample service.'})
    except Exception as e: 
        app.logger.error(f'Error in GET request: {str(e)}')
        return jsonify({'error': str(e)}), 500


# POST route to enqueue tasks instead of processing immediately
@app.route(f'{BASE_PATH}/post_data', methods=['POST'])
def post_data():
    try: 
        app.logger.debug('POST request received.')
        raw_data = request.get_json()
        app.logger.debug(f'Raw data received: {raw_data}')

        outer_entity_type = raw_data.get('type')
        if outer_entity_type is None:
            err_msg = 'Entity type not provided.'
            app.logger.error(err_msg)
            return jsonify({'error': err_msg}), 400
        
        if outer_entity_type == "Notification":
            # Actual entity is inside the data field
            entity = raw_data.get('data')[0]
            entity_type = entity.get('type')
        else:
            entity = raw_data
            entity_type = outer_entity_type
            app.logger.debug(f"Received outer entity type: {outer_entity_type} instead of Notification")
        
        # Quick validation check
        if entity_type == "Alert":
            uav_id = entity["uav_id"]["value"]
            flight_number = entity["flight_number"]["value"]
            bm_id = entity["bm_id"]["value"]
            alert_ref = entity["alert_ref"]["object"]
            set_uav_id(redis_conn, uav_id)
            set_flight_number(redis_conn, flight_number)
            set_bm_id(redis_conn, bm_id)
            set_alert_ref_id(redis_conn, alert_ref)
            msg = f"Received Alert with bm_id and alert_ref and uav_id and flight_number saved to redis: {bm_id}, {alert_ref}, uav_id: {uav_id}, flight_number: {flight_number}"
            return jsonify({'message': msg}), 200
        
        explanator = get_explanator()
            
        if entity_type not in explanator.VALID_ENTITY_TYPES:
            err_msg = f'Invalid entity type: {entity_type}.'
            app.logger.error(err_msg)
            return jsonify({'error': err_msg}), 400

        current_bm_id = get_bm_id(redis_conn)
        current_alert_ref_id = get_alert_ref_id(redis_conn)
        current_uav_id = get_uav_id(redis_conn)
        current_flight_number = get_flight_number(redis_conn)   
        app.logger.debug(f"Current alert_ref: {current_alert_ref_id}")
        app.logger.debug(f"Current bm_id: {current_bm_id}")
        app.logger.debug(f"Current uav_id: {current_uav_id}")
        app.logger.debug(f"Current flight_number: {current_flight_number}")

        # Extract image information
        posted_uav_id = entity["uav_id"]["value"]
        if current_uav_id is None:
            app.logger.info(f"No cached uav_id; storing value {posted_uav_id}.")
            set_uav_id(redis_conn, posted_uav_id)
            current_uav_id = posted_uav_id
        elif posted_uav_id != current_uav_id:
            app.logger.warning(f"Received uav_id: {posted_uav_id} does not match current uav_id: {current_uav_id}; updating stored uav_id.")
            set_uav_id(redis_conn, posted_uav_id)
            current_uav_id = posted_uav_id

        posted_flight_number = entity["flight_number"]["value"]
        if current_flight_number is None:
            app.logger.info(f"No cached flight_number; storing value {posted_flight_number}.")
            set_flight_number(redis_conn, posted_flight_number)
            current_flight_number = posted_flight_number
        elif posted_flight_number != current_flight_number:
            app.logger.warning(f"Received flight_number: {posted_flight_number} does not match current flight_number: {current_flight_number}; updating stored flight_number.")
            set_flight_number(redis_conn, posted_flight_number)
            current_flight_number = posted_flight_number
             


        posted_bm_id = entity["bm_id"]["value"]
        if current_bm_id is None:
            app.logger.info(f"No cached bm_id; storing value {posted_bm_id}.")
            set_bm_id(redis_conn, posted_bm_id)
            current_bm_id = posted_bm_id
        elif posted_bm_id != current_bm_id:
            app.logger.warning(f"Received bm_id: {posted_bm_id} does not match current bm_id: {current_bm_id}; updating stored bm_id.")
            set_bm_id(redis_conn, posted_bm_id)
            current_bm_id = posted_bm_id

        posted_alert_ref = entity["alert_ref"]["object"]
        if current_alert_ref_id is None:
            app.logger.info(f"No cached alert_ref; storing value {posted_alert_ref}.")
            set_alert_ref_id(redis_conn, posted_alert_ref)
        elif posted_alert_ref != current_alert_ref_id:
            app.logger.warning(f"Received alert_ref: {posted_alert_ref} does not match current alert_ref: {current_alert_ref_id}; updating stored alert_ref.")
            set_alert_ref_id(redis_conn, posted_alert_ref)
            current_alert_ref_id = posted_alert_ref
        src_image_filename = entity["filename"]["value"]
        src_image_bucket = entity["bucket"]["value"]
        
        # Submit tasks for both PersonVehicleDetection and FloodSegmentation
        # entities_to_explain = ['PersonVehicleDetection']
        # entities_to_explain = ['FloodSegmentation']
        entities_to_explain = ['FloodSegmentation', 'PersonVehicleDetection']
        
        task_ids = []
        for entity_type in entities_to_explain:
            # Generate a unique task ID
            task_id = str(uuid.uuid4())
            
            # Store the initial job status
            job_status = {
                'status': 'queued',
                'progress': 0,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'entity_type': entity_type,
                'src_image_bucket': src_image_bucket,
                'minio_filename': src_image_filename,
                'uav_id': posted_uav_id,
                'flight_number': posted_flight_number,
                'bm_id': posted_bm_id,
                'alert_ref': posted_alert_ref
            }
            update_job_status(redis_conn, task_id, job_status)
            
            # Enqueue the task
            job = job_queue.enqueue(
                process_image_task,
                entity_type,
                src_image_bucket,
                src_image_filename,
                task_id,
                uav_id=posted_uav_id,
                flight_number=posted_flight_number,
                bm_id=posted_bm_id,
                alert_ref=posted_alert_ref,
                job_timeout='12h'  # Set an appropriate timeout
            )
            task_ids.append(task_id)
        
        return jsonify({
            'message': 'Task queued successfully',
            'task_ids': task_ids
        }), 202  # Return 202 Accepted for async processing
        
    except Exception as e:
        app.logger.error(f'Unexpected error: {str(e)}')
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# Route for the POST request
@app.route(f'{BASE_PATH}/post_data_old', methods=['POST']) 
def post_data_old():
    try: 
        app.logger.debug('POST request received.') 
        raw_data = request.get_json() 

        app.logger.debug(f'Raw data received: {raw_data}')

        outer_entity_type = raw_data.get('type')

        if outer_entity_type is None:
            err_msg = f'Entity type not provided.'
            app.logger.error(err_msg)
            return jsonify({'error': err_msg}), 400
        
        if outer_entity_type == "Notification":
            # Actual entity is inside the data field
            entity = raw_data.get('data')[0]
            entity_type = entity.get('type')
        else:
            entity = raw_data
            entity_type = outer_entity_type
            app.logger.debug(f"Received outer entity type: {outer_entity_type} instead of Notification")
        
        if entity_type == "Alert":
            uav_id = entity["uav_id"]["value"]
            flight_number = entity["flight_number"]["value"]
            bm_id = entity["bm_id"]["value"]
            alert_ref = entity["alert_ref"]["value"]
            set_alert_ref_id(redis_conn, alert_ref)
            set_bm_id(redis_conn, bm_id)
            set_uav_id(redis_conn, uav_id)
            set_flight_number(redis_conn, flight_number)
            msg = f"Received Alert with bm_id:  {bm_id} and alert_ref: {alert_ref} saved to redis"
            return jsonify({'message': msg}), 200

        bm_id = get_bm_id(redis_conn)
        uav_id = get_uav_id(redis_conn)
        flight_number = get_flight_number(redis_conn)
        alert_ref = get_alert_ref_id(redis_conn)
        app.logger.debug(f"Current alert_ref: {alert_ref}")

        app.logger.debug(f"Current bm_id: {bm_id}")
        app.logger.debug(f"Current uav_id: {uav_id}")
        app.logger.debug(f"Current flight_number: {flight_number}")
        explanator = get_explanator()

        if entity_type not in explanator.VALID_ENTITY_TYPES:
            err_msg = f'Invalid entity type: {entity_type}.'
            app.logger.error(err_msg)
            return jsonify({'error': err_msg}), 400


        # Load the input image from MinIO for AUTH entities
        filename = entity["parameters"]["value"]["FileName"]
        src_image_bucket = entity["bucket"]["value"]

        minio_filename = f"{filename}"

        minio_client = get_minio_client()

        app.logger.info("Downloading image from MinIO")
        img = minio_client.download_image(src_image_bucket, minio_filename)
        if img is None:
            return jsonify({'error': f'Error downloading image from MinIO: {src_image_bucket}/{minio_filename}'}), 500

        app.logger.info("Explaining entity")
        explanation_entity, explanation_images, exp_img_filenames = explanator.explain(entity_type, entity, img)
        
        app.logger.info("Uploading explanation images to MinIO")
        for explanation_image, explanation_image_filename in zip(explanation_images, exp_img_filenames):
            minio_client.upload_image(FHHI_MINIO_BUCKET, explanation_image_filename, explanation_image)

        app.logger.info("Sending explanation entity to Orion")
        orion_response = update_entity(explanation_entity)

        status_map = {
            204: {'json_response': {'message': 'Entity successfully updated on Orion.'}},
            400: {'json_response': {'error': 'Bad request to Orion'}},
            401: {'json_response': {'error': 'Unauthorized request to Orion'}},
            404: {'json_response': {'error': 'Entity not found on Orion'}},
            'default': {'json_response': {'error': 'Unexpected response from Orion'}}
        }

        status_info = status_map.get(orion_response.status_code, status_map['default'])
        orion_json_response = status_info['json_response']

        if orion_response.status_code != 204:
            return jsonify(orion_json_response), orion_response.status_code 

        explanation_image = explanation_images[0]  
        print(np.asarray(Image.fromarray(explanation_image)).shape)
        explanation_image = Image.fromarray(explanation_image)
        explanation_image.show()

        print(np.asarray(Image.fromarray(img)).shape)
        img = Image.fromarray(img)
        img.show()

        return jsonify({'message': 'Explanation successful', 'explanation_entity': explanation_entity, 'orion_response': orion_json_response}), 200
        
    except Exception as e:
        app.logger.error(f'Unexpected error: {str(e)}')
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Unexpected error: {str(e)}, traceback: {traceback.format_exc()}'}), 500






@app.route(f'{BASE_PATH}/delete_entity', methods=['POST'])
def delete_entity():
    try:
        app.logger.debug('POST request to delete entity received.')
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided in the request.'}), 400
        
        entity_id = data.get('entity_id')

        if entity_id is None:
            return jsonify({'error': 'Entity ID not provided.'}), 400
            
        # Construct the delete URL
        broker_url = os.environ.get('BROKER_URL', 'https://orion.tema.digital-enabler.eng.it')
        delete_entity_url = f"{broker_url}/ngsi-ld/v1/entities/{entity_id}"
        
        # Send the delete request to the Orion Context Broker
        app.logger.debug(f'Sending DELETE request to: {delete_entity_url}')
        response = requests.delete(delete_entity_url)
        
        status_map = {
            204: {'log_message': 'Entity successfully deleted from Orion.', 'json_response': {'message': 'Entity successfully deleted.'}},
            400: {'log_message': 'Bad request to Orion.', 'json_response': {'error': 'Bad request to Orion'}},
            401: {'log_message': 'Unauthorized request to Orion.', 'json_response': {'error': 'Unauthorized request to Orion'}},
            404: {'log_message': 'Entity not found on Orion.', 'json_response': {'error': 'Entity not found on Orion'}},
            'default': {'log_message': 'Unexpected response from Orion.', 'json_response': {'error': 'Unexpected response from Orion'}}
        }
        
        status_info = status_map.get(response.status_code, status_map['default'])
        log_message = status_info['log_message']
        json_response = status_info['json_response']
        
        app.logger.debug(f'{log_message} Status code: {response.status_code}')
        return jsonify(json_response), response.status_code
        
    except Exception as e:
        app.logger.error(f'Error in delete entity request: {str(e)}')
        return jsonify({'error': str(e)}), 500


@app.route(f'{BASE_PATH}/subscribe_to_context_broker', methods=['POST'])
def subscribe_to_context_broker():
    try:
        app.logger.debug('POST request to subscribe to context broker received.')
        
        # Delete existing subscription if it exists
        subscription_id = "urn:ngsi-ld:fhhi:one_subscription_to_rule_them_all"

        broker_url = os.environ.get('BROKER_URL', 'https://orion.tema.digital-enabler.eng.it')
        delete_url = f"{broker_url}/ngsi-ld/v1/subscriptions/{subscription_id}"
        
        app.logger.debug(f'Attempting to delete existing subscription: {delete_url}')
        delete_response = requests.delete(delete_url)
        app.logger.debug(f'Delete subscription response: {delete_response.status_code}')
        
        # Get the notification endpoint
        host = request.host_url.rstrip('/')
        post_data_url = f"{host}{BASE_PATH}/post_data"
        app.logger.info(f'POST_DATA_URL: {post_data_url}')
        
        # Create subscription payload
        subscription_payload = {
            "id": subscription_id,
            "type": "Subscription",
            "name": "TFA02 inputs subscription",
            "description": "Subscription description",
            "entities": [
                {"type": "BurntSegmentation"},
                {"type": "FireSegmentation"},
                {"type": "FloodSegmentation"},
                {"type": "PersonVehicleDetection"},
                {"type": "SmokeSegmentation"}
            ],
            "notification": {
                "endpoint": {
                    "uri": post_data_url,
                    "accept": "application/json"
                }
            }
        }
        
        # Create new subscription
        subscription_url = f"{broker_url}/ngsi-ld/v1/subscriptions"
        app.logger.debug(f'Creating subscription at: {subscription_url}')
        
        create_response = requests.post(
            subscription_url,
            headers={'Content-Type': 'application/json'},
            json=subscription_payload
        )
 
        status_map = {
            201: {'message': 'Subscription created successfully.'},
            204: {'message': 'Subscription updated successfully.'},
            400: {'error': 'Bad request to Orion Context Broker.'},
            401: {'error': 'Unauthorized request to Orion Context Broker.'},
            403: {'error': 'Forbidden request to Orion Context Broker.'},
            422: {'error': 'Unprocessable Entity - Invalid subscription payload.'},
            'default': {'error': f'Unexpected response from Orion Context Broker: {create_response.status_code}'}
        }

        response_info = status_map.get(create_response.status_code, status_map['default'])
        app.logger.info(f'Subscription response: {create_response.status_code} - {response_info}')
 
        if create_response.status_code in [201, 204]:
            return jsonify(response_info), create_response.status_code
        else:
            return jsonify(response_info), create_response.status_code

    except Exception as e:
        app.logger.error(f'Error in subscribe to context broker request: {str(e)}')
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# Run the Flask application
if __name__ == '__main__':
    debug = os.environ.get('DEBUG', '').lower() in ('true', '1')
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=debug)
