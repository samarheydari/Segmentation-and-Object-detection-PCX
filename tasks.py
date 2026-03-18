import os
import logging
import redis
from rq import Queue
import traceback

# Set PyTorch CUDA memory management environment variables
# Using PyTorch 1.13.1 compatible settings

from src.explanator import Explanator
from src.minio_client import MinIOClient, FHHI_MINIO_BUCKET, NAPLES_MINIO_BUCKET
from common_app_funcs import update_entity, get_bm_id, get_alert_ref_id,get_flight_number,get_uav_id, update_job_status, get_job_status, get_redis_conn, get_job_queue

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

# Set up Redis connection and queue
redis_conn = get_redis_conn() 
job_queue = get_job_queue(redis_conn) 

explanator_logger = logging.getLogger('explanator')
explanator_logger.setLevel(logging.DEBUG)

# This function will be executed by the worker in the background
def process_image_task(entity_type, image_bucket, image_filename, task_id, bm_id, uav_id, flight_number, alert_ref):
    try:
        # Create instances for the worker process
        minio_client = MinIOClient()  # Create a new instance for the worker
        explanator = Explanator(project_root=PROJECT_ROOT, logger=explanator_logger)  # Create in worker process
        
        # Download the image
        logging.info(f"Task {task_id}: Downloading image from MinIO {image_bucket}/{image_filename}")
        img = minio_client.download_image(image_bucket, image_filename)
        if img is None:
            job_status = {
                'status': 'failed',
                'error': f'Error downloading image from MinIO: {image_bucket}/{image_filename}'
            }
            update_job_status(redis_conn, task_id, job_status)
            return
        
        # Update job status
        job_status = get_job_status(redis_conn, task_id)
        job_status['status'] = 'processing'
        job_status['progress'] = 50
        update_job_status(redis_conn, task_id, job_status)
        
        # Generate explanation
        logging.info(f"Task {task_id}: Explaining entity")
        explanation_entity, explanation_images, exp_img_filenames = explanator.explain(entity_type, image_bucket, image_filename, img, bm_id=bm_id,uav_id=uav_id, flight_number=flight_number, alert_ref=alert_ref)
        
        # Upload explanation images
        logging.info(f"Task {task_id}: Uploading explanation images to MinIO")
        for explanation_image, explanation_image_filename in zip(explanation_images, exp_img_filenames):
            minio_client.upload_image(FHHI_MINIO_BUCKET, explanation_image_filename, explanation_image)
        
        # Send explanation to Orion
        logging.info(f"Task {task_id}: Sending explanation entity to Orion")
        orion_response = update_entity(explanation_entity)
        
        # Update job status with results
        if orion_response.status_code == 204:
            job_status = {
                'status': 'completed',
                'progress': 100,
                'explanation_entity_id': explanation_entity[0].get('id', 'unknown'),
                'explanation_images': exp_img_filenames
            }
            update_job_status(redis_conn, task_id, job_status)
        else:
            job_status = {
                'status': 'failed',
                'error': f'Error sending to Orion: Status {orion_response.status_code}'
            }
            update_job_status(redis_conn, task_id, job_status)
            
    except Exception as e:
        logging.error(f"Task {task_id}: Error processing image: {str(e)}")
        logging.error(traceback.format_exc())
        job_status = {
            'status': 'failed',
            'error': str(e)
        }
        update_job_status(redis_conn, task_id, job_status)
