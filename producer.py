import sys
import types
import cv2
import threading
import time
import configparser
import hashlib
from concurrent.futures import ThreadPoolExecutor
from kafka import KafkaProducer
from kafka.errors import KafkaError
import numpy as np
from queue import Queue
import logging
import atexit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoStreamProducer:
    def __init__(self, bootstrap_servers='localhost:9092', producer_id=None):
        self.producer_id = producer_id or f"producer-{threading.current_thread().ident}"
        self.producer = self.create_producer(bootstrap_servers)
        self.active_streams = set()
        self.shutdown_flag = threading.Event()
        

    @staticmethod
    def create_producer(bootstrap_servers):
        return KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            linger_ms=5,  # Reduced from 10 to decrease latency
            batch_size=32768,  # Increased for better throughput
            compression_type='gzip',
            buffer_memory=67108864,  # 64MB buffer
            max_request_size=5242880,  # 5MB max request
            retries=3
        )

    @staticmethod
    def hash_camera_to_producer(camera_url, num_producers):
        """
        Hash function to assign cameras to producers consistently.
        Uses SHA-256 hash of the camera URL to ensure consistent assignment.
        """
        hash_object = hashlib.sha256(camera_url.encode())
        hash_int = int(hash_object.hexdigest(), 16)
        return hash_int % num_producers

    @staticmethod
    def get_even_camera_distribution(camera_urls, num_producers):
        """
        Distribute cameras evenly across producers using round-robin with hash-based consistency.
        Ensures each producer gets roughly the same number of cameras.
        """
        total_cameras = len(camera_urls)
        base_cameras_per_producer = total_cameras // num_producers
        extra_cameras = total_cameras % num_producers
        
        # Create a list to track how many cameras each producer should get
        cameras_per_producer = [base_cameras_per_producer] * num_producers
        
        # Distribute the extra cameras to the first few producers
        for i in range(extra_cameras):
            cameras_per_producer[i] += 1
        
        # Create camera-to-producer mapping using hash for consistency
        camera_producer_pairs = []
        for idx, camera_url in enumerate(camera_urls):
            # Use hash to create a stable sort key for consistent assignment
            hash_key = VideoStreamProducer.hash_camera_to_producer(camera_url, 999999)  # Large number for better distribution
            camera_producer_pairs.append((idx, camera_url, hash_key))
        
        # Sort by hash key to ensure consistent ordering
        camera_producer_pairs.sort(key=lambda x: x[2])
        
        # Assign cameras to producers in round-robin fashion
        producer_assignments = [[] for _ in range(num_producers)]
        producer_index = 0
        cameras_assigned_to_current = 0
        
        for idx, camera_url, _ in camera_producer_pairs:
            producer_assignments[producer_index].append((idx, camera_url))
            cameras_assigned_to_current += 1
            
            # Move to next producer if current one has enough cameras
            if cameras_assigned_to_current >= cameras_per_producer[producer_index]:
                producer_index += 1
                cameras_assigned_to_current = 0
        
        return producer_assignments, cameras_per_producer

    
    def validate_even_distribution(camera_urls, num_producers):
        """
        Validate that the camera distribution is as even as possible.
        """
        producer_assignments, expected_counts = VideoStreamProducer.get_even_camera_distribution(camera_urls, num_producers)
        
        # Check if all cameras are assigned
        total_assigned = sum(len(assignment) for assignment in producer_assignments)
        if total_assigned != len(camera_urls):
            raise ValueError(f"Camera assignment error: {total_assigned} assigned, {len(camera_urls)} expected")
        
        # Check if distribution is even (difference should be at most 1)
        actual_counts = [len(assignment) for assignment in producer_assignments]
        min_count = min(actual_counts)
        max_count = max(actual_counts)
        
        if max_count - min_count > 1:
            raise ValueError(f"Uneven distribution: min={min_count}, max={max_count}")
        
        # Verify expected vs actual counts match
        for i, (expected, actual) in enumerate(zip(expected_counts, actual_counts)):
            if expected != actual:
                raise ValueError(f"Producer {i} count mismatch: expected {expected}, got {actual}")
        
        # Check for duplicate assignments
        all_assigned_cameras = set()
        for assignment in producer_assignments:
            for cam_idx, _ in assignment:
                if cam_idx in all_assigned_cameras:
                    raise ValueError(f"Duplicate camera assignment: camera {cam_idx}")
                all_assigned_cameras.add(cam_idx)
        
        logger.info("✅ Camera distribution validation passed!")
        return True

    @staticmethod
    def get_assigned_cameras(camera_urls, producer_index, num_producers):
        """
        Get cameras assigned to a specific producer with even distribution.
        """
        producer_assignments, _ = VideoStreamProducer.get_even_camera_distribution(camera_urls, num_producers)
        return producer_assignments[producer_index] if producer_index < len(producer_assignments) else []

    def emit_video(self, camera_index, rtsp_url, topic, width=640, height=640, frame_skip=10):
        try:
            logger.info(f'Producer {self.producer_id} starting video stream from camera {camera_index}: {rtsp_url}')
            video = cv2.VideoCapture(rtsp_url)
            if not video.isOpened():
                logger.error(f"Failed to open stream: {rtsp_url}")
                return

            # Set OpenCV buffer size
            video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Create a thread-safe counter
            class FrameCounter:
                def __init__(self):
                    self.count = 0
                    self.lock = threading.Lock()
                
                def increment(self):
                    with self.lock:
                        self.count += 1
                        return self.count

            frame_counter = FrameCounter()
            frame_buffer = Queue(maxsize=10)  # Buffer for frame preprocessing

            def preprocess_frame():
                while not self.shutdown_flag.is_set():
                    success, frame = video.read()
                    if not success:
                        logger.warning(f"Failed to read frame from {rtsp_url}")
                        break

                    current_frame = frame_counter.increment()
                    if current_frame % frame_skip == 0:
                        resized_frame = cv2.resize(frame, (width, height))
                        _, encoded_frame = cv2.imencode('.jpeg', resized_frame, 
                                                        [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frame_buffer.put(encoded_frame.tobytes())

            # Start preprocessing thread
            preprocess_thread = threading.Thread(target=preprocess_frame)
            preprocess_thread.start()
            self.active_streams.add(preprocess_thread)

            # Main sending loop
            while not self.shutdown_flag.is_set():
                try:
                    frame_data = frame_buffer.get(timeout=1.0)
                    future = self.producer.send(topic, frame_data)
                    future.get(timeout=2.0)  
                except Exception as e:
                    logger.error(f"Error in stream {rtsp_url}: {str(e)}")
                    break

        finally:
            if 'video' in locals():
                video.release()
            if 'preprocess_thread' in locals() and preprocess_thread in self.active_streams:
                self.active_streams.remove(preprocess_thread)

    def start_assigned_streams(self, camera_urls, producer_index, num_producers):
        """
        Start streams for cameras assigned to this producer using hash function.
        """
        assigned_cameras = self.get_assigned_cameras(camera_urls, producer_index, num_producers)
        
        if not assigned_cameras:
            logger.info(f"Producer {self.producer_id} has no assigned cameras")
            return

        logger.info(f"Producer {self.producer_id} assigned to {len(assigned_cameras)} cameras")
        
        with ThreadPoolExecutor(max_workers=len(assigned_cameras)) as executor:
            futures = []
            for camera_index, rtsp_url in assigned_cameras:
                topic = f'camera-stream-{camera_index}'
                logger.info(f"Producer {self.producer_id} handling camera {camera_index}: {rtsp_url}")
                futures.append(
                    executor.submit(self.emit_video, camera_index, rtsp_url, topic)
                )
            
            # Wait for all streams to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Stream failed with error: {str(e)}")

    def cleanup(self):
        logger.info(f"Shutting down video streams for producer {self.producer_id}...")
        self.shutdown_flag.set()
        
        # Wait for all preprocessing threads to complete
        for thread in list(self.active_streams):  # Create a copy to avoid modification during iteration
            thread.join(timeout=5.0)
            
        if self.producer:
            self.producer.flush()
            self.producer.close()

class MultiProducerManager:
    """
    Manager class to coordinate multiple producers with hash-based camera assignment.
    """
    def __init__(self, bootstrap_servers='localhost:9092', num_producers=3):
        self.bootstrap_servers = bootstrap_servers
        self.num_producers = num_producers
        self.producers = []
        self.producer_threads = []

    
    def start_all_producers(self, camera_urls):
        """
        Start all producers with their assigned cameras based on hash function.
        """
        logger.info(f"Starting {self.num_producers} producers for {len(camera_urls)} cameras")
        
        # Log camera assignments for debugging
        producer_assignments, cameras_per_producer = VideoStreamProducer.get_even_camera_distribution(camera_urls, self.num_producers)
        
        for i in range(self.num_producers):
            assigned_cameras = producer_assignments[i]
            expected_count = cameras_per_producer[i]
            actual_count = len(assigned_cameras)
            logger.info(f"Producer {i} assigned {actual_count}/{expected_count} cameras: {[cam[0] for cam in assigned_cameras]}")
            
            if actual_count != expected_count:
                logger.warning(f"Producer {i} camera count mismatch: expected {expected_count}, got {actual_count}")

        def run_producer(producer_index):
            producer = VideoStreamProducer(
                bootstrap_servers=self.bootstrap_servers,
                producer_id=f"producer-{producer_index}"
            )
            self.producers.append(producer)
            producer.start_assigned_streams(camera_urls, producer_index, self.num_producers)

        # Start producer threads
        for i in range(self.num_producers):
            thread = threading.Thread(target=run_producer, args=(i,))
            thread.start()
            self.producer_threads.append(thread)

        # Wait for all producer threads
        try:
            for thread in self.producer_threads:
                thread.join()
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            self.cleanup()

    def cleanup(self):
        """
        Clean up all producers.
        """
        logger.info("Shutting down all producers...")
        for producer in self.producers:
            producer.cleanup()

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    try:
        num_cameras = int(config['settings']['num_cameras'])
        camera_urls = [url.strip() for url in config['settings']['camera_urls'].split('\n') 
                      if url.strip()]
        
        # Optional: load number of producers from config
        num_producers = int(config.get('settings', 'num_producers', fallback=3))
        
        return num_cameras, camera_urls, num_producers
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def demonstrate_hash_assignment(camera_urls, num_producers):
    """
    Utility function to demonstrate how cameras are assigned to producers with even distribution.
    """
    producer_assignments, cameras_per_producer = VideoStreamProducer.get_even_camera_distribution(camera_urls, num_producers)
    
    print(f"\nCamera Assignment Distribution (Even Distribution with Hash-based Consistency):")
    print(f"Total cameras: {len(camera_urls)}")
    print(f"Number of producers: {num_producers}")
    print(f"Base cameras per producer: {len(camera_urls) // num_producers}")
    print(f"Extra cameras to distribute: {len(camera_urls) % num_producers}")
    print("-" * 70)
    
    total_assigned = 0
    for producer_idx in range(num_producers):
        assigned_cameras = producer_assignments[producer_idx]
        expected_count = cameras_per_producer[producer_idx]
        actual_count = len(assigned_cameras)
        total_assigned += actual_count
        
        print(f"Producer {producer_idx}: {actual_count} cameras (expected: {expected_count})")
        for cam_idx, cam_url in assigned_cameras:
            # Show truncated URL for readability
            short_url = cam_url[:50] + "..." if len(cam_url) > 50 else cam_url
            print(f"  - Camera {cam_idx}: {short_url}")
        
        if actual_count != expected_count:
            print(f"  ⚠️  WARNING: Count mismatch!")
            
    print("-" * 70)
    print(f"Total cameras assigned: {total_assigned}/{len(camera_urls)}")
    
    # Verify even distribution
    min_cameras = min(len(assignment) for assignment in producer_assignments)
    max_cameras = max(len(assignment) for assignment in producer_assignments)
    distribution_diff = max_cameras - min_cameras
    
    print(f"Distribution quality:")
    print(f"  - Min cameras per producer: {min_cameras}")
    print(f"  - Max cameras per producer: {max_cameras}")
    print(f"  - Distribution difference: {distribution_diff}")
    
    if distribution_diff <= 1:
        print("  ✅ Excellent: Cameras are evenly distributed!")
    else:
        print(f"  ⚠️  Warning: Uneven distribution detected!")
    
    return producer_assignments, cameras_per_producer

if __name__ == '__main__':
    try:
        num_cameras, camera_urls, num_producers = load_config()
        
        # Optional: demonstrate the assignment before starting
        demonstrate_hash_assignment(camera_urls, num_producers)
        
        # Start the multi-producer system
        manager = MultiProducerManager(num_producers=num_producers)
        manager.start_all_producers(camera_urls)
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)