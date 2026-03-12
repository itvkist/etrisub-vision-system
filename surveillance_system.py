import threading
import cv2
import base64
import time
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, render_template, Response
from kafka import KafkaConsumer
from vi2enmodel.translate import translate_en2vi
import redis
import json

from config_manager import Config
from elastic_client import ElasticsearchClient
from frame_processor import FrameProcessor
from camera_manager import CameraManager
from text_processor import TextProcessor

class SurveillanceSystem:
    def __init__(self):
        self.config = Config()
        self.es_client = ElasticsearchClient()
        self.frame_processor = FrameProcessor()
        self.camera_manager = CameraManager()
        self.text_processor = TextProcessor()
        self.app = self._create_app()
        
        # Redis cache configuration
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.cache_key = 'surveillance_cache'
        self.cache_size = 10  # Number of items to cache before processing
        self.cache_timeout = 300  # Cache timeout in seconds
        
        self._start_cache_flush_thread()
        
    def _create_app(self):
        app = Flask(__name__, static_folder='static',
                template_folder='templates')
        self._register_routes(app)
        return app
        
    def _start_cache_flush_thread(self):
        def flush_cache_periodically():
            while True:
                self._flush_cache()
                time.sleep(5)  # Flush every 5 seconds

        thread = threading.Thread(target=flush_cache_periodically, daemon=True)
        thread.start()

    def _flush_cache(self):
        # Get all cached items
        cached_items = self.redis_client.lrange(self.cache_key, 0, -1)
        if not cached_items:
            return
            
        # Process cached items in batch
        docs = []
        captions = []
        
        for item in cached_items:
            doc = json.loads(item.decode('utf-8'))
            docs.append(doc)
            captions.append(doc['filtered_content'])
        
        # if captions:
        #     # Batch translate
        #     translated_captions = translate_en2vi(captions)
            
        #     # Update and index documents
        #     for doc, trans_caption in zip(docs, translated_captions):
        #         doc['filtered_content_vi'] = trans_caption
        #         self.es_client.index_analysis(doc)
        
        # Clear cache
        self.redis_client.delete(self.cache_key)

    def _register_routes(self, app):
        @app.route('/')
        def index():
            return render_template('index.html', num_cameras=self.config.num_cameras)

        @app.route('/camera/<int:camera_id>')
        def video_feed(camera_id):
            topic = f'camera-stream-{camera_id}'
            consumer = KafkaConsumer(topic, bootstrap_servers='localhost:9092')
            return Response(
                self._kafka_stream_handler(consumer, topic, camera_id),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @app.route('/process_frame/<int:camera_id>')
        def process_frame(camera_id):
            frame_info = self.camera_manager.get_frame(camera_id)
            if not frame_info:
                return jsonify({'error': 'No frames available for this camera'}), 404

            try:
                processed_frame, classes, caption_results, image_results = \
                    self.frame_processor.process_detailed_with_classification(frame_info['frame_data'])
                # print(classes)
                encoded_images = []
                for img in image_results:
                    success, jpeg = cv2.imencode('.jpeg', img)
                    if success:
                        encoded_images.append(base64.b64encode(jpeg).decode('utf-8'))

                if encoded_images:
                    for idx, image in enumerate(encoded_images):
                        cls = classes[idx].get('<DETAILED_CAPTION>', '')
                        caption = caption_results[idx].get('<DETAILED_CAPTION>', '')
                        attributes = self.text_processor.preprocess_caption(caption)
                        
                        doc = {
                            'timestamp': frame_info['timestamp'],
                            'class': cls,
                            'image': image, 
                            'filtered_content': caption,
                            'attributes': attributes,
                            'camera_id': camera_id
                        }
                        
                        # Cache document in Redis
                        # self.redis_client.rpush(self.cache_key, json.dumps(doc))
                        self.es_client.index_analysis(doc)
                        # Flush if cache size limit reached
                        # if self.redis_client.llen(self.cache_key) >= self.cache_size:
                        # self._flush_cache()
                
                return jsonify({
                    'timestamp': frame_info['timestamp'],
                    'fps': frame_info['fps'],
                    'captions': classes,
                    'caption_results': caption_results,
                    'image_results': encoded_images
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @app.route('/search')
        def search_page():
            return render_template('search.html')

        @app.route('/search_results')
        def search_results():
            search_term = request.args.get('q', '')
            try:
                results = self.es_client.search(search_term)
                # Results are already formatted from elasticsearch_client
                return jsonify(results)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @app.route('/frames/<int:camera_id>')
        def frames_page(camera_id):
            return render_template('frames.html', camera_id=camera_id)

    def start_consumers(self, num_consumers):
        consumers = [
            KafkaConsumer(f'camera-stream-{i}', 
                         bootstrap_servers='localhost:9092')
            for i in range(self.config.num_cameras)
        ]
        threads = []
        for idx in range(num_consumers):
            consumer = consumers[idx % self.config.num_cameras]
            topic = f'camera-stream-{idx}'
            thread = threading.Thread(
                target=self._kafka_stream_handler,
                args=(consumer, topic, idx)
            )
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    def _kafka_stream_handler(self, consumer, topic, camera_id):
        for message in consumer:
            processed_frame, fps = self.frame_processor.process_stream(
                message.value, camera_id
            )
            self.camera_manager.update_frame(camera_id, message.value, fps)
            
            _, jpeg = cv2.imencode('.jpeg', processed_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   jpeg.tobytes() + b'\r\n\r\n')
    
    def run(self):
        self.start_consumers(num_consumers=5)
        self.app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
