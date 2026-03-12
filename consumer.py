import sys
import types
import cv2
import numpy as np
import configparser
import threading
import json
import base64
import urllib3
import os
import re
from datetime import datetime
from collections import deque, defaultdict
from flask import Flask, Flask, render_template_string, request, jsonify,render_template,Response
from kafka import KafkaConsumer
from elasticsearch import Elasticsearch
from human_att.code.florence2Class import Florence2Model
from attributes import attributes
from vehicles.carMMTClass import CarMMTRecognizer

from concurrent.futures import ThreadPoolExecutor

# Mock kafka.vendor.six.moves module
m = types.ModuleType('kafka.vendor.six.moves', 'Mock module')
setattr(m, 'range', range)
sys.modules['kafka.vendor.six.moves'] = m

class Config:
    def __init__(self, config_path='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
    @property
    def num_cameras(self):
        return int(self.config['settings']['num_cameras'])

class ElasticsearchClient:
    def __init__(self):
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.es = Elasticsearch(
            # hosts=["https://localhost:9200"],
            # basic_auth=['elastic', 'haK9PfmRkM*BRv+K5DX3'],
            # verify_certs=False
            hosts=["http://localhost:9200"]
        )
    
    def index_analysis(self, doc):
        return self.es.index(index='surveillance_analysis', body=doc)
    
    def search(self, search_term):
        query = {
            "query": {
                "multi_match": {
                    "query": search_term,
                    "fields": ["filtered_content", "attributes.*.*.keyword"]
                }
            }
        }
        return self.es.search(index="surveillance_analysis", body=query)
    
    def delete_index(self,doc):
        self.es.indices.delete(index=doc,ignore=[400,404])
        
    def create_index(self,doc):
        self.es.indices.create(
    index='surveillance_analysis',
    body={
        'mappings': {
            'properties': {
                'timestamp': {'type': 'date'},
                'image': {'type': 'text'},
                'filtered_content': {'type': 'text'},
                'attributes': {'type': 'object'},
                'camera_id': {'type': 'integer'}
            }
        }
    }
)
class FrameProcessor:
    def __init__(self):
        self.humanModel = Florence2Model()
        self.carModel = CarMMTRecognizer()
        self.last_frame_time = {}
        self.frame_counters = {}

    def process_stream_raw(self, frame_data, camera_id):
        """Process frames for live streaming"""
        frame_rgb = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Calculate FPS
        current_time = datetime.now().timestamp()
        if camera_id not in self.last_frame_time:
            self.last_frame_time[camera_id] = current_time
            fps = 0.0
        else:
            time_diff = current_time - self.last_frame_time[camera_id]
            fps = 2 / time_diff if time_diff > 0 else 0.0
        self.last_frame_time[camera_id] = current_time
        
        return frame_rgb, fps
    
    def process_stream_raw_1(self, frame_data, camera_id):
        """Process frames for live streaming"""
        frame_rgb = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Calculate FPS
        current_time = datetime.now().timestamp()
        if camera_id not in self.last_frame_time:
            self.last_frame_time[camera_id] = current_time
            fps = 0.0
        else:
            time_diff = current_time - self.last_frame_time[camera_id]
            fps = 2 / time_diff if time_diff > 0 else 0.0
        self.last_frame_time[camera_id] = current_time
        
        # Process frame
        processed_frame, _ = self.humanModel.process_detect(frame_rgb)
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return processed_frame, fps
    
    def process_stream_raw_2(self, frame_data, camera_id):
        """Process frames for live streaming"""
        frame_rgb = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Calculate FPS
        current_time = datetime.now().timestamp()
        if camera_id not in self.last_frame_time:
            self.last_frame_time[camera_id] = current_time
            fps = 0.0
        else:
            time_diff = current_time - self.last_frame_time[camera_id]
            fps = 2 / time_diff if time_diff > 0 else 0.0
        self.last_frame_time[camera_id] = current_time
        
        # Process frame
        listBox_vehicle = self.carModel.process_detect(frame_rgb)
        processed_frame = self.carModel.detector.drawBoxes(frame_rgb, listBox_vehicle)
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return processed_frame, fps
        
    def process_stream(self, frame_data, camera_id):
        """Process frames for live streaming"""
        frame_rgb = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Calculate FPS
        current_time = datetime.now().timestamp()
        if camera_id not in self.last_frame_time:
            self.last_frame_time[camera_id] = current_time
            fps = 0.0
        else:
            time_diff = current_time - self.last_frame_time[camera_id]
            fps = 2 / time_diff if time_diff > 0 else 0.0
        self.last_frame_time[camera_id] = current_time
        
        # Process frame
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_human = executor.submit(self.humanModel.process_detect, frame_rgb)
            future_vehicle = executor.submit(self.carModel.process_detect_with_attributes, frame_rgb)

            processed_frame, _ = future_human.result()
            listBox_vehicle = future_vehicle.result()

        # 2 YOLO models on a same GPU -> GPU context switching. OOM!
        # Another way: 2 YOLO on 2 GPU, no CUDA context switching!
        # Also considering batch processing, examples from batchFlorence2.py
        #rocessed_human_frame, _ = self.humanModel.process_detect(frame_rgb)
        #listBox_vehicle = self.carModel.process_detect(frame_rgb)
        if len(listBox_vehicle) > 0:
            #processed_frame = self.carModel.draw_box_with_att(processed_frame, listBox_vehicle)
            processed_frame = self.carModel.draw_box_with_att(processed_frame, listBox_vehicle)
        
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return processed_frame, fps
    
    def process_stream_squence_stable_fps(self, frame_data, camera_id):
        """Process frames for live streaming"""
        target_fps = 25.0
        frame_interval = 1.0 / target_fps
        current_time = datetime.now().timestamp()

        if camera_id not in self.last_frame_time:
            self.last_frame_time[camera_id] = current_time
            fps = 0.0
            return None, fps
        else:
            time_diff = current_time - self.last_frame_time[camera_id]
            if time_diff < frame_interval:
                return None, None  # Skip to maintain target FPS
            fps = 2 / time_diff if time_diff > 0 else 0.0
            self.last_frame_time[camera_id] = current_time

            frame_rgb = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        
            # Process frame
            processed_frame, _ = self.humanModel.process_detect(frame_rgb)
            #listBox_vehicle = self.carModel.process_detect(frame_rgb)
            listBox_vehicle = self.carModel.process_detect_with_attributes(frame_rgb)
        
            if len(listBox_vehicle) > 0:
                processed_frame = self.carModel.draw_box_with_att(processed_frame, listBox_vehicle)
                #processed_frame = self.carModel.detector.drawBoxes(processed_frame, listBox_vehicle)
        
            cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
            return processed_frame, fps
    
    def process_stream_squence(self, frame_data, camera_id):
        """Process frames for live streaming"""
        frame_rgb = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Calculate FPS
        current_time = datetime.now().timestamp()
        if camera_id not in self.last_frame_time:
            self.last_frame_time[camera_id] = current_time
            fps = 0.0
        else:
            time_diff = current_time - self.last_frame_time[camera_id]
            fps = 2 / time_diff if time_diff > 0 else 0.0
        self.last_frame_time[camera_id] = current_time
        
        # Process frame
        processed_frame, _ = self.humanModel.process_detect(frame_rgb)
        #listBox_vehicle = self.carModel.process_detect(frame_rgb)
        listBox_vehicle = self.carModel.process_detect_with_attributes(frame_rgb)
        
        if len(listBox_vehicle) > 0:
            processed_frame = self.carModel.draw_box_with_att(processed_frame, listBox_vehicle)
            #processed_frame = self.carModel.detector.drawBoxes(processed_frame, listBox_vehicle)
        
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return processed_frame, fps
    
    def process_detailed_human(self, frame_data):
        """Process frame with detailed analysis about human"""
        frame_rgb = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        return self.humanModel.process_frame_tasks(frame_rgb)
    
    def process_detailed_vehicles(self, frame_data):
        """Process frame with detailed analysis about vehicles"""
        frame_rgb = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        return self.carModel.process_frame_tasks(frame_rgb)

class CameraManager:
    def __init__(self):
        self.latest_frames = {}
        self.frame_locks = defaultdict(threading.Lock)
        self.message_counter = defaultdict(int)
        
    def update_frame(self, camera_id, frame_data, fps):
        if camera_id not in self.frame_locks:
            self.frame_locks[camera_id] = threading.Lock()
            
        with self.frame_locks[camera_id]:
            self.latest_frames[camera_id] = {
                'frame_data': frame_data,
                'timestamp': datetime.now().isoformat(),
                'fps': fps
            }
    
    def get_frame(self, camera_id):
        if camera_id not in self.latest_frames:
            return None
            
        with self.frame_locks[camera_id]:
            return self.latest_frames[camera_id].copy()

class TextProcessor:
    @staticmethod
    def preprocess_caption(caption):
        found_attributes = {category: [] for category in attributes.keys()}
        
        for category, attrs in attributes.items():
            for attr in attrs:
                type_match = re.search(rf'\b{attr["type"]}\b', caption, re.IGNORECASE)
                if type_match:
                    color_match = re.search(rf'\b(?P<color>\w+)\s+{attr["type"]}\b', 
                                          caption, re.IGNORECASE)
                    color = color_match.group("color") if color_match else None
                    found_attributes[category].append({
                        "type": attr["type"], 
                        "color": color
                    })
                    
        return found_attributes

class SurveillanceSystem:
    def __init__(self):
        self.config = Config()
        self.es_client = ElasticsearchClient()
        # self.es_client.delete_index("surveillance_analysis")
        self.frame_processor = FrameProcessor()
        self.camera_manager = CameraManager()
        self.text_processor = TextProcessor()
        self.app = self._create_app()
        
    def _create_app(self):
        app = Flask(__name__,static_folder='static',
                template_folder='templates')
        self._register_routes(app)
        return app
    def _register_routes(self, app):
        @app.route('/')
        def index():
            return render_template('index.html', num_cameras=self.config.num_cameras)
        @app.route('/camera/<int:camera_id>')
        def video_feed(camera_id):
            topic = f'camera-stream-{camera_id}'
            consumer = KafkaConsumer(topic, bootstrap_servers='localhost:9092')
            return Response(
                #self._kafka_stream_handler(consumer, topic, camera_id),
                self._kafka_stream_handler_stable_fps(consumer, topic, camera_id),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @app.route('/process_frame/<int:camera_id>/human')
        def process_human_frame(camera_id):
            frame_info = self.camera_manager.get_frame(camera_id)
            if not frame_info:
                return jsonify({'error': 'No frames available for this camera'}), 404

            try:
                # Process frame with detailed analysis
                processed_frame, captions, caption_results, image_results = \
                    self.frame_processor.process_detailed_human(frame_info['frame_data'])

                # Encode processed images
                encoded_images = []
                for img in image_results:
                    success, jpeg = cv2.imencode('.jpeg', img)
                    if success:
                        encoded_images.append(base64.b64encode(jpeg).decode('utf-8'))

                if encoded_images:
                    # Process and store results for each image
                    for idx, image in enumerate(encoded_images):
                        caption = caption_results[idx].get('<DETAILED_CAPTION>', '')
                        
                        # Process the caption
                        attributes = self.text_processor.preprocess_caption(caption)
                        
                        # Prepare and store document
                        doc = {
                            'timestamp': frame_info['timestamp'],
                            'image': image,
                            'filtered_content': caption,
                            'attributes': attributes,
                            'camera_id': camera_id
                        }
                        #elf.es_client.index_analysis(doc)

                    return jsonify({
                        'timestamp': frame_info['timestamp'],
                        'fps': frame_info['fps'],
                        'captions': captions,
                        'caption_results': caption_results,
                        'image_results': encoded_images
                    })
                else:
                    return jsonify({
                        'timestamp': frame_info['timestamp'],
                        'fps': frame_info['fps'],
                        'image_results': []
                    })

            except Exception as e:
                return jsonify({'error': str(e)}), 500
            
        @app.route('/process_frame/<int:camera_id>/vehicle')
        def process_vehicle_frame(camera_id):
            frame_info = self.camera_manager.get_frame(camera_id)
            if not frame_info:
                return jsonify({'error': 'No frames available for this camera'}), 404
            
            try:
                crops, attributes = self.frame_processor.process_detailed_vehicles(frame_info['frame_data'])
                
                encoded_images = []
                for img in crops:
                    success, jpeg = cv2.imencode('.jpeg', img)
                    if success:
                        encoded_images.append(base64.b64encode(jpeg).decode('utf-8'))
                
                if encoded_images:
                    for idx, image in enumerate(encoded_images):
                        indiv_att = attributes[idx]
                        if len(indiv_att) < 3:
                            doc = {
                            'timestamp': frame_info['timestamp'],
                            'image': image,
                            'type': indiv_att[0],
                            'camera_id': camera_id
                            }
                        else:
                            doc = {
                            'timestamp': frame_info['timestamp'],
                            'image': image,
                            'type': indiv_att[0],
                            'make': indiv_att[1],
                            'model': indiv_att[2],
                            'camera_id': camera_id
                            }
                        
                        #self.es_client.index_analysis(doc)

                    return jsonify({
                        'timestamp': frame_info['timestamp'],
                        'fps': frame_info['fps'],
                        'attributes_everycar': attributes, 
                        'image_results': encoded_images
                    })
                else:
                    return jsonify({
                        'timestamp': frame_info['timestamp'],
                        'fps': frame_info['fps'],
                        'image_results': []
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
                formatted_results = []
                
                for hit in results['hits']['hits']:
                    source = hit['_source']
                    formatted_results.append({
                        'timestamp': source.get('timestamp'),
                        'image': source.get('image'),
                        'filtered_content': source.get('filtered_content'),
                        'attributes': source.get('attributes', {})
                    })
                
                return jsonify(formatted_results)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @app.route('/frames/<int:camera_id>/humans')
        def frames_human_page(camera_id):
            return render_template('frames_human.html', camera_id=camera_id)
        
        @app.route('/frames/<int:camera_id>/vehicles')
        def frames_vehicles_page(camera_id):
            return render_template('frames_vehicle.html', camera_id=camera_id)
            
    
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
                #target=self._kafka_stream_handler,
                target=self._kafka_stream_handler_stable_fps,
                args=(consumer, topic, idx)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
    
    def _kafka_stream_handler(self, consumer, topic, camera_id):
        for message in consumer:
            processed_frame, fps = self.frame_processor.process_stream_squence(
                message.value, camera_id
            )
            self.camera_manager.update_frame(camera_id, message.value, fps)
            
            # resize processed_frame.
            h, w = processed_frame.shape[:2]
            if w > 320:
                scale = 320 / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                processed_frame = cv2.resize(processed_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            _, jpeg = cv2.imencode('.jpeg', processed_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   jpeg.tobytes() + b'\r\n\r\n')
            
    def _kafka_stream_handler_stable_fps(self, consumer, topic, camera_id):
        for message in consumer:
            processed_frame, fps = self.frame_processor.process_stream_squence_stable_fps(
                message.value, camera_id
            )
            self.camera_manager.update_frame(camera_id, message.value, fps)
            
            if processed_frame is None:
                continue  # Skip frame to maintain stable FPS
            # resize processed_frame.
            h, w = processed_frame.shape[:2]
            if w > 320:
                scale = 320 / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                processed_frame = cv2.resize(processed_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            _, jpeg = cv2.imencode('.jpeg', processed_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   jpeg.tobytes() + b'\r\n\r\n')
    
    def run(self):
        self.start_consumers(num_consumers=5)
        self.app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    surveillance_system = SurveillanceSystem()
    surveillance_system.run()