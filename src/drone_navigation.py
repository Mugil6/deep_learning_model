from ultralytics import YOLO
import torch
import cv2
import time
import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


class DroneNavigation:
    def __init__(self, model_path='models/best.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f" Using device: {self.device.upper()}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def process_video(self, video_source, save_output=True):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f" Cannot open video source: {video_source}")
        if save_output:
            os.makedirs('output', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                'output/output.mp4',
                fourcc,
                20.0,
                (int(cap.get(3)), int(cap.get(4)))
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_times = []
        print(f"🎥 Processing {total_frames} frames... Press 'q' to stop early.\n")

        
        for _ in tqdm(range(total_frames), desc="Processing Video"):
            ret, frame = cap.read()
            if not ret:
                print("Video processing completed.")
                break

            start_time = time.time()

            
            results = self.model(frame, verbose=False)
            annotated_frame = results[0].plot()

            
            detected_classes = [self.model.names[int(box.cls)] for box in results[0].boxes]
            control_signal = self.autonomous_decision(detected_classes)
            cv2.putText(
                annotated_frame, f'Action: {control_signal}',
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2
            )

           
            process_time = time.time() - start_time
            frame_times.append(process_time)
            fps = 1 / process_time if process_time > 0 else 0

            
            cv2.putText(
                annotated_frame, f'FPS: {fps:.2f}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            
            cv2.imshow('Drone Navigation', annotated_frame)
            if save_output:
                out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Stopped manually by user.")
                break

        
        cap.release()
        if save_output:
            out.release()
        cv2.destroyAllWindows()

        
        if frame_times:
            return {
                'avg_fps': round(1 / np.mean(frame_times), 2),
                'max_fps': round(1 / min(frame_times), 2),
                'min_fps': round(1 / max(frame_times), 2)
            }
        else:
            return {'avg_fps': 0, 'max_fps': 0, 'min_fps': 0}

    def autonomous_decision(self, detected_classes):
        if 'person' in detected_classes:
            return "Hover - Human nearby"
        elif 'car' in detected_classes or 'truck' in detected_classes:
            return "Avoid - Vehicle detected"
        elif 'bird' in detected_classes:
            return "Ascend - Flying object detected"
        elif 'chair' in detected_classes or 'table' in detected_classes:
            return "Descend slightly - Indoor object"
        else:
            return "Move Forward"

    def calculate_accuracy(self, pred, gt):
        """Dummy accuracy metric (replace with true eval if available)"""
        return np.random.uniform(0.7, 0.99)

    def calculate_speed(self, pred):
        """Dummy FPS-based speed metric"""
        return np.random.uniform(15, 30)

    def get_confidence(self, pred):
        """Average confidence of detections"""
        return float(pred[0].boxes.conf.mean()) if len(pred[0].boxes) else 0

    def evaluate_model(self, test_data):
        """Evaluate detection accuracy, speed, and confidence"""
        results = {'accuracy': [], 'speed': [], 'confidence': []}
        for image, gt in test_data:
            pred = self.model(image)
            results['accuracy'].append(self.calculate_accuracy(pred, gt))
            results['speed'].append(self.calculate_speed(pred))
            results['confidence'].append(self.get_confidence(pred))
        return results

    def create_visualization(self, results):
        """Visualize accuracy and speed"""
        os.makedirs('output', exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(results['accuracy'])
        plt.title('Detection Accuracy Over Time')
        plt.savefig('output/accuracy_plot.png')

        plt.figure(figsize=(10, 5))
        plt.plot(results['speed'])
        plt.title('Processing Speed (FPS)')
        plt.savefig('output/speed_plot.png')
        print("📊 Plots saved in output folder.")


def main():

    drone_nav = DroneNavigation(model_path='models/best.pt')
    video_path = 'input/test_2.mp4'  # replace with your path
    print("Running YOLO-based Drone Navigation...\n")
    metrics = drone_nav.process_video(video_source=video_path, save_output=True)
    print("\n Performance Metrics:", metrics)

    # Dummy evaluation visualization
    dummy_data = [(np.zeros((640, 480, 3)), {}) for _ in range(10)]
    eval_results = drone_nav.evaluate_model(dummy_data)
    drone_nav.create_visualization(eval_results)
    print("Model evaluation completed.")


if __name__ == "__main__":
    main()
