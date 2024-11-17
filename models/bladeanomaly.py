import numpy as np
import pandas as pd
from sklearn import preprocessing
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
import threading
import time
from datetime import datetime

class MachineMonitor:
    def __init__(self, model_path=None):
        """
        Initialize the machine monitoring system
        
        Args:
            model_path (str, optional): Path to saved model. If None, a new model will be created.
        """
        self.columns = [
            'pCut::Motor_Torque',
            'pCut::CTRL_Position_controller::Lag_error',
            'pCut::CTRL_Position_controller::Actual_position',
            'pCut::CTRL_Position_controller::Actual_speed',
            'pSvolFilm::CTRL_Position_controller::Actual_position',
            'pSvolFilm::CTRL_Position_controller::Actual_speed',
            'pSvolFilm::CTRL_Position_controller::Lag_error',
            'pSpintor::VAX_speed'
        ]
        
        # Initialize scaler
        self.scaler = preprocessing.StandardScaler()
        
        # Initialize model
        if model_path:
            try:
                self.model = tf.keras.models.load_model(model_path)
                print(f"Model loaded from {model_path}")
            except:
                print("Could not load model, initializing new one")
                self.model = self._build_model()
        else:
            self.model = self._build_model()
            
        self.is_monitoring = False
        self.anomaly_threshold = 0.1
        self.current_state = {col: None for col in self.columns}
        
        # Add state tracking for Streamlit
        self.is_trained = False
        self.latest_error = 0.0
        self.latest_prediction = None
        self.anomaly_history = []  # Store recent anomalies for display
        
    def _build_model(self):
        """Build the autoencoder model"""
        model = Sequential([
            Dense(50, activation='elu', input_shape=(len(self.columns),)),
            Dense(10, activation='elu'),
            Dense(50, activation='elu'),
            Dense(len(self.columns))
        ])
        model.compile(loss='mse', optimizer='adam')
        return model
    
    def process_sensor_data(self, sensor_data, return_prediction=False):
        """
        Process incoming sensor data and optionally return prediction
        
        Args:
            sensor_data (dict): Dictionary containing sensor readings
            return_prediction (bool): Whether to return prediction results
            
        Returns:
            tuple: (is_anomaly, error) if return_prediction is True
        """
        # Update current state
        for col in self.columns:
            if col in sensor_data:
                self.current_state[col] = sensor_data[col]
        
        if return_prediction and self.is_trained:
            is_anomaly, error = self.predict_anomaly()
            self.latest_error = error
            if is_anomaly:
                self.anomaly_history.append({
                    'timestamp': datetime.now(),
                    'error': error,
                    'state': self.current_state.copy()
                })
                # Keep only last 100 anomalies
                if len(self.anomaly_history) > 100:
                    self.anomaly_history.pop(0)
            return is_anomaly, error
        return None, None
    
    def get_current_state(self):
        """Get the current state of all sensors"""
        return {col: self.current_state[col] if self.current_state[col] is not None else None 
                for col in self.columns}
    
    def predict_anomaly(self):
        """
        Predict if current state is anomalous
        
        Returns:
            tuple: (is_anomaly, reconstruction_error)
        """
        if not self.is_trained:
            return False, 0.0
            
        if any(self.current_state[col] is None for col in self.columns):
            return False, 0.0
        
        current_data = np.array([[self.current_state[col] for col in self.columns]])
        scaled_data = self.scaler.transform(current_data)
        predicted_state = self.model.predict(scaled_data, verbose=0)
        
        reconstruction_error = np.mean(np.abs(scaled_data - predicted_state))
        self.latest_prediction = predicted_state
        
        return reconstruction_error > self.anomaly_threshold, reconstruction_error
    
    def train_model(self, training_data, epochs=10, batch_size=200, validation_split=0.1):
        """
        Train the model on historical data
        
        Args:
            training_data: numpy array or pandas DataFrame with training data
            epochs: number of training epochs
            batch_size: batch size for training
            validation_split: fraction of data to use for validation
            
        Returns:
            history: training history
        """
        if isinstance(training_data, pd.DataFrame):
            training_data = training_data[self.columns].values
            
        # Fit the scaler and transform the data
        self.scaler.fit(training_data)
        scaled_data = self.scaler.transform(training_data)
        
        # Train the model
        history = self.model.fit(
            scaled_data, scaled_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        self.is_trained = True
        return history
    
    def save_model(self, model_path='anomaly_detection_model.keras'):
        """Save the current model to disk"""
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def get_monitoring_status(self):
        """Get current monitoring status for Streamlit display"""
        return {
            'is_monitoring': self.is_monitoring,
            'is_trained': self.is_trained,
            'latest_error': self.latest_error,
            'anomaly_count': len(self.anomaly_history),
            'current_state': self.get_current_state()
        }
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            is_anomaly, error = self.predict_anomaly()
            if is_anomaly:
                self._handle_anomaly(error)
            print(f"Current Error: {error:.4f}")
            time.sleep(1)
    
    def _handle_anomaly(self, error):
        """Handle detected anomalies"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] ANOMALY DETECTED - Error: {error:.4f}")
    
    def start_monitoring(self):
        """Start the monitoring process"""
        if not self.is_monitoring and self.is_trained:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.start()
            print("Monitoring started")
        else:
            print("Cannot start monitoring: Model not trained or already monitoring")
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.is_monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        print("Monitoring stopped")

# Example usage with Streamlit
if __name__ == "__main__":
    # This is the standalone version for testing
    monitor = MachineMonitor()
    
    # Create synthetic training data
    n_samples = 1000
    training_data = np.array([
        [np.random.normal(0.120634, 0.607871) for _ in range(n_samples)],
        [np.random.normal(-0.000055, 0.121212) for _ in range(n_samples)],
        [np.random.normal(3.371415e+08, 5.466868e+08) for _ in range(n_samples)],
        [np.random.normal(1945.794809, 4873.922235) for _ in range(n_samples)],
        [np.random.normal(1.488169e+08, 2.711355e+08) for _ in range(n_samples)],
        [np.random.normal(5367.031778, 3382.193664) for _ in range(n_samples)],
        [np.random.normal(0.984759, 0.343720) for _ in range(n_samples)],
        [np.random.normal(1927.328330, 655.904709) for _ in range(n_samples)]
    ]).T

    # Train the model
    print("Training model with synthetic data...")
    monitor.train_model(training_data)
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        while True:
            sensor_data = {
                'pCut::Motor_Torque': np.random.normal(0.120634, 0.607871),
                'pCut::CTRL_Position_controller::Lag_error': np.random.normal(-0.000055, 0.121212),
                'pCut::CTRL_Position_controller::Actual_position': np.random.normal(3.371415e+08, 5.466868e+08),
                'pCut::CTRL_Position_controller::Actual_speed': np.random.normal(1945.794809, 4873.922235),
                'pSvolFilm::CTRL_Position_controller::Actual_position': np.random.normal(1.488169e+08, 2.711355e+08),
                'pSvolFilm::CTRL_Position_controller::Actual_speed': np.random.normal(5367.031778, 3382.193664),
                'pSvolFilm::CTRL_Position_controller::Lag_error': np.random.normal(0.984759, 0.343720),
                'pSpintor::VAX_speed': np.random.normal(1927.328330, 655.904709)
            }
            
            # Process data and get prediction
            is_anomaly, error = monitor.process_sensor_data(sensor_data, return_prediction=True)
            if is_anomaly:
                print(f"Anomaly detected! Error: {error:.4f}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("\nMonitoring stopped by user")