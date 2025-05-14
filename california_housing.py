import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Configure TensorFlow to use GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"Found {len(physical_devices)} GPU(s)")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print(f" - {device.name} with memory growth enabled")
else:
    print("No GPU found. Using CPU instead.")

# Set TensorFlow logging level to reduce warnings
tf.get_logger().setLevel('ERROR')

class CaliforniaHousingPredictor:
    def __init__(self):
        self.model = None
        self.fuzzy_system = None
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.feature_names = None
        
    def load_dataset(self):
        """Load California Housing dataset"""
        print("Loading California housing dataset...")
        housing = fetch_california_housing()
        X = pd.DataFrame(housing.data, columns=housing.feature_names)
        y = pd.DataFrame(housing.target, columns=['PRICE'])
        
        # Use all available features for better accuracy
        # Original features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
        print(f"Available features: {', '.join(X.columns)}")
        # Keep all features for better accuracy
        # X = X[['MedInc', 'AveRooms', 'AveOccup']]  # Previous limited selection
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        print(f"Dataset loaded with {X.shape[0]} samples and {X.shape[1]} features")
        print(f"Features used: {', '.join(X.columns)}")
        
        return X, y
    
    def prepare_data(self, X, y):
        """Prepare data for model training"""
        print("\nPreparing data for neural network...")
        # Scale the data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """Build and compile neural network for price prediction"""
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(input_shape,)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        
        # Use a learning rate scheduler for better convergence
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        print(f"Model built with {model.count_params()} parameters")
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, epochs=100, batch_size=32):
        """Train the neural network"""
        print("\nBuilding and training neural network...")
        
        # Add early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Add learning rate reduction on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0,
            callbacks=[early_stopping, reduce_lr]
        )
        
        print(f"Model trained for {len(history.history['loss'])} epochs")
        return history
    
    def predict_price(self, features):
        """Predict housing price using the trained model"""
        # Convert to DataFrame with feature names if input is numpy array
        if isinstance(features, np.ndarray) and self.feature_names is not None:
            if features.ndim == 1:
                features = features.reshape(1, -1)
            features = pd.DataFrame(features, columns=self.feature_names)
            
        # Scale the input
        scaled_features = self.scaler_X.transform(features)
        # Predict
        prediction = self.model.predict(scaled_features, verbose=0)
        # Inverse transform to get actual price
        prediction = self.scaler_y.inverse_transform(prediction)
        
        return prediction
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        # Get original feature values
        X_original = self.scaler_X.inverse_transform(X_test)
        y_original = self.scaler_y.inverse_transform(y_test)
        
        # Predict prices
        y_pred = self.model.predict(X_test)
        
        # Calculate error metrics
        mse = np.mean((y_pred - y_test) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate R-squared (coefficient of determination) as accuracy measure
        y_mean = np.mean(y_test)
        ss_total = np.sum((y_test - y_mean) ** 2)
        ss_residual = np.sum((y_test - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        
        print(f"\nModel Evaluation:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Accuracy (R-squared): {r_squared:.4f} or {r_squared*100:.2f}%")
        
        # Plot actual vs predicted prices for a sample
        sample_size = min(50, len(X_test))
        indices = np.random.choice(range(len(X_test)), sample_size, replace=False)
        
        actual_prices = self.scaler_y.inverse_transform(y_test[indices])
        predicted_prices = self.scaler_y.inverse_transform(y_pred[indices])
        
        plt.figure(figsize=(10, 6))
        plt.scatter(actual_prices, predicted_prices)
        plt.plot([actual_prices.min(), actual_prices.max()], [actual_prices.min(), actual_prices.max()], 'k--')
        plt.xlabel('Actual Prices ($100k)')
        plt.ylabel('Predicted Prices ($100k)')
        plt.title('Actual vs Predicted Housing Prices')
        plt.tight_layout()
        plt.savefig('california_housing_prediction.png')
        # plt.show()  # Commented out to avoid display issues
        
        return mse, rmse, r_squared

    def build_fuzzy_system(self):
        """Build fuzzy logic system for housing recommendation"""
        print("\nBuilding fuzzy logic system...")
        try:
            # Define fuzzy variables with explicit universe ranges
            price = ctrl.Antecedent(np.arange(0, 51, 1), 'price')
            rooms = ctrl.Antecedent(np.arange(3, 9, 0.1), 'rooms')
            location_quality = ctrl.Antecedent(np.arange(0, 101, 1), 'location_quality')
            recommendation = ctrl.Consequent(np.arange(0, 101, 1), 'recommendation')
            
            # Define membership functions for price
            price['low'] = fuzz.trimf(price.universe, [0, 0, 20])
            price['medium'] = fuzz.trimf(price.universe, [15, 25, 35])
            price['high'] = fuzz.trimf(price.universe, [30, 50, 50])
            
            # Define membership functions for rooms
            rooms['small'] = fuzz.trimf(rooms.universe, [3, 3, 5])
            rooms['medium'] = fuzz.trimf(rooms.universe, [4.5, 6, 7.5])
            rooms['large'] = fuzz.trimf(rooms.universe, [7, 9, 9])
            
            # Define membership functions for location quality
            location_quality['poor'] = fuzz.trimf(location_quality.universe, [0, 0, 40])
            location_quality['average'] = fuzz.trimf(location_quality.universe, [30, 50, 70])
            location_quality['excellent'] = fuzz.trimf(location_quality.universe, [60, 100, 100])
            
            # Define membership functions for recommendation
            recommendation['not_recommended'] = fuzz.trimf(recommendation.universe, [0, 0, 40])
            recommendation['consider'] = fuzz.trimf(recommendation.universe, [30, 50, 70])
            recommendation['highly_recommended'] = fuzz.trimf(recommendation.universe, [60, 100, 100])
            
            # Define fuzzy rules
            rule1 = ctrl.Rule(price['low'] & rooms['large'] & location_quality['excellent'], recommendation['highly_recommended'])
            rule2 = ctrl.Rule(price['low'] & rooms['medium'] & location_quality['excellent'], recommendation['highly_recommended'])
            rule3 = ctrl.Rule(price['low'] & rooms['small'] & location_quality['excellent'], recommendation['consider'])
            
            rule4 = ctrl.Rule(price['medium'] & rooms['large'] & location_quality['excellent'], recommendation['highly_recommended'])
            rule5 = ctrl.Rule(price['medium'] & rooms['medium'] & location_quality['average'], recommendation['consider'])
            rule6 = ctrl.Rule(price['medium'] & rooms['small'] & location_quality['poor'], recommendation['not_recommended'])
            
            rule7 = ctrl.Rule(price['high'] & rooms['large'] & location_quality['excellent'], recommendation['consider'])
            rule8 = ctrl.Rule(price['high'] & rooms['medium'] & location_quality['average'], recommendation['not_recommended'])
            rule9 = ctrl.Rule(price['high'] & rooms['small'] & location_quality['poor'], recommendation['not_recommended'])

            # Additional rules for better coverage
            rule10 = ctrl.Rule(price['medium'] & rooms['small'] & location_quality['excellent'], recommendation['consider'])
            rule11 = ctrl.Rule(price['medium'] & rooms['large'] & location_quality['average'], recommendation['consider'])
            rule12 = ctrl.Rule(price['low'] & rooms['medium'] & location_quality['average'], recommendation['highly_recommended'])
            
            # Create control system
            control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12])
            self.fuzzy_system = ctrl.ControlSystemSimulation(control_system)
            
            # Test the system to ensure it works
            self.fuzzy_system.input['price'] = 25.0
            self.fuzzy_system.input['rooms'] = 6.0
            self.fuzzy_system.input['location_quality'] = 50.0
            self.fuzzy_system.compute()
            test_output = self.fuzzy_system.output['recommendation']
            print(f"Fuzzy system initialized successfully. Test output: {test_output}")
            
            # System is ready for actual use after testing
            return self.fuzzy_system
        except Exception as e:
            print(f"Error building fuzzy system: {e}")
            # Create a simplified fallback system if the main one fails
            try:
                # Create a simplified system with fewer rules
                price = ctrl.Antecedent(np.arange(0, 51, 1), 'price')
                rooms = ctrl.Antecedent(np.arange(3, 9, 0.1), 'rooms')
                location_quality = ctrl.Antecedent(np.arange(0, 101, 1), 'location_quality')
                recommendation = ctrl.Consequent(np.arange(0, 101, 1), 'recommendation')
                
                # Simplified membership functions
                price['low'] = fuzz.trimf(price.universe, [0, 0, 25])
                price['high'] = fuzz.trimf(price.universe, [25, 50, 50])
                
                rooms['small'] = fuzz.trimf(rooms.universe, [3, 3, 6])
                rooms['large'] = fuzz.trimf(rooms.universe, [6, 9, 9])
                
                location_quality['poor'] = fuzz.trimf(location_quality.universe, [0, 0, 50])
                location_quality['good'] = fuzz.trimf(location_quality.universe, [50, 100, 100])
                
                recommendation['not_recommended'] = fuzz.trimf(recommendation.universe, [0, 0, 50])
                recommendation['recommended'] = fuzz.trimf(recommendation.universe, [50, 100, 100])
                
                # Test the fallback system to ensure it works
                print("Testing fallback fuzzy system...")
                
                # Simplified rules
                rule1 = ctrl.Rule(price['low'] & rooms['large'] & location_quality['good'], recommendation['recommended'])
                rule2 = ctrl.Rule(price['high'] & rooms['small'] & location_quality['poor'], recommendation['not_recommended'])
                
                # Create control system
                control_system = ctrl.ControlSystem([rule1, rule2])
                self.fuzzy_system = ctrl.ControlSystemSimulation(control_system)
                
                # Test the fallback system
                self.fuzzy_system.input['price'] = 25.0
                self.fuzzy_system.input['rooms'] = 6.0
                self.fuzzy_system.input['location_quality'] = 50.0
                self.fuzzy_system.compute()
                test_output = self.fuzzy_system.output['recommendation']
                print(f"Fallback fuzzy system created successfully. Test output: {test_output}")
                
                # Fallback system is ready for actual use after testing
                return self.fuzzy_system
            except Exception as e2:
                print(f"Failed to create fallback fuzzy system: {e2}")
                return None
    
    def get_fuzzy_recommendation(self, price, rooms, location_quality):
        """Get recommendation from fuzzy system"""
        # Ensure inputs are within valid ranges for the fuzzy system
        price = max(0, min(price, 50))  # Clamp between 0 and 50
        rooms = max(3, min(rooms, 8.9))  # Clamp between 3 and 8.9
        location_quality = max(0, min(location_quality, 100))  # Clamp between 0 and 100
        
        try:
            self.fuzzy_system.input['price'] = float(price)
            self.fuzzy_system.input['rooms'] = float(rooms)
            self.fuzzy_system.input['location_quality'] = float(location_quality)
            
            # print(f"DEBUG: Fuzzy inputs before compute: price={price}, rooms={rooms}, location_quality={location_quality}")
            self.fuzzy_system.compute()
            # print(f"DEBUG: Fuzzy output dict after compute: {self.fuzzy_system.output}")
            # For more detailed debugging, if needed later, uncomment:
            # self.fuzzy_system.print_state()
            # Use .get() to provide a default value if 'recommendation' is not in the output
            recommendation_value = self.fuzzy_system.output.get('recommendation', 50) 
            # if recommendation_value == 50 and 'recommendation' not in self.fuzzy_system.output:
                # print("DEBUG: 'recommendation' key not found in fuzzy_system.output. Defaulting to 50.")
            return recommendation_value
        except Exception as e:
            # Handle potential errors in fuzzy computation with more detailed error message
            print(f"Error in fuzzy computation: {e}. Using default value.")
            return 50
    
    def evaluate_properties(self, X_test, y_test, num_samples=10):
        """Evaluate properties using both neural network and fuzzy system"""
        # Get original feature values
        X_original = self.scaler_X.inverse_transform(X_test)
        y_original = self.scaler_y.inverse_transform(y_test)
        
        # Select a subset of samples
        indices = np.random.choice(range(len(X_test)), num_samples, replace=False)
        
        results = []
        for i in indices:
            # Get features and actual price
            features = X_original[i]
            actual_price = y_original[i][0]
            
            # Predict price using neural network
            predicted_price = self.predict_price(features.reshape(1, -1))[0][0]
            
            # Get fuzzy recommendation - use feature indices based on all features
            # For 8 features: MedInc(0), HouseAge(1), AveRooms(2), AveBedrms(3), Population(4), AveOccup(5), Latitude(6), Longitude(7)
            rooms = features[2] if len(features) > 2 else 5  # AveRooms or default
            location_quality = min(features[0] * 20, 100)  # MedInc (scaled)
            
            # Ensure rooms is within valid range for fuzzy system
            rooms = max(3, min(rooms, 8.9))  # Clamp between 3 and 8.9
            
            # Scale price to 0-50 range for fuzzy system
            # Original prices are in $100k (e.g., 0.5 to 5.0)
            # Fuzzy price universe is 0-50. Multiply by 10.
            scaled_price = min(predicted_price * 10, 50)
            
            # Use a default recommendation value
            recommendation = 50
            
            # Only try to get fuzzy recommendation if fuzzy system exists
            if self.fuzzy_system is not None:
                try:
                    # Don't try to check input variables - just use the get_fuzzy_recommendation method
                    # which already has proper error handling
                    recommendation = self.get_fuzzy_recommendation(scaled_price, rooms, location_quality)
                except Exception as e:
                    print(f"Fuzzy error in evaluate_properties: {e}. Using default value.")
            
            results.append({
                'features': features,
                'actual_price': actual_price,
                'predicted_price': predicted_price,
                'recommendation': recommendation
            })
        
        return results
    
    def visualize_fuzzy_membership(self):
        """Visualize fuzzy membership functions"""
        # print("\nVisualizing fuzzy membership functions...")
        # Create the fuzzy variables again for visualization
        price = ctrl.Antecedent(np.arange(0, 51, 1), 'price')
        rooms = ctrl.Antecedent(np.arange(3, 9, 0.1), 'rooms')
        location_quality = ctrl.Antecedent(np.arange(0, 101, 1), 'location_quality')
        recommendation = ctrl.Consequent(np.arange(0, 101, 1), 'recommendation')
        
        # Define membership functions for price
        price['low'] = fuzz.trimf(price.universe, [0, 0, 20])
        price['medium'] = fuzz.trimf(price.universe, [15, 25, 35])
        price['high'] = fuzz.trimf(price.universe, [30, 50, 50])
        
        # Define membership functions for rooms
        rooms['small'] = fuzz.trimf(rooms.universe, [3, 3, 5])
        rooms['medium'] = fuzz.trimf(rooms.universe, [4.5, 6, 7.5])
        rooms['large'] = fuzz.trimf(rooms.universe, [7, 9, 9])
        
        # Define membership functions for location quality
        location_quality['poor'] = fuzz.trimf(location_quality.universe, [0, 0, 40])
        location_quality['average'] = fuzz.trimf(location_quality.universe, [30, 50, 70])
        location_quality['excellent'] = fuzz.trimf(location_quality.universe, [60, 100, 100])
        
        # Define membership functions for recommendation
        recommendation['not_recommended'] = fuzz.trimf(recommendation.universe, [0, 0, 40])
        recommendation['consider'] = fuzz.trimf(recommendation.universe, [30, 50, 70])
        recommendation['highly_recommended'] = fuzz.trimf(recommendation.universe, [60, 100, 100])
        
        # Visualize
        plt.figure(figsize=(15, 10))

        plt.subplot(4, 1, 4)
        recommendation.view()
        plt.title('Recommendation Membership')
        
        plt.tight_layout()
        plt.savefig('california_fuzzy_membership.png')
        # plt.show()  # Commented out to avoid display issues
    
    def visualize_results(self, results, feature_names):
        """Visualize evaluation results"""
        # print("\nVisualizing evaluation results...")
        # Extract data for visualization
        actual_prices = [r['actual_price'] for r in results]
        predicted_prices = [r['predicted_price'] for r in results]
        recommendations = [r['recommendation'] for r in results]
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot price comparison
        plt.subplot(2, 1, 1)
        x = range(len(results))
        width = 0.35
        plt.bar([i - width/2 for i in x], actual_prices, width, label='Actual Price')
        plt.bar([i + width/2 for i in x], predicted_prices, width, label='Predicted Price')
        plt.xlabel('Property Index')
        plt.ylabel('Price ($100k)')
        plt.title('Neural Network Price Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot recommendations
        plt.subplot(2, 1, 2)
        colors = ['red' if r < 40 else 'yellow' if r < 70 else 'green' for r in recommendations]
        plt.bar(x, recommendations, color=colors)
        plt.xlabel('Property Index')
        plt.ylabel('Recommendation Score')
        plt.title('Fuzzy Logic Recommendation')
        plt.axhline(y=40, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=70, color='g', linestyle='--', alpha=0.5)
        plt.text(len(results)-1, 20, 'Not Recommended', ha='right')
        plt.text(len(results)-1, 55, 'Consider', ha='right')
        plt.text(len(results)-1, 85, 'Highly Recommended', ha='right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('california_housing_evaluation.png')
        # plt.show()  # Commented out to avoid display issues

# Main execution
def main():
    # Create the predictor
    predictor = CaliforniaHousingPredictor()
    
    # Load dataset
    X, y = predictor.load_dataset()
    
    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(X, y)
    
    # Build and train model
    predictor.build_model(X_train.shape[1])
    history = predictor.train_model(X_train, y_train, epochs=100)
    
    # Evaluate model performance
    mse, rmse, accuracy = predictor.evaluate_model(X_test, y_test)
    print("\n" + "="*50)
    print(f"MODEL ACCURACY: {accuracy*100:.2f}%")
    print("="*50 + "\n")
    
    # Build fuzzy system
    predictor.build_fuzzy_system()
    
    # Visualize fuzzy membership functions
    predictor.visualize_fuzzy_membership()
    
    # Evaluate properties using both neural network and fuzzy logic
    results = predictor.evaluate_properties(X_test, y_test, num_samples=10)
    
    # Visualize results
    predictor.visualize_results(results, X.columns)
    
    print("\nCalifornia Housing Price Prediction with Fuzzy Logic completed!")

if __name__ == "__main__":
    main()