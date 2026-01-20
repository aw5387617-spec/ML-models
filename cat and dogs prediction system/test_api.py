"""
Test client for the Cat vs Dog Classification API

This script demonstrates how to use the API endpoints.
"""

import requests
import base64
import json

# API configuration
API_BASE_URL = "http://localhost:5000"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"


def test_health_check():
    """Test the health check endpoint."""
    try:
        response = requests.get(HEALTH_ENDPOINT)
        if response.status_code == 200:
            data = response.json()
            print("✓ Health check successful")
            print(f"  Status: {data['status']}")
            print(f"  Model loaded: {data['model_loaded']}")
            return True
        else:
            print(f"✗ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


def test_prediction_with_file(image_path):
    """Test prediction using file upload."""
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(PREDICT_ENDPOINT, files=files)

        if response.status_code == 200:
            data = response.json()
            if data['success']:
                prediction = data['prediction']
                print("✓ File upload prediction successful")
                print(f"  Predicted class: {prediction['predicted_class']}")
                print(f"  Confidence: {prediction['confidence']:.4f}")
                print(
                    f"  Cat probability: {prediction['probabilities']['cat']:.4f}")
                print(
                    f"  Dog probability: {prediction['probabilities']['dog']:.4f}")
                return True
            else:
                print(
                    f"✗ Prediction failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(
                f"✗ Prediction request failed with status {response.status_code}")
            print(f"  Error: {response.text}")
            return False
    except Exception as e:
        print(f"✗ File upload prediction error: {e}")
        return False


def test_prediction_with_base64(image_path):
    """Test prediction using base64 encoded image."""
    try:
        # Read and encode the image as base64
        with open(image_path, 'rb') as f:
            image_data = f.read()
            base64_encoded = base64.b64encode(image_data).decode('utf-8')

        # Send as JSON payload
        payload = {'image': base64_encoded}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            PREDICT_ENDPOINT, data=json.dumps(payload), headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data['success']:
                prediction = data['prediction']
                print("✓ Base64 prediction successful")
                print(f"  Predicted class: {prediction['predicted_class']}")
                print(f"  Confidence: {prediction['confidence']:.4f}")
                print(
                    f"  Cat probability: {prediction['probabilities']['cat']:.4f}")
                print(
                    f"  Dog probability: {prediction['probabilities']['dog']:.4f}")
                return True
            else:
                print(
                    f"✗ Prediction failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(
                f"✗ Prediction request failed with status {response.status_code}")
            print(f"  Error: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Base64 prediction error: {e}")
        return False


def test_invalid_requests():
    """Test various invalid request scenarios."""
    print("\nTesting invalid requests:")

    # Test with no image
    try:
        response = requests.post(PREDICT_ENDPOINT, json={})
        print(f"  No image test: Status {response.status_code}")
        if response.status_code == 400:
            print("  ✓ Correctly rejected request with no image")
        else:
            print("  ✗ Should have rejected request with no image")
    except Exception as e:
        print(f"  Error testing no image: {e}")

    # Test with invalid file type
    try:
        # Create a temporary text file to test invalid upload
        with open('temp_test.txt', 'w') as f:
            f.write('This is not an image')

        with open('temp_test.txt', 'rb') as f:
            files = {'image': f}
            response = requests.post(PREDICT_ENDPOINT, files=files)

        print(f"  Invalid file test: Status {response.status_code}")
        if response.status_code != 200:
            print("  ✓ Correctly rejected invalid file type")
        else:
            print("  ? Accepted request (might be expected if PIL can handle it)")

        # Clean up
        import os
        os.remove('temp_test.txt')
    except Exception as e:
        print(f"  Error testing invalid file: {e}")


def main():
    print("Cat vs Dog Classification API Test Client")
    print("=" * 50)

    # Test health endpoint
    print("\n1. Testing health endpoint:")
    health_ok = test_health_check()

    if not health_ok:
        print("\nAPI is not healthy. Please check that the API server is running and the model is loaded.")
        return

    # Find a test image from the dataset
    import os
    import glob

    # Look for test images in the dataset
    test_images = glob.glob("test_set/cats/*.jpg") + \
        glob.glob("test_set/dogs/*.jpg")

    if not test_images:
        print("\nNo test images found in the dataset directories.")
        print("Please ensure you have images in test_set/cats/ and test_set/dogs/ directories.")
        return

    test_image_path = test_images[0]
    print(f"\n2. Testing with image: {os.path.basename(test_image_path)}")

    # Test file upload prediction
    print("\n3. Testing file upload prediction:")
    test_prediction_with_file(test_image_path)

    # Test base64 prediction
    print("\n4. Testing base64 prediction:")
    test_prediction_with_base64(test_image_path)

    # Test invalid requests
    test_invalid_requests()

    print("\n" + "=" * 50)
    print("Test completed. The API is ready to use!")
    print("\nUsage examples:")
    print("1. File upload: curl -X POST -F 'image=@path/to/image.jpg' http://localhost:5000/predict")
    print(
        "2. Base64 JSON: curl -X POST -H 'Content-Type: application/json' -d '{\"image\":\"<base64_data>\"}' http://localhost:5000/predict")


if __name__ == "__main__":
    main()
