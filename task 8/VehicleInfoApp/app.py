from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# -----------------------
# Helper Functions
# -----------------------
def fetch_vehicle_info(vin: str) -> dict:
    """Fetch vehicle info from NHTSA API using VIN."""
    api_url = f"https://vpic.nhtsa.dot.gov/api/vehicles/decodevin/{vin}?format=json"
    response = requests.get(api_url, timeout=10)
    response.raise_for_status()
    data = response.json()
    if 'Results' in data:
        vehicle_data = data['Results']
        return {item['Variable']: item['Value'] for item in vehicle_data if item['Value']}
    return {}

def fetch_manufacturers() -> list:
    """Get all vehicle manufacturers from NHTSA API."""
    url = "https://vpic.nhtsa.dot.gov/api/vehicles/getallmanufacturers?format=json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    return [{"Name": m.get("Mfr_Name"), "Country": m.get("Country")} for m in data.get("Results", [])]

def fetch_models_by_make(make: str) -> list:
    """Get models of a given manufacturer/make."""
    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/getmodelsformake/{make}?format=json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    return [model.get("Model_Name") for model in data.get("Results", [])]


# -----------------------
# Routes
# -----------------------
@app.route('/')
def home():
    return jsonify({
        "message": "Vehicle Info App API is running.",
        "endpoints": {
            "GET /vehicle-info?vin=<VIN>": "Get vehicle info by VIN",
            "GET /manufacturers": "Get all vehicle manufacturers",
            "GET /models?make=<MAKE>": "Get models by manufacturer/make"
        }
    })


@app.route('/vehicle-info', methods=['GET'])
def vehicle_info():
    vin = request.args.get('vin')
    if not vin:
        return jsonify({"error": "VIN is required"}), 400
    try:
        result = fetch_vehicle_info(vin)
        if result:
            return jsonify({"vin": vin, "vehicle_info": result})
        return jsonify({"error": "No vehicle info found for this VIN"}), 404
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to fetch data from NHTSA API", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500


@app.route('/manufacturers', methods=['GET'])
def manufacturers():
    try:
        data = fetch_manufacturers()
        return jsonify({"total_manufacturers": len(data), "manufacturers": data})
    except Exception as e:
        return jsonify({"error": "Failed to fetch manufacturers", "details": str(e)}), 500


@app.route('/models', methods=['GET'])
def models():
    make = request.args.get('make')
    if not make:
        return jsonify({"error": "Make is required"}), 400
    try:
        models_list = fetch_models_by_make(make)
        if models_list:
            return jsonify({"make": make, "models": models_list})
        return jsonify({"error": f"No models found for make '{make}'"}), 404
    except Exception as e:
        return jsonify({"error": "Failed to fetch models", "details": str(e)}), 500


# -----------------------
# Run the App
# -----------------------
if __name__ == '__main__':
    app.run(debug=True)