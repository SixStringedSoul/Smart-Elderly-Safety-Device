from flask import Flask, render_template, jsonify, request
from datetime import datetime

app = Flask(__name__)

# =========================================================
# GUARDIAN APPLICATION STATE
# =========================================================

latest_location = {
    "latitude": None,
    "longitude": None,
    "accuracy": None,
    "timestamp": None
}

system_state = {
    "status": "online",
    "fall_detected": False,
    "fall_probability": 0.0,
    "heart_rate": 72,
    "spo2": 98,
    "battery": 87,
    "device_connected": True,
    "model_status": "ready"
}

event_log = []

# =========================================================
# ELDERLY PERSON PROFILE
# =========================================================

person_profile = {
    "name": "Rajesh Kumar",
    "age": 72,
    "device_id": "GUARDIAN-001",
    "emergency_contact": "Family Emergency Contact"
}

# =========================================================
# SOS STATE
# =========================================================

latest_sos = {
    "active": False,
    "person_name": person_profile["name"],
    "latitude": None,
    "longitude": None,
    "timestamp": None,
    "emergency_type": None
}


# =========================================================
# MAIN WEBSITE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# SYSTEM STATUS API
# =========================================================

@app.route("/api/status", methods=["GET"])
def api_status():

    return jsonify({
        "status": system_state["status"],
        "message": "Guardian backend is connected",

        "person": person_profile,

        "device": {
            "connected": system_state["device_connected"]
        },

        "ai": {
            "model": "BiGRU",
            "status": system_state["model_status"],
            "fall_detected": system_state["fall_detected"],
            "fall_probability": system_state["fall_probability"]
        },

        "health": {
            "heart_rate": system_state["heart_rate"],
            "spo2": system_state["spo2"],
            "battery": system_state["battery"]
        },

        "location": latest_location,

        "sos": latest_sos,

        "events": event_log
    })


# =========================================================
# PERSON PROFILE API
# =========================================================

@app.route("/api/person", methods=["GET"])
def get_person():

    return jsonify({
        "success": True,
        "person": person_profile
    })


# =========================================================
# GPS LOCATION API
# =========================================================

@app.route("/api/location", methods=["POST"])
def receive_location():

    global latest_location

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "No location data received"
            }), 400

        latitude = data.get("latitude")
        longitude = data.get("longitude")
        accuracy = data.get("accuracy")
        timestamp = data.get("timestamp")

        if latitude is None or longitude is None:
            return jsonify({
                "success": False,
                "message": "Latitude or longitude missing"
            }), 400

        latest_location = {
            "latitude": latitude,
            "longitude": longitude,
            "accuracy": accuracy,
            "timestamp": timestamp or datetime.now().isoformat()
        }

        print("\n" + "=" * 60)
        print("GUARDIAN GPS UPDATE")
        print("=" * 60)
        print(f"Person    : {person_profile['name']}")
        print(f"Latitude  : {latitude}")
        print(f"Longitude : {longitude}")
        print(f"Accuracy  : {accuracy} meters")
        print(f"Time      : {latest_location['timestamp']}")
        print("=" * 60)

        return jsonify({
            "success": True,
            "message": "Location received successfully",
            "location": latest_location
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": "Error processing GPS data",
            "error": str(e)
        }), 500


# =========================================================
# GET CURRENT GPS LOCATION
# =========================================================

@app.route("/api/location", methods=["GET"])
def get_location():

    return jsonify({
        "success": True,
        "person": person_profile["name"],
        "location": latest_location
    })


# =========================================================
# MAP LINK API
# =========================================================

@app.route("/api/location/map", methods=["GET"])
def get_map_link():

    latitude = latest_location["latitude"]
    longitude = latest_location["longitude"]

    if latitude is None or longitude is None:

        return jsonify({
            "success": False,
            "message": "GPS location not available"
        }), 404

    map_url = (
        "https://www.google.com/maps/"
        f"search/?api=1&query={latitude},{longitude}"
    )

    return jsonify({
        "success": True,
        "latitude": latitude,
        "longitude": longitude,
        "map_url": map_url
    })


# =========================================================
# TEST GPS LOCATION
# =========================================================

@app.route("/api/location/test", methods=["POST"])
def test_location():

    global latest_location

    latest_location = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "accuracy": 10,
        "timestamp": datetime.now().isoformat()
    }

    print("\n" + "=" * 60)
    print("GUARDIAN TEST GPS LOCATION")
    print("=" * 60)
    print(f"Person    : {person_profile['name']}")
    print(f"Latitude  : {latest_location['latitude']}")
    print(f"Longitude : {latest_location['longitude']}")
    print(f"Accuracy  : {latest_location['accuracy']} meters")
    print("=" * 60)

    return jsonify({
        "success": True,
        "message": "Test GPS location created",
        "location": latest_location
    })


# =========================================================
# FALL DETECTION API
# =========================================================

@app.route("/api/fall", methods=["POST"])
def update_fall_state():

    global system_state

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "No fall data received"
            }), 400

        fall_detected = bool(
            data.get("fall_detected", False)
        )

        probability = float(
            data.get("fall_probability", 0.0)
        )

        system_state["fall_detected"] = fall_detected
        system_state["fall_probability"] = probability

        if fall_detected:

            event = {
                "type": "FALL DETECTED",
                "person_name": person_profile["name"],
                "probability": probability,
                "timestamp": datetime.now().isoformat(),
                "location": latest_location.copy()
            }

            event_log.insert(0, event)

            if len(event_log) > 50:
                event_log.pop()

            print("\n" + "=" * 70)
            print("🚨 GUARDIAN FALL ALERT")
            print("=" * 70)
            print(f"Person      : {person_profile['name']}")
            print(f"Probability : {probability:.2f}")
            print(f"Location    : {latest_location}")
            print("=" * 70)

        return jsonify({
            "success": True,
            "fall_detected": fall_detected,
            "fall_probability": probability
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": "Error processing fall data",
            "error": str(e)
        }), 500


# =========================================================
# EVENT HISTORY
# =========================================================

@app.route("/api/events", methods=["GET"])
def get_events():

    return jsonify({
        "success": True,
        "count": len(event_log),
        "events": event_log
    })


# =========================================================
# CLEAR EVENT HISTORY
# =========================================================

@app.route("/api/events/clear", methods=["POST"])
def clear_events():

    global event_log

    event_log = []

    return jsonify({
        "success": True,
        "message": "Event history cleared"
    })


# =========================================================
# HEALTH DATA API
# =========================================================

@app.route("/api/health", methods=["POST"])
def update_health():

    global system_state

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "No health data received"
            }), 400

        if "heart_rate" in data:
            system_state["heart_rate"] = data["heart_rate"]

        if "spo2" in data:
            system_state["spo2"] = data["spo2"]

        if "battery" in data:
            system_state["battery"] = data["battery"]

        return jsonify({
            "success": True,
            "health": {
                "heart_rate": system_state["heart_rate"],
                "spo2": system_state["spo2"],
                "battery": system_state["battery"]
            }
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": "Error processing health data",
            "error": str(e)
        }), 500


# =========================================================
# DEVICE CONNECTION STATUS
# =========================================================

@app.route("/api/device", methods=["POST"])
def update_device():

    global system_state

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "No device data received"
            }), 400

        system_state["device_connected"] = bool(
            data.get("connected", False)
        )

        return jsonify({
            "success": True,
            "device_connected": system_state["device_connected"]
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": "Error processing device status",
            "error": str(e)
        }), 500


# =========================================================
# SOS API — ACTIVATE EMERGENCY
# =========================================================

@app.route("/api/sos", methods=["POST"])
def receive_sos():

    global latest_sos

    try:

        data = request.get_json() or {}

        # Use latest GPS automatically if coordinates aren't supplied
        latitude = data.get(
            "latitude",
            latest_location["latitude"]
        )

        longitude = data.get(
            "longitude",
            latest_location["longitude"]
        )

        latest_sos = {
            "active": True,

            "person_name": data.get(
                "person_name",
                person_profile["name"]
            ),

            "latitude": latitude,
            "longitude": longitude,

            "timestamp": data.get(
                "timestamp",
                datetime.now().isoformat()
            ),

            "emergency_type": data.get(
                "emergency_type",
                "MANUAL SOS"
            )
        }

        # Add SOS to event history
        event = {
            "type": "SOS ACTIVATED",
            "person_name": latest_sos["person_name"],
            "emergency_type": latest_sos["emergency_type"],
            "timestamp": latest_sos["timestamp"],
            "location": {
                "latitude": latitude,
                "longitude": longitude
            }
        }

        event_log.insert(0, event)

        if len(event_log) > 50:
            event_log.pop()

        print("\n" + "=" * 70)
        print("🚨🚨 GUARDIAN EMERGENCY SOS 🚨🚨")
        print("=" * 70)
        print(f"Person      : {latest_sos['person_name']}")
        print(f"Emergency   : {latest_sos['emergency_type']}")
        print(f"Latitude    : {latest_sos['latitude']}")
        print(f"Longitude   : {latest_sos['longitude']}")
        print(f"Timestamp   : {latest_sos['timestamp']}")
        print("=" * 70)

        return jsonify({
            "success": True,
            "message": "SOS activated successfully",
            "sos": latest_sos
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": "Error activating SOS",
            "error": str(e)
        }), 500


# =========================================================
# GET SOS STATUS
# =========================================================

@app.route("/api/sos", methods=["GET"])
def get_sos():

    return jsonify({
        "success": True,
        "sos": latest_sos
    })


# =========================================================
# CANCEL SOS
# =========================================================

@app.route("/api/sos/cancel", methods=["POST"])
def cancel_sos():

    global latest_sos

    latest_sos["active"] = False

    print("\nGuardian SOS cancelled.")

    return jsonify({
        "success": True,
        "message": "SOS cancelled",
        "sos": latest_sos
    })


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/api/health-check", methods=["GET"])
def health_check():

    return jsonify({
        "success": True,
        "server": "Guardian Flask Backend",
        "status": "running",
        "time": datetime.now().isoformat()
    })


# =========================================================
# 404 HANDLER
# =========================================================

@app.errorhandler(404)
def page_not_found(error):

    return jsonify({
        "success": False,
        "message": "API endpoint not found"
    }), 404


# =========================================================
# RUN GUARDIAN SERVER
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("       GUARDIAN SMART EMERGENCY ALERT SYSTEM")
    print("=" * 70)
    print()
    print(f"Registered person : {person_profile['name']}")
    print(f"Device ID         : {person_profile['device_id']}")
    print()
    print("Website:")
    print("http://127.0.0.1:5000")
    print()
    print("System API:")
    print("http://127.0.0.1:5000/api/status")
    print()
    print("Person API:")
    print("http://127.0.0.1:5000/api/person")
    print()
    print("GPS API:")
    print("http://127.0.0.1:5000/api/location")
    print()
    print("Events API:")
    print("http://127.0.0.1:5000/api/events")
    print()
    print("Health API:")
    print("http://127.0.0.1:5000/api/health")
    print()
    print("SOS API:")
    print("http://127.0.0.1:5000/api/sos")
    print()
    print("Server ready.")
    print("=" * 70)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )