// ==========================================
// GUARDIAN REAL-TIME GPS MAP
// ==========================================

let guardianMap;
let guardianMarker;
let accuracyCircle;


// Initialize map
function initializeGuardianMap() {

    // Default location: Chennai
    const defaultLocation = [13.0827, 80.2707];

    guardianMap = L.map("guardianMap").setView(
        defaultLocation,
        15
    );

    L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
            maxZoom: 19,
            attribution: "&copy; OpenStreetMap contributors"
        }
    ).addTo(guardianMap);

    guardianMarker = L.marker(defaultLocation)
        .addTo(guardianMap)
        .bindPopup(
            "<b>Guardian</b><br>Waiting for live GPS..."
        );

    accuracyCircle = L.circle(
        defaultLocation,
        {
            radius: 0
        }
    ).addTo(guardianMap);
}


// Get GPS data from Flask
async function updateGuardianLocation() {

    try {

        const response = await fetch(
            "/api/location"
        );

        const data = await response.json();

        if (
            !data.success ||
            !data.location ||
            data.location.latitude === null
        ) {

            setGPSStatus("WAITING");

            return;
        }

        const latitude =
            Number(data.location.latitude);

        const longitude =
            Number(data.location.longitude);

        const accuracy =
            Number(data.location.accuracy || 0);

        const newLocation = [
            latitude,
            longitude
        ];


        // Move marker
        guardianMarker.setLatLng(newLocation);


        // Move accuracy circle
        accuracyCircle.setLatLng(newLocation);
        accuracyCircle.setRadius(accuracy);


        // Move map smoothly
        guardianMap.flyTo(
            newLocation,
            17,
            {
                duration: 1
            }
        );


        // Update popup
        guardianMarker.bindPopup(
            `
            <b>Guardian elderly safety device</b>
            <br>
            Latitude: ${latitude.toFixed(6)}
            <br>
            Longitude: ${longitude.toFixed(6)}
            <br>
            Accuracy: ${accuracy.toFixed(1)} m
            `
        );


        // Update dashboard values
        document.getElementById(
            "gpsLatitude"
        ).textContent = latitude.toFixed(6);

        document.getElementById(
            "gpsLongitude"
        ).textContent = longitude.toFixed(6);

        document.getElementById(
            "gpsAccuracy"
        ).textContent =
            accuracy.toFixed(1) + " m";


        if (data.location.timestamp) {

            const time =
                new Date(
                    data.location.timestamp
                );

            document.getElementById(
                "gpsTime"
            ).textContent =
                time.toLocaleTimeString();

        }


        setGPSStatus("LIVE");

    }

    catch (error) {

        console.error(
            "Guardian GPS error:",
            error
        );

        setGPSStatus("OFFLINE");
    }
}


// GPS status indicator
function setGPSStatus(status) {

    const element =
        document.getElementById("gpsStatus");

    if (!element) return;

    element.textContent = status;

    element.classList.remove(
        "live",
        "waiting"
    );

    if (status === "LIVE") {

        element.classList.add("live");

    } else {

        element.classList.add("waiting");
    }
}


// Start map
document.addEventListener(
    "DOMContentLoaded",
    function () {

        initializeGuardianMap();

        updateGuardianLocation();

        // Update every 2 seconds
        setInterval(
            updateGuardianLocation,
            2000
        );
    }
);