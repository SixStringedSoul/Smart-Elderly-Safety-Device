/* =========================================================
   GUARDIAN GPS / LOCATION MODULE
   Uses the browser Geolocation API.
   No API key required.
   ========================================================= */

let guardianLocation = {
    latitude: null,
    longitude: null,
    accuracy: null,
    timestamp: null
};

let guardianWatchId = null;


/* ---------------------------------------------------------
   Start continuous location tracking
   --------------------------------------------------------- */

function startGuardianGPS() {

    if (!navigator.geolocation) {
        console.error("Geolocation is not supported by this browser.");
        return;
    }

    console.log("Guardian GPS: starting location tracking...");

    guardianWatchId = navigator.geolocation.watchPosition(

        function (position) {

            guardianLocation.latitude =
                position.coords.latitude;

            guardianLocation.longitude =
                position.coords.longitude;

            guardianLocation.accuracy =
                position.coords.accuracy;

            guardianLocation.timestamp =
                new Date().toISOString();


            console.log("Guardian GPS location:", {
                latitude: guardianLocation.latitude,
                longitude: guardianLocation.longitude,
                accuracy: guardianLocation.accuracy
            });


            /*
             * Send location to your Flask backend.
             */
            sendLocationToBackend();

        },

        function (error) {

            console.error(
                "Guardian GPS error:",
                error.message
            );

        },

        {
            enableHighAccuracy: true,
            maximumAge: 5000,
            timeout: 10000
        }
    );
}


/* ---------------------------------------------------------
   Send current GPS coordinates to Flask
   --------------------------------------------------------- */

async function sendLocationToBackend() {

    if (
        guardianLocation.latitude === null ||
        guardianLocation.longitude === null
    ) {
        return;
    }

    try {

        const response = await fetch("/api/location", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                latitude: guardianLocation.latitude,

                longitude: guardianLocation.longitude,

                accuracy: guardianLocation.accuracy,

                timestamp: guardianLocation.timestamp
            })
        });


        if (!response.ok) {
            throw new Error(
                `HTTP ${response.status}`
            );
        }


        const data = await response.json();

        console.log(
            "Guardian location sent to backend:",
            data
        );

    }

    catch (error) {

        console.error(
            "Could not send GPS location:",
            error
        );

    }
}


/* ---------------------------------------------------------
   Stop GPS tracking
   --------------------------------------------------------- */

function stopGuardianGPS() {

    if (
        guardianWatchId !== null &&
        navigator.geolocation
    ) {

        navigator.geolocation.clearWatch(
            guardianWatchId
        );

        guardianWatchId = null;

        console.log(
            "Guardian GPS tracking stopped."
        );
    }
}


/* ---------------------------------------------------------
   Start automatically when website loads
   --------------------------------------------------------- */

document.addEventListener(
    "DOMContentLoaded",
    function () {

        startGuardianGPS();

    }
);