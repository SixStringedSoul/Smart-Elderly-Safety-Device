(() => {
  const $ = (id) => document.getElementById(id);
  const toast = (message) => {
    const node = $("toast");
    if (!node) return;
    node.textContent = message;
    node.classList.add("show");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => node.classList.remove("show"), 2600);
  };

  // Interactive Three.js concept model. If CDN/WebGL is unavailable, the rest of the dashboard still works.
  function mountDevice(targetId, compact = false) {
    const host = $(targetId);
    if (!host || !window.THREE) return;
    let renderer, scene, camera, root, frameId;
    try {
      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(34, Math.max(host.clientWidth, 1) / Math.max(host.clientHeight, 1), 0.1, 100);
      camera.position.set(0, 0.15, compact ? 6.5 : 6.7);
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(host.clientWidth || 360, host.clientHeight || 280);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.setClearColor(0x000000, 0);
      host.appendChild(renderer.domElement);

      scene.add(new THREE.AmbientLight(0x9adce8, 1.8));
      const keyLight = new THREE.PointLight(0x65dce9, 28, 20);
      keyLight.position.set(3, 3, 4); scene.add(keyLight);
      const rimLight = new THREE.PointLight(0x8b78ff, 20, 18);
      rimLight.position.set(-3, -1, -3); scene.add(rimLight);
      const fillLight = new THREE.DirectionalLight(0xe5f8ff, 1.3);
      fillLight.position.set(0, 4, 3); scene.add(fillLight);

      root = new THREE.Group(); scene.add(root);
      const shellMat = new THREE.MeshPhysicalMaterial({ color: 0x1a3041, metalness: .78, roughness: .24, clearcoat: .8, clearcoatRoughness: .18 });
      const edgeMat = new THREE.MeshStandardMaterial({ color: 0x63dce7, metalness: .7, roughness: .23, emissive: 0x124b58, emissiveIntensity: .7 });
      const darkMat = new THREE.MeshStandardMaterial({ color: 0x09121d, metalness: .55, roughness: .28 });
      const glassMat = new THREE.MeshPhysicalMaterial({ color: 0x122838, metalness: .25, roughness: .12, transmission: .12, transparent: true, opacity: .92 });
      const body = new THREE.Mesh(new THREE.BoxGeometry(1.42, 1.85, .38), shellMat);
      body.position.y = .04; root.add(body);
      const bevel = new THREE.Mesh(new THREE.BoxGeometry(1.30, 1.73, .405), darkMat);
      bevel.position.z = .012; bevel.position.y = .04; root.add(bevel);
      const face = new THREE.Mesh(new THREE.BoxGeometry(1.18, 1.58, .035), glassMat);
      face.position.set(0, .04, .225); root.add(face);
      const frame = new THREE.Mesh(new THREE.BoxGeometry(1.23, 1.63, .04), edgeMat);
      frame.position.set(0, .04, .202); root.add(frame);
      const screen = new THREE.Mesh(new THREE.BoxGeometry(1.15, 1.55, .04), darkMat);
      screen.position.set(0, .04, .232); root.add(screen);
      // Fine cyan screen strokes / visual sensor UI
      const lineMat = new THREE.MeshBasicMaterial({ color: 0x70e1e9 });
      const dimLineMat = new THREE.MeshBasicMaterial({ color: 0x2c6878 });
      function bar(x, y, w, h, mat = dimLineMat, z = .26) {
        const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, .008), mat);
        m.position.set(x, y, z); root.add(m); return m;
      }
      bar(0, .61, .64, .018, dimLineMat);
      bar(-.29, .45, .05, .16, lineMat); bar(-.20, .48, .05, .22, lineMat);
      bar(-.11, .42, .05, .10, lineMat); bar(-.02, .50, .05, .26, lineMat);
      bar(.07, .46, .05, .18, lineMat); bar(.16, .53, .05, .32, lineMat);
      bar(.25, .44, .05, .14, lineMat);
      // Circular sensor glyph
      const ring = new THREE.Mesh(new THREE.TorusGeometry(.19, .014, 8, 64), edgeMat);
      ring.position.set(0, -.04, .28); root.add(ring);
      const core = new THREE.Mesh(new THREE.SphereGeometry(.075, 24, 24), new THREE.MeshStandardMaterial({ color: 0x72eff0, emissive: 0x38c7dc, emissiveIntensity: 1.7, metalness: .3, roughness: .2 }));
      core.position.set(0, -.04, .29); root.add(core);
      const ring2 = new THREE.Mesh(new THREE.TorusGeometry(.29, .004, 6, 64), dimLineMat);
      ring2.position.set(0, -.04, .27); root.add(ring2);
      bar(0, -.39, .72, .012, dimLineMat);
      bar(-.25, -.53, .10, .05, edgeMat); bar(-.08, -.53, .10, .05, edgeMat); bar(.09, -.53, .10, .05, edgeMat); bar(.26, -.53, .10, .05, edgeMat);
      // Small side button
      const button = new THREE.Mesh(new THREE.CylinderGeometry(.075, .075, .09, 24), edgeMat);
      button.rotation.z = Math.PI / 2; button.position.set(.75, .2, 0); root.add(button);
      // Wrist/strap lugs for a wearable-device concept
      const strapMat = new THREE.MeshStandardMaterial({ color: 0x101b29, metalness: .25, roughness: .48 });
      const strapTop = new THREE.Mesh(new THREE.BoxGeometry(.78, .42, .20), strapMat);
      strapTop.position.set(0, 1.07, -.04); strapTop.rotation.z = -.04; root.add(strapTop);
      const strapBottom = new THREE.Mesh(new THREE.BoxGeometry(.78, .42, .20), strapMat);
      strapBottom.position.set(0, -1.00, -.04); strapBottom.rotation.z = .04; root.add(strapBottom);
      for (let i = 0; i < 3; i++) {
        const hole = new THREE.Mesh(new THREE.BoxGeometry(.13, .035, .01), edgeMat);
        hole.position.set(0, -1.02 - i * .09, .075); root.add(hole);
      }
      root.rotation.set(-.08, -.35, -.04);
      root.scale.setScalar(compact ? .78 : 1.0);

      let dragging = false, lastX = 0, lastY = 0, velocityY = 0, velocityX = 0;
      renderer.domElement.addEventListener("pointerdown", e => { dragging = true; lastX = e.clientX; lastY = e.clientY; renderer.domElement.setPointerCapture?.(e.pointerId); });
      renderer.domElement.addEventListener("pointermove", e => {
        if (!dragging) return;
        velocityY = (e.clientX - lastX) * .009;
        velocityX = (e.clientY - lastY) * .006;
        root.rotation.y += velocityY; root.rotation.x += velocityX;
        lastX = e.clientX; lastY = e.clientY;
      });
      renderer.domElement.addEventListener("pointerup", () => { dragging = false; });
      renderer.domElement.addEventListener("pointercancel", () => { dragging = false; });
      let clock = 0;
      const animate = () => {
        frameId = requestAnimationFrame(animate); clock += .012;
        if (!dragging) {
          root.rotation.y += .0035;
          root.rotation.x += (Math.sin(clock * .8) * .055 - root.rotation.x) * .012;
        }
        root.position.y = Math.sin(clock) * .055;
        ring.rotation.z += .006;
        ring2.rotation.z -= .003;
        renderer.render(scene, camera);
      };
      animate();
      const resize = () => {
        if (!host.clientWidth || !host.clientHeight) return;
        camera.aspect = host.clientWidth / host.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(host.clientWidth, host.clientHeight);
      };
      if (window.ResizeObserver) new ResizeObserver(resize).observe(host);
      else window.addEventListener("resize", resize);
      host.dataset.mounted = "true";
    } catch (error) {
      console.warn("3D concept could not initialize:", error);
      host.innerHTML = '<div class="webgl-fallback">3D preview unavailable in this browser.<br>Dashboard controls remain available.</div>';
    }
  }
  mountDevice("device3d", false);
  mountDevice("device3dMini", true);

  // Motion chart uses clearly labelled synthetic preview values until actual sensor data is connected.
  const chartCanvas = $("motionChart");
  let motionChart = null;
  if (chartCanvas && window.Chart) {
    const labels = Array.from({ length: 60 }, (_, i) => (i / 20).toFixed(2));
    const wave = (amp, phase, drift = 0) => labels.map((_, i) => Number((Math.sin(i * .41 + phase) * amp + Math.sin(i * 1.13) * amp * .24 + drift).toFixed(3)));
    const acc = wave(.68, .3, .18);
    const gyro = wave(.36, 1.8, -.05);
    motionChart = new Chart(chartCanvas, {
      type: "line",
      data: { labels, datasets: [
        { label: "Acceleration preview", data: acc, borderColor: "#70e1e9", backgroundColor: "#70e1e910", borderWidth: 1.7, pointRadius: 0, tension: .28, fill: true },
        { label: "Angular velocity preview", data: gyro, borderColor: "#aa9bff", borderWidth: 1.5, pointRadius: 0, tension: .3, fill: false }
      ]},
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 450 },
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { display: false }, tooltip: { backgroundColor: "#172437", borderColor: "#31465e", borderWidth: 1, titleColor: "#e7f2ff", bodyColor: "#b4c6da", displayColors: true } },
        scales: {
          x: { grid: { color: "#202d40" }, ticks: { color: "#6f839c", maxTicksLimit: 7, font: { size: 9 } }, title: { display: true, text: "Time (seconds) · preview", color: "#61758e", font: { size: 9 } } },
          y: { grid: { color: "#202d40" }, ticks: { color: "#6f839c", maxTicksLimit: 5, font: { size: 9 } }, border: { display: false } }
        }
      }
    });
    $("signalMode")?.addEventListener("change", e => {
      const mode = e.target.value;
      motionChart.data.datasets[0].hidden = mode === "gyro";
      motionChart.data.datasets[1].hidden = mode === "accel";
      motionChart.update();
    });
  }

  async function refreshStatus() {
    const btn = $("refreshBtn");
    if (btn) btn.textContent = "…";
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      const data = await response.json();
      $("deviceStatus").textContent = data.device_connected ? "Online" : "Offline";
      $("deviceDetail").textContent = data.device_connected ? "Receiving telemetry" : "No ESP32 stream";
      $("modelStatus").textContent = data.prediction === "WAITING" ? "Model prepared" : "Inference active";
      $("alertStatus").textContent = data.mode === "live" ? "Connected" : "Not connected";
      $("heroStatus").textContent = data.message;
      if (data.fall_probability !== null && data.fall_probability !== undefined) {
        const probability = Number(data.fall_probability);
        $("fallProbability").textContent = probability.toFixed(4);
        $("probabilityFill").style.width = `${Math.max(0, Math.min(1, probability)) * 100}%`;
        const isFall = probability >= Number(data.threshold || .60);
        $("fallState").textContent = isFall ? "Potential fall" : "Normal movement";
        $("fallSub").textContent = `Model probability ${probability.toFixed(3)} · threshold ${Number(data.threshold || .60).toFixed(2)}`;
      }
    } catch (err) {
      $("heroStatus").textContent = "Backend status unavailable — check Flask server";
      toast("Couldn't refresh backend status.");
    } finally {
      if (btn) btn.textContent = "↻";
    }
  }
  $("refreshBtn")?.addEventListener("click", refreshStatus);
  refreshStatus();

  $("mobileMenu")?.addEventListener("click", () => $("sidebar")?.classList.toggle("open"));
  document.querySelectorAll(".nav-link").forEach(link => {
    link.addEventListener("click", () => {
      document.querySelectorAll(".nav-link").forEach(n => n.classList.remove("active"));
      link.classList.add("active");
      $("sidebar")?.classList.remove("open");
    });
  });
  const observer = "IntersectionObserver" in window ? new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      document.querySelectorAll(".nav-link").forEach(n => n.classList.toggle("active", n.dataset.section === entry.target.id));
    });
  }, { rootMargin: "-20% 0px -65% 0px" }) : null;
  if (observer) ["overview", "monitoring", "health", "alerts", "device"].forEach(id => { const section = $(id); if (section) observer.observe(section); });

  $("exportEvents")?.addEventListener("click", () => {
    const csv = "timestamp,event,probability,status\\n";
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "guardian_event_log_preview.csv"; a.click();
    URL.revokeObjectURL(url);
    toast("Exported an empty preview event log. Live event integration is pending.");
  });

  // Clickable profile, project card, and overflow menu.
  const profileButton = $("profileButton");
  const profileMenu = $("profileMenu");
  const moreButton = $("sidebarMoreButton");
  const moreMenu = $("sidebarMoreMenu");
  const modal = $("infoModal");
  const openModal = (kind = "about") => {
    if (!modal) return;
    const title = $("infoModalTitle");
    const text = $("infoModalText");
    if (kind === "status") {
      title.textContent = "System status";
      text.textContent = "Guardian is currently in preview mode. No live ESP32 stream, BiGRU inference, health readings, GPS, or emergency dispatch is connected.";
    } else {
      title.textContent = "Guardian prototype";
      text.textContent = "Smart Emergency Assistance for the Elderly — a capstone research prototype. The 3D wearable is a visual concept; trained-model and hardware integration are planned next.";
    }
    modal.hidden = false;
    $("closeInfoModal")?.focus();
  };
  const closeModal = () => { if (modal) modal.hidden = true; };
  profileButton?.addEventListener("click", (event) => {
    event.stopPropagation();
    const open = profileMenu.hidden;
    profileMenu.hidden = !open;
    profileButton.setAttribute("aria-expanded", String(open));
    if (moreMenu) moreMenu.hidden = true;
    moreButton?.setAttribute("aria-expanded", "false");
  });
  moreButton?.addEventListener("click", (event) => {
    event.stopPropagation();
    const open = moreMenu.hidden;
    moreMenu.hidden = !open;
    moreButton.setAttribute("aria-expanded", String(open));
    if (profileMenu) profileMenu.hidden = true;
    profileButton?.setAttribute("aria-expanded", "false");
  });
  $("sidebarProfileButton")?.addEventListener("click", () => openModal("about"));
  document.querySelectorAll("[data-profile-action]").forEach(button => {
    button.addEventListener("click", () => {
      if (profileMenu) profileMenu.hidden = true;
      profileButton?.setAttribute("aria-expanded", "false");
      openModal(button.dataset.profileAction);
    });
  });
  document.querySelectorAll("[data-sidebar-action]").forEach(button => {
    button.addEventListener("click", () => {
      if (moreMenu) moreMenu.hidden = true;
      moreButton?.setAttribute("aria-expanded", "false");
      const action = button.dataset.sidebarAction;
      if (action === "export") $("exportEvents")?.click();
      else openModal(action);
    });
  });
  $("closeInfoModal")?.addEventListener("click", closeModal);
  $("modalDone")?.addEventListener("click", closeModal);
  modal?.addEventListener("click", event => { if (event.target === modal) closeModal(); });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") {
      closeModal();
      if (profileMenu) profileMenu.hidden = true;
      if (moreMenu) moreMenu.hidden = true;
      profileButton?.setAttribute("aria-expanded", "false");
      moreButton?.setAttribute("aria-expanded", "false");
    }
  });
  document.addEventListener("click", event => {
    if (profileMenu && !profileMenu.hidden && !profileMenu.contains(event.target) && !profileButton?.contains(event.target)) {
      profileMenu.hidden = true; profileButton?.setAttribute("aria-expanded", "false");
    }
    if (moreMenu && !moreMenu.hidden && !moreMenu.contains(event.target) && !moreButton?.contains(event.target)) {
      moreMenu.hidden = true; moreButton?.setAttribute("aria-expanded", "false");
    }
  });

})();

  // ============================================================
  // GUARDIAN ML MODEL TESTING
  // ============================================================

    async function getGuardianStatus() {
    try {
        const response = await fetch("/api/status");

        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }

        const data = await response.json();

        console.log("Guardian Backend Data:");
        console.log(data);

        // Update website elements
        updateDashboard(data);

    } catch (error) {
        console.error("Could not connect to Guardian backend:", error);
    }
}


function updateDashboard(data) {

    // Connection status
    const statusElement = document.getElementById("backend-status");

    if (statusElement) {
        statusElement.textContent =
            data.status === "online"
                ? "Guardian System Online"
                : "Backend Offline";
    }


    // Heart rate
    const heartRateElement = document.getElementById("heart-rate");

    if (heartRateElement) {
        heartRateElement.textContent =
            data.heart_rate + " BPM";
    }


    // Battery
    const batteryElement = document.getElementById("battery");

    if (batteryElement) {
        batteryElement.textContent =
            data.battery + "%";
    }


    // Fall detection
    const fallElement = document.getElementById("fall-status");

    if (fallElement) {

        if (data.fall_detected) {
            fallElement.textContent = "FALL DETECTED";
        } else {
            fallElement.textContent = "No Fall Detected";
        }
    }


    // Location
    const locationElement = document.getElementById("location");

    if (locationElement) {

        locationElement.textContent =
            `${data.location.latitude}, ${data.location.longitude}`;
    }
}


// Get data when website loads
document.addEventListener("DOMContentLoaded", () => {

    getGuardianStatus();

});
/* =========================================================
   GUARDIAN SENSOR SIMULATION ENGINE
   Software simulation for project demonstration
========================================================= */

(function () {

    console.log("==========================================");
    console.log("GUARDIAN SENSOR SIMULATION STARTED");
    console.log("==========================================");

    const SIMULATION = true;

    /* ---------------------------------------------------------
       SIMULATED SENSOR STATE
    --------------------------------------------------------- */

    let sensorState = {
        latitude: 13.0827,
        longitude: 80.2707,

        heartRate: 72,
        spo2: 98,
        battery: 87,

        fallProbability: 0.08,
        fallDetected: false,

        acceleration: {
            x: 0,
            y: 0,
            z: 9.81
        },

        gyroscope: {
            x: 0,
            y: 0,
            z: 0
        },

        timestamp: null
    };


    /* ---------------------------------------------------------
       SIMULATED GPS ROUTE
       Chennai demonstration route
    --------------------------------------------------------- */

    const gpsRoute = [
        [13.0827, 80.2707],
        [13.0835, 80.2714],
        [13.0842, 80.2722],
        [13.0850, 80.2730],
        [13.0858, 80.2737],
        [13.0865, 80.2745],
        [13.0872, 80.2752],
        [13.0880, 80.2760],
        [13.0887, 80.2768],
        [13.0894, 80.2775],
        [13.0900, 80.2782],
        [13.0908, 80.2790]
    ];

    let gpsIndex = 0;


    /* ---------------------------------------------------------
       LEAFLET MAP
    --------------------------------------------------------- */

    let guardianMap = null;
    let guardianMarker = null;
    let guardianAccuracyCircle = null;
    let guardianRoute = null;

    function initializeGPSMap() {

        const mapElement = document.getElementById("guardianMap");

        if (!mapElement) {
            console.warn("GPS map element not found.");
            return;
        }

        if (typeof L === "undefined") {
            console.error("Leaflet is not loaded.");
            return;
        }

        guardianMap = L.map("guardianMap").setView(
            [
                sensorState.latitude,
                sensorState.longitude
            ],
            16
        );

        L.tileLayer(
            "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            {
                maxZoom: 19,
                attribution: "&copy; OpenStreetMap contributors"
            }
        ).addTo(guardianMap);

        guardianMarker = L.marker([
            sensorState.latitude,
            sensorState.longitude
        ]).addTo(guardianMap);

        guardianMarker.bindPopup(
            "<b>GUARDIAN</b><br>" +
            "Simulated elderly wearable<br>" +
            "GPS tracking active"
        );

        guardianAccuracyCircle = L.circle(
            [
                sensorState.latitude,
                sensorState.longitude
            ],
            {
                radius: 5,
                color: "#4ee0bd",
                fillColor: "#4ee0bd",
                fillOpacity: 0.12
            }
        ).addTo(guardianMap);

        guardianRoute = L.polyline(
            gpsRoute,
            {
                color: "#4ee0bd",
                weight: 3,
                opacity: 0.7
            }
        ).addTo(guardianMap);

        console.log("GPS map initialized.");

    }


    /* ---------------------------------------------------------
       UPDATE GPS
    --------------------------------------------------------- */

    function updateGPS() {

        if (gpsIndex >= gpsRoute.length) {
            gpsIndex = 0;
        }

        const point = gpsRoute[gpsIndex];

        sensorState.latitude = point[0];
        sensorState.longitude = point[1];

        sensorState.timestamp =
            new Date().toISOString();

        gpsIndex++;

        updateGPSDisplay();

        sendLocationToBackend();

    }


    /* ---------------------------------------------------------
       UPDATE GPS UI
    --------------------------------------------------------- */

    function updateGPSDisplay() {

        const latElement =
            document.getElementById("gpsLatitude");

        const lonElement =
            document.getElementById("gpsLongitude");

        const accuracyElement =
            document.getElementById("gpsAccuracy");

        const timestampElement =
            document.getElementById("gpsTimestamp");

        if (latElement) {
            latElement.textContent =
                sensorState.latitude.toFixed(6);
        }

        if (lonElement) {
            lonElement.textContent =
                sensorState.longitude.toFixed(6);
        }

        if (accuracyElement) {
            accuracyElement.textContent =
                "±5 m";
        }

        if (timestampElement) {

            timestampElement.textContent =
                new Date().toLocaleTimeString();
        }

        if (guardianMarker) {

            const position = [
                sensorState.latitude,
                sensorState.longitude
            ];

            guardianMarker.setLatLng(position);

            guardianAccuracyCircle.setLatLng(position);

            guardianMap.panTo(position);
        }

    }


    /* ---------------------------------------------------------
       SEND GPS DATA TO FLASK
    --------------------------------------------------------- */

    async function sendLocationToBackend() {

        try {

            const response = await fetch(
                "/api/location",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        latitude:
                            sensorState.latitude,

                        longitude:
                            sensorState.longitude,

                        accuracy: 5,

                        timestamp:
                            sensorState.timestamp
                    })
                }
            );

            const result =
                await response.json();

            console.log(
                "GPS → Flask:",
                result
            );

        } catch (error) {

            console.warn(
                "Could not send GPS to backend:",
                error
            );

        }

    }


    /* ---------------------------------------------------------
       SIMULATE HEART RATE
    --------------------------------------------------------- */

    function simulateHeartRate() {

        const variation =
            Math.round(
                (Math.random() - 0.5) * 10
            );

        sensorState.heartRate =
            Math.max(
                65,
                Math.min(
                    95,
                    72 + variation
                )
            );

        updateHealthCards();

    }


    /* ---------------------------------------------------------
       SIMULATE SpO2
    --------------------------------------------------------- */

    function simulateSpO2() {

        const variation =
            Math.random() * 2 - 1;

        sensorState.spo2 =
            Math.max(
                95,
                Math.min(
                    99,
                    97 + variation
                )
            );

        updateHealthCards();

    }


    /* ---------------------------------------------------------
       SIMULATE BATTERY
    --------------------------------------------------------- */

    function simulateBattery() {

        sensorState.battery -= 0.01;

        if (sensorState.battery < 10) {
            sensorState.battery = 87;
        }

        updateHealthCards();

    }


    /* ---------------------------------------------------------
       UPDATE HEALTH UI
    --------------------------------------------------------- */

    function updateHealthCards() {

        const cards =
            document.querySelectorAll(
                ".vital-card"
            );

        if (!cards.length) {
            return;
        }


        /* HEART RATE */

        const heartValue =
            cards[0].querySelector(
                ".vital-value"
            );

        const heartStatus =
            cards[0].querySelector(
                ".not-connected"
            );

        const heartText =
            cards[0].querySelector("p");

        if (heartValue) {

            heartValue.innerHTML =
                `${Math.round(sensorState.heartRate)}
                 <small>BPM</small>`;

        }

        if (heartStatus) {
            heartStatus.textContent =
                "SIMULATED";
        }

        if (heartText) {
            heartText.textContent =
                "Live simulated MAX30102 reading";
        }


        /* SpO2 */

        const oxygenValue =
            cards[1].querySelector(
                ".vital-value"
            );

        const oxygenStatus =
            cards[1].querySelector(
                ".not-connected"
            );

        const oxygenText =
            cards[1].querySelector("p");

        if (oxygenValue) {

            oxygenValue.innerHTML =
                `${sensorState.spo2.toFixed(1)}
                 <small>% SpO₂</small>`;

        }

        if (oxygenStatus) {
            oxygenStatus.textContent =
                "SIMULATED";
        }

        if (oxygenText) {
            oxygenText.textContent =
                "Live simulated MAX30102 reading";
        }


        /* BATTERY */

        const batteryValue =
            cards[2].querySelector(
                ".vital-value"
            );

        const batteryStatus =
            cards[2].querySelector(
                ".not-connected"
            );

        const batteryText =
            cards[2].querySelector("p");

        if (batteryValue) {

            batteryValue.innerHTML =
                `${Math.round(sensorState.battery)}
                 <small>%</small>`;

        }

        if (batteryStatus) {
            batteryStatus.textContent =
                "SIMULATED";
        }

        if (batteryText) {
            batteryText.textContent =
                "Simulated ESP32 battery telemetry";
        }

    }


    /* ---------------------------------------------------------
       SIMULATE MPU6050
    --------------------------------------------------------- */

    function simulateMotion() {

        sensorState.acceleration.x =
            (Math.random() - 0.5) * 2;

        sensorState.acceleration.y =
            (Math.random() - 0.5) * 2;

        sensorState.acceleration.z =
            9.81 +
            (Math.random() - 0.5) * 1.5;


        sensorState.gyroscope.x =
            (Math.random() - 0.5) * 10;

        sensorState.gyroscope.y =
            (Math.random() - 0.5) * 10;

        sensorState.gyroscope.z =
            (Math.random() - 0.5) * 10;

    }


    /* ---------------------------------------------------------
       SIMULATE BI-GRU FALL PREDICTION
    --------------------------------------------------------- */

    function simulateAI() {

        /*
           Most of the time:
           normal activity = low probability

           Occasionally:
           simulated fall event
        */

        const randomEvent =
            Math.random();

        if (randomEvent < 0.04) {

            /*
               Simulated fall
            */

            sensorState.fallProbability =
                0.75 +
                Math.random() * 0.20;

            sensorState.fallDetected =
                true;

        } else {

            sensorState.fallProbability =
                0.05 +
                Math.random() * 0.25;

            sensorState.fallDetected =
                false;
        }

        updateAIUI();

    }


    /* ---------------------------------------------------------
       UPDATE AI UI
    --------------------------------------------------------- */

    function updateAIUI() {

        const fallState =
            document.getElementById(
                "fallState"
            );

        const fallSub =
            document.getElementById(
                "fallSub"
            );

        const probability =
            document.getElementById(
                "fallProbability"
            );

        const probabilityFill =
            document.getElementById(
                "probabilityFill"
            );

        if (probability) {

            probability.textContent =
                sensorState.fallProbability
                    .toFixed(2);

        }

        if (probabilityFill) {

            probabilityFill.style.width =
                (
                    sensorState.fallProbability *
                    100
                ) + "%";

        }


        if (sensorState.fallDetected) {

            if (fallState) {
                fallState.textContent =
                    "FALL DETECTED";
            }

            if (fallSub) {
                fallSub.textContent =
                    "Simulated high-risk movement";
            }

            createSimulationEvent();

        } else {

            if (fallState) {
                fallState.textContent =
                    "NORMAL";
            }

            if (fallSub) {
                fallSub.textContent =
                    "Motion within normal range";
            }

        }

    }


    /* ---------------------------------------------------------
       EVENT LOG
    --------------------------------------------------------- */

    let lastEventTime = 0;

    function createSimulationEvent() {

        const now =
            Date.now();

        /*
           Prevent multiple events
           every second.
        */

        if (
            now - lastEventTime <
            5000
        ) {
            return;
        }

        lastEventTime = now;

        const eventList =
            document.getElementById(
                "eventList"
            );

        if (!eventList) {
            return;
        }

        const event =
            document.createElement(
                "div"
            );

        event.className =
            "simulation-event";

        event.innerHTML = `
            <div>
                <strong>⚠ Simulated fall detected</strong>
                <small>
                    BiGRU confidence:
                    ${(
                        sensorState.fallProbability *
                        100
                    ).toFixed(1)}%
                </small>
            </div>

            <span>
                ${new Date().toLocaleTimeString()}
            </span>
        `;

        eventList.prepend(event);

        console.log(
            "SIMULATED FALL EVENT",
            sensorState.fallProbability
        );

    }


    /* ---------------------------------------------------------
       UPDATE SYSTEM STATUS
    --------------------------------------------------------- */

    function updateSystemStatus() {

        const deviceStatus =
            document.getElementById(
                "deviceStatus"
            );

        const deviceDetail =
            document.getElementById(
                "deviceDetail"
            );

        const modelStatus =
            document.getElementById(
                "modelStatus"
            );

        const alertStatus =
            document.getElementById(
                "alertStatus"
            );

        const heroStatus =
            document.getElementById(
                "heroStatus"
            );


        if (deviceStatus) {
            deviceStatus.textContent =
                "Online";
        }

        if (deviceDetail) {
            deviceDetail.textContent =
                "Simulation stream active";
        }

        if (modelStatus) {
            modelStatus.textContent =
                "BiGRU simulation";
        }

        if (alertStatus) {
            alertStatus.textContent =
                "Simulation active";
        }

        if (heroStatus) {
            heroStatus.textContent =
                "Simulation active · sensor telemetry streaming";
        }

    }


    /* ---------------------------------------------------------
       INITIALIZATION
    --------------------------------------------------------- */

    function startGuardianSimulation() {

        console.log(
            "Starting Guardian sensor simulation..."
        );

        updateSystemStatus();

        updateHealthCards();

        updateAIUI();

        initializeGPSMap();

        updateGPSDisplay();


        /*
           GPS:
           Every 3 seconds
        */

        setInterval(
            updateGPS,
            3000
        );


        /*
           Health:
           Every 2 seconds
        */

        setInterval(
            simulateHeartRate,
            2000
        );

        setInterval(
            simulateSpO2,
            2500
        );


        /*
           Battery:
           Every 5 seconds
        */

        setInterval(
            simulateBattery,
            5000
        );


        /*
           Motion:
           500 ms
        */

        setInterval(
            simulateMotion,
            500
        );


        /*
           AI inference:
           Every 3 seconds
        */

        setInterval(
            simulateAI,
            3000
        );

    }


    /* ---------------------------------------------------------
       START AFTER PAGE LOAD
    --------------------------------------------------------- */

    if (
        document.readyState ===
        "loading"
    ) {

        document.addEventListener(
            "DOMContentLoaded",
            startGuardianSimulation
        );

    } else {

        startGuardianSimulation();

    }

})();
// ============================================================
// GUARDIAN SOS SYSTEM
// ============================================================

const sosButton = document.getElementById("sosButton");

if (sosButton) {

    sosButton.addEventListener("click", async function () {

        const confirmed = confirm(
            "ACTIVATE EMERGENCY SOS?\n\n" +
            "This will create an emergency event for Rajesh Kumar."
        );

        if (!confirmed) {
            return;
        }

        // Get current location if browser GPS is available
        navigator.geolocation.getCurrentPosition(

            async function(position) {

                const latitude = position.coords.latitude;
                const longitude = position.coords.longitude;

                await sendSOS(latitude, longitude);
            },

            async function(error) {

                console.log("GPS unavailable. Using last known location.");

                // Demo coordinates
                await sendSOS(
                    13.0827,
                    80.2707
                );
            }
        );
    });
}


// ============================================================
// SEND SOS TO FLASK
// ============================================================

async function sendSOS(latitude, longitude) {

    try {

        const response = await fetch("/api/sos", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                person_name: "Rajesh Kumar",

                latitude: latitude,

                longitude: longitude,

                timestamp: new Date().toISOString(),

                emergency_type: "MANUAL SOS"

            })
        });

        const data = await response.json();

        if (data.success) {

            activateSOSInterface(
                latitude,
                longitude
            );

        } else {

            alert("Unable to activate SOS.");

        }

    } catch (error) {

        console.error("SOS API error:", error);

        // Still activate UI for demonstration
        activateSOSInterface(
            latitude,
            longitude
        );
    }
}


// ============================================================
// UPDATE WEBSITE AFTER SOS
// ============================================================

function activateSOSInterface(latitude, longitude) {

    const status =
        document.getElementById("personStatus");

    const location =
        document.getElementById("profileLocation");

    const coordinates =
        document.getElementById("profileCoordinates");

    if (status) {

        status.innerText =
            "🚨 SOS ACTIVE";

    }

    if (location) {

        location.innerText =
            "Emergency location detected";

    }

    if (coordinates) {

        coordinates.innerText =
            latitude.toFixed(6) +
            ", " +
            longitude.toFixed(6);

    }

    // Change SOS button
    sosButton.classList.add("sos-active");

    sosButton.innerHTML = `
        <span class="sos-icon">!</span>
        <span>
            <strong>SOS ACTIVE</strong>
            <small>Emergency response initiated</small>
        </span>
    `;

    // Add event to alert history
    addSOSEvent(
        latitude,
        longitude
    );

    // Show notification
    alert(
        "🚨 EMERGENCY SOS ACTIVATED\n\n" +
        "Person: Rajesh Kumar\n" +
        "Location: " +
        latitude.toFixed(6) +
        ", " +
        longitude.toFixed(6)
    );
}


// ============================================================
// ADD EVENT TO ALERT HISTORY
// ============================================================

function addSOSEvent(latitude, longitude) {

    const eventList =
        document.getElementById("eventList");

    if (!eventList) {
        return;
    }

    const event = document.createElement("div");

    event.className = "guardian-sos-event";

    event.innerHTML = `

        <div class="sos-event-icon">!</div>

        <div class="sos-event-content">

            <strong>Emergency SOS Activated</strong>

            <span>
                Rajesh Kumar · Manual SOS
            </span>

            <small>
                Location: ${latitude.toFixed(6)},
                ${longitude.toFixed(6)}
            </small>

            <small>
                ${new Date().toLocaleString()}
            </small>

        </div>

        <span class="sos-event-status">
            ACTIVE
        </span>
    `;

    eventList.prepend(event);

    // Remove "no events" message
    const emptyEvents =
        document.querySelector(".empty-events");

    if (emptyEvents) {
        emptyEvents.style.display = "none";
    }
}