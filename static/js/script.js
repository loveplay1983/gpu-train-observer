document.addEventListener("DOMContentLoaded", () => {
  const statusText = document.getElementById("status-text");
  const lastUpdate = document.getElementById("last-update");
  const alertsList = document.getElementById("alerts-list");
  const tableBody = document.getElementById("gpu-table-body");

  const startBtn = document.getElementById("start-btn");
  const stopBtn = document.getElementById("stop-btn");
  const alertModal = document.getElementById("alert-modal");
  const modalBody = document.getElementById("modal-body");
  const closeModal = document.getElementById("close-modal");

  // --- Modal controls ---
  function showModal(message) {
    modalBody.textContent = message;
    alertModal.style.display = "block";
    setTimeout(hideModal, 20000);
  }
  function hideModal() {
    alertModal.style.display = "none";
  }
  closeModal.onclick = hideModal;
  window.onclick = (e) => { if (e.target === alertModal) hideModal(); };

  // --- Dynamic button updates ---
  startBtn.addEventListener("click", () => {
    startBtn.textContent = "Started";
    stopBtn.textContent = "Stop";
    showModal("Monitoring started!");
  });

  stopBtn.addEventListener("click", () => {
    stopBtn.textContent = "Stopped";
    startBtn.textContent = "Start";
    showModal("Monitoring stopped.");
  });

  // --- Periodic status refresh ---
  async function refreshStatus() {
    try {
      const response = await fetch("/status_json");
      if (!response.ok) throw new Error("Network response was not ok");
      const data = await response.json();

      // Update status text
      statusText.textContent = data.running ? "Running" : "Stopped";

      // Update alerts
      alertsList.innerHTML = "";
      if (data.alerts && data.alerts.length > 0) {
        data.alerts.forEach(a => {
          const li = document.createElement("li");
          li.textContent = a;
          alertsList.appendChild(li);
        });
        showModal("⚠️ Alert detected:\n" + data.alerts.join("\n"));
      } else {
        const li = document.createElement("li");
        li.textContent = "No active alerts.";
        li.style.color = "#777";
        alertsList.appendChild(li);
      }

      // Update GPU table
      tableBody.innerHTML = "";
      if (data.last_sample && data.last_sample.length > 0) {
        data.last_sample.forEach(r => {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td>${r.gpu_index}</td>
            <td>${r.gpu_name}</td>
            <td>${r.nvml_memory_used_MB}</td>
            <td>${r.nvml_memory_total_MB}</td>
            <td>${r["nvml_gpu_util_%"]}</td>
            <td>${r["nvml_mem_util_%"]}</td>
            <td>${r.nvml_temperature_C ?? ""}</td>
            <td>${r.pytorch_allocated_MB ?? ""}</td>
            <td>${r.pytorch_reserved_MB ?? ""}</td>
            <td>${r.pytorch_fragmentation ?? ""}</td>
            <td>${r.cpu_percent ?? ""}</td>
            <td>${r.ram_used_GB ?? ""}</td>
            <td>${r.ram_total_GB ?? ""}</td>
            <td>${r.alerts ?? ""}</td>
          `;
          tableBody.appendChild(tr);
        });
      } else {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="14">No data yet.</td>`;
        tableBody.appendChild(tr);
      }

      // Update last updated time
      const now = new Date().toLocaleTimeString();
      lastUpdate.textContent = now;

    } catch (err) {
      console.error("Refresh error:", err);
    }
  }

  refreshStatus();
  setInterval(refreshStatus, 3000);
});
