// Backend addresses used by the frontend
const API_BASE = "http://127.0.0.1:5051";
const API_ENDPOINT = `${API_BASE}/api/tryon`;

// Icons and default text used in the interface
const personIcon = `
  <svg viewBox="0 0 24 24">
    <path d="M20 21a8 8 0 0 0-16 0"></path>
    <circle cx="12" cy="7" r="4"></circle>
  </svg>
`;

const hangerIcon = `
  <svg viewBox="0 0 24 24">
    <path d="M12 7a2 2 0 1 0-2-2"></path>
    <path d="M12 7v3"></path>
    <path d="M5 20h14l-7-8z"></path>
  </svg>
`;

const resultPlaceholder = `
  <span>Result will appear here</span>
`;

// Brands shown in the recommendation section
const recommendedBrands = [
  {
    name: "FENGSYSTEM",
    display: "Fengsystem",
    url: "https://fengsystem.com"
  },
  {
    name: "JADEDLONDON",
    display: "Jaded",
    url: "https://jadedldn.com"
  },
  {
    name: "MISBHV",
    display: "misbhv",
    url: "https://misbhv.com"
  },
  {
    name: "SCULPTOR",
    display: "SCULPTOR.",
    url: "https://sculptor-worldwide.com"
  }
];

// Get the HTML elements that JavaScript needs
const personInput =
  document.getElementById("person-input");

const clothInput =
  document.getElementById("cloth-input");

const uploadPersonBtn =
  document.getElementById("upload-person-btn");

const uploadClothBtn =
  document.getElementById("upload-cloth-btn");

const clearPersonBtn =
  document.getElementById("clear-person-btn");

const clearClothBtn =
  document.getElementById("clear-cloth-btn");

const thumbPerson =
  document.getElementById("thumb-person");

const thumbCloth =
  document.getElementById("thumb-cloth");

const previewPersonBox =
  document.getElementById("preview-person-box");

const previewClothBox =
  document.getElementById("preview-cloth-box");

const resultDisplay =
  document.getElementById("result-display");

const tryBtn =
  document.getElementById("try-btn");

const saveBtn =
  document.getElementById("save-btn");

const downloadBtn =
  document.getElementById("download-btn");

const status =
  document.getElementById("status");

const storeLinkInput =
  document.getElementById("store-link");

const wardrobeGrid =
  document.getElementById("wardrobe-grid");

const brandsGrid =
  document.getElementById("brands-grid");

const guestAuth =
  document.getElementById("guest-auth");

const userAuth =
  document.getElementById("user-auth");

const currentUserEmail =
  document.getElementById("current-user-email");

const openLoginBtn =
  document.getElementById("open-login-btn");

const openRegisterBtn =
  document.getElementById("open-register-btn");

const logoutBtn =
  document.getElementById("logout-btn");

const authModal =
  document.getElementById("auth-modal");

const authModalBackdrop =
  document.getElementById("auth-modal-backdrop");

const authCloseBtn =
  document.getElementById("auth-close-btn");

const authForm =
  document.getElementById("auth-form");

const authTitle =
  document.getElementById("auth-title");

const authEmail =
  document.getElementById("auth-email");

const authPassword =
  document.getElementById("auth-password");

const authSubmitBtn =
  document.getElementById("auth-submit-btn");

const authSwitchBtn =
  document.getElementById("auth-switch-btn");

const authError =
  document.getElementById("auth-error");

// Current files, generated result and logged-in user
let personFile = null;
let clothFile = null;

let resultImageUrl = null;
let currentRequestId = null;

let currentUser = null;
let authMode = "login";

// Reset the result area
function resetResultPlaceholder() {
  resultDisplay.innerHTML = resultPlaceholder;
}

// Remove the selected person image
function clearPerson() {
  personFile = null;
  personInput.value = "";

  thumbPerson.innerHTML = personIcon;
  previewPersonBox.innerHTML = "";
}

// Remove the selected garment image
function clearCloth() {
  clothFile = null;
  clothInput.value = "";

  thumbCloth.innerHTML = hangerIcon;
  previewClothBox.innerHTML = "";
}

// FileReader is asynchronous, so it is wrapped in a Promise
function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      resolve(reader.result);
    };

    reader.onerror = () => {
      reject(
        new Error("Could not read the selected file.")
      );
    };

    reader.readAsDataURL(file);
  });
}

// Check the optional store link
function isValidUrl(value) {
  try {
    new URL(value);
    return true;
  } catch {
    return false;
  }
}

// Escape text before adding it to HTML
function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// Read a server response without blocking the page
async function readJsonResponse(response) {
  const contentType =
    response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    return response.json();
  }

  const text = await response.text();

  return {
    error: text || "Unexpected server response."
  };
}

// Show a preview after selecting the person image
personInput.addEventListener(
  "change",
  async (event) => {
    const file = event.target.files[0];

    if (!file) {
      return;
    }

    personFile = file;

    const url = await fileToDataURL(file);

    thumbPerson.innerHTML = `
      <img src="${url}" alt="Person">
    `;

    previewPersonBox.innerHTML = `
      <img src="${url}" alt="Person preview">
    `;
  }
);

// Show a preview after selecting the garment image
clothInput.addEventListener(
  "change",
  async (event) => {
    const file = event.target.files[0];

    if (!file) {
      return;
    }

    clothFile = file;

    const url = await fileToDataURL(file);

    thumbCloth.innerHTML = `
      <img src="${url}" alt="Garment">
    `;

    previewClothBox.innerHTML = `
      <img src="${url}" alt="Garment preview">
    `;
  }
);

uploadPersonBtn.addEventListener(
  "click",
  () => {
    personInput.click();
  }
);

uploadClothBtn.addEventListener(
  "click",
  () => {
    clothInput.click();
  }
);

clearPersonBtn.addEventListener(
  "click",
  clearPerson
);

clearClothBtn.addEventListener(
  "click",
  clearCloth
);

// Add drag and drop support to an upload area
function makeDrop(drop, input) {
  drop.addEventListener(
    "dragover",
    (event) => {
      event.preventDefault();
      drop.classList.add("dragover");
    }
  );

  drop.addEventListener(
    "dragleave",
    () => {
      drop.classList.remove("dragover");
    }
  );

  drop.addEventListener(
    "drop",
    (event) => {
      event.preventDefault();

      drop.classList.remove("dragover");

      const file =
        event.dataTransfer.files[0];

      if (!file) {
        return;
      }

      input.files =
        event.dataTransfer.files;

      input.dispatchEvent(
        new Event("change")
      );
    }
  );

  drop.addEventListener(
    "click",
    (event) => {
      if (event.target.closest("button")) {
        return;
      }

      input.click();
    }
  );
}

makeDrop(
  document.getElementById("drop-person"),
  personInput
);

makeDrop(
  document.getElementById("drop-cloth"),
  clothInput
);

// Send both images to the backend and wait for the generated result
tryBtn.addEventListener(
  "click",
  async () => {
    if (!personFile || !clothFile) {
      alert(
        "Please upload both person image and garment image."
      );

      return;
    }

    status.textContent = "Generating...";

    tryBtn.disabled = true;
    saveBtn.disabled = true;
    downloadBtn.disabled = true;

    currentRequestId = null;

    resultDisplay.innerHTML = `
      <span>Generating...</span>
    `;

    const formData = new FormData();

    formData.append(
      "person",
      personFile
    );

    formData.append(
      "cloth",
      clothFile
    );

    try {
      const response = await fetch(
        API_ENDPOINT,
        {
          method: "POST",
          credentials: "include",
          body: formData
        }
      );

      if (!response.ok) {
        const errorData =
          await readJsonResponse(response);

        throw new Error(
          errorData.error ||
          `Server error: ${response.status}`
        );
      }

      currentRequestId =
        response.headers.get("X-Request-ID");

      if (!currentRequestId) {
        throw new Error(
          "Server did not return a request ID."
        );
      }

      const blob =
        await response.blob();

      if (resultImageUrl) {
        URL.revokeObjectURL(
          resultImageUrl
        );
      }

      resultImageUrl =
        URL.createObjectURL(blob);

      resultDisplay.innerHTML = `
        <img
          class="result-image"
          src="${resultImageUrl}"
          alt="Generated try-on result"
        >
      `;

      status.textContent = "Completed";

      saveBtn.disabled = false;
      downloadBtn.disabled = false;

    } catch (error) {
      console.error(error);

      alert(
        "Error: " + error.message
      );

      status.textContent = "Error";

      resetResultPlaceholder();

    } finally {
      tryBtn.disabled = false;
    }
  }
);

// Download the current generated image
downloadBtn.addEventListener(
  "click",
  () => {
    if (!resultImageUrl) {
      return;
    }

    const link =
      document.createElement("a");

    link.href = resultImageUrl;

    link.download =
      "virtual-try-on-result.jpg";

    link.click();
  }
);

// Save button uses an asynchronous POST request
saveBtn.addEventListener(
  "click",
  async () => {
    if (!resultImageUrl || !currentRequestId) {
      alert(
        "Generate a try-on result first."
      );

      return;
    }

    if (!currentUser) {
      openAuthModal("login");

      authError.textContent =
        "Please log in before saving a result.";

      return;
    }

    const storeUrl =
      storeLinkInput.value.trim();

    if (
      storeUrl &&
      !isValidUrl(storeUrl)
    ) {
      alert(
        "Please enter a valid store link."
      );

      return;
    }

    saveBtn.disabled = true;
    status.textContent = "Saving...";

    try {
      const response = await fetch(
        `${API_BASE}/api/results`,
        {
          method: "POST",
          credentials: "include",

          headers: {
            "Content-Type":
              "application/json"
          },

          body: JSON.stringify({
            request_id:
              currentRequestId,

            store_url:
              storeUrl || null
          })
        }
      );

      const data =
        await readJsonResponse(response);

      if (!response.ok) {
        if (response.status === 401) {
          currentUser = null;

          renderAuthState();
          openAuthModal("login");
        }

        throw new Error(
          data.error ||
          "Could not save the result."
        );
      }

      status.textContent = "Saved";

      await loadWardrobe();

    } catch (error) {
      console.error(error);

      status.textContent =
        "Save error";

      alert(error.message);

    } finally {
      saveBtn.disabled = false;
    }
  }
);

// Load saved results of the current user from the backend
async function loadWardrobe() {
  if (!currentUser) {
    renderLoggedOutWardrobe();
    return;
  }

  wardrobeGrid.innerHTML = `
    <div class="empty">
      Loading saved results...
    </div>
  `;

  try {
    const response = await fetch(
      `${API_BASE}/api/results`,
      {
        method: "GET",
        credentials: "include"
      }
    );

    const data =
      await readJsonResponse(response);

    if (!response.ok) {
      if (response.status === 401) {
        currentUser = null;

        renderAuthState();
        renderLoggedOutWardrobe();

        return;
      }

      throw new Error(
        data.error ||
        "Could not load saved results."
      );
    }

    const items =
      Array.isArray(data)
        ? data
        : data.results || [];

    renderWardrobe(items);

  } catch (error) {
    console.error(error);

    wardrobeGrid.innerHTML = `
      <div class="empty">
        Could not load saved results
      </div>
    `;
  }
}

// Show a message when the user is not logged in
function renderLoggedOutWardrobe() {
  wardrobeGrid.innerHTML = `
    <div class="empty">
      Log in to view your saved results
    </div>
  `;
}

// Create wardrobe cards from the received data
function renderWardrobe(items) {
  if (!Array.isArray(items) || items.length === 0) {
    wardrobeGrid.innerHTML = `
      <div class="empty">
        Saved results appear here
      </div>
    `;

    return;
  }

  wardrobeGrid.innerHTML =
    items.map((item) => {
      const id =
        Number(item.id);

      const storeUrl =
        item.store_url || "";

      const imageUrl =
        `${API_BASE}/api/results/${id}/image`;

      let linkHtml = `
        <span>No store link</span>
      `;

      if (
        storeUrl &&
        isValidUrl(storeUrl)
      ) {
        linkHtml = `
          <a
            href="${escapeHtml(storeUrl)}"
            target="_blank"
            rel="noopener noreferrer"
          >
            Open link
          </a>
        `;
      }

      return `
        <div class="wardrobe-card">

          <img
            src="${imageUrl}"
            alt="Saved result"
          >

          <div class="wardrobe-info">

            <strong>
              Saved Look
            </strong>

            ${linkHtml}

            <button
              class="ghost"
              type="button"
              data-delete-result="${id}"
            >
              Delete
            </button>

          </div>
        </div>
      `;
    }).join("");
}

// Handle delete buttons inside the wardrobe
wardrobeGrid.addEventListener(
  "click",
  async (event) => {
    const deleteButton =
      event.target.closest(
        "[data-delete-result]"
      );

    if (!deleteButton) {
      return;
    }

    const resultId =
      deleteButton.dataset.deleteResult;

    const confirmed =
      window.confirm(
        "Delete this saved result?"
      );

    if (!confirmed) {
      return;
    }

    deleteButton.disabled = true;

    try {
      const response = await fetch(
        `${API_BASE}/api/results/${resultId}`,
        {
          method: "DELETE",
          credentials: "include"
        }
      );

      const data =
        await readJsonResponse(response);

      if (!response.ok) {
        throw new Error(
          data.error ||
          "Could not delete the result."
        );
      }

      await loadWardrobe();

    } catch (error) {
      console.error(error);

      deleteButton.disabled = false;

      alert(error.message);
    }
  }
);

// Create cards for recommended brands
function renderBrands() {
  brandsGrid.innerHTML =
    recommendedBrands.map(
      (brand) => {
        return `
          <div class="brand-card">

            <div class="brand-preview">
              ${escapeHtml(brand.display)}
            </div>

            <div class="brand-info">

              <strong>
                ${escapeHtml(brand.name)}
              </strong>

              <a
                href="${escapeHtml(brand.url)}"
                target="_blank"
                rel="noopener noreferrer"
              >
                Open store
              </a>

            </div>
          </div>
        `;
      }
    ).join("");
}

// Open the login or registration form
function openAuthModal(mode) {
  authMode = mode;

  authError.textContent = "";
  authForm.reset();

  if (mode === "register") {
    authTitle.textContent =
      "Register";

    authSubmitBtn.textContent =
      "Register";

    authSwitchBtn.textContent =
      "Already have an account? Log in";

    authPassword.autocomplete =
      "new-password";

  } else {
    authTitle.textContent =
      "Log in";

    authSubmitBtn.textContent =
      "Log in";

    authSwitchBtn.textContent =
      "Create an account";

    authPassword.autocomplete =
      "current-password";
  }

  authModal.hidden = false;

  setTimeout(
    () => authEmail.focus(),
    0
  );
}

// Close the authentication window
function closeAuthModal() {
  authModal.hidden = true;

  authError.textContent = "";
  authForm.reset();
}

// Update the header depending on login state
function renderAuthState() {
  const authenticated =
    Boolean(currentUser);

  guestAuth.hidden =
    authenticated;

  userAuth.hidden =
    !authenticated;

  currentUserEmail.textContent =
    authenticated
      ? currentUser.email
      : "";
}

// Ask the backend if a user session already exists
async function checkCurrentUser() {
  try {
    const response = await fetch(
      `${API_BASE}/api/me`,
      {
        method: "GET",
        credentials: "include"
      }
    );

    const data =
      await readJsonResponse(response);

    if (!response.ok) {
      throw new Error(
        data.error ||
        "Could not check authentication."
      );
    }

    currentUser =
      data.authenticated
        ? data.user
        : null;

  } catch (error) {
    console.error(error);

    currentUser = null;
  }

  renderAuthState();

  if (currentUser) {
    await loadWardrobe();
  } else {
    renderLoggedOutWardrobe();
  }
}

openLoginBtn.addEventListener(
  "click",
  () => {
    openAuthModal("login");
  }
);

openRegisterBtn.addEventListener(
  "click",
  () => {
    openAuthModal("register");
  }
);

authCloseBtn.addEventListener(
  "click",
  closeAuthModal
);

authModalBackdrop.addEventListener(
  "click",
  closeAuthModal
);

authSwitchBtn.addEventListener(
  "click",
  () => {
    if (authMode === "login") {
      openAuthModal("register");
    } else {
      openAuthModal("login");
    }
  }
);

document.addEventListener(
  "keydown",
  (event) => {
    if (
      event.key === "Escape" &&
      !authModal.hidden
    ) {
      closeAuthModal();
    }
  }
);

// Send login or registration data to the backend
authForm.addEventListener(
  "submit",
  async (event) => {
    event.preventDefault();

    const email =
      authEmail.value.trim();

    const password =
      authPassword.value;

    authError.textContent = "";
    authSubmitBtn.disabled = true;

    try {
      const endpoint =
        authMode === "register"
          ? "/api/register"
          : "/api/login";

      const response = await fetch(
        `${API_BASE}${endpoint}`,
        {
          method: "POST",
          credentials: "include",

          headers: {
            "Content-Type":
              "application/json"
          },

          body: JSON.stringify({
            email,
            password
          })
        }
      );

      const data =
        await readJsonResponse(response);

      if (!response.ok) {
        throw new Error(
          data.error ||
          "Authentication failed."
        );
      }

      currentUser =
        data.user;

      renderAuthState();
      closeAuthModal();

      await loadWardrobe();

    } catch (error) {
      authError.textContent =
        error.message;

    } finally {
      authSubmitBtn.disabled =
        false;
    }
  }
);

// End the current user session
logoutBtn.addEventListener(
  "click",
  async () => {
    try {
      const response = await fetch(
        `${API_BASE}/api/logout`,
        {
          method: "POST",
          credentials: "include"
        }
      );

      const data =
        await readJsonResponse(response);

      if (!response.ok) {
        throw new Error(
          data.error ||
          "Logout failed."
        );
      }

      currentUser = null;
      currentRequestId = null;

      renderAuthState();
      renderLoggedOutWardrobe();

    } catch (error) {
      alert(error.message);
    }
  }
);

// Load static and user data when the page opens
renderBrands();
checkCurrentUser();