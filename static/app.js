"use strict";

const MAX_MESSAGE_LENGTH = 2000;
const FIELD_DEFINITIONS = [
  { path: "full_name", label: "Full name" },
  { path: "home_address", label: "Home address" },
  { path: "covers_worldwide_assets", label: "Worldwide assets" },
  { path: "has_children", label: "Has children" },
  { path: "children_names", label: "Children's names" },
  { path: "executor.name", label: "Executor name" },
  { path: "executor.relationship", label: "Executor relationship" },
  { path: "gifts", label: "Specific gifts" },
  { path: "additional_wishes", label: "Additional wishes" },
];

const elements = {
  chat: document.getElementById("chat-messages"),
  count: document.getElementById("character-count"),
  document: document.getElementById("draft-document"),
  error: document.getElementById("error-banner"),
  fields: document.getElementById("collected-fields"),
  form: document.getElementById("message-form"),
  input: document.getElementById("message-input"),
  newConversation: document.getElementById("new-conversation"),
  send: document.getElementById("send-button"),
  thinking: document.getElementById("thinking-indicator"),
  warnings: document.getElementById("warning-list"),
};

let sessionId = null;
let busy = false;

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers,
    },
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload?.error?.message || "The request could not be completed.");
  }
  return payload;
}

function createSession() {
  return requestJson("/api/sessions", { method: "POST" });
}

function sendMessage(id, message) {
  return requestJson(`/api/sessions/${encodeURIComponent(id)}/messages`, {
    method: "POST",
    body: JSON.stringify({ message }),
  });
}

function addChatMessage(role, content) {
  const message = document.createElement("article");
  message.className = `chat-message chat-message-${role}`;

  const speaker = document.createElement("span");
  speaker.className = "message-speaker";
  speaker.textContent = role === "user" ? "You" : "Assistant";
  const body = document.createElement("p");
  body.className = "message-content";
  body.textContent = content;

  message.append(speaker, body);
  elements.chat.append(message);
  elements.chat.scrollTop = elements.chat.scrollHeight;
}

function valueForDisplay(field) {
  if (!field || field.value === null || field.value === undefined) {
    return "-";
  }
  if (typeof field.value === "boolean") {
    return field.value ? "Yes" : "No";
  }
  if (Array.isArray(field.value)) {
    return field.value.length ? field.value.join(", ") : "None";
  }
  return String(field.value);
}

function getTrackedField(state, path) {
  return path.split(".").reduce((value, key) => value?.[key], state);
}

function renderCollectedInformation(state) {
  elements.fields.replaceChildren();
  for (const definition of FIELD_DEFINITIONS) {
    const field = getTrackedField(state, definition.path);
    const item = document.createElement("li");
    item.className = "collected-field";

    const details = document.createElement("div");
    details.className = "field-details";
    const label = document.createElement("span");
    label.className = "field-label";
    label.textContent = definition.label;
    const value = document.createElement("span");
    value.className = "field-value";
    value.textContent = valueForDisplay(field);
    details.append(label, value);

    const status = document.createElement("span");
    const statusName = field?.status || "missing";
    status.className = `status-badge status-${statusName}`;
    status.textContent = statusName;
    item.append(details, status);
    elements.fields.append(item);
  }
}

function renderResponse(response) {
  addChatMessage("assistant", response.assistant_message);
  renderCollectedInformation(response.state);
  elements.document.textContent = response.document;
  renderWarnings(response.warnings || []);
}

function renderWarnings(warnings) {
  elements.warnings.replaceChildren();
  for (const warning of warnings) {
    const note = document.createElement("div");
    note.className = "warning-note";
    const text = document.createElement("p");
    text.textContent = warning;
    const dismiss = document.createElement("button");
    dismiss.className = "dismiss-warning";
    dismiss.type = "button";
    dismiss.setAttribute("aria-label", "Dismiss warning");
    dismiss.textContent = "Dismiss";
    dismiss.addEventListener("click", () => note.remove());
    note.append(text, dismiss);
    elements.warnings.append(note);
  }
}

function showError(message) {
  elements.error.textContent = message;
  elements.error.hidden = false;
}

function clearError() {
  elements.error.textContent = "";
  elements.error.hidden = true;
}

function updateCharacterCount() {
  elements.count.textContent = `${elements.input.value.length} / ${MAX_MESSAGE_LENGTH}`;
}

function setBusy(isBusy) {
  busy = isBusy;
  elements.input.disabled = isBusy || !sessionId;
  elements.send.disabled = isBusy || !sessionId;
  elements.newConversation.disabled = isBusy;
  elements.thinking.hidden = !isBusy;
}

async function startNewConversation() {
  clearError();
  setBusy(true);
  sessionId = null;
  elements.chat.replaceChildren();
  elements.warnings.replaceChildren();
  elements.fields.replaceChildren();
  elements.document.textContent = "";
  elements.input.value = "";
  updateCharacterCount();

  try {
    const response = await createSession();
    sessionId = response.session_id;
    renderResponse(response);
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
    if (sessionId) {
      elements.input.focus();
    }
  }
}

async function handleSubmit(event) {
  event.preventDefault();
  if (busy || !sessionId) {
    return;
  }
  const message = elements.input.value;
  if (!message.trim()) {
    return;
  }

  clearError();
  elements.warnings.replaceChildren();
  addChatMessage("user", message);
  setBusy(true);
  try {
    const response = await sendMessage(sessionId, message);
    elements.input.value = "";
    updateCharacterCount();
    renderResponse(response);
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
    elements.input.focus();
  }
}

elements.form.addEventListener("submit", handleSubmit);
elements.input.addEventListener("input", updateCharacterCount);
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});
elements.newConversation.addEventListener("click", startNewConversation);

renderCollectedInformation({});
startNewConversation();