# AI Log

## Implemented stages

1. [Stage 0] Analysed the project brief and defined the overall architecture.
   We chose a lightweight Python app with no RAG layer, no framework-heavy setup,
   in-memory session state, and a mock provider plus a Gemini provider path.

2. [Stage 1] Designed the state schema in `app/schemas.py`.
   We introduced per-field status tracking (`missing`, `unconfirmed`, `confirmed`)
   and kept LLM updates as plain strings to avoid one bad field update invalidating
   the whole response.

3. [Stage 2] Designed the deterministic document renderer in `app/document.py`.
   The document is not LLM-written; only confirmed values are rendered, while
   unconfirmed entries show `[Awaiting confirmation]`, and confirmed empty lists show
   `none` as a real value, not as missing data.

4. [Stage 3] Implemented validation rules for raw LLM output.
   The code validates proposed edits before applying them, converting model results
   into safe, typed updates with explicit warnings and conflict handling.

5. [Stage 4] Built the state manager and transition logic.
   The app validates proposed state changes, applies only safe updates, and ensures
   the session state follows consistent rules without letting invalid data leak in.

6. [Stage 5] Added the service layer for conversation orchestration.
   The service loads the session, decides what still needs attention, builds prompts,
   validates provider responses, and saves state only after a successful turn.

7. [Stage 6] Implemented the LLM abstraction and provider clients.
   We created a clean interface, a deterministic mock implementation for local testing,
   and a Gemini adapter for real provider integration.

8. [Stage 7] Added session management and persistence support in the app layer.
   Sessions are created, loaded, and updated in a structured way so the conversation
   can progress through multiple user turns without losing state.

9. [Stage 8] Built the API and frontend integration.
   We added the JSON API for creating sessions, sending messages, fetching state, and
   checking health, along with a simple browser UI served from `static/`.

10. [Stage 9] Completed testing, verification, and documentation.
    We covered schemas, validation, prompts, mock/Gemini behavior, service logic,
    scenario flows, and API-level checks with pytest, ensuring the main flow works
    without calling real external LLM services.

## Remaining steps / Future Improvements

- Add persistent storage so sessions survive restarts.
- Add authentication and per-user session handling.
- Add rate limiting, logging, and privacy-safe observability.
- Strengthen provider error handling and configuration validation.
- Add CI/CD and deployment automation.
- Review the document content for legal/production readiness beyond the demo context.

