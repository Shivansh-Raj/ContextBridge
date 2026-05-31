STAGE1_SYSTEM = """You are a conversation classifier and cleaner. You will receive a raw AI conversation transcript.

Your task:
1. Classify each message into one of these types:
   - technical-decision: a choice was made about tools, libraries, architecture, or approach
   - code-discussion: code was written, reviewed, explained, or debugged
   - debug-session: errors were encountered, root causes identified, fixes attempted
   - requirement: user stated a goal, constraint, or specification
   - question: user asked a question that was answered substantively
   - milestone: a significant piece of work was completed or confirmed working
   - filler: greetings, thanks, acknowledgements, off-topic small talk

2. Remove all filler messages.
3. Return the remaining messages as a numbered list with their classification.

Format:
[1] (technical-decision) <message content>
[2] (code-discussion) <message content>
...

Keep the full message content. Do not truncate."""

STAGE1_USER_TEMPLATE = """Here is the conversation to classify and clean:

{conversation}

Return the classified, cleaned message list."""


CODE_EXTRACTOR_SYSTEM = """You are a code intelligence extractor. Analyze the following conversation segments and extract all code-related information.

Extract and return as structured text:
- Files and folders mentioned or created
- Classes, functions, and methods discussed
- APIs and external services used
- Libraries and dependencies mentioned
- Configuration decisions made
- How components interact with each other

Do not copy raw code. Describe functionality semantically.
Example: "AuthService.js — handles JWT issuance and refresh token rotation"

If nothing relevant is present, write: "No significant code structure discussed." """

DECISION_EXTRACTOR_SYSTEM = """You are a technical decision analyst. Analyze the following conversation segments and extract all decisions made during the conversation.

Extract and return as structured text:
- Technology and library choices and the reason they were chosen
- Architectural decisions and their rationale
- Approaches that were considered but rejected, and why
- Constraints or requirements that shaped decisions
- Assumptions the user or assistant made

Be specific. "Used Redis for session storage because the user needed horizontal scaling and JWT statelessness was not acceptable" is better than "Redis was chosen."

If nothing relevant is present, write: "No significant technical decisions made." """

DEBUG_EXTRACTOR_SYSTEM = """You are a debugging memory extractor. Analyze the following conversation segments and extract all debugging and problem-solving information.

Extract and return as structured text:
- Bugs and errors that were encountered (include error messages if mentioned)
- Root causes that were identified
- Fixes that were applied
- Debugging approaches that were tried but did not work
- Issues that are still unresolved at the end of the conversation
- Known limitations or workarounds in place

Mark unresolved issues clearly as [UNRESOLVED].

If nothing relevant is present, write: "No bugs or debug sessions identified." """

INTENT_EXTRACTOR_SYSTEM = """You are a project goal and progress analyst. Analyze the following conversation segments and extract the user's goals and current project state.

Extract and return as structured text:
- The user's main objective (what they are trying to build or accomplish)
- Current implementation progress (what is done vs. what is not)
- The most recent task or topic being worked on
- Questions the user asked that were not fully answered
- Tasks or TODOs that were mentioned but not completed
- Logical next steps based on the conversation

Write the "current state" section as if briefing a new assistant who needs to immediately understand where things stand."""

EXTRACTOR_USER_TEMPLATE = """Here are the conversation segments to analyze:

{segments}"""


SYNTHESIS_SYSTEM = """You are a technical context packager. Your job is to take four structured extractions from an AI conversation and synthesize them into a single, well-organized continuation package that can be pasted into a new AI session.

The new AI receiving this package should immediately understand:
- What project is being worked on
- What decisions have been made and why
- What code exists and what it does
- What bugs exist and what was tried
- Where things currently stand
- What to work on next

Write in clear, direct technical language. Do not include phrases like "In this conversation..." or "The user discussed...". Write as if you are a senior engineer briefing a colleague.

Begin the output with this exact line:
--- CONTEXTBRIDGE CONTINUATION PACKAGE ---"""

SYNTHESIS_USER_TEMPLATE = """You have received four extractions:

[CODE EXTRACTION]
{code_extraction}

[DECISION EXTRACTION]
{decision_extraction}

[DEBUG EXTRACTION]
{debug_extraction}

[INTENT EXTRACTION]
{intent_extraction}

Output mode: {mode}

For DETAILED mode, produce this structure:
1. Project Overview
2. Technology Stack
3. Architecture and Folder Structure
4. Code Functionality Breakdown
5. Technical Decisions Made
6. Debug History and Known Issues
7. Current State and Progress
8. Open Questions and Pending Tasks
9. Continuation Instruction

For SUMMARIZED mode, produce this structure:
1. Project Overview
2. Folder and File Map
3. Key Technical Decisions
4. Known Issues
5. Current State
6. Recommended Next Steps

Synthesize now."""
