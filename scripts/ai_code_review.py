"""
AI Code Review Script for CI Pipeline.

Uses OpenAI's GPT-5-mini to review code changes in pull requests
and posts the review as a comment on the PR via the GitHub API.
"""

import json
import os
import sys

import requests
from openai import OpenAI

REVIEW_PROMPT = """You are an expert code reviewer. Analyze the following code diff \
from a pull request and provide a constructive review.

The following tools have ALREADY been executed successfully in CI:
- mypy (type checking)
- black (formatting)
- pylint (linting & style)
- bandit (security static analysis)
- pytest (unit tests)
- pytest with coverage enforcement

DO NOT:
- Comment on formatting
- Comment on type annotations completeness
- Report linting/style issues
- Report common static security warnings
- Suggest adding docstrings unless critical
- Suggest adding basic tests
- Suggest increasing coverage
- Repeat what these tools already validate

Instead, focus ONLY on:
- Logical bugs
- Hidden edge cases
- Architectural concerns
- Real-world security risks beyond static scanning
- Performance risks
- Concurrency issues
- Data consistency issues
- Transaction boundaries
- Misuse of frameworks/libraries
- API contract breaks
- Maintainability risks at scale

Please provide:

1. 🚨 Critical Issues
   - Bugs
   - Security vulnerabilities
   - Data corruption risks
   - Breaking changes

2. ⚠️ Major Concerns
   - Performance problems
   - Architectural violations
   - Scalability risks
   - Risky assumptions

3. 🧠 Design & Maintainability Risks
   - Tight coupling
   - Hidden side effects
   - Poor separation of concerns
   - Long-term maintainability issues

4. 🧪 Test Logic Gaps (NOT coverage issues)
   - Missing edge-case logic validation
   - Incorrect assertions
   - Flaky test risk

5. ✅ Positive Observations (if any)

Rules:
- If no issues are found in a category, explicitly state: "No issues found."
- Reference file names and functions where possible.
- Be concise but specific.
- Do NOT rewrite the entire code.
- Do NOT restate the diff.
- Do NOT provide generic best practices.

Respond in well-structured Markdown suitable for a GitHub comment.

Here is the diff:

```diff
{diff}
```
"""


def get_pr_diff(repo: str, pr_number: str, token: str) -> str:
    """Fetch the diff for a pull request from the GitHub API."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.diff",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def review_diff(diff: str, openai_api_key: str) -> str:
    """Send the diff to GPT-5-mini and return the review."""
    client = OpenAI(api_key=openai_api_key)

    # Truncate very large diffs to stay within token limits
    max_diff_chars = 60000
    if len(diff) > max_diff_chars:
        diff = diff[:max_diff_chars] + "\n\n... (diff truncated due to size)"

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a senior software engineer performing a strict, production-level code review. 
                Your role is to identify high-value issues that automated linters and static analysis tools typically miss. 
                Be precise, technical, and concise. Do not explain basic programming concepts. Do not provide generic advice. 
                Focus only on meaningful, non-trivial issues.""",
            },
            {"role": "user", "content": REVIEW_PROMPT.format(diff=diff)},
        ],
        max_completion_tokens=4096,
    )

    choice = response.choices[0]
    print(f"📊 API response - finish_reason: {choice.finish_reason}")
    print(f"📊 Content present: {choice.message.content is not None}")
    if hasattr(choice.message, "refusal") and choice.message.refusal:
        print(f"⚠️ Model refusal: {choice.message.refusal}")

    content = choice.message.content
    if not content or not content.strip():
        return (
            "⚠️ The AI model returned an empty review. This may be a transient issue."
        )
    return content


def post_pr_comment(repo: str, pr_number: str, token: str, body: str) -> None:
    """Post a comment on the pull request with the review results."""
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {"body": body}
    response = requests.post(url, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    print(f"✅ Review comment posted: {response.json().get('html_url')}")


def main() -> None:
    """Run the AI code review pipeline."""
    # Read environment variables
    github_token = os.environ.get("GITHUB_TOKEN")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    github_event_path = os.environ.get("GITHUB_EVENT_PATH")
    github_repository = os.environ.get("GITHUB_REPOSITORY")

    if not all([github_token, openai_api_key, github_event_path, github_repository]):
        print("❌ Missing required environment variables.")
        sys.exit(1)

    # Assert to narrow types for mypy after the None check above
    assert github_token is not None
    assert openai_api_key is not None
    assert github_event_path is not None
    assert github_repository is not None

    # Parse the PR number from the GitHub event payload
    with open(github_event_path, "r", encoding="utf-8") as f:
        event = json.load(f)

    pr_number = str(event["pull_request"]["number"])
    print(f"📋 Reviewing PR #{pr_number} in {github_repository}")

    # Step 1: Get the diff
    print("📥 Fetching PR diff...")
    diff = get_pr_diff(github_repository, pr_number, github_token)

    if not diff.strip():
        print("ℹ️  No changes found in the PR diff. Skipping review.")
        return

    # Step 2: Run AI review
    print("🤖 Running AI code review with GPT-5-mini...")
    review = review_diff(diff, openai_api_key)

    # Step 3: Post the review as a PR comment
    comment_body = (
        "## 🤖 AI Code Review (GPT-5-mini)\n\n"
        f"{review}\n\n"
        "---\n"
        "*This review was generated automatically by the AI Code Review CI job. "
        "It is advisory only and does not block the pipeline.*"
    )

    print("💬 Posting review comment on PR...")
    post_pr_comment(github_repository, pr_number, github_token, comment_body)
    print("✅ AI code review completed successfully.")


if __name__ == "__main__":
    main()
