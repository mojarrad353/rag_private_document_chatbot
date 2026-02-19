#!/usr/bin/env python3
"""
AI-powered code review script using OpenAI.

This script reviews changed Python files in a PR and posts
feedback as a GitHub PR comment. It is designed to be run
as a non-blocking CI step.
"""

import json
import os
import subprocess
import sys

from openai import OpenAI

REVIEW_PROMPT = """You are an expert Python code reviewer. Analyze the following \
code changes and provide a concise, actionable review.

Focus on:
1. **Bugs & Logic Errors** - Potential bugs, edge cases, or incorrect logic
2. **Security** - Vulnerabilities, unsafe patterns, or data exposure risks
3. **Performance** - Inefficiencies, unnecessary computations, or memory issues
4. **Readability** - Unclear naming, missing docstrings, or confusing structure
5. **Best Practices** - Python idioms, design patterns, and maintainability

Rules:
- Only comment on genuine issues or meaningful improvements.
- Do NOT repeat what linters (pylint, mypy, black, bandit) already catch.
- Be concise. Use bullet points.
- If the code looks good, say so briefly.
- Rate overall quality: ⭐ (needs work) to ⭐⭐⭐⭐⭐ (excellent).

File: {filename}
```python
{code}
```
"""


def get_changed_files() -> list[str]:
    """Get list of changed Python files compared to the base branch."""
    base_ref = os.getenv("GITHUB_BASE_REF", "main")
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"origin/{base_ref}...HEAD", "--", "*.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        # Filter to only existing files (exclude deleted)
        return [f for f in files if os.path.exists(f)]
    except subprocess.CalledProcessError:
        print("Warning: Could not get diff, falling back to all Python files in src/")
        result = subprocess.run(
            ["find", "src", "-name", "*.py", "-type", "f"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]


def review_file(client: OpenAI, filepath: str, model: str) -> str:
    """Send a file to OpenAI for code review."""
    with open(filepath, encoding="utf-8") as f:
        code = f.read()

    # Skip very small files (e.g., __init__.py)
    if len(code.strip()) < 10:
        return ""

    prompt = REVIEW_PROMPT.format(filename=filepath, code=code)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content or ""


def post_github_comment(body: str) -> None:
    """Post a comment on the GitHub PR."""
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    pr_number = os.getenv("PR_NUMBER")

    if not all([token, repo, pr_number]):
        print("Missing GitHub context. Printing review to stdout instead.")
        print(body)
        return

    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = json.dumps({"body": body})

    subprocess.run(
        [
            "curl",
            "-s",
            "-X",
            "POST",
            "-H",
            f"Authorization: {headers['Authorization']}",
            "-H",
            f"Accept: {headers['Accept']}",
            "-H",
            "Content-Type: application/json",
            "-d",
            data,
            url,
        ],
        check=True,
    )
    print(f"Posted review comment to PR #{pr_number}")


def main() -> None:
    """Main entry point for the AI code review."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Skipping AI review.")
        sys.exit(0)  # Exit 0 so CI doesn't fail

    model = os.getenv("REVIEW_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    changed_files = get_changed_files()
    if not changed_files:
        print("No changed Python files found. Skipping review.")
        sys.exit(0)

    print(f"Reviewing {len(changed_files)} file(s): {changed_files}")

    reviews: list[str] = []
    for filepath in changed_files:
        print(f"  Reviewing: {filepath}")
        review = review_file(client, filepath, model)
        if review:
            reviews.append(f"### 📄 `{filepath}`\n\n{review}")

    if not reviews:
        print("No review comments generated.")
        sys.exit(0)

    # Build the full comment
    header = "## 🤖 AI Code Review\n\n"
    header += "_Automated review powered by OpenAI. "
    header += "This is advisory only and does not affect CI status._\n\n---\n\n"
    full_review = header + "\n\n---\n\n".join(reviews)

    # Post or print
    post_github_comment(full_review)


if __name__ == "__main__":
    main()
