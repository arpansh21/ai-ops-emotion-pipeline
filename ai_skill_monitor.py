import argparse
import json
import os
from pathlib import Path

AI_KEYWORDS = [
    'agent', 'skill', 'workflow', 'prompt', 'llm', 'chat', 'reasoning',
    'embedding', 'tool', 'vector', 'chain', 'adapter', 'planning',
    'policy', 'monitor', 'autosuggest', 'agent', 'plugin'
]

ISSUE_RULES = [
    'error', 'exception', 'failed', 'timeout', 'unavailable', 'crash',
    'deprecated', 'unsupported', 'warning', 'invalid', 'mismatch'
]


def scan_file(path: Path):
    with path.open('r', encoding='utf-8', errors='ignore') as handle:
        text = handle.read().lower()

    ai_score = sum(keyword in text for keyword in AI_KEYWORDS)
    issues = [rule for rule in ISSUE_RULES if rule in text]
    return ai_score, sorted(set(issues))


def scan_directory(root: Path):
    report = []
    for path in root.rglob('*'):
        if path.is_file() and path.suffix in {'.py', '.md', '.yml', '.yaml', '.json', '.txt'}:
            ai_score, issues = scan_file(path)
            if ai_score or issues:
                report.append({
                    'path': str(path.relative_to(root)),
                    'ai_score': ai_score,
                    'issues': issues,
                })
    return sorted(report, key=lambda item: (-item['ai_score'], item['path']))


def build_markdown_report(report):
    lines = [
        '# AI Skill Monitor Report',
        '',
        'This report identifies files with AI-related content and potential issues.',
        '',
        '| File | AI Keyword Score | Issues Found |',
        '|---|---|---|',
    ]

    if not report:
        lines.append('| _none detected_ | 0 | _none_ |')
    else:
        for item in report:
            issues = ', '.join(item['issues']) if item['issues'] else 'None'
            lines.append(f"| {item['path']} | {item['ai_score']} | {issues} |")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='AI Skill workflow analyzer')
    parser.add_argument('--root', default='.', help='Repository root path')
    parser.add_argument('--output', default='ai_issue_report.md', help='Report file name')
    parser.add_argument('--fail-on-issue', action='store_true', help='Fail when issues are detected')
    args = parser.parse_args()

    root = Path(args.root)
    report = scan_directory(root)
    markdown = build_markdown_report(report)
    with open(root / args.output, 'w', encoding='utf-8') as handle:
        handle.write(markdown)

    print(markdown)
    if args.fail_on_issue and any(item['issues'] for item in report):
        raise SystemExit('Detected AI-related issues in repository')


if __name__ == '__main__':
    main()
