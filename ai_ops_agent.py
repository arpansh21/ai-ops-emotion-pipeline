from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

ISSUE_KEYWORDS = [
    'error', 'exception', 'failed', 'timeout', 'unavailable', 'crash',
    'deprecated', 'unsupported', 'warning', 'invalid', 'mismatch', 'todo',
    'fixme', 'stale', 'drift', 'bias'
]

AI_KEYWORDS = [
    'agent', 'skill', 'workflow', 'prompt', 'llm', 'chat', 'reasoning',
    'embedding', 'tool', 'vector', 'chain', 'adapter', 'planning',
    'policy', 'monitor', 'autosuggest', 'plugin', 'monitoring'
]


@dataclass
class SkillResult:
    name: str
    description: str
    findings: List[str]
    recommendations: List[str]
    score: int


class BaseSkill:
    name = 'base'
    description = ''

    def run(self, root: Path) -> SkillResult:
        raise NotImplementedError


class DataQualitySkill(BaseSkill):
    name = 'Data Quality Skill'
    description = 'Evaluate training data structure, label balance, and class coverage.'

    def run(self, root: Path) -> SkillResult:
        findings: List[str] = []
        recommendations: List[str] = []
        score = 0

        train_dir = root / 'train'
        if not train_dir.exists() or not train_dir.is_dir():
            findings.append('Training dataset directory `train/` is missing.')
            recommendations.append('Add training data under `train/<class>/...` to support model health.')
            return SkillResult(self.name, self.description, findings, recommendations, score)

        classes = [d for d in train_dir.iterdir() if d.is_dir()]
        if not classes:
            findings.append('No labeled class directories found under `train/`.')
            recommendations.append('Create class folders and place representative examples in each class.')
            return SkillResult(self.name, self.description, findings, recommendations, score)

        counts = {cls.name: sum(1 for _ in cls.glob('*') if _.is_file()) for cls in classes}

        if len(classes) < 5:
            findings.append(f'Only {len(classes)} classes detected; AI emotion tasks usually need 5+ classes.')
            recommendations.append('Add missing emotion categories under `train/` to prevent class gaps.')
            score -= 2

        empty_classes = [name for name, count in counts.items() if count == 0]
        if empty_classes:
            findings.append(f'Empty class directories found: {empty_classes}.')
            recommendations.append('Populate empty class directories with representative data.')
            score -= 2

        if counts and min(counts.values()) > 0:
            max_count = max(counts.values())
            min_count = min(counts.values())
            if max_count / min_count > 4:
                findings.append('Detected class imbalance where one class has more than four times the examples of another.')
                recommendations.append('Rebalance the dataset by adding or downsampling examples.')
                score -= 1

        if not findings:
            recommendations.append('Training data structure and balance checks passed.')
            score += 3

        return SkillResult(self.name, self.description, findings, recommendations, score)


class ModelHealthSkill(BaseSkill):
    name = 'Model Health Skill'
    description = 'Validate saved model metadata, artifact size, and label consistency.'

    def run(self, root: Path) -> SkillResult:
        findings: List[str] = []
        recommendations: List[str] = []
        score = 0

        model_path = root / 'model.h5'
        labels_path = root / 'labels.json'

        if not model_path.exists():
            findings.append('Model artifact `model.h5` is missing.')
            recommendations.append('Train the model and save weights to `model.h5`.')
            return SkillResult(self.name, self.description, findings, recommendations, score)

        if not labels_path.exists():
            findings.append('Label metadata `labels.json` is missing.')
            recommendations.append('Write class label indices to `labels.json` during training.')
            score -= 2
        else:
            labels = self._load_labels(labels_path)
            if labels is None:
                findings.append('Could not parse `labels.json`; file is not valid JSON.')
                recommendations.append('Fix `labels.json` formatting and rerun training metadata export.')
                score -= 2
            elif not labels:
                findings.append('`labels.json` contains no class labels.')
                recommendations.append('Populate `labels.json` with class index mappings from training.')
                score -= 2
            else:
                recommendations.append('Label metadata exists and appears valid.')
                score += 1

        size_mb = model_path.stat().st_size / (1024 * 1024)
        if size_mb < 1:
            findings.append('Model file size is unusually small; it may be missing weights or be corrupt.')
            score -= 1
        elif size_mb > 200:
            recommendations.append('Large model size may increase latency; consider compression or smaller architectures.')

        return SkillResult(self.name, self.description, findings, recommendations, score)

    def _load_labels(self, path: Path):
        try:
            with path.open('r', encoding='utf-8') as handle:
                return json.load(handle)
        except Exception:
            return None


class PromptWorkflowSkill(BaseSkill):
    name = 'Prompt Workflow Skill'
    description = 'Detect AI workflow and agent awareness across docs and automation files.'

    def run(self, root: Path) -> SkillResult:
        findings: List[str] = []
        recommendations: List[str] = []
        score = 0

        workflows_dir = root / '.github' / 'workflows'
        workflows = list(workflows_dir.glob('*')) if workflows_dir.exists() else []
        if not workflows:
            findings.append('No GitHub workflow files are configured under `.github/workflows`.')
            recommendations.append('Add workflow automation to monitor and deploy AI components automatically.')
            score -= 1

        prompt_assets = []
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.md', '.yml', '.yaml'}:
                text = path.read_text(encoding='utf-8', errors='ignore').lower()
                if any(keyword in text for keyword in AI_KEYWORDS):
                    prompt_assets.append(path.relative_to(root))

        if not prompt_assets:
            findings.append('No AI prompt or agent-related language detected in repository content.')
            recommendations.append('Document your AI components, prompts, and agent behavior in README or workflow files.')
            score -= 1
        else:
            score += 1
            recommendations.append('AI-related workflow language found across repository assets.')

        return SkillResult(self.name, self.description, findings, recommendations, score)


class IssueScannerSkill(BaseSkill):
    name = 'Issue Scanner Skill'
    description = 'Scan repository text assets for unresolved issues, warnings, and drift indicators.'

    def run(self, root: Path) -> SkillResult:
        findings: List[str] = []
        recommendations: List[str] = []
        score = 0

        scanned = 0
        for path in root.rglob('*'):
            if not path.is_file() or path.suffix not in {'.py', '.md', '.yml', '.yaml', '.json', '.txt'}:
                continue
            scanned += 1
            text = path.read_text(encoding='utf-8', errors='ignore').lower()
            for keyword in ISSUE_KEYWORDS:
                if keyword in text:
                    findings.append(f'Found `{keyword}` in `{path.relative_to(root)}`.')
                    score -= 1

        if scanned == 0:
            findings.append('No repository text assets were scanned.')
            score -= 1
        elif not findings:
            recommendations.append('No issue-related terms detected in scanned files.')
            score += 2

        return SkillResult(self.name, self.description, findings, recommendations, score)


class DataDriftSkill(BaseSkill):
    name = 'Data Drift Skill'
    description = 'Compare training and test data coverage to detect drift and class mismatches.'

    def run(self, root: Path) -> SkillResult:
        findings: List[str] = []
        recommendations: List[str] = []
        score = 0

        train_dir = root / 'train'
        test_dir = root / 'test'

        if not train_dir.exists() or not train_dir.is_dir():
            findings.append('Training dataset directory `train/` is missing; cannot compare drift.')
            recommendations.append('Ensure `train/` exists before running drift checks.')
            return SkillResult(self.name, self.description, findings, recommendations, score)

        if not test_dir.exists() or not test_dir.is_dir():
            findings.append('Test dataset directory `test/` is missing; cannot validate deployment drift.')
            recommendations.append('Add a `test/` split for drift and performance validation.')
            score -= 2
            return SkillResult(self.name, self.description, findings, recommendations, score)

        train_classes = {d.name for d in train_dir.iterdir() if d.is_dir()}
        test_classes = {d.name for d in test_dir.iterdir() if d.is_dir()}

        missing_in_test = sorted(train_classes - test_classes)
        extra_in_test = sorted(test_classes - train_classes)

        if missing_in_test:
            findings.append(f'Classes present in train but missing in test: {missing_in_test}.')
            recommendations.append('Add missing test classes or align the data splits for consistency.')
            score -= 2
        if extra_in_test:
            findings.append(f'Classes present in test but not in train: {extra_in_test}.')
            recommendations.append('Remove unsupported test classes or include them in training data.')
            score -= 1

        if not findings:
            recommendations.append('Train/test class coverage is aligned.')
            score += 2

        return SkillResult(self.name, self.description, findings, recommendations, score)


class SecurityScanSkill(BaseSkill):
    name = 'Security Scan Skill'
    description = 'Scan code and documentation for leaked credentials or high-risk tokens.'

    SECRET_PATTERNS = [
        r'api[_-]?key',
        r'secret',
        r'token',
        r'password',
        r'client_secret',
        r'aws_access_key_id',
        r'aws_secret_access_key',
        r'[{<]\s*[A-Za-z0-9_-]{20,}\s*[}>]',
    ]

    def run(self, root: Path) -> SkillResult:
        findings: List[str] = []
        recommendations: List[str] = []
        score = 0

        for path in root.rglob('*'):
            if not path.is_file() or path.suffix not in {'.py', '.md', '.yml', '.yaml', '.json', '.txt'}:
                continue
            text = path.read_text(encoding='utf-8', errors='ignore')
            for pattern in self.SECRET_PATTERNS:
                if any(token in text.lower() for token in ['api_key', 'secret', 'token', 'password']) and re.search(pattern, text, re.IGNORECASE):
                    findings.append(f'Potential secret pattern in `{path.relative_to(root)}`: {pattern}')
                    score -= 2

        if findings:
            recommendations.append('Remove secrets from source and use GitHub Secrets or environment variables.')
        else:
            recommendations.append('No obvious secret patterns detected.')
            score += 1

        return SkillResult(self.name, self.description, findings, recommendations, score)


class DocumentationSkill(BaseSkill):
    name = 'Documentation Skill'
    description = 'Verify repository documentation coverage for AI Ops, deployment, and usage.'

    def run(self, root: Path) -> SkillResult:
        findings: List[str] = []
        recommendations: List[str] = []
        score = 0

        readme = root / 'README.md'
        if not readme.exists():
            findings.append('README.md is missing.')
            recommendations.append('Add README.md to document the AI workflow, installation, and release process.')
            return SkillResult(self.name, self.description, findings, recommendations, score)

        text = readme.read_text(encoding='utf-8', errors='ignore').lower()
        missing = []
        for topic in ['ai ops', 'workflow', 'model', 'training', 'release', 'monitor']:
            if topic not in text:
                missing.append(topic)

        if missing:
            findings.append(f'Readme is missing AI Ops documentation topics: {missing}.')
            recommendations.append('Update README.md to describe the AI Ops pipeline and repository responsibilities.')
            score -= 1
        else:
            recommendations.append('Documentation appears to cover the AI Ops process.')
            score += 2

        return SkillResult(self.name, self.description, findings, recommendations, score)


class AIOpsAgent:
    def __init__(self, root: Path):
        self.root = root
        self.skills: List[BaseSkill] = [
            DataQualitySkill(),
            ModelHealthSkill(),
            DataDriftSkill(),
            PromptWorkflowSkill(),
            DocumentationSkill(),
            SecurityScanSkill(),
            IssueScannerSkill(),
        ]

    def run(self) -> List[SkillResult]:
        return [skill.run(self.root) for skill in self.skills]

    def build_report(self, results: List[SkillResult]) -> str:
        total_score = sum(result.score for result in results)
        lines = [
            '# AI Ops Agent Report',
            '',
            'This report evaluates AI repository readiness and real-time risk signals.',
            '',
            '## Summary',
            '',
            f'- Total score: **{total_score}**',
            f'- Skills evaluated: **{len(results)}**',
            '',
        ]

        for result in results:
            lines.extend([
                f'## {result.name}',
                f'*{result.description}*',
                '',
                f'- Findings: {len(result.findings)}',
                f'- Recommendations: {len(result.recommendations)}',
                f'- Skill score: {result.score}',
                '',
            ])
            if result.findings:
                lines.append('### Findings')
                lines.extend(f'- {finding}' for finding in result.findings)
                lines.append('')
            if result.recommendations:
                lines.append('### Recommendations')
                lines.extend(f'- {recommendation}' for recommendation in result.recommendations)
                lines.append('')

        return '\n'.join(lines)

    def build_json(self, results: List[SkillResult]) -> dict:
        return {
            'summary': {
                'total_score': sum(result.score for result in results),
                'skills': len(results),
            },
            'skills': [
                {
                    'name': result.name,
                    'description': result.description,
                    'findings': result.findings,
                    'recommendations': result.recommendations,
                    'score': result.score,
                }
                for result in results
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description='AI Ops Agent that detects real-time AI risks.')
    parser.add_argument('--root', default='.', help='Repository root path')
    parser.add_argument('--output', default='ai_ops_report.md', help='Markdown report path')
    parser.add_argument('--output-json', default='ai_ops_report.json', help='JSON report path')
    parser.add_argument('--fail-on-issue', action='store_true', help='Exit with failure when risks are detected')
    args = parser.parse_args()

    root = Path(args.root)
    agent = AIOpsAgent(root)
    results = agent.run()
    report_md = agent.build_report(results)
    report_json = agent.build_json(results)

    Path(args.output).write_text(report_md, encoding='utf-8')
    Path(args.output_json).write_text(json.dumps(report_json, indent=2), encoding='utf-8')

    print(report_md)
    if args.fail_on_issue and any(result.findings for result in results):
        raise SystemExit('AI Ops Agent detected issues and is failing the pipeline.')


if __name__ == '__main__':
    main()
