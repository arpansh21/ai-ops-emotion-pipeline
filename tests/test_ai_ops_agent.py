import json
import subprocess
import sys
from pathlib import Path

from ai_ops_agent import (
    AIOpsAgent,
    DataQualitySkill,
    ModelHealthSkill,
    IssueScannerSkill,
    DataDriftSkill,
    DocumentationSkill,
    SecurityScanSkill,
)


def test_data_quality_skill_reports_missing_train(tmp_path):
    result = DataQualitySkill().run(tmp_path)

    assert any('missing' in finding.lower() for finding in result.findings)
    assert result.score == 0


def test_model_health_skill_reports_missing_model(tmp_path):
    result = ModelHealthSkill().run(tmp_path)

    assert any('model artifact' in finding.lower() for finding in result.findings)
    assert result.score == 0


def test_issue_scanner_detects_issue_keywords(tmp_path):
    file_path = tmp_path / 'deploy.md'
    file_path.write_text('This AI workflow has a TODO and a warning.')

    result = IssueScannerSkill().run(tmp_path)

    assert any('todo' in finding.lower() for finding in result.findings)
    assert any('warning' in finding.lower() for finding in result.findings)
    assert result.score < 0


def test_data_drift_skill_detects_missing_test(tmp_path):
    (tmp_path / 'train' / 'happy').mkdir(parents=True)
    (tmp_path / 'train' / 'happy' / 'sample.jpg').write_text('dummy')

    result = DataDriftSkill().run(tmp_path)

    assert any('test dataset directory' in finding.lower() for finding in result.findings)
    assert result.score < 0


def test_documentation_skill_reports_missing_readme(tmp_path):
    result = DocumentationSkill().run(tmp_path)

    assert any('readme.md is missing' in finding.lower() for finding in result.findings)
    assert result.score == 0


def test_security_scan_detects_secret_pattern(tmp_path):
    file_path = tmp_path / 'config.py'
    file_path.write_text("API_KEY = 'AKIA1234567890ABCDEF'\n")

    result = SecurityScanSkill().run(tmp_path)

    assert any('potential secret pattern' in finding.lower() for finding in result.findings)
    assert result.score < 0


def test_cli_writes_reports(tmp_path):
    script_path = Path(__file__).resolve().parents[1] / 'ai_ops_agent.py'
    output_md = tmp_path / 'report.md'
    output_json = tmp_path / 'report.json'

    completed = subprocess.run(
        [sys.executable, str(script_path), '--root', str(tmp_path), '--output', str(output_md), '--output-json', str(output_json)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_md.exists()
    assert output_json.exists()
    report_data = json.loads(output_json.read_text(encoding='utf-8'))
    assert report_data['summary']['skills'] == 7
