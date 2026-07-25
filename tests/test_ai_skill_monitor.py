import json
import subprocess
import sys
from pathlib import Path

from ai_ops_agent import AIOpsAgent, DataQualitySkill, ModelHealthSkill, IssueScannerSkill, PromptWorkflowSkill


def test_data_quality_skill_reports_missing_train(tmp_path):
    agent = AIOpsAgent(tmp_path)
    result = DataQualitySkill().run(tmp_path)

    assert 'missing' in result.findings[0].lower()
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
