"""Append test outcomes to the report's Markdown results table."""

import fcntl
from pathlib import Path
from typing import Any


_TABLE_HEADER = (
    '| STT | Câu lệnh | Văn bản nhận diện (STT) | '
    'Chuỗi khu vực trích xuất | Tọa độ đích (x, y) | '
    'Thời gian di chuyển (s) | '
    'Kết quả (Thành công / Báo lỗi vùng không tồn tại) |'
)
_TABLE_SEPARATOR = (
    '|---|---|---|---|---|---:|---|'
)


def default_results_path() -> Path:
    """Choose the workspace report path when available."""
    source_root = Path(__file__).resolve().parent.parent
    candidates = [
        source_root / 'report_assets' / 'data',
        Path.cwd() / 'report_assets' / 'data',
    ]
    try:
        from ament_index_python.packages import get_package_prefix

        workspace_root = Path(
            get_package_prefix('tb4_voice_nav')
        ).resolve().parents[1]
        candidates.append(workspace_root / 'report_assets' / 'data')
    except (ImportError, LookupError):
        pass

    for directory in candidates:
        if directory.is_dir():
            return directory / 'test_results.md'
    return source_root / 'report_assets' / 'data' / 'test_results.md'


def _cell(value: Any) -> str:
    text = str(value).replace('\r', ' ').replace('\n', ' ').strip()
    return text.replace('|', r'\|')


def append_test_result(path: Path, result: dict[str, Any]) -> None:
    """Append one safely escaped row, initializing the table if necessary."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+', encoding='utf-8') as markdown_file:
        fcntl.flock(markdown_file.fileno(), fcntl.LOCK_EX)
        markdown_file.seek(0)
        existing_lines = markdown_file.readlines()
        if not existing_lines:
            markdown_file.write(
                _TABLE_HEADER + '\n' + _TABLE_SEPARATOR + '\n'
            )
            result_number = 1
        else:
            result_number = 1 + sum(
                line.startswith('|')
                and line.strip() not in (_TABLE_HEADER, _TABLE_SEPARATOR)
                for line in existing_lines
            )

        row = '| ' + ' | '.join(
            _cell(value)
            for value in (
                result_number,
                result.get('command', ''),
                result.get('transcription', ''),
                result.get('route', ''),
                result.get('coordinates', ''),
                result.get('duration', ''),
                result.get('outcome', ''),
            )
        ) + ' |'
        markdown_file.seek(0, 2)
        markdown_file.write(row + '\n')
        markdown_file.flush()
        fcntl.flock(markdown_file.fileno(), fcntl.LOCK_UN)
