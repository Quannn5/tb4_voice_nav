from tb4_voice_nav.reporting import append_test_result


def test_append_test_result_initializes_table_and_escapes_cells(tmp_path):
    results_path = tmp_path / 'data' / 'test_results.md'
    row = {
        'command': 'Đi đến A',
        'transcription': 'Đi đến | khu vực A',
        'route': 'khu_vuc_a',
        'coordinates': '(-7.40, 7.60)',
        'duration': '3.25',
        'outcome': 'Thành công',
    }

    append_test_result(results_path, row)
    append_test_result(results_path, row)

    content = results_path.read_text(encoding='utf-8')
    assert content.count('| STT |') == 1
    assert content.count(r'Đi đến \| khu vực A') == 2
    assert len(content.strip().splitlines()) == 4
    data_rows = content.strip().splitlines()[2:]
    assert data_rows[0].startswith('| 1 |')
    assert data_rows[1].startswith('| 2 |')
