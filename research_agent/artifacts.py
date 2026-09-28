"""Portable exports and safe table rendering from structured cell coordinates."""
from __future__ import annotations

import html
from io import BytesIO
import json
import zipfile

from .models import RunResult, TableCell
from .visuals import image_bytes


def table_html(cells: list[TableCell]) -> str:
    occupied, origins = {}, {}
    for cell in cells:
        for row in range(cell.row, cell.row + cell.row_span):
            for col in range(cell.column, cell.column + cell.column_span):
                if row >= 200 or col >= 100 or (row, col) in occupied:
                    raise ValueError('表格存在重叠单元格或超出尺寸限制，不能可靠还原合并表头。')
                occupied[row, col] = cell
        origins[cell.row, cell.column] = cell
    if not occupied:
        return ''
    rows = ['<table style="border-collapse:collapse;width:100%">']
    for row in range(max(r for r, _ in occupied) + 1):
        rows.append('<tr>')
        for col in range(max(c for _, c in occupied) + 1):
            cell = origins.get((row, col))
            if cell:
                tag = 'th' if cell.header else 'td'
                rows.append(f'<{tag} rowspan="{cell.row_span}" colspan="{cell.column_span}" '
                            f'style="border:1px solid #cbd5e1;padding:6px">{html.escape(cell.text)}</{tag}>')
            elif (row, col) not in occupied:
                rows.append('<td></td>')
        rows.append('</tr>')
    return ''.join(rows) + '</table>'


def evidence_bundle(run: RunResult) -> bytes:
    buffer = BytesIO()
    manifest = []
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('report.md', run.report)
        exported = run.model_dump()
        for paper_index, paper in enumerate(run.papers):
            for asset_index, asset in enumerate(paper.assets):
                filename = f'images/paper-{paper_index + 1}-page-{asset.page}.png'
                try:
                    archive.writestr(filename, image_bytes(asset))
                    exported['papers'][paper_index]['assets'][asset_index]['image_path'] = filename
                    manifest.append({'asset_id': asset.id, 'file': filename, 'sha256': asset.image_sha256})
                    if asset.reading:
                        archive.writestr(filename.replace('.png', '.json'), asset.reading.model_dump_json(indent=2))
                except (OSError, RuntimeError) as exc:
                    manifest.append({'asset_id': asset.id, 'error': str(exc)})
        archive.writestr('run.json', json.dumps(exported, ensure_ascii=False, indent=2))
        archive.writestr('images/manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    return buffer.getvalue()
