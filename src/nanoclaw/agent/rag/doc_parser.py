"""教材文档解析模块：把常见格式的文件内容提取为纯文本，供 RAG 入库。

支持的格式（按文件扩展名分发，解析依赖缺失时给出友好提示）：
- 纯文本类：.txt / .md / .markdown / .csv          （多编码尝试：UTF-8 → GBK → GB18030）
- PDF：     .pdf                                   （优先 PyMuPDF，退回 pypdf）
- Word：    .docx（python-docx）；.doc（Word COM 转换，需本机装有 MS Word）
- PPT：     .pptx（python-pptx）；.ppt（PowerPoint COM 转换）
- Excel：   .xlsx（openpyxl）；.xls（Excel COM 转换）
- 网页：    .html / .htm                           （BeautifulSoup 去标签）
- 图片 OCR：.jpg / .jpeg / .png / .bmp / .webp     （RapidOCR 中英文识别，onnxruntime 推理）

所有重型依赖都在对应分支内延迟导入：某个库缺失只影响该格式，
其他格式的导入不受影响。COM 转换运行在工作线程中，必须先
CoInitialize（asyncio.to_thread 的线程不是 COM 默认单元线程）。
"""

import csv
import io
import os
import tempfile
from pathlib import Path

# 单文件解析上限（30MB），防止超大 PDF / 图片拖垮内存
MAX_FILE_SIZE = 30 * 1024 * 1024

# 图片 OCR 引擎的进程级单例（首次加载模型约 1-2 秒，之后复用）
_ocr_engine = None

SUPPORTED_TEXT = "文本类 .txt .md .csv ｜ PDF ｜ Word .doc .docx ｜ PPT .ppt .pptx ｜ Excel .xls .xlsx ｜ 网页 .html ｜ 图片 .jpg .png .bmp .webp（OCR）"


class DocParseError(Exception):
    """文档解析失败（含依赖缺失、格式不支持、内容为空等）。"""


def parse_document(filename: str, data: bytes) -> str:
    """按扩展名把文件字节流解析为纯文本。失败抛 DocParseError。"""
    if len(data) > MAX_FILE_SIZE:
        raise DocParseError(f"文件超过 {MAX_FILE_SIZE // (1024 * 1024)}MB 上限")
    if not data:
        raise DocParseError("文件内容为空")

    suffix = Path(filename or "").suffix.lower().lstrip(".")
    dispatch = {
        # 纯文本类
        "txt": _parse_text, "md": _parse_text, "markdown": _parse_text, "csv": _parse_text,
        # PDF
        "pdf": _parse_pdf,
        # Word
        "docx": _parse_docx, "doc": _parse_doc,
        # PPT
        "pptx": _parse_pptx, "ppt": _parse_ppt,
        # Excel
        "xlsx": _parse_xlsx, "xls": _parse_xls,
        # 网页
        "html": _parse_html, "htm": _parse_html,
        # 图片 OCR
        "jpg": _parse_image, "jpeg": _parse_image, "png": _parse_image,
        "bmp": _parse_image, "webp": _parse_image,
    }
    parser = dispatch.get(suffix)
    if parser is None:
        raise DocParseError(f"暂不支持 .{suffix or '无扩展名'} 格式。支持：{SUPPORTED_TEXT}")
    text = parser(data).strip()
    if not text:
        raise DocParseError("未能从文件中提取到文字（可能是扫描件/空文档）")
    return text


# ============ 纯文本类 ============

def _decode_text(data: bytes) -> str:
    """中文教材常见编码依次尝试：UTF-8（含 BOM）→ GBK → GB18030。"""
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _parse_text(data: bytes) -> str:
    text = _decode_text(data)
    # CSV 按行重组成"列1 列2 列3"的形式，保持语义可分块
    if text and ("," in text or "\t" in text) and "\n" in text:
        try:
            rows = list(csv.reader(io.StringIO(text)))
            return "\n".join(" | ".join(cell for cell in row if cell) for row in rows if any(row))
        except csv.Error:
            pass
    return text


# ============ PDF ============

def _parse_pdf(data: bytes) -> str:
    parts = []
    try:  # PyMuPDF：中文提取质量最好
        import fitz
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                parts.append(page.get_text())
        return "\n".join(parts)
    except ImportError:
        pass
    try:  # 退回 pypdf
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except ImportError as e:
        raise DocParseError("PDF 解析依赖未安装（pip install pymupdf 或 pypdf）") from e


# ============ Word ============

def _parse_docx(data: bytes) -> str:
    try:
        import docx
    except ImportError as e:
        raise DocParseError("docx 解析依赖未安装（pip install python-docx）") from e
    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:  # 表格内容也要入库
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _parse_doc(data: bytes) -> str:
    """老版 .doc：用本机 Word COM 打开后另存为 .docx 再解析。"""
    docx_bytes = _office_com_convert(data, ".doc", ".docx", 16, "Word.Application", "Word")
    if docx_bytes is not None:
        return _parse_docx(docx_bytes)
    raise DocParseError(".doc 需要本机安装 Microsoft Word 进行转换（或先另存为 .docx / .txt）")


# ============ PPT ============

def _parse_pptx(data: bytes) -> str:
    try:
        from pptx import Presentation
    except ImportError as e:
        raise DocParseError("pptx 解析依赖未安装（pip install python-pptx）") from e
    parts = []
    presentation = Presentation(io.BytesIO(data))
    for index, slide in enumerate(presentation.slides, 1):
        parts.append(f"--- 第 {index} 页 ---")
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if text:
                    parts.append(text)
        if slide.has_notes_slide:  # 演讲备注往往包含讲解要点
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"（备注：{notes}）")
    return "\n".join(parts)


def _parse_ppt(data: bytes) -> str:
    pptx_bytes = _office_com_convert(data, ".ppt", ".pptx", 24, "PowerPoint.Application", "PowerPoint")
    if pptx_bytes is not None:
        return _parse_pptx(pptx_bytes)
    raise DocParseError(".ppt 需要本机安装 Microsoft PowerPoint 进行转换（或先另存为 .pptx）")


# ============ Excel ============

def _parse_xlsx(data: bytes) -> str:
    try:
        import openpyxl
    except ImportError as e:
        raise DocParseError("xlsx 解析依赖未安装（pip install openpyxl）") from e
    parts = []
    workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    for sheet in workbook.worksheets:
        parts.append(f"--- 工作表：{sheet.title} ---")
        for row in sheet.iter_rows(values_only=True):
            cells = [str(v) for v in row if v is not None and str(v).strip()]
            if cells:
                parts.append(" | ".join(cells))
    workbook.close()
    return "\n".join(parts)


def _parse_xls(data: bytes) -> str:
    xlsx_bytes = _office_com_convert(data, ".xls", ".xlsx", 51, "Excel.Application", "Excel")
    if xlsx_bytes is not None:
        return _parse_xlsx(xlsx_bytes)
    raise DocParseError(".xls 需要本机安装 Microsoft Excel 进行转换（或先另存为 .xlsx / .csv）")


# ============ 网页 ============

def _parse_html(data: bytes) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise DocParseError("HTML 解析依赖未安装（pip install beautifulsoup4）") from e
    soup = BeautifulSoup(_decode_text(data), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n")


# ============ 图片 OCR ============

def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise DocParseError("图片 OCR 依赖未安装（pip install rapidocr_onnxruntime）") from e
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _parse_image(data: bytes) -> str:
    result, _ = _get_ocr_engine()(data)
    if not result:
        return ""
    # result: [[box, text, score], ...] 按阅读顺序（已按版面排序）拼接
    return "\n".join(item[1] for item in result if len(item) >= 2 and item[1])


# ============ MS Office COM 转换（.doc/.ppt/.xls → 新格式） ============

def _office_com_convert(data: bytes, old_suffix: str, new_suffix: str, save_format: int,
                        prog_id: str, office_name: str) -> bytes | None:
    """用本机 Office 把老格式另存为新格式，返回新文件字节；Office 不可用时返回 None。

    COM 调用必须在工作线程里先 CoInitialize；全程用独立临时目录，
    finally 里保证 Word/Excel/PowerPoint 进程退出，不留僵尸进程。
    """
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        return None

    workdir = tempfile.mkdtemp(prefix="rag_convert_")
    src = Path(workdir) / f"input{old_suffix}"
    dst = Path(workdir) / f"output{new_suffix}"
    src.write_bytes(data)
    app = None
    try:
        pythoncom.CoInitialize()
        app = win32com.client.DispatchEx(prog_id)
        app.DisplayAlerts = False
        if office_name == "Word":
            doc = app.Documents.Open(str(src), ReadOnly=True)
            doc.SaveAs2(str(dst), FileFormat=save_format)
            doc.Close(False)
        elif office_name == "Excel":
            book = app.Workbooks.Open(str(src), ReadOnly=True)
            book.SaveAs(str(dst), FileFormat=save_format)
            book.Close(False)
        else:  # PowerPoint
            pres = app.Presentations.Open(str(src), ReadOnly=True, WithWindow=False)
            pres.SaveAs(str(dst), FileFormat=save_format)
            pres.Close()
        return dst.read_bytes()
    except Exception:
        # Office 未安装 / 打开失败：返回 None 让上层给出"请先转换格式"提示
        return None
    finally:
        if app is not None:
            try:
                app.Quit()
            except Exception:
                pass
        pythoncom.CoUninitialize()
        try:
            src.unlink(missing_ok=True)
            dst.unlink(missing_ok=True)
            os.rmdir(workdir)
        except OSError:
            pass
