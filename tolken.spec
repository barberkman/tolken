# PyInstaller's recipe for tolken.exe: tolken in one file that needs nothing installed (no Python,
# Textual or PyMuPDF), to put in a folder on PATH, such as %USERPROFILE%\.local\bin.
#
#   pip install -r requirements.txt pyinstaller
#   pyinstaller tolken.spec          (writes dist\tolken.exe)
#
# .github/workflows/release.yml builds it this way, tests it with test_exe.py and publishes it.
from PyInstaller.utils.hooks import collect_submodules

# Packages that PyMuPDF requires but tolken does not use: its page layout analysis (pymupdf.layout, with
# 50 MB of models, loaded only by Page.get_layout()) and what that runs on.
UNUSED = ["pymupdf.layout", "pymupdf_layout", "pymupdf4llm", "onnxruntime", "numpy", "networkx", "tabulate", "yaml",
          "psutil", "tkinter"]

a = Analysis(
    ["tolken.py"],
    # textual.widgets imports a widget only when it is first used, by a name PyInstaller cannot see.
    hiddenimports=collect_submodules("textual.widgets"),
    excludes=UNUSED,
)
# PyMuPDF's hook collects all of its data files: the layout models and the headers to compile against it.
a.datas = [entry for entry in a.datas if not entry[0].replace("\\", "/").startswith(("pymupdf/layout/", "pymupdf/mupdf-devel/"))]

exe = EXE(
    PYZ(a.pure),
    a.scripts,
    a.binaries,
    a.datas,
    name="tolken",
    console=True,  # a terminal program
    upx=False,
)
