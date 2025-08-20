# -*- mode: python ; coding: utf-8 -*-
import glob

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    # 'datas' is ONLY for non-code assets.
    # Your .py files will be found automatically via imports in app.py.
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
    ] + [
        # This correctly finds your models and puts them in a 'models' folder
        # inside the final app bundle.
        (f, 'models') for f in glob.glob('Run_Predictions/*.hdf5')
    ],
    hiddenimports=[
        'tensorflow',
        'keras',
        'h5py',
        'scipy',
        'pandas',
        'werkzeug.serving'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    # This creates the folder 'ecg-predictor' inside the 'dist' directory.
    name='ecg-predictor'
)