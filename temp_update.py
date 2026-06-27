import os

files_to_modify = {
    'd:/pythonunion/pycharm/Segdefect/project/config.py': {
        "encoder         = 'timm-efficientnet-b5'": "encoder         = 'resnet50'",
    },
    'd:/pythonunion/pycharm/Segdefect/project/model.py': {
        'EfficientNet-B5 Encoder': 'ResNet50 Encoder',
        "encoder='timm-efficientnet-b5'": "encoder='resnet50'",
    },
    'd:/pythonunion/pycharm/Segdefect/project/validate_pipeline.py': {
        "cfg.encoder          = 'timm-efficientnet-b1'": "cfg.encoder          = 'resnet18'",
    },
}

for filepath, replacements in files_to_modify.items():
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    for old, new in replacements.items():
        content = content.replace(old, new)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Updated: {filepath}')

print('Done!')
