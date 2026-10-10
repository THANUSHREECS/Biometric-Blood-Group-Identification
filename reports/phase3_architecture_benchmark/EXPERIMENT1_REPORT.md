# EXPERIMENT 1: ARCHITECTURE BENCHMARK

## Validation Rankings
| Architecture | Val Macro F1 | Val Accuracy | Val O+ Share |
|--------------|--------------|--------------|--------------|
| EfficientNet-B0 | 0.5667 | 0.7500 | 0.5625 |
| DenseNet121 | 0.3173 | 0.5000 | 0.4375 |
| VGG16 | 0.3110 | 0.4375 | 0.6250 |
| Custom CNN | 0.2929 | 0.5000 | 0.8750 |
| EfficientNetV2-B0 | 0.2786 | 0.3750 | 0.5000 |
| ConvNeXt-Tiny | 0.2659 | 0.3125 | 0.0625 |
| ResNet50 | 0.2582 | 0.4375 | 0.4375 |


## Final Locked Test Performance
| Architecture | Test Macro F1 | Test Accuracy | Test O+ Share |
|--------------|---------------|---------------|---------------|
| EfficientNet-B0 | 0.1250 | 0.1905 | 0.3810 |
| DenseNet121 | 0.3750 | 0.5238 | 0.5714 |
| VGG16 | 0.3100 | 0.5238 | 0.8095 |
| Custom CNN | 0.1200 | 0.2857 | 0.8095 |
| EfficientNetV2-B0 | 0.1929 | 0.3333 | 0.7143 |
| ConvNeXt-Tiny | 0.3268 | 0.3333 | 0.1429 |
| ResNet50 | 0.1357 | 0.2381 | 0.5714 |


## Best Architecture: EfficientNet-B0
- Test Accuracy: 0.1905
- Test Macro F1: 0.1250
- 95% Bootstrap CI (Accuracy): [0.0476, 0.3810]
- 95% Bootstrap CI (Macro F1): [0.0278, 0.2271]
