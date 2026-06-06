# Gradio Demo Verification

Demo model:

`outputs/experiments/mobilenet_v3_transfer_fc_then_features_tail/best_model.pth`

Local URL:

`http://127.0.0.1:7860/`

## UI Check

- Page title: Chinese UI title for the tomato disease recognition system
- Upload area: present
- Submit button: present
- Top-3 output component: present

## Prediction Samples

### Healthy Leaf

Input image:

`tomato/val/Tomato___healthy/000bf685-b305-408b-91f4-37030f8e62db___GH_HL Leaf 308.1.JPG`

Top-3 output:

| Rank | Class | Confidence |
| ---: | --- | ---: |
| 1 | Tomato___healthy | 0.999996 |
| 2 | Tomato___Target_Spot | 0.000002 |
| 3 | Tomato___Late_blight | 0.000001 |

### Tomato Yellow Leaf Curl Virus

Input image:

`tomato/val/Tomato___Tomato_Yellow_Leaf_Curl_Virus/1af07f2b-027b-4792-80c5-2c20a4ed538c___YLCV_NREC 0179.JPG`

Top-3 output:

| Rank | Class | Confidence |
| ---: | --- | ---: |
| 1 | Tomato___Tomato_Yellow_Leaf_Curl_Virus | 0.999882 |
| 2 | Tomato___Target_Spot | 0.000094 |
| 3 | Tomato___Spider_mites Two-spotted_spider_mite | 0.000020 |

Conclusion:

The Gradio interface and prediction API are both functional. The demo satisfies
the advanced requirement for uploading tomato leaf images and returning Top-3
classification results with confidence scores.
