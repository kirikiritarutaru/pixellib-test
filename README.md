# pixellib-test
pixcellibのお試しリポジトリ

# 導入
- `mkdir inputs outputs models`
- `pip install tensorflow-gpu pixellib pycocotools`
- チュートリアルに記載の学習済みモデルをダウンロード
  - xception_pascalvoc.pb
  - mask_rcnn_coco.h5
  - deeplabv3_xception65_ade20k.h5

# Pytorch実装追加
- `mkdir inputs outputs models`
- `pip install pytorch opencv-python pixellib pycocotools`
- 学習済みモデルをダウンロードしmodelsにおく
  - pointrend_resnet50.pkl
