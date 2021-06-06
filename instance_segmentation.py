import cv2
from pixellib.instance import instance_segmentation


# instance_segmentationとbbox表示
def inst_seg():
    segment_image = instance_segmentation()
    segment_image.load_model('models/mask_rcnn_coco.h5')
    segment_image.segmentImage(
        'inputs/cycle.jpg',
        output_image_name='outputs/inst_seg.jpg',
        show_bboxes=True
    )


# 同じ場所に切り抜いた画像が保存されるのなんとかならんか
def extract_inst_seg():
    seg = instance_segmentation()
    seg.load_model('models/mask_rcnn_coco.h5')
    seg.segmentImage(
        'inputs/man_and_cars.jpg',
        show_bboxes=True,
        output_image_name='outputs/inst_seg2.jpg',
        extract_segmented_objects=True,
        save_extracted_objects=True
    )


def inst_seg_target_class():
    segment_image = instance_segmentation()
    segment_image.load_model('models/mask_rcnn_coco.h5')
    target_classes = segment_image.select_target_classes(person=True)
    segment_image.segmentImage(
        'inputs/cycle.jpg',
        segment_target_classes=target_classes,
        show_bboxes=True,
        output_image_name='outputs/seg_person.jpg'
    )


def inst_seg_video():
    segment_frame = instance_segmentation(infer_speed='rapid')
    segment_frame.load_model('models/mask_rcnn_coco.h5')

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        segment_frame.segmentFrame(
            frame,
            show_bboxes=True
        )
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    inst_seg_video()
