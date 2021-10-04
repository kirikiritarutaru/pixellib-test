import cv2
from pixellib.torchbackend.instance import instanceSegmentation


def image_segmentation():
    ins = instanceSegmentation()
    ins.load_model('models/pointrend_resnet50.pkl')
    ins.segmentImage(
        'inputs/sample.jpg',
        show_bboxes=True,
        output_image_name='outputs/output_image.jpg'
    )


def usb_camera_segmentation():
    capture = cv2.VideoCapture(0)

    seg_video = instanceSegmentation()
    seg_video.load_model('models/pointrend_resnet50.pkl')
    target_classes = seg_video.select_target_classes(person=True)
    seg_video.process_camera(
        capture,
        show_bboxes=False,  # bboxは表示できず（2021-10-04）
        frames_per_second=5,
        segment_target_classes=target_classes,
        show_frames=True,
        frame_name='frame'
    )


if __name__ == '__main__':
    usb_camera_segmentation()
