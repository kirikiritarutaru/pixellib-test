import cv2
import pixellib
from pixellib.tune_bg import alter_bg


# 人以外をblur
def blur_bg():
    change_bg = alter_bg(model_type='pb')
    change_bg.load_pascalvoc_model('xception_pascalvoc.pb')
    change_bg.blur_bg(
        'inputs/woman.jpg',
        extreme=True,
        detect='person',
        output_image_name='outputs/blur_img2.jpg'
    )


# モデルで検出できる全てのオブジェクトをペースト
def paste_person():
    change_bg = alter_bg(model_type='pb')
    change_bg.load_pascalvoc_model('xception_pascalvoc.pb')
    change_bg.change_bg_img(
        f_image_path='inputs/man_and_cars.jpg',
        b_image_path='inputs/flowers.jpg',
        output_image_name='outputs/new_img2.jpg'
    )


# 人を検出してペースト
def paste_detected_obj(set_detect: str = 'person'):
    change_bg = alter_bg(model_type='pb')
    change_bg.load_pascalvoc_model('xception_pascalvoc.pb')
    change_bg.change_bg_img(
        f_image_path='inputs/man_and_cars.jpg',
        b_image_path='inputs/flowers.jpg',
        output_image_name='outputs/new_img4.jpg',
        detect=set_detect
    )


# ブルーバックに人をペースト
def paste_detected_obj_to_blue_bg(set_detect: str = 'person'):
    change_bg = alter_bg(model_type='pb')
    change_bg.load_pascalvoc_model('xception_pascalvoc.pb')
    change_bg.color_bg(
        'inputs/man_and_cars.jpg',
        colors=(0, 0, 255),
        output_image_name='outputs/new_img5.jpg',
        detect=set_detect
    )


def blur_bg_video():
    change_bg = alter_bg(model_type='pb')
    change_bg.load_pascalvoc_model('xception_pascalvoc.pb')

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output = change_bg.blur_frame(
            frame,
            extreme=True,
            detect='person'
        )

        cv2.imshow('frame', output)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def gray_bg_video():
    change_bg = alter_bg(model_type='pb')
    change_bg.load_pascalvoc_model('xception_pascalvoc.pb')

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output = change_bg.gray_frame(frame, detect='person')
        cv2.imshow('frame', output)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    gray_bg_video()
