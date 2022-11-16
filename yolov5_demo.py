import yolov5_py
import cv2

def plot_one_box(x, im, label=None, line_thickness=3):#画框
    color=(128, 128, 128)
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(
        0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

yolov5z=yolov5_py.yolov5()
if(yolov5z.initialization("yolov5s.engine",640,640,22)):
    print("ok")
cap=cv2.VideoCapture(r"D:\c++\tensor_yv5\Release\a.mp4")
label= ["train_id","train_infor","train_module","0","1","2","3","4","5","6","7","8","9","A","C","E","K","T","H","B","train_separate","train_head"]#模型类别
while (cap.isOpened()):
    _,frame=cap.read()
    res=yolov5z.get_result(frame)
    for i in res:
        plot_one_box(i.bbox, frame, label=label[int(i.class_id)], line_thickness=3)
    cv2.imshow("a",frame)
    cv2.waitKey(25)
