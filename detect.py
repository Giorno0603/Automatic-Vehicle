import argparse
import time
from pathlib import Path

import serial
import sys
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
state = False
state2 = False
state3 = False
def send_command_to_arduino(command):
    ser = serial.Serial('/dev/ttyTHS1', 57600)
    ser.write(command.encode())
    time.sleep(0.0001) 
    ser.close()



def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images «O¦s±À²z¹Ï¤ù
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories ¥Ø¿ý
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run ¼W¶q¹B¦æ
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir »s§@¥Ø¿ý

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier ²Ä¤G¶¥¤ÀÃþ¾¹
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader ³]©w¾É¤J¸ê®Æ
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference ¥[§Ö±ÀÂ_³t«×
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors 
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference ¹B¦æ±À²z
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup ´Nºü
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference ±À²z
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections ³B²zÀË´ú
        for i, det in enumerate(pred):  # detections per image ¹Ï¤ùÀË´ú
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                mainobject  = 350
                area2 = 1
                area3 = 1
                cls1 = False
                cls2 = False
                cls3 = False
                cls4 = False
                cls5 = False
                global state
                global state2
                global state3
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        
                    #**    
                    if int(cls) == 0 and conf > 0.5:
                        cls1 = True
                        reg=[]
                        for i in xyxy:
                            reg.append(int(i))
                        a = (reg[0]+reg[2])/2-320
                        b = (reg[2]-reg[0])*(reg[3]-reg[1])
                        if abs(a) < abs(mainobject):
                            mainobject = a
                            area = b
                    elif int(cls) == 56 and conf > 0.6:
                        cls4 = True
                        reg=[]
                        for i in xyxy:
                            reg.append(int(i))
                        a = (reg[0]+reg[2])/2-320
                        b = (reg[2]-reg[0])*(reg[3]-reg[1])
                        if b > area2:
                            mainobject2 = a
                            area2 = b
                    elif int(cls) == 9 and conf > 0.7:
                        cls3 = True
                        reg=[]
                        for i in xyxy:
                            reg.append(int(i))
                        b = (reg[2]-reg[0])*(reg[3]-reg[1])
                        if b > area3:
                            area3 = b
                    elif int(cls) == 100 and conf > 0.7:
                        cls5 = True
                if cls3 or cls5:
                    if not state2:
                        state2 = True
                        print("warn")
                        send_command_to_arduino("W\n")
                else:
                    state2 = False
                if area3 > 20000:
                    state = False
                    if not state3:
                        state3 = True
                        print("stop")
                        send_command_to_arduino("S\n")
                        print("wwarn")
                        send_command_to_arduino("WW\n")
                elif state3:
                    state3 = False
                elif area2 > 76800 and abs(mainobject2) < 200 and cls3:
                    state = False
                    print("stop")
                    send_command_to_arduino("S\n")
                elif area2 >85000 and abs(mainobject2) < 200 :
                    cls2 = True
                    if not state:
                        print("stop")
                        send_command_to_arduino("S\n")
                        resp = input ("avoid obstacle?")
                        if resp == 'y':
                            #cls2 = True
                        # if not state:
                            state = True
                            if mainobject2 < 0:
                                print("turnright")
                                send_command_to_arduino("R\n")
                                start = time.process_time()
                            else :
                                print("turnleft")
                                send_command_to_arduino("L\n")
                                start = time.process_time()
                elif state and abs(mainobject2) > 300:
                    end = time.process_time()
                    state = False 
                    if mainobject2 < 0:
                        print("forward")
                        send_command_to_arduino("F\n")
                        time.sleep(2*(1.2+end-start))
                        print("turnleft")
                        send_command_to_arduino("L\n")
                        time.sleep(end-start)
                        print("stop")
                        send_command_to_arduino("S\n")
                    else :
                        print("forward")
                        send_command_to_arduino("F\n")
                        time.sleep(2*(1.2+end-start))
                        print("turnright")
                        send_command_to_arduino("R\n")
                        time.sleep(end-start)
                        print("stop")
                        send_command_to_arduino("S\n")
                elif state and not cls2:
                    end = time.process_time()
                    state = False 
                    if mainobject2 < 0:
                        print("forward")
                        send_command_to_arduino("F\n")
                        time.sleep(2*(1.2+end-start))
                        print("turnleft")
                        send_command_to_arduino("L\n")
                        time.sleep(end-start)
                        print("stop")
                        send_command_to_arduino("S\n")
                    else :
                        print("forward")
                        send_command_to_arduino("F\n")
                        time.sleep(2*(1.2+end-start))
                        print("turnright")
                        send_command_to_arduino("R\n")
                        time.sleep(end-start)
                        print("stop")
                        send_command_to_arduino("S\n")
                elif not state:
                    if cls1:
                        if mainobject > 320 or mainobject < -320:
                            send_command_to_arduino("S\n")
                            print("stop")
                        elif mainobject < -100 and area < 61440:
                            print("fl")
                            send_command_to_arduino("FL\n")
                        elif mainobject > 100 and area < 61440:
                            print("fr")
                            send_command_to_arduino("FR\n")
                        elif mainobject < -100:
                            print("turnleft")
                            send_command_to_arduino("L\n")
                        elif mainobject > 100:
                            print("turnright")
                            send_command_to_arduino("R\n")
                        elif area < 50000:
                            print("forward")
                            send_command_to_arduino("F\n")
                        elif area < 61440:
                            print("sforward")
                            send_command_to_arduino("SF\n")
                        else:
                            send_command_to_arduino("S\n")
                            print("stop")
                    elif cls4:
                        print("forward")
                        send_command_to_arduino("F\n")
                    else:
                        send_command_to_arduino("S\n")
                        print("stop")
            elif state:
                end = time.process_time()
                state = False 
                if mainobject2 < 0:
                    print("forward")
                    send_command_to_arduino("F\n")
                    time.sleep(2*(1.2+end-start))
                    print("turnleft")
                    send_command_to_arduino("L\n")
                    time.sleep(end-start)
                    print("stop")
                    send_command_to_arduino("S\n")
                else :
                    print("forward")
                    send_command_to_arduino("F\n")
                    time.sleep(2*(1.2+end-start))
                    print("turnright")
                    send_command_to_arduino("R\n")
                    time.sleep(end-start)
                    print("stop")
                    send_command_to_arduino("S\n")
            else:
                print("stop")
                send_command_to_arduino("S\n")
            # Print time (inference + NMS)
            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.75, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7-tiny.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
    
