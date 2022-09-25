# use pip install -r requirements.txt to install dependencies
# github repo: https://github.com/davh565/ENGMT558-YOLOv4-DEEPSORT

# code is forked from https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3 and
# heavily modified. Code from https://github.com/theAIGuysCode/yolov4-deepsort
# and https://github.com/theAIGuysCode/yolov4-custom-functions was also referenced
# and adapted to fit the needs of this project.

# To use, place video file in /input/. Results will appear in 
# /output/<original_filename>/
# run ENGMT558_Assignment3.py

from object_tracker import *
import shutil

print("_________________________________________________________________________________________________________")
print("Loading YOLOv4 Model")
print(" ")
yolo = Load_Yolo_model()
print("_________________________________________________________________________________________________________")
print("Creating Directories")
print(" ")
filenames = os.listdir('./input')
for filename in filenames:
    # name, _ = os.path.splitext(filename)
    file_path = filename.split('.')[0]+"/"
    crop_path = "./working/"+file_path+"crop/"
    plate_path = "./working/"+file_path+"plates/"
    rembg_path = "./working/"+file_path+"rembg/"
    output_path = "./output/"+file_path
    make_dirs(crop_path)
    make_dirs(rembg_path)
    make_dirs(plate_path)
    make_dirs(output_path)

    print("_________________________________________________________________________________________________________")
    print("Tracking Vehicles in "+filename)
    print(" ")
    df = Object_tracking(yolo,
                    filename.split(".")[0],
                    "./input/"+filename,
                    "./working/"+filename,
                    input_size=YOLO_INPUT_SIZE,
                    show=False, 
                    iou_threshold=0.1,
                    rectangle_colors=(0,0,255),
                    Track_only = ["Car", "Truck", "Bus", "Van"],
                    n_init = 3,
                    max_age=10,
                    skip_frames=9)

    # df = pd.read_csv(output_path+"df.csv")
    df['plate'] = "$no_plate$"
    df['colour'] = "$none$"
    print("_________________________________________________________________________________________________________")
    print("Reading Plates and Colours in "+filename)
    print(" ")
    for file in df.path.iteritems():
        df = detect_plate(yolo,df,crop_path+file[1] ,plate_path+file[1] , input_size=YOLO_INPUT_SIZE, show=False, CLASSES=YOLO_COCO_CLASSES, rectangle_colors=(255,0,0),score_threshold=0.1, iou_threshold=0.2)
        add_plate_timestamp(crop_path+file[1],df.timestamp[df.path == file[1]].values[0],df.plate[df.path == file[1]].values[0])
    df = df.drop_duplicates(subset='plate', keep="last")
    df = df[df.plate != "$no_plate$"]
    for file in df.path.iteritems():
        df = detect_plate(yolo,df,crop_path+file[1] ,plate_path+file[1] , input_size=YOLO_INPUT_SIZE, show=False, CLASSES=YOLO_COCO_CLASSES, rectangle_colors=(255,0,0),score_threshold=0.1, iou_threshold=0.2)
        df.loc[df['path'] == file[1],["colour"]] = get_colour(3,crop_path+file[1],rembg_path+file[1])
        row = df.loc[df['path'] == file[1]]
        row = row.iloc[0]
        output_filename = (row.plate+"-"+row.colour+"-"+row.timestamp)
        shutil.copyfile(crop_path+file[1], output_path+output_filename+".png")
        
        pathname, _ = os.path.splitext(output_path+file[1])
    
    df.to_csv(output_path+"df.csv", index=False)
        
    print(df)
