import cv2
import tensorflow as tf
import keras
import mediapipe as mp
from tensorflow.keras import layers
import math
import numpy as np
import torch
from torchvision import transforms

kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

def conv3x3(x, out_planes, stride=1, name=None):
    x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
    return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=name)(x)

def basic_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
    out = layers.ReLU(name=f'{name}.relu1')(out)

    out = conv3x3(out, planes, name=f'{name}.conv2')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add(name=f'{name}.add')([identity, out])
    out = layers.ReLU(name=f'{name}.relu2')(out)

    return out

def make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
        ]

    x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
    for i in range(1, blocks):
        x = basic_block(x, planes, name=f'{name}.{i}')

    return x

def resnet(x, blocks_per_layer, num_classes=1000):
    x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal, name='conv1')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

    x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
    x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
    x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
    x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

    x = layers.GlobalAveragePooling2D(name='avgpool')(x)
    initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
    x = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)

    return x

def resnet18(x, **kwargs):
    return resnet(x, [2, 2, 2, 2], **kwargs)

def background_blur(img):
    tensor = preprocess(img)
    batch = tensor.unsqueeze(0)

    with torch.no_grad():
        output = segmentation(batch)['out'][0]
        predictions = output.argmax(0)

    output = predictions.byte().cpu().numpy()
    mask = cv2.resize(output, dsize=(img.shape[1], img.shape[0]))

    mask[mask != 0] = 1
    mask[mask == 0] = 0

    return mask

def emotion_detection(imgGRAY):
    # get bounding boxes for faces
    faces = face_cascade.detectMultiScale(imgGRAY, scaleFactor=1.1, minNeighbors=5, minSize=(250, 250))

    face_boxes = []
    emotions = np.zeros(len(faces))

    # pass each face through model
    for i in range(len(faces)):
        (x, y, w, h) = faces[i]

        face_img = imgGRAY[y: y + h, x: x + w]
        face_img = tf.expand_dims(face_img, axis=2)
        face_img = tf.expand_dims(face_img, axis=0)
        face_img = tf.image.resize(face_img, (48, 48))
        face_img /= 255.0

        pred = face_model.predict(face_img, verbose = False)
        emotions[i] = np.argmax(pred)
        face_boxes.append([(x, y), (x + w, y + h)])

    return face_boxes, emotions

def asl(imgRGB, imgGRAY):
    results = hands.process(imgRGB)
    hand_boxes = []
    preds = []
    h, w = imgGRAY.shape
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Initialize min and max coordinates for bounding box
            min_x, min_y = w, h
            max_x, max_y = 0, 0

            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Update min and max coordinates
                min_x = min(min_x, cx)
                min_y = min(min_y, cy)
                max_x = max(max_x, cx)
                max_y = max(max_y, cy)

            hand_boxes.append([(min_x, min_y), (max_x, max_y)])

            padding = 100

            min_x -= padding
            min_y -= padding
            max_x += padding
            max_y += padding

            # ensure hand is in frame
            min_x = max(min_x, 0)
            min_y = max(min_y, 0)
            max_x = min(max_x, w)
            max_y = min(max_y, h)

            hand_img = imgGRAY[min_y: max_y, min_x: max_x]
            hand_img = tf.expand_dims(hand_img, axis=2)
            hand_img = tf.expand_dims(hand_img, axis=0)
            hand_img = tf.image.resize_with_pad(hand_img, 28, 28)

            pred = asl_model.predict(hand_img, verbose = False)
            preds.append(pred)
    return hand_boxes, np.array(preds)


if __name__ == "__main__":
    # asl

    asl_inputs = layers.Input(shape=(28, 28, 1))
    asl_outputs = resnet18(asl_inputs, num_classes=25)
    asl_model = keras.Model(asl_inputs, asl_outputs)

    asl_model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate = 0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    asl_model.load_weights('asl_weights/')

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    # emotion detection

    face_inputs = layers.Input(shape=(48, 48, 1))
    face_outputs = resnet18(face_inputs, num_classes=10)
    face_model = keras.Model(face_inputs, face_outputs)

    face_model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate = 0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    face_model.load_weights('face_weights/')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    emotions_list = np.array(["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt", "unknown", "NF"])
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

    # background blur

    segmentation = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    segmentation.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),  # Convert NumPy array to PIL Image
        transforms.Resize((256, 256)),  # Resize the image to match model input
        transforms.ToTensor(),  # Convert PIL Image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgGRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = background_blur(imgRGB)

        face_boxes, emotions = emotion_detection(imgGRAY)
        hand_boxes, letters = asl(imgRGB, imgGRAY)

        blur_padding = 25
        for hand in hand_boxes:
            mask[hand[0][0] - blur_padding:hand[0][1] + blur_padding, hand[1][0] - blur_padding:hand[1][1] + blur_padding] = 1
        blur = cv2.blur(img,(40,40),0)
        img[mask == 0] = blur[mask == 0]

        for hand, label, color in zip(hand_boxes, letters, colors):
            mask[hand[0][0]:hand[0][1], hand[1][0]:hand[1][1]] = 1
            cv2.rectangle(img, hand[0], hand[1], color, 3)
            cv2.putText(img, str(np.argmax(letters)), (hand[0][0], hand[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

        for face, emotion, color in zip(face_boxes, emotions, colors):
            cv2.rectangle(img, face[0], face[1], color, 3)
            cv2.putText(img, emotions_list[int(emotion)], (face[0][0], face[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)


        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()